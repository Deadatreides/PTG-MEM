# -*- coding: utf-8 -*-
"""In-process embedders. No server, no port, no API key.

Two backends:

  * ``st``   — sentence-transformers (Hugging Face models). The default for new
               projects is Qwen/Qwen3-Embedding-0.6B: multilingual, 1024-d, runs on
               CPU, same family as the 4B model the thresholds were calibrated on.
  * ``gguf`` — llama.cpp through llama-cpp-python, for GGUF files. This is how the
               reference archive (Qwen3-Embedding-4B-Q4_K_M, 2560-d) was built.

DTYPE IS NOT COSMETIC. Measured 20.09.2026 on a GTX 1660 SUPER (Turing): a model
trained in bfloat16 run in float16 returned 14 NaN vectors out of 40 — silently,
AUC 0.517 — and float32 on the same card was 3.7x FASTER (no tensor cores; fp16
only pays for casts). So float32 is the default, and every batch is checked for
NaN: a poisoned vector costs more than a crash.

IDLE UNLOAD. A memory daemon lives for days; a 4B model holds 5 GB of VRAM that the
user wants back for their own models. The model loads on first use and is released
after ``idle_unload_seconds`` without requests.
"""
from __future__ import annotations

import gc
import os
import threading
import time

import numpy as np

EMBED_MAX_CHARS = 6000        # same cut as the graph engine: longer texts are truncated


class EmbedderError(RuntimeError):
    pass


class BaseEmbedder:
    backend = "base"

    def __init__(self, model: str, idle_unload_seconds: float = 600, **opts):
        self.model = model
        self.opts = opts
        self.idle_unload_seconds = float(idle_unload_seconds)
        self.dim: int | None = opts.get("dim")
        self._lock = threading.RLock()
        self._last_used = 0.0
        self._loaded_at = 0.0
        self._error: str | None = None
        self.stats = {"texts": 0, "seconds": 0.0, "loads": 0, "unloads": 0}

    # to be provided by backends
    def _load(self):
        raise NotImplementedError

    def _unload(self):
        raise NotImplementedError

    def _is_loaded(self) -> bool:
        raise NotImplementedError

    def _encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    # --- public ----------------------------------------------------------------
    def embed(self, texts) -> np.ndarray:
        texts = [str(t)[:EMBED_MAX_CHARS] for t in texts]
        if not texts:
            return np.zeros((0, self.dim or 0), dtype="float32")
        with self._lock:
            if not self._is_loaded():
                t0 = time.time()
                try:
                    self._load()
                except Exception as ex:          # noqa: BLE001 — surfaced in status()
                    self._error = "load failed: %s" % ex
                    raise EmbedderError(self._error) from ex
                self._loaded_at = time.time()
                self.stats["loads"] += 1
                self.stats["last_load_seconds"] = round(time.time() - t0, 2)
            t0 = time.time()
            out = np.asarray(self._encode(texts), dtype="float32")
            if out.ndim == 1:
                out = out.reshape(1, -1)
            bad = int(np.isnan(out).any(axis=1).sum())
            if bad:
                raise EmbedderError("%d of %d vectors are NaN (backend %s, model %s). On GPUs "
                                    "without native bfloat16 run the model in float32."
                                    % (bad, len(out), self.backend, self.model))
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1.0, norms)
            self.dim = int(out.shape[1])
            self._last_used = time.time()
            self.stats["texts"] += len(texts)
            self.stats["seconds"] += time.time() - t0
            self._error = None
            return out

    def maybe_unload(self) -> bool:
        """Called periodically by the daemon. Returns True if it freed the model."""
        if self.idle_unload_seconds <= 0:
            return False
        if not self._lock.acquire(blocking=False):
            return False                         # busy right now — not idle
        try:
            if self._is_loaded() and time.time() - self._last_used > self.idle_unload_seconds:
                self._unload()
                gc.collect()
                try:
                    import torch                  # noqa: WPS433
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:                 # noqa: BLE001
                    pass
                self.stats["unloads"] += 1
                return True
            return False
        finally:
            self._lock.release()

    def unload(self):
        with self._lock:
            if self._is_loaded():
                self._unload()
                gc.collect()
                self.stats["unloads"] += 1

    def status(self) -> dict:
        return {"backend": self.backend, "model": self.model, "dim": self.dim,
                "loaded": self._is_loaded(),
                "idle_seconds": round(time.time() - self._last_used, 1) if self._last_used else None,
                "idle_unload_seconds": self.idle_unload_seconds,
                "error": self._error, **{k: (round(v, 2) if isinstance(v, float) else v)
                                        for k, v in self.stats.items()}}

    # --- the interface ptg_core.Archive expects from its embedder -----------------
    def test_connection(self, preferred_model=None) -> bool:
        try:
            self.embed(["ping"])
            return True
        except Exception as ex:                  # noqa: BLE001
            self._error = str(ex)
            return False

    @property
    def probe_error(self):
        return self._error


class STEmbedder(BaseEmbedder):
    """sentence-transformers backend."""
    backend = "st"

    def __init__(self, model: str, device: str | None = None, dtype: str = "float32",
                 batch_size: int = 16, max_seq_length: int = 2048, **kw):
        super().__init__(model, **kw)
        self.device = device
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.max_seq_length = int(max_seq_length)
        self._m = None

    def _is_loaded(self):
        return self._m is not None

    def _load(self):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as ex:
            raise EmbedderError("sentence-transformers is not installed: pip install "
                                "\"ptg-mem[st]\"") from ex
        dev = self.device or ("cuda" if torch.cuda.is_available() else
                              ("mps" if getattr(torch.backends, "mps", None) and
                               torch.backends.mps.is_available() else "cpu"))
        dt = self.dtype if dev != "cpu" else "float32"
        self._m = SentenceTransformer(self.model, device=dev, trust_remote_code=True,
                                      model_kwargs={"torch_dtype": getattr(torch, dt)})
        self._m.max_seq_length = self.max_seq_length
        self.device = dev

    def _unload(self):
        self._m = None

    def _encode(self, texts):
        return self._m.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)


def _win_llama_boot() -> None:
    """Make a CUDA build of llama-cpp-python importable on Windows without the CUDA
    Toolkit. The runtime DLLs it links against ship inside torch/lib; adding the
    folder to the DLL search path is NOT enough — ggml-cuda.dll only loads after
    cudart/cublasLt/cublas are preloaded explicitly (verified: without the preload
    ERROR_MOD_NOT_FOUND). A CUDA_PATH pointing to a missing folder breaks
    add_dll_directory, so a dangling one is dropped."""
    if os.name != "nt":
        return
    import ctypes
    import importlib.util
    cp = os.environ.get("CUDA_PATH")
    if cp and not os.path.isdir(cp):
        os.environ.pop("CUDA_PATH", None)
    dirs = {}
    for mod, sub in (("torch", "lib"), ("llama_cpp", "lib")):
        spec = importlib.util.find_spec(mod)
        if spec and spec.submodule_search_locations:
            d = os.path.join(list(spec.submodule_search_locations)[0], sub)
            if os.path.isdir(d):
                dirs[mod] = d
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
    for mod, names in (("torch", ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll")),
                       ("llama_cpp", ("ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll"))):
        for name in names:
            p = os.path.join(dirs.get(mod, ""), name)
            if dirs.get(mod) and os.path.exists(p):
                try:
                    ctypes.CDLL(p, winmode=0)
                except OSError:
                    pass                          # CPU build without ggml-cuda.dll is fine


class GGUFEmbedder(BaseEmbedder):
    """llama.cpp backend (llama-cpp-python), one text per call.

    Measured on the reference machine: batching 1/4/16/32 texts gives the same
    ~190 tok/s, while one-by-one keeps progress honest and isolates errors.
    n_ctx 3072 + flash attention held 205-219 tok/s against 166-218 at 4096 on a
    6 GB card (at 4096 buffers push compute out of VRAM)."""
    backend = "gguf"

    def __init__(self, model: str, n_ctx: int = 3072, n_gpu_layers: int = -1,
                 flash_attn: bool = True, **kw):
        super().__init__(model, **kw)
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.flash_attn = bool(flash_attn)
        self._llm = None

    def _is_loaded(self):
        return self._llm is not None

    def _load(self):
        if not os.path.isfile(self.model):
            raise EmbedderError("GGUF file not found: %s" % self.model)
        try:
            _win_llama_boot()
            from llama_cpp import Llama
        except ImportError as ex:
            raise EmbedderError("llama-cpp-python is not installed: pip install "
                                "\"ptg-mem[gguf]\"") from ex
        self._llm = Llama(model_path=self.model, embedding=True, n_ctx=self.n_ctx,
                          n_batch=self.n_ctx, n_ubatch=self.n_ctx,
                          n_gpu_layers=self.n_gpu_layers, flash_attn=self.flash_attn,
                          verbose=False)

    def _unload(self):
        llm, self._llm = self._llm, None
        try:
            llm.close()
        except Exception:                        # noqa: BLE001
            pass
        del llm

    def _one(self, text: str):
        max_tokens = self.n_ctx - 64
        # Tokenising only to check the length costs 0.1-0.2 s per text; 3000
        # characters cannot exceed the window in any script (Cyrillic ~2.2 chars/token).
        if len(text) > 3000:
            toks = self._llm.tokenize(text.encode("utf-8"), add_bos=True, special=False)
            if len(toks) > max_tokens:
                keep = int(len(text) * max_tokens / len(toks)) - 16
                text = text[:max(1, keep)]
        r = self._llm.create_embedding(text)
        return r["data"][0]["embedding"]

    def _encode(self, texts):
        return np.asarray([self._one(t) for t in texts], dtype="float32")


class HashEmbedder(BaseEmbedder):
    """Deterministic bag-of-words vectors. For tests and CI only: no model, no
    download, same text -> same vector, shared words -> positive cosine."""
    backend = "hash"

    def __init__(self, model: str = "hash-256", dim: int = 256, **kw):
        super().__init__(model, **kw)
        self.dim = int(dim)
        self._on = False

    def _is_loaded(self):
        return self._on

    def _load(self):
        self._on = True

    def _unload(self):
        self._on = False

    def _encode(self, texts):
        import hashlib
        import re as _re
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in _re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
            out[i, 0] += 0.5                          # never an all-zero vector
        return out


def make_embedder(spec: dict, idle_unload_seconds: float = 600) -> BaseEmbedder:
    spec = dict(spec or {})
    backend = spec.pop("backend", "st")
    model = spec.pop("model", None)
    if backend == "hash":
        return HashEmbedder(model or "hash-256", idle_unload_seconds=idle_unload_seconds, **spec)
    if not model:
        raise EmbedderError("embedder spec has no model")
    if backend == "gguf":
        return GGUFEmbedder(model, idle_unload_seconds=idle_unload_seconds, **spec)
    if backend == "st":
        return STEmbedder(model, idle_unload_seconds=idle_unload_seconds, **spec)
    raise EmbedderError("unknown embedder backend: %s" % backend)
