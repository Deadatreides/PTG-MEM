# -*- coding: utf-8 -*-
r"""code_ptg/local_embedder.py — эмбеддер внутри процесса, без LM Studio.

ЗАЧЕМ. Прежний путь шёл через HTTP на localhost:1234: сначала LM Studio, потом
собственный `embed_server.py`. Это единая точка отказа — без поднятого сервера
кодовый граф слепнет, — и ровно то место, на котором падал деплой. Для
шароварного распространения оно смертельно: никто не будет ставить отдельный
сервер моделей, чтобы попробовать индексатор.

Здесь модель живёт в том же процессе: ни порта, ни сервера, ни ключа, ни сети
после первой загрузки. Интерфейс тот же, что у `graph_engine.Embedder`
(`test_connection`, `embed`, `dim`), поэтому подменяется он без правок вызовов.

ДТИП — НЕ КОСМЕТИКА. Замер 20.09.2026 на GTX 1660 SUPER (Turing, compute 7.5):
модель, обученная в bfloat16, на fp16 выдала **14 NaN-векторов из 40**, и это
не упало, а тихо отравило выдачу — AUC 0.517, то есть ровно случайность. У bf16
диапазон много шире fp16, а нативного bf16 на Turing нет. При этом float32 там
же оказался **в 3.7 раза БЫСТРЕЕ** (11.79 против 3.22 текста/с): тензорных ядер
на Turing нет, счётного выигрыша fp16 не даёт, а приведения и переливы
оплачивает. Поэтому по умолчанию float32, а результат проверяется на NaN сразу:
молча испорченные векторы дороже падения.

ВЫБОР МОДЕЛИ. Карточка кода — короткий текст (`callable:function create_dump`
плюс докстринг, медиана 48 символов), и большая модель здесь не нужна. Для
распространения правильный умолчательный выбор — маленькая CPU-модель на
384 измерения: она ставится вместе с пакетом и не требует видеокарты, а
`graph_engine` уже держит 384 как запасную размерность. Путь к модели
задаётся снаружи, чтобы не зашивать в код ни конкретное имя, ни конкретный диск.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


class LocalEmbedder:
    """Тот же интерфейс, что у graph_engine.Embedder, но без сети."""

    def __init__(self, model: Optional[str] = None, device: Optional[str] = None,
                 dtype: str = "float32", batch_size: int = 32,
                 max_seq_length: int = 512):
        self.model_name = model or os.environ.get("CODE_PTG_MODEL", "")
        self.device = device or os.environ.get("CODE_PTG_DEVICE", "")
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.dim: Optional[int] = None
        self.backend = "local"
        self.model = self.model_name          # graph_engine логирует это поле
        self._m = None
        self._error: Optional[str] = None

    # ---------- загрузка ----------
    def _load(self):
        """Ленивая: импорт torch стоит секунды, а нужен он только при индексации."""
        if self._m is not None:
            return self._m
        if not self.model_name:
            raise RuntimeError(
                "не задана модель: передайте model=... или переменную окружения "
                "CODE_PTG_MODEL с путём к модели sentence-transformers")
        import torch
        from sentence_transformers import SentenceTransformer

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # на CPU float16 не считается быстрее и часто вовсе не поддержан
        dt = self.dtype if dev == "cuda" else "float32"
        self._m = SentenceTransformer(self.model_name, trust_remote_code=True,
                                      device=dev,
                                      model_kwargs={"dtype": getattr(torch, dt)})
        self._m.max_seq_length = self.max_seq_length
        self.device = dev
        return self._m

    # ---------- интерфейс Embedder ----------
    def test_connection(self, preferred_model: str | None = None) -> bool:
        """Проверка не сети, а того, что модель грузится и считает без NaN."""
        try:
            v = self.embed(["ping"])
        except Exception as ex:                     # noqa: BLE001 — причину отдаём наружу
            self._error = str(ex)
            return False
        ok = bool(len(v)) and not bool(np.isnan(v[0]).any())
        if not ok:
            self._error = "модель вернула NaN на пробном тексте"
        return ok

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        m = self._load()
        out = m.encode(list(texts), batch_size=self.batch_size,
                       show_progress_bar=False, convert_to_numpy=True,
                       normalize_embeddings=True).astype("float32")
        if out.ndim == 1:
            out = out.reshape(1, -1)
        bad = int(np.isnan(out).any(axis=1).sum())
        if bad:
            raise RuntimeError(
                "эмбеддер вернул %d NaN-векторов из %d (дтип %s, устройство %s). "
                "На картах без нативного bfloat16 это перелив fp16 — считайте в "
                "float32." % (bad, len(out), self.dtype, self.device))
        self.dim = int(out.shape[1])
        return [out[i] for i in range(out.shape[0])]

    # ---------- диагностика ----------
    def status(self) -> dict:
        return {"backend": self.backend, "model": self.model_name or None,
                "device": self.device or None, "dtype": self.dtype,
                "dim": self.dim, "загружена": self._m is not None,
                "ошибка": self._error}
