# -*- coding: utf-8 -*-
r"""Ревизия метадерева, замер 4: готовые адаптеры Drift-Adapter против нашего ортогонального правила.

Постановка — та же, что в ptg_mem/v4/seam_rules.py (21.09.2026): карточки кодового графа store_code_v2; движок A —
Jasper-600M (векторы из хранилища), движок B — Qwen3-Embedding-4B Q4_K_M в процессе через llama_cpp; 70 % — пары на
обучение, 30 % — проверка. Мера — совпадение 10 соседей: запрос, переведённый A → B, ищет среди векторов B
проверочной части, и его соседи сравниваются с родными соседями той же карточки в B. Тогда было: без перевода
0,4989, ортогональный Прокруст 0,6489.

Адаптеры — из Drift-Adapter (Vejendla, EMNLP 2025): ортогональный Прокруст, линейный (гребневая регрессия),
малоранговый линейный и остаточный MLP с косинусной потерей.

    .venv-rev\Scripts\python bench\drift_adapter_probe.py [карточек]
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"H:\работа\сайты\идеи\память\for codex (1.1) — копия"
STORE = os.path.join(BASE, r"trace-probe\ptg\_snapshot_run\store_code_v2\.code_ptg")
GGUF = os.path.join(BASE, r"trace-probe\models\Qwen3-Embedding-4B-GGUF\Qwen3-Embedding-4B-Q4_K_M.gguf")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEED = 20260921
CACHE = os.path.join(HERE, "out", f"drift_B_{N}.npz")


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def load_a(n):
    with io.open(os.path.join(STORE, "meta_tree.json"), encoding="utf-8") as f:
        order = json.load(f)["id_order"]
    V = np.load(os.path.join(STORE, "faiss.index.npy"))
    with io.open(os.path.join(STORE, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    rng = np.random.default_rng(SEED)
    idx = [i for i, nid in enumerate(order) if nid in nodes and (nodes[nid].get("text") or "").strip()]
    idx = list(rng.permutation(idx))[:n]
    return unit(V[idx].astype("float32")), [nodes[order[i]]["text"] for i in idx]


def load_b(texts):
    if os.path.exists(CACHE):
        return np.load(CACHE)["B"]
    from llama_cpp import Llama
    llm = Llama(model_path=GGUF, embedding=True, n_ctx=512, n_batch=512, n_ubatch=512, n_gpu_layers=0, verbose=False)
    t0, out = time.time(), []
    for i, t in enumerate(texts):
        out.append(np.asarray(llm.create_embedding(t[:2000])["data"][0]["embedding"], dtype="float32"))
        if (i + 1) % 500 == 0:
            print(f"    B: {i + 1}/{len(texts)}, {(i + 1) / (time.time() - t0):.1f} карточек/с", flush=True)
    B = unit(np.stack(out))
    np.savez(CACHE, B=B)
    return B


def agreement(P, T, k=10):
    """Запрос P ищет среди T; его соседи против родных соседей той же точки в T (без неё самой)."""
    sp, st = unit(P) @ T.T, T @ T.T
    np.fill_diagonal(sp, -np.inf)
    np.fill_diagonal(st, -np.inf)
    a = np.argpartition(-sp, k, axis=1)[:, :k]
    b = np.argpartition(-st, k, axis=1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def own_space(X, Y, k=10):
    """Без перевода: соседи каждого в своём пространстве — потолок согласия движков."""
    def top(M):
        S = M @ M.T
        np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, axis=1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(top(X), top(Y))]))


def procrustes(A, B):
    U, _s, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return lambda X: X @ (U @ Vt)


def ridge(A, B, lam=1e-2, rank=None):
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=A.dtype), A.T @ B)
    if rank:
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        W = (U[:, :rank] * s[:rank]) @ Vt[:rank]
    return lambda X: X @ W


def mlp(A, B, hidden=1024, epochs=300, seed=SEED):
    """Остаточный MLP Drift-Adapter: f(x) = xW + g(x), косинусная потеря, останов по отложенной части."""
    import torch
    torch.manual_seed(seed)
    n = len(A)
    cut = int(0.85 * n)
    At, Bt = torch.tensor(A), torch.tensor(B)
    base = torch.nn.Linear(A.shape[1], B.shape[1], bias=False)
    with torch.no_grad():
        U, _s, Vt = np.linalg.svd(A[:cut].T @ B[:cut], full_matrices=False)
        base.weight.copy_(torch.tensor((U @ Vt).T))
    g = torch.nn.Sequential(torch.nn.Linear(A.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, B.shape[1]))
    with torch.no_grad():
        g[2].weight.mul_(0.01)
        g[2].bias.zero_()
    params = list(base.parameters()) + list(g.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    best, best_state = -1.0, None
    for _ in range(epochs):
        opt.zero_grad()
        pred = base(At[:cut]) + g(At[:cut])
        loss = 1 - torch.nn.functional.cosine_similarity(pred, Bt[:cut]).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            v = torch.nn.functional.cosine_similarity(base(At[cut:]) + g(At[cut:]), Bt[cut:]).mean().item()
        if v > best:
            best, best_state = v, ([p.detach().clone() for p in params])
    with torch.no_grad():
        for p, s in zip(params, best_state):
            p.copy_(s)

    def apply(X):
        with torch.no_grad():
            T = torch.tensor(X)
            return (base(T) + g(T)).numpy()
    return apply


def main():
    A, texts = load_a(N)
    print(f"A (Jasper, из хранилища): {A.shape}", flush=True)
    B = load_b(texts)
    print(f"B (Qwen3-4B): {B.shape}", flush=True)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(A))
    cut = int(0.7 * len(A))
    tr, te = perm[:cut], perm[cut:]
    rows = {"без перевода (каждый в своём)": own_space(A[te], B[te])}
    ways = {"Прокруст (наш, 21.09)": procrustes(A[tr], B[tr]), "линейный (гребневой)": ridge(A[tr], B[tr]),
            "малоранговый 256": ridge(A[tr], B[tr], rank=256), "остаточный MLP": mlp(A[tr], B[tr])}
    for name, f in ways.items():
        rows[name] = agreement(f(A[te]), B[te])
    print(f"\nпар на обучение {len(tr)}, проверка {len(te)}; совпадение 10 соседей:")
    for name, v in rows.items():
        print(f"  {name:<32} {v:.4f}")
    with io.open(os.path.join(HERE, "out", "drift_adapter_probe.json"), "w", encoding="utf-8") as f:
        json.dump({"n": len(A), "train": int(len(tr)), "test": int(len(te)), "top10": rows}, f, ensure_ascii=False, indent=1)
    print("записано: bench/out/drift_adapter_probe.json")


if __name__ == "__main__":
    main()
