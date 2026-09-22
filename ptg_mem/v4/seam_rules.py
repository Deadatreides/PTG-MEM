# -*- coding: utf-8 -*-
r"""PTG_V4/seam_rules.py — можно ли достраивать базу другой моделью.

ВОПРОС. База собрана одной моделью. Вторая модель даёт векторы в другом
пространстве, и `ptg_core` честно останавливает сборку при смене модели.
Обойти это можно только правилом конвертации, вычисленным по якорям — одним
и тем же текстам, посчитанным обеими моделями. Вопрос в том, существует ли
такое правило с приемлемым качеством, и если да — одно на всё пространство
или своё на каждый участок («шов»).

ЧТО УЖЕ ИЗМЕРЕНО. Глобальная ортогональная привязка (Прокруст) между
Qwen3-Embedding-4B и Jasper-600M на текстовом архиве, 20.09.2026: косинус
проекции к цели 0.7939, **корреляция матриц сходства 0.6531** при воротах
0.90. Не прошло. Порядок сходств — именно то, что нужно поиску и графтингу,
и он разрушается.

ГИПОТЕЗА ЭТОГО МОДУЛЯ. Одно преобразование на всё пространство — слишком
грубо: соответствие между моделями не обязано быть глобально ортогональным,
но локально, внутри смыслового кластера, может быть близко к нему. Тогда
правило считается не одно, а по одному на шов, и применяется по ближайшему
шву. Это и проверяется: развёртка по числу швов и по числу якорей на шов.

ЧТО ИЗМЕРЕНО 21.09.2026 (1200 карточек кодового графа; A — Jasper-600M, которым
собран `store_code_v2`, B — Qwen3-Embedding-4B Q4_K_M в том же процессе через
`llama_cpp`, без сервера; 840 якорей на обучение, 360 на проверку):

    швов   косинус  корреляция   top10   якорей на шов
      1     0.8507     0.4926    0.6489        840
      4     0.8103     0.5440    0.6228        220
     16     0.8102     0.5263    0.6331         41
     64     0.8386     0.5038    0.6458         10

    якорей  косинус  корреляция   top10          (один шов)
        64   0.6647     0.4926    0.4603
       210   0.7613     0.4926    0.5781
       420   0.8122     0.4926    0.6175
       840   0.8507     0.4926    0.6489

    согласие движков по top-10 БЕЗ конвертации: 0.4989

ГИПОТЕЗА О ШВАХ НЕ ПОДТВЕРДИЛАСЬ. Кусочные правила не бьют одно общее: при
четырёх и шестнадцати швах top-10 даже ниже, при шестидесяти четырёх — вровень.
Разрывность на границах съедает ровно то, что даёт локальность.

ЧТО ПОДТВЕРДИЛОСЬ. Конвертация всё же полезна, и видно насколько: без неё два
движка сходятся в половине соседей (0.4989), с одним общим ортогональным
правилом — в двух третях (0.6489). Плюс шестнадцать пунктов, и они настоящие.

ЧЕГО ОНА НЕ ДАЁТ. Ворота по сохранению порядка (корреляция ≥ 0.90) не проходит
ни один вариант: 0.49–0.54. Треть соседей после конвертации — чужие. Значит
дописывать в базу векторы другой модели как свои НЕЛЬЗЯ: пороги графтинга
(0.47/0.57/0.65) откалиброваны под распределение косинусов одного движка, а
треть промаха разрушит и ветвление, и возвраты.

ГДЕ ЭТО ЗАКОННО. Две трети — хорошая доля для ПРЕДЛОЖЕНИЯ кандидатов: правило
переносит запрос в чужое пространство, тот отдаёт кандидатов, а решение
принимается уже в родном. То есть конвертация годится как вход в поиск и не
годится как слияние. Это ровно то же разделение, что в METATREE-ENGINES.md:
ранг переносится, координата — нет.

ЯКОРЕЙ НУЖНО МНОГО. Кривая не выходит на полку к 840: 64 якоря дают 0.46,
840 — 0.65, и рост не кончился. «Усиленный шов» имеет смысл не как отдельное
правило на участок, а как ТРЕБОВАНИЕ К ЧИСЛУ ЯКОРЕЙ: общих текстов между двумя
базами нужны сотни, а не десятки.

ЧЕСТНАЯ ГРАНИЦА. Кусочная привязка по построению разрывна на границах швов:
две близкие точки по разные стороны границы получают разные преобразования.
Поэтому меряется не только косинус к цели, но и СОХРАНЕНИЕ ПОРЯДКА
(корреляция матриц сходства и совпадение top-k) — величина, которую
разрывность портит первой.

Запуск:  python seam_rules.py [сколько_карточек]
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
ROOT = os.path.dirname(HERE)
STORE = os.path.join(ROOT, r"trace-probe\ptg\_snapshot_run\store_code_v2\.code_ptg")
GGUF = os.path.join(ROOT, r"trace-probe\models\Qwen3-Embedding-4B-GGUF"
                          r"\Qwen3-Embedding-4B-Q4_K_M.gguf")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEED = 20260921


# ---------------------------------------------------------------------------
def load_engine_a(n: int):
    """Движок A — тот, которым база собрана. Векторы берутся ИЗ архива."""
    with io.open(os.path.join(STORE, "meta_tree.json"), encoding="utf-8") as f:
        order = json.load(f)["id_order"]
    V = np.load(os.path.join(STORE, "faiss.index.npy"))
    with io.open(os.path.join(STORE, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    rng = np.random.default_rng(SEED)
    idx = [i for i, nid in enumerate(order)
           if nid in nodes and (nodes[nid].get("text") or "").strip()]
    idx = list(rng.permutation(idx))[:n]
    texts = [nodes[order[i]]["text"] for i in idx]
    A = V[idx].astype("float32")
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    return A, texts


def load_engine_b(texts: list[str], n_ctx: int = 512, gpu_layers: int = 0):
    """Движок B — другая модель, в том же процессе (без сервера)."""
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(model_path=GGUF, embedding=True, n_ctx=n_ctx, n_batch=n_ctx,
                n_ubatch=n_ctx, n_gpu_layers=gpu_layers, verbose=False)
    print("  модель B поднята за %.0f с, dim=%d" % (time.time() - t0, llm.n_embd()))
    out = []
    t0 = time.time()
    for i, t in enumerate(texts):
        r = llm.create_embedding(t[:2000])
        out.append(np.asarray(r["data"][0]["embedding"], dtype="float32"))
        if (i + 1) % 500 == 0:
            print("    %d/%d, %.1f карточек/с" % (i + 1, len(texts), (i + 1) / (time.time() - t0)))
    B = np.stack(out)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    print("  движок B: %s за %.0f с" % (B.shape, time.time() - t0))
    return B


# ---------------------------------------------------------------------------
def procrustes(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Ортогональное W: src @ W ~ dst. Ортогональность обязательна —
    она сохраняет косинусную геометрию; произвольная линейная подгонка
    «улучшает» метрику, подстраиваясь под обучающие пары."""
    U, _s, Vt = np.linalg.svd(src.T @ dst, full_matrices=False)
    return U @ Vt


def kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = SEED):
    """Швы — это участки пространства. Кластеры считаются по движку A,
    потому что именно в его координатах приходит новая точка."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmax(X @ C.T, axis=1)
        for j in range(k):
            m = lab == j
            if m.any():
                v = X[m].mean(axis=0)
                nrm = np.linalg.norm(v)
                if nrm > 0:
                    C[j] = v / nrm
    return C, np.argmax(X @ C.T, axis=1)


def order_agreement(P: np.ndarray, T: np.ndarray, k: int = 10) -> float:
    """Совпадение top-k соседей: практическая мера того, переживёт ли
    конвертацию поиск. Корреляция сходств — та же величина мягче."""
    sp = P @ T.T
    st = T @ T.T
    np.fill_diagonal(sp, -np.inf)
    np.fill_diagonal(st, -np.inf)
    a = np.argpartition(-sp, k, axis=1)[:, :k]
    b = np.argpartition(-st, k, axis=1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def topk_overlap(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Согласие двух движков по составу top-k соседей на одном множестве.

    Координаты несравнимы, а СОСТАВ соседей — сравним: это множества
    одних и тех же карточек. Величина и есть потолок для любого правила
    конвертации: правило переносит координаты, а не создаёт согласие."""
    def top(M):
        S = M @ M.T
        np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, axis=1)[:, :k]
    a, b = top(X), top(Y)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def evaluate(P: np.ndarray, T: np.ndarray) -> dict:
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    cos = float(np.einsum("ij,ij->i", P, T).mean())
    iu = np.triu_indices(len(T), 1)
    corr = float(np.corrcoef((P @ P.T)[iu], (T @ T.T)[iu])[0, 1])
    return {"косинус": round(cos, 4), "корреляция": round(corr, 4),
            "top10": round(order_agreement(P, T), 4)}


def fit_seams(A_tr, B_tr, A_te, k: int):
    """Правило на шов: k-means по A, свой Прокруст внутри каждого шва."""
    if k == 1:
        W = procrustes(A_tr, B_tr)
        return A_te @ W, {0: len(A_tr)}
    C, lab = kmeans(A_tr, k)
    Ws, sizes = {}, {}
    glob = procrustes(A_tr, B_tr)
    for j in range(k):
        m = lab == j
        sizes[j] = int(m.sum())
        # шов, на котором якорей меньше размерности, переопределён: своего
        # правила у него нет, берём общее
        Ws[j] = procrustes(A_tr[m], B_tr[m]) if m.sum() >= 64 else glob
    lab_te = np.argmax(A_te @ C.T, axis=1)
    P = np.zeros((len(A_te), B_tr.shape[1]), dtype="float32")
    for j in range(k):
        m = lab_te == j
        if m.any():
            P[m] = A_te[m] @ Ws[j]
    return P, sizes


# ---------------------------------------------------------------------------
def main():
    print("движок A (тот, которым собрана база) — из архива")
    A, texts = load_engine_a(N)
    print("  %s, текст карточки: медиана %d симв."
          % (A.shape, sorted(len(t) for t in texts)[len(texts) // 2]))
    print("движок B (другая модель, в процессе)")
    B = load_engine_b(texts, gpu_layers=int(os.environ.get("GPU_LAYERS", "0")))

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(A))
    cut = int(0.7 * len(A))
    tr, te = perm[:cut], perm[cut:]
    A_tr, B_tr, A_te, B_te = A[tr], B[tr], A[te], B[te]
    print("\nякорей на обучение %d, на проверку %d" % (len(tr), len(te)))
    print("ворота: косинус >= 0.80, корреляция >= 0.90\n")

    print("%-8s %10s %12s %10s %14s" % ("швов", "косинус", "корреляция", "top10", "якорей на шов"))
    rows = {}
    for k in (1, 4, 16, 64):
        P, sizes = fit_seams(A_tr, B_tr, A_te, k)
        m = evaluate(P, B_te)
        med = int(np.median(list(sizes.values())))
        rows[k] = dict(m, якорей_на_шов=med)
        print("%-8d %10.4f %12.4f %10.4f %14d"
              % (k, m["косинус"], m["корреляция"], m["top10"], med))

    # сколько якорей нужно шву — развёртка по объёму обучения при одном шве
    print("\nсколько якорей нужно правилу (один шов):")
    print("%-10s %10s %12s %10s" % ("якорей", "косинус", "корреляция", "top10"))
    budget = {}
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        n_tr = max(64, int(frac * len(tr)))
        W = procrustes(A_tr[:n_tr], B_tr[:n_tr])
        m = evaluate(A_te @ W, B_te)
        budget[n_tr] = m
        print("%-10d %10.4f %12.4f %10.4f" % (n_tr, m["косинус"], m["корреляция"], m["top10"]))

    # Потолок: насколько два движка согласны в соседях САМИ ПО СЕБЕ, без
    # всякой конвертации. Выше этого никакое правило подняться не может —
    # оно переносит координаты, а не создаёт согласие.
    base = {"потолок_top10_согласия_движков": round(topk_overlap(A_te, B_te), 4)}
    print("\nпотолок: согласие движков по top-10 без конвертации = %.4f"
          % base["потолок_top10_согласия_движков"])
    out = {"n": int(N), "швы": rows, "бюджет_якорей": budget, "опора": base}
    with io.open(os.path.join(HERE, "out", "seam_rules.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nзаписано: PTG_V4/out/seam_rules.json")


if __name__ == "__main__":
    main()
