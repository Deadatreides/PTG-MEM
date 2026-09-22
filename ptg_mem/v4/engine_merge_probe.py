# -*- coding: utf-8 -*-
r"""PTG_V4/engine_merge_probe.py — опыт, проверяющий правила слияния движков.

ВОПРОС. Базы, проэмбежженные разными моделями, несравнимы поатомно: косинус из
одного пространства ничего не значит в другом, а пороги графтинга откалиброваны
под конкретное распределение косинусов (ptg_core останавливает сборку при смене
модели — ПАТЧ 18). Значит слияние возможно только слоями. Этот модуль меряет,
что слияние даёт и чего стоит, на ОДНОМ множестве атомов с двумя движками:

  * движок A — Qwen3-Embedding-4B, векторы берутся из самого архива (не
    пересчитываются: сравнивать надо ровно то, на чём архив живёт);
  * движок B — Jasper-Token-Compression-600M, считается здесь же.

ЧТО ПРОВЕРЯЕТСЯ.

1. РАЗДЕЛЯЮЩАЯ СИЛА (кому верить). AUC разделения пар «из одного файла» против
   «из разных». Принадлежность файлу внешняя обеим моделям — ветки для этой роли
   не годятся, они построены векторами A и к ним предвзяты.

2. ВЫИГРЫШ ОТ ОБЪЕДИНЕНИЯ (ради чего сливать). Слой близости строится отдельно
   по каждому движку правилом из sandwich_graft, затем берётся объединение.
   Объединение монотонно — рёбра только добавляются, — поэтому средняя степень
   растёт, а условие режима `d >> ln n` становится ДОСТИЖИМЕЕ, не труднее.
   Проверяется главное: падает ли доля изолированных ниже, чем у каждого движка
   по отдельности, и к нулевой модели `exp(-c)`.

3. ПЕРЕНОСИМОСТЬ ПРОСТРАНСТВ (можно ли вообще в общие координаты). Один и тот же
   текст, посчитанный обоими движками, — это якорь. По якорям строится
   ортогональное преобразование Прокруста B -> A (оно сохраняет косинусную
   геометрию, в отличие от произвольной линейной подгонки). Качество меряется на
   ОТЛОЖЕННЫХ якорях, которых в подгонке не было. Без этого «векторы совмещены»
   было бы верой, а не измерением.

Запуск:  python engine_merge_probe.py [сколько_атомов] [store]
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SNAP = os.path.join(ROOT, "trace-probe", "ptg", "_snapshot_run")
sys.path.insert(0, HERE)

import sandwich_graft                                   # noqa: E402

JASPER = os.environ.get("PTG_ENGINE_B", "")  # второй движок-эмбеддер
MAX_CHARS = 6000
N_ATOMS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
STORE_NAME = sys.argv[2] if len(sys.argv) > 2 else "store_text_v3"
STORE = os.path.join(SNAP, STORE_NAME, ".ptg")
OUT = os.path.join(HERE, "out")


# ---------------------------------------------------------------------------
def load_engine_a(n_atoms):
    t0 = time.time()
    with io.open(os.path.join(STORE, "meta_tree.json"), encoding="utf-8") as f:
        meta = json.load(f)
    order = meta["id_order"]
    print("архив: %s, модель %s, dim %s, узлов %d"
          % (STORE_NAME, meta.get("embedder"), meta.get("embedder_dim"), len(order)))
    vecs = np.load(os.path.join(STORE, "faiss.index.npy"))
    with io.open(os.path.join(STORE, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    pos = {nid: i for i, nid in enumerate(order)}

    byfile = {}
    for nid in order:
        nd = nodes.get(nid)
        if not nd or not (nd.get("text") or "") or not nd.get("file"):
            continue
        byfile.setdefault(nd["file"], []).append(nid)
    byfile = {k: v for k, v in byfile.items() if len(v) >= 4}
    cand = [nid for v in byfile.values() for nid in v]
    random.seed(20260920)
    random.shuffle(cand)
    sel = cand[:n_atoms]

    texts = [nodes[nid]["text"][:MAX_CHARS] for nid in sel]
    files = [nodes[nid]["file"] for nid in sel]
    A = vecs[[pos[nid] for nid in sel]].astype("float32")
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    print("взято атомов: %d из %d файлов, %.0f с" % (len(sel), len(byfile), time.time() - t0))
    return sel, texts, files, A


def load_engine_b(texts, batch=1, dtype="float32"):
    """Jasper.

    ДТИП. Модель обучена в bfloat16, а у bf16 диапазон много шире, чем у fp16.
    На Turing (1660 SUPER) bf16 нативно не поддержан, и попытка считать в fp16
    даёт переливы: замер 20.09.2026 отдал сплошные NaN — косинусы nan, AUC
    0.517 (ровно случайность), SVD не сошёлся. Поэтому по умолчанию float32,
    а результат проверяется сразу: молча испорченные векторы дороже падения.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    m = SentenceTransformer(JASPER, trust_remote_code=True, device="cuda",
                            model_kwargs={"dtype": getattr(torch, dtype)})
    m.max_seq_length = 3072
    B = m.encode(texts, batch_size=batch, show_progress_bar=False,
                 convert_to_numpy=True, normalize_embeddings=True)
    dt = time.time() - t0
    B = B.astype("float32")
    bad = int(np.isnan(B).any(axis=1).sum())
    print("Jasper (%s): %s за %.0f с (%.2f атом/с), NaN-векторов: %d"
          % (dtype, B.shape, dt, len(texts) / dt, bad))
    if bad:
        raise SystemExit("ОСТАНОВЛЕНО: %d векторов из %d — NaN. Считать по ним "
                         "нечего, все числа ниже были бы шумом." % (bad, len(B)))
    del m
    torch.cuda.empty_cache()
    return B


# ---------------------------------------------------------------------------
def auc_same_file(V, files, n_pairs=200_000, seed=7):
    rnd = random.Random(seed)
    idx = {}
    for i, fl in enumerate(files):
        idx.setdefault(fl, []).append(i)
    groups = [v for v in idx.values() if len(v) >= 2]
    pp = np.array([rnd.sample(rnd.choice(groups), 2) for _ in range(n_pairs // 2)])
    nn = []
    while len(nn) < n_pairs // 2:
        a, b = rnd.randrange(len(files)), rnd.randrange(len(files))
        if files[a] != files[b]:
            nn.append((a, b))
    nn = np.array(nn)
    sp = np.einsum("ij,ij->i", V[pp[:, 0]], V[pp[:, 1]])
    sn = np.einsum("ij,ij->i", V[nn[:, 0]], V[nn[:, 1]])
    s = np.sort(sn)
    lo = np.searchsorted(s, sp, side="left")
    hi = np.searchsorted(s, sp, side="right")
    auc = float((lo + 0.5 * (hi - lo)).mean() / len(s))
    return auc, float(sp.mean()), float(sn.mean())


def propose(V, ids, per_node=64):
    """Предложения движка: пары top-k, упорядоченные по ЕГО косинусу."""
    n = len(ids)
    S = V @ V.T
    np.fill_diagonal(S, -np.inf)
    k = min(per_node, n - 1)
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    cand = {}
    for i in range(n):
        for j in top[i]:
            j = int(j)
            a, b = (i, j) if i < j else (j, i)
            cand[(a, b)] = float(S[a, b])
    out = [(ids[a], ids[b], s) for (a, b), s in cand.items()]
    out.sort(key=lambda e: -e[2])
    return out


def proximity_edges(V, ids, budget, per_node=64):
    """Слой одного движка: предложения -> раскладка с бюджетом."""
    return sandwich_graft.graft(propose(V, ids, per_node), budget=budget)


def fused_edges(proposals, budget, rrf_k=60):
    """П4: сливаются ПРЕДЛОЖЕНИЯ по рангам, затем ОДНА раскладка с тем же
    бюджетом. Косинусы разных пространств несравнимы, ранг пары внутри своего
    движка — сравним. Потолок степени остаётся тот же, что у одного движка,
    поэтому сравнение честное: равный бюджет, равный hard_cap."""
    rank = {}
    for prop in proposals:
        for r, (u, v, _s) in enumerate(prop):
            rank.setdefault((u, v), []).append(r)
    fused = [(u, v, sum(1.0 / (rrf_k + r + 1) for r in rs))
             for (u, v), rs in rank.items()]
    fused.sort(key=lambda e: -e[2])
    return sandwich_graft.graft(fused, budget=budget)


def graph_stats(edges, ids):
    deg = {i: 0 for i in ids}
    simple = set()
    for u, v, _s in edges:
        key = (u, v) if u < v else (v, u)
        if key in simple:
            continue
        simple.add(key)
        deg[u] += 1
        deg[v] += 1
    n = len(ids)
    c = 2.0 * len(simple) / n
    iso = sum(1 for d in deg.values() if d == 0)
    return {"рёбер": len(simple), "средняя_степень": round(c, 2),
            "изолированных": iso, "доля_изолированных": round(100.0 * iso / n, 2),
            "по_нулевой_модели": round(100.0 * math.exp(-c), 2),
            "максимум": max(deg.values()) if deg else 0,
            "глубина_обхода": sandwich_graft.traversal_depth(n, c)}


def procrustes(A, B, train_frac=0.7, seed=11):
    """Ортогональное W: B @ W ~ A. Качество — на отложенных якорях."""
    n = len(A)
    rnd = np.random.default_rng(seed)
    perm = rnd.permutation(n)
    cut = int(train_frac * n)
    tr, te = perm[:cut], perm[cut:]
    M = B[tr].T @ A[tr]
    U, _s, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    Bp = B[te] @ W
    Bp /= np.maximum(np.linalg.norm(Bp, axis=1, keepdims=True), 1e-9)
    cos = np.einsum("ij,ij->i", Bp, A[te])
    # сохраняется ли ПОРЯДОК: сравнение матриц сходства на отложенных
    Sa = A[te] @ A[te].T
    Sb = Bp @ Bp.T
    iu = np.triu_indices(len(te), 1)
    r = float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])
    return {"якорей_обучение": int(cut), "якорей_проверка": int(len(te)),
            "косинус_проекции_к_цели": round(float(cos.mean()), 4),
            "медиана": round(float(np.median(cos)), 4),
            "корреляция_матриц_сходства": round(r, 4)}


# ---------------------------------------------------------------------------
def main():
    ids, texts, files, A = load_engine_a(N_ATOMS)
    B = load_engine_b(texts, batch=int(os.environ.get("BS", "1")),
                      dtype=os.environ.get("DTYPE", "float32"))
    n = len(ids)
    budget = sandwich_graft.suggest_budget(n)
    print("\nбюджет связей по правилу d >> ln n: %d (n=%d, ln n=%.2f)"
          % (budget, n, math.log(n)))

    print("\n--- 1. разделяющая сила (AUC «свой файл» против «чужой») ---")
    res = {}
    for name, V in (("Qwen3-4B", A), ("Jasper-600M", B)):
        auc, mp, mn = auc_same_file(V, files)
        res[name] = {"auc": auc, "свой": mp, "чужой": mn}
        print("%-12s AUC %.4f   косинус: свой %.3f, чужой %.3f, разрыв %.3f"
              % (name, auc, mp, mn, mp - mn))

    print("\n--- 2. слои близости и их объединение ---")
    pa, pb = propose(A, ids), propose(B, ids)
    ea = sandwich_graft.graft(pa, budget=budget)
    eb = sandwich_graft.graft(pb, budget=budget)
    ef = fused_edges([pa, pb], budget)
    union = {}
    for u, v, s in list(ea) + list(eb):
        key = (u, v) if u < v else (v, u)
        union[key] = max(union.get(key, -1.0), s)

    eu = [(u, v, s) for (u, v), s in union.items()]
    print("%-16s %8s %10s %14s %12s %8s %8s"
          % ("слой", "рёбер", "ср.степень", "изолированных", "нулевая", "макс", "обход"))
    stats = {}
    for name, e in (("A (Qwen 4B)", ea), ("B (Jasper)", eb),
                    ("сырое A ∪ B", eu), ("слияние рангов", ef)):
        st = graph_stats(e, ids)
        stats[name] = st
        print("%-16s %8d %10.2f %8d (%4.2f%%) %11.2f%% %8d %8d"
              % (name, st["рёбер"], st["средняя_степень"], st["изолированных"],
                 st["доля_изолированных"], st["по_нулевой_модели"], st["максимум"],
                 st["глубина_обхода"]))
    overlap = len(set((u, v) if u < v else (v, u) for u, v, _ in ea)
                  & set((u, v) if u < v else (v, u) for u, v, _ in eb))
    print("общих рёбер у двух движков: %d (%.1f %% от A)"
          % (overlap, 100.0 * overlap / max(1, len(ea))))

    print("\n--- 3. переносимость пространств (Прокруст на отложенных якорях) ---")
    pr = procrustes(A, B)
    for k, v in pr.items():
        print("  %-28s %s" % (k, v))

    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "engine_merge_probe.json"), "w", encoding="utf-8") as f:
        json.dump({"n": n, "budget": budget, "auc": res, "layers": stats,
                   "overlap_edges": overlap, "procrustes": pr}, f,
                  ensure_ascii=False, indent=1)
    print("\nзаписано: out/engine_merge_probe.json")


if __name__ == "__main__":
    main()
