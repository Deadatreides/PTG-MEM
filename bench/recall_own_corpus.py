# -*- coding: utf-8 -*-
r"""Полнота на собственном корпусе: находится ли своя же мысль своими словами.

ЗАЧЕМ. Обещание «свежая сессия стартует полной, не упустив ни одной вашей мысли»
до сих пор не подкреплено ни одним числом. Здесь оно превращается в замер.

КАК УСТРОЕНА РАЗМЕТКА, И ПОЧЕМУ ЕЙ МОЖНО ВЕРИТЬ. Разметку не сочиняем: берём
файлы памяти проекта — это выжимки, написанные автором по ходу работ, у каждой
есть **опорное число или редкий термин** (28.43, ERROR_USER_MAPPED_FILE,
MYC_OBS_TEMP…). Запрос — человеческое описание из индекса памяти, БЕЗ опоры.
Попадание — если в выдаче есть атом, содержащий опору буквально.

То есть меряется ровно то, что обещано: спросил своими словами — вернулась своя
мысль. Качество ответа при этом НЕ меряется, только находимость.

ЧЕСТНОСТЬ ВЫБОРКИ. В замер берутся только те опоры, которые **вправду есть в
архиве**. Если факта в корпусе нет (память написана после даты архива), полнота
для него не определена, и он отбрасывается до начала — иначе мы мерили бы
собственную забывчивость и выдавали её за промах поиска.

ТРИ СПОСОБА, ОДНА ВЫБОРКА:

  1. **плоский косинус** — то, что делает обычный RAG;
  2. **branch-first** — сначала ветви по центроидам, потом атомы внутри них
     (то, чем PTG отличается на поиске);
  3. **граф** — плоский косинус плюс расширение по типизированным рёбрам.

Запрос эмбеддится ТОЙ ЖЕ моделью, которой собран архив (Qwen3-Embedding-4B), в
том же процессе через llama_cpp — сервер не нужен.

Запуск:  python recall_own_corpus.py [top_k] [сколько_запросов]
"""
from __future__ import annotations

import collections
import io
import json
import os
import re
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SNAP = os.path.join(ROOT, "trace-probe", "ptg", "_snapshot_run")
STORE = os.path.join(SNAP, "store_text_v3", ".ptg")
GGUF = os.path.join(ROOT, r"trace-probe\models\Qwen3-Embedding-4B-GGUF"
                          r"\Qwen3-Embedding-4B-Q4_K_M.gguf")
MEM = os.environ.get("PTG_NOTES_DIR", "")  # каталог с собственными заметками

TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MAX_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# опора — число с десятичной частью, редкий ИДЕНТИФИКАТОР или термин в бэктиках
ANCHOR = re.compile(r"`([^`]{4,40})`|\b(\d+\.\d+)\b|\b([A-Z][A-Z_0-9]{5,})\b")
STOP_ANCHOR = {"MEMORY", "CLAUDE", "PYTHON"}


def load_queries():
    """Из индекса памяти: описание -> запрос, опора -> чем проверять попадание."""
    idx = io.open(os.path.join(MEM, "MEMORY.md"), encoding="utf-8").read()
    out = []
    for line in idx.splitlines():
        m = re.match(r"^- \[([^\]]+)\]\(([^)]+)\)\s+—\s+(.+)$", line.strip())
        if not m:
            continue
        title, fname, desc = m.group(1), m.group(2), m.group(3)
        anchors = []
        for a in ANCHOR.finditer(desc):
            val = a.group(1) or a.group(2) or a.group(3)
            if val and val not in STOP_ANCHOR and not val.isdigit():
                anchors.append(val)
        if not anchors:
            continue
        # запрос — описание БЕЗ опор, чтобы не искать по подстроке
        query = desc
        for a in anchors:
            query = query.replace(a, " ")
        query = re.sub(r"\s+", " ", query).strip(" ;—,.")
        if len(query) < 25:
            continue
        out.append({"название": title, "файл": fname, "запрос": query,
                    "опоры": anchors})
    return out


def main():
    t0 = time.time()
    with io.open(os.path.join(STORE, "meta_tree.json"), encoding="utf-8") as f:
        meta = json.load(f)
    order = meta["id_order"]
    with io.open(os.path.join(STORE, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    with io.open(os.path.join(STORE, "edges.json"), encoding="utf-8") as f:
        edges = json.load(f)
    V = np.load(os.path.join(STORE, "faiss.index.npy")).astype("float32")
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    print("архив: %d атомов, %d рёбер, векторы %s, чтение %.0f с"
          % (len(order), len(edges), V.shape, time.time() - t0), flush=True)

    texts = [(nodes.get(nid, {}).get("text") or "") for nid in order]
    low = [t.lower() for t in texts]

    # --- разметка ---------------------------------------------------------
    qs = load_queries()
    print("выжимок памяти с опорой: %d" % len(qs))
    kept = []
    for q in qs:
        present = [a for a in q["опоры"]
                   if any(a.lower() in t for t in low)]
        if present:
            q["опоры"] = present
            kept.append(q)
    print("из них опора ВПРАВДУ есть в архиве: %d  (остальные отброшены — "
          "факт написан позже архива)" % len(kept), flush=True)
    if MAX_Q:
        kept = kept[:MAX_Q]

    # --- ветви ------------------------------------------------------------
    pos = {nid: i for i, nid in enumerate(order)}
    by_branch = collections.defaultdict(list)
    for nid in order:
        b = nodes.get(nid, {}).get("branch")
        if b:
            by_branch[b].append(pos[nid])
    bids = list(by_branch)
    C = np.zeros((len(bids), V.shape[1]), dtype="float32")
    for j, b in enumerate(bids):
        v = V[by_branch[b]].mean(axis=0)
        n = np.linalg.norm(v)
        C[j] = v / n if n > 0 else v
    print("ветвей: %d" % len(bids), flush=True)

    adj = collections.defaultdict(set)
    for e in edges:
        a, b = pos.get(e["from"]), pos.get(e["to"])
        if a is not None and b is not None:
            adj[a].add(b)
            adj[b].add(a)

    # --- эмбеддер ---------------------------------------------------------
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(model_path=GGUF, embedding=True, n_ctx=512, n_batch=512,
                n_ubatch=512, n_gpu_layers=0, verbose=False)
    print("эмбеддер поднят за %.0f с, dim=%d" % (time.time() - t0, llm.n_embd()), flush=True)

    def embed(text):
        r = llm.create_embedding(text[:1500])
        v = np.asarray(r["data"][0]["embedding"], dtype="float32")
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # --- три способа ------------------------------------------------------
    def flat(qv, k):
        return list(np.argsort(-(V @ qv))[:k])

    def branch_first(qv, k, top_b=8):
        bidx = np.argsort(-(C @ qv))[:top_b]
        cand = []
        for j in bidx:
            cand.extend(by_branch[bids[int(j)]])
        if not cand:
            return flat(qv, k)
        cand = np.array(sorted(set(cand)))
        s = V[cand] @ qv
        return list(cand[np.argsort(-s)][:k])

    def graph(qv, k, seed_n=3):
        seeds = list(np.argsort(-(V @ qv))[:seed_n])
        pool = set(seeds)
        for s in seeds:
            pool |= adj.get(s, set())
        pool = np.array(sorted(pool))
        sc = V[pool] @ qv
        return list(pool[np.argsort(-sc)][:k])

    methods = {"плоский косинус": flat, "branch-first": branch_first, "граф": graph}
    hits = {m: 0 for m in methods}
    ranks = {m: [] for m in methods}

    t0 = time.time()
    for i, q in enumerate(kept, 1):
        qv = embed(q["запрос"])
        for name, fn in methods.items():
            idxs = fn(qv, TOP_K)
            got = None
            for r, ix in enumerate(idxs):
                t = low[ix]
                if any(a.lower() in t for a in q["опоры"]):
                    got = r + 1
                    break
            if got:
                hits[name] += 1
                ranks[name].append(got)
        if i % 5 == 0:
            print("  %d/%d, %.0f с" % (i, len(kept), time.time() - t0), flush=True)

    n = len(kept)
    print("\n%-20s %10s %12s %14s" % ("способ", "recall@%d" % TOP_K, "попаданий", "медиана ранга"))
    for name in methods:
        med = int(np.median(ranks[name])) if ranks[name] else 0
        print("%-20s %9.1f%% %12s %14s"
              % (name, 100.0 * hits[name] / max(n, 1), "%d/%d" % (hits[name], n),
                 med if med else "—"))

    with io.open(os.path.join(HERE, "out", "recall_own.json"), "w", encoding="utf-8") as f:
        json.dump({"top_k": TOP_K, "запросов": n,
                   "recall": {m: round(100.0 * hits[m] / max(n, 1), 1) for m in methods},
                   "медиана_ранга": {m: (int(np.median(ranks[m])) if ranks[m] else None)
                                     for m in methods},
                   "промахи_у_всех": [q["название"] for q in kept]
                   if all(h == 0 for h in hits.values()) else []},
                  f, ensure_ascii=False, indent=1)
    print("\nзаписано: PTG_V5/out/recall_own.json")


if __name__ == "__main__":
    main()
