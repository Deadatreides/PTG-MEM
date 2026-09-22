# -*- coding: utf-8 -*-
r"""Разбор замера полноты: что не нашлось и что даёт граф сверх находки.

ДВА ВОПРОСА, НА КОТОРЫЕ ПЕРВЫЙ ЗАМЕР НЕ ОТВЕТИЛ.

1. **Какие именно мысли не возвращаются.** 70 % recall@10 значит, что три из
   десяти собственных выводов автора не находятся. Какие — важнее, чем сколько.

2. **Правильно ли вообще мерили граф.** Первый замер спрашивал «найди атом,
   похожий на мой пересказ». Это задача векторного поиска, и он её решает:
   медиана ранга 1. Граф заявлен не на это — он заявлен на то, что ПОСЛЕ
   находки покажет, чем её отменили, чему она противоречит и куда линия пошла
   дальше. Здесь это и проверяется: от найденного атома идём по типизированным
   рёбрам и смотрим, что там лежит.

Второе — честный тест для PTG, первый был честным тестом для RAG.
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
sys.path.insert(0, HERE)
import recall_own_corpus as R                          # noqa: E402

TOP_K = 10
TYPED = ("supersedes", "contradicts", "fixes", "refines", "returns_to")


def main():
    with io.open(os.path.join(STORE, "meta_tree.json"), encoding="utf-8") as f:
        order = json.load(f)["id_order"]
    with io.open(os.path.join(STORE, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    with io.open(os.path.join(STORE, "edges.json"), encoding="utf-8") as f:
        edges = json.load(f)
    V = np.load(os.path.join(STORE, "faiss.index.npy")).astype("float32")
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    texts = [(nodes.get(nid, {}).get("text") or "") for nid in order]
    low = [t.lower() for t in texts]
    pos = {nid: i for i, nid in enumerate(order)}

    typed = collections.defaultdict(list)
    for e in edges:
        if e["type"] in TYPED:
            a, b = pos.get(e["from"]), pos.get(e["to"])
            if a is not None and b is not None:
                typed[a].append((b, e["type"]))
                typed[b].append((a, e["type"]))

    qs = [q for q in R.load_queries()
          if any(a.lower() in t for a in q["опоры"] for t in low)]
    for q in qs:
        q["опоры"] = [a for a in q["опоры"] if any(a.lower() in t for t in low)]
    print("запросов: %d" % len(qs), flush=True)

    from llama_cpp import Llama
    llm = Llama(model_path=GGUF, embedding=True, n_ctx=512, n_batch=512,
                n_ubatch=512, n_gpu_layers=0, verbose=False)

    hit_rows, miss_rows = [], []
    typed_reach = 0
    typed_kinds = collections.Counter()
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        r = llm.create_embedding(q["запрос"][:1500])
        qv = np.asarray(r["data"][0]["embedding"], dtype="float32")
        qv /= max(float(np.linalg.norm(qv)), 1e-9)
        idxs = list(np.argsort(-(V @ qv))[:TOP_K])
        rank = None
        seed = None
        for j, ix in enumerate(idxs):
            if any(a.lower() in low[ix] for a in q["опоры"]):
                rank, seed = j + 1, ix
                break
        if rank is None:
            # где лежит опора на самом деле
            where = [k for k in range(len(low))
                     if any(a.lower() in low[k] for a in q["опоры"])]
            best = None
            if where:
                s = V[np.array(where)] @ qv
                best = (int(np.array(where)[int(np.argmax(s))]), float(s.max()))
            miss_rows.append((q["название"], len(where), best))
        else:
            kinds = collections.Counter(t for _n, t in typed.get(seed, []))
            if kinds:
                typed_reach += 1
                typed_kinds.update(kinds)
            hit_rows.append((q["название"], rank, dict(kinds)))
        if i % 10 == 0:
            print("  %d/%d, %.0f с" % (i, len(qs), time.time() - t0), flush=True)

    print("\n=== НЕ НАЙДЕНО (%d из %d) ===" % (len(miss_rows), len(qs)))
    print("  %-46s %8s %10s" % ("выжимка", "атомов с опорой", "лучший косинус"))
    for name, cnt, best in miss_rows:
        print("  %-46s %8d %10s" % (name[:46], cnt,
                                    ("%.3f" % best[1]) if best else "—"))

    print("\n=== ЧТО ГРАФ ДАЁТ СВЕРХ НАХОДКИ ===")
    print("  у найденного атома есть типизированные рёбра: %d из %d (%.0f %%)"
          % (typed_reach, len(hit_rows), 100.0 * typed_reach / max(1, len(hit_rows))))
    print("  типы:", dict(typed_kinds.most_common()))
    print("\n  %-46s %5s  %s" % ("выжимка", "ранг", "рёбра от найденного"))
    for name, rank, kinds in hit_rows[:20]:
        print("  %-46s %5d  %s" % (name[:46], rank, kinds or "—"))


if __name__ == "__main__":
    main()
