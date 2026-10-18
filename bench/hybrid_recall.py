# -*- coding: utf-8 -*-
r"""Ревизия PTG-MEM, замер 1: готовый гибридный поиск против плоского косинуса на LongMemEval.

Те же 60 вопросов, что в longmemeval_recall.py (зерно 20260922, по 10 на тип), та же метрика — полнота
реплики-улики (has_answer) и сессии-улики (answer_session_ids) при k = 1, 3, 5, 10. Отличие одно: вместо
графа PTG сравниваются готовые способы поиска.

  * плоский косинус — прежний базис (там 45,0 / 75,0 / 86,7 / 96,7 % по реплике с эмбеддером Jasper);
  * BM25 — bm25s;
  * гибрид LanceDB — одна таблица: вектор, полнотекстовый индекс и слияние рангов RRF
    (`search(query_type="hybrid")` + `RRFReranker`);
  * RRF вручную — плоский косинус и bm25s, k = 60, для сверки с LanceDB.

Эмбеддер — умолчание PTG-MEM, Qwen/Qwen3-Embedding-0.6B (на CPU, если видеопамять занята демоном). Векторы
каждого стога кэшируются в out/lme_cache — повторный прогон их не считает.

    .venv-rev\Scripts\python bench\hybrid_recall.py [вопросов] [устройство]
"""
from __future__ import annotations

import collections
import io
import json
import os
import random
import sys
import tempfile
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("LME_DATA", r"H:\работа\сайты\идеи\память\for codex (1.1) — копия\PTG_V5\data\longmemeval_s.json")
MODEL = os.environ.get("PTG_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 60
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cpu"
KS = (1, 3, 5, 10)
SEED = 20260922
CACHE = os.path.join(HERE, "out", "lme_cache", MODEL.replace("/", "_"))


def build_atoms(item):
    """Атом = реплика, как в longmemeval_recall.py."""
    atoms = []
    for sid, sess in zip(item["haystack_session_ids"], item["haystack_sessions"]):
        for ti, turn in enumerate(sess):
            txt = (turn.get("content") or "").strip()
            if txt:
                atoms.append({"text": txt, "session": sid, "turn": ti, "evidence": bool(turn.get("has_answer"))})
    return atoms


def pick(data):
    rng = random.Random(SEED)
    by_type = collections.defaultdict(list)
    for it in data:
        by_type[it["question_type"]].append(it)
    picked, per = [], max(1, N_Q // len(by_type))
    for _, items in by_type.items():
        rng.shuffle(items)
        picked.extend(items[:per])
    return picked[:N_Q]


def rrf(*rankings, k=60):
    score = collections.defaultdict(float)
    for ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] += 1.0 / (k + r + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])]


def lance_hybrid(texts, V, question, qv, limit):
    import lancedb
    import pyarrow as pa
    from lancedb.rerankers import RRFReranker
    tmp = tempfile.mkdtemp(prefix="lme-lance-")
    db = lancedb.connect(tmp)
    schema = pa.schema([("id", pa.int32()), ("text", pa.string()), ("vector", pa.list_(pa.float32(), V.shape[1]))])
    table = pa.table({"id": pa.array(range(len(texts)), pa.int32()), "text": texts,
                      "vector": pa.FixedSizeListArray.from_arrays(pa.array(V.reshape(-1), pa.float32()), V.shape[1])}, schema=schema)
    tbl = db.create_table("atoms", data=table)
    tbl.create_fts_index("text", replace=True)
    rows = (tbl.search(query_type="hybrid").vector(qv.tolist()).text(question).limit(limit)
            .rerank(RRFReranker()).to_list())
    return [int(r["id"]) for r in rows]


def main():
    print(f"читаю стенд ... модель {MODEL}, устройство {DEVICE}", flush=True)
    data = json.load(io.open(DATA, encoding="utf-8"))
    picked = pick(data)
    print("вопросов: %d %s" % (len(picked), dict(collections.Counter(p["question_type"] for p in picked))), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    import bm25s
    enc = None
    res = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    t0 = time.time()
    for qi, item in enumerate(picked, 1):
        atoms = build_atoms(item)
        texts = [a["text"][:1500] for a in atoms]
        cache = os.path.join(CACHE, item["question_id"] + ".npz")
        if os.path.exists(cache):
            z = np.load(cache)
            V, qv = z["V"], z["q"]
        else:
            if enc is None:
                from sentence_transformers import SentenceTransformer
                enc = SentenceTransformer(MODEL, device=DEVICE)
                enc.max_seq_length = 512
            V = enc.encode(texts, batch_size=16, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            qv = enc.encode([item["question"]], prompt_name="query", convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]
            np.savez(cache, V=V, q=qv)
        ev_turns = {i for i, a in enumerate(atoms) if a["evidence"]}
        ev_sess = set(item.get("answer_session_ids") or [])
        qt = item["question_type"]
        tot[qt] += 1
        depth = min(100, len(texts))
        dense = list(np.argsort(-(V @ qv))[:depth])
        retr = bm25s.BM25()
        retr.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
        ids, _ = retr.retrieve(bm25s.tokenize([item["question"]], stopwords="en", show_progress=False), k=depth, show_progress=False)
        lexical = [int(i) for i in ids[0]]
        ways = {"плоский": dense, "BM25": lexical, "RRF вручную": rrf(dense, lexical),
                "гибрид LanceDB": lance_hybrid(texts, V, item["question"], qv, max(KS))}
        for name, ranking in ways.items():
            for k in KS:
                cut = ranking[:k]
                for unit, hit in (("turn", any(i in ev_turns for i in cut)), ("sess", any(atoms[i]["session"] in ev_sess for i in cut))):
                    res[name][f"{unit}{k}_all"] += int(hit)
                    res[name][f"{unit}{k}_{qt}"] += int(hit)
        if qi % 5 == 0:
            print(f"  {qi}/{len(picked)}, {time.time() - t0:.0f} с", flush=True)
    n = sum(tot.values())
    names = list(res)
    print(f"\nвопросов: {n}")
    for unit, title in (("turn", "реплика-улика"), ("sess", "сессия-улика")):
        print(f"\n{title}: " + " | ".join(f"{m}" for m in names))
        for k in KS:
            print(f"  k={k:<3}" + "".join(f"{100.0 * res[m][f'{unit}{k}_all'] / n:>18.1f}%" for m in names))
    print("\nпо типам, сессия-улика при k=3:")
    for qt in sorted(tot):
        print(f"  {qt:<28} n={tot[qt]:<3}" + "".join(f"{m}: {100.0 * res[m]['sess3_' + qt] / tot[qt]:.1f}%  " for m in names))
    out = {"модель": MODEL, "вопросов": n, "recall": {m: {f"{u}{k}": round(100.0 * res[m][f"{u}{k}_all"] / n, 1)
                                                          for u in ("turn", "sess") for k in KS} for m in names},
           "по_типам_sess3": {qt: {m: round(100.0 * res[m]["sess3_" + qt] / tot[qt], 1) for m in names} for qt in tot}}
    with io.open(os.path.join(HERE, "out", "hybrid_recall.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nзаписано: bench/out/hybrid_recall.json")


if __name__ == "__main__":
    main()
