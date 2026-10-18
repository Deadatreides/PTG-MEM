# -*- coding: utf-8 -*-
r"""Ревизия PTG-MEM, замер 8: поиск LanceDB без индекса = плоский косинус?

Замер 11.10 (bench/hybrid_recall.py) мерил плоский косинус в numpy: 98,3 % по реплике при k = 10 на LongMemEval.
engine3 ищет в LanceDB без векторного индекса. Если LanceDB при этом даёт ровно ту же десятку, что numpy, замер
11.10 переносится на engine3 без повторного прогона на видеокарте.

20 000 единичных векторов размерности 1024 (как у Qwen3-Embedding-0.6B), 200 запросов; совпадение десяток.

    PTG-MEM\rev-shell\.venv\Scripts\python bench\lance_exact.py
"""
import sys
import tempfile
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
import lancedb  # noqa: E402
import pyarrow as pa  # noqa: E402

N, DIM, Q, K = 20000, 1024, 200, 10
rng = np.random.default_rng(11)
base = rng.normal(size=(N, DIM)).astype("float32")
base /= np.linalg.norm(base, axis=1, keepdims=True)
# запросы — рядом с векторами базы, как у настоящих вопросов к памяти
idx = rng.integers(0, N, size=Q)
queries = base[idx] + 0.6 * rng.normal(size=(Q, DIM)).astype("float32") / np.sqrt(DIM)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

db = lancedb.connect(tempfile.mkdtemp())
schema = pa.schema([pa.field("id", pa.int64()), pa.field("status", pa.string()), pa.field("vector", pa.list_(pa.float32(), DIM))])
t = db.create_table("atoms", data=pa.Table.from_pylist([{"id": i, "status": "active", "vector": v.tolist()} for i, v in enumerate(base)], schema=schema))
same, overlap, t_lance = 0, 0, 0.0
for q in queries:
    want = np.argsort(-(base @ q), kind="stable")[:K].tolist()
    t0 = time.time()
    got = [r["id"] for r in t.search(q).metric("cosine").where("status = 'active'", prefilter=True).select(["id"]).limit(K).to_list()]
    t_lance += time.time() - t0
    same += got == want
    overlap += len(set(got) & set(want))
print(f"векторов {N}, размерность {DIM}, запросов {Q}")
print(f"десятка совпала целиком: {same} из {Q}; общих в десятках: {overlap / (Q * K):.4f}")
print(f"LanceDB на запрос: {1000 * t_lance / Q:.1f} мс")
