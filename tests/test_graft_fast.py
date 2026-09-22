# -*- coding: utf-8 -*-
r"""Сверка graft_fast с движком: тот же поток атомов, два Archive, сравнение.

Атомы и векторы берутся настоящие — из store_text_cal, а не синтетические:
распределение косинусов и есть то, на чём решается argmax, и подменять его
случайными векторами значит проверять не тот вопрос.

Сравнивается не «похоже», а поатомно: ветка каждого узла, полный список рёбер
и структура веток. Плюс время обеих сборок на одном и том же потоке.

Запуск:  python test_graft_fast.py [сколько_атомов]
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SNAP = os.path.join(ROOT, "trace-probe", "ptg", "_snapshot_run")
PTG_TEXT = os.path.join(ROOT, "trace-probe", "ptg", "PTG+MCP1.3")
STORE = os.path.join(SNAP, "store_text_cal", ".ptg")
sys.path.insert(0, HERE)
sys.path.insert(0, SNAP)
sys.path.insert(0, PTG_TEXT)

import ptg_core                                        # noqa: E402
import graft_fast                                      # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
# Сверка перебором стоит ровно столько же, сколько старый путь, и потому
# искажает замер времени. Держим её отдельным аргументом: сверка на малом N,
# время на большом.
VERIFY = int(sys.argv[2]) if len(sys.argv) > 2 else min(150, N)
# Пишем во временный каталог ОС, а не в out/: тестовые архивы проекту не нужны,
# а логи ptg_core на каждый прогон дают десяток мегабайт.
SCRATCH = os.environ.get("SCRATCH", os.path.join(tempfile.gettempdir(), "ptg_v4_test_graft"))


def load_atoms(n):
    with io.open(os.path.join(STORE, "meta_tree.json"), encoding="utf-8") as f:
        meta = json.load(f)
    order = meta["id_order"]
    vecs = np.load(os.path.join(STORE, "faiss.index.npy"))
    with io.open(os.path.join(STORE, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    out = []
    for i, nid in enumerate(order):
        if len(out) >= n:
            break
        nd = nodes.get(nid)
        if nd is None:
            continue
        v = np.asarray(vecs[i], dtype="float32")
        nrm = float(np.linalg.norm(v))
        if nrm == 0:
            continue
        a = {k: nd[k] for k in nd
             if k not in ("vec", "index", "branch", "semantic_epoch", "placement_info",
                          "path_coherence", "short_path_coherence", "long_path_coherence",
                          "reinforcement_count", "reinforcement_links", "repetition_epochs")}
        a["vec"] = v / nrm
        out.append(a)
    return out, int(vecs.shape[1])


def fresh(dim, tag):
    d = os.path.join(SCRATCH, tag)
    os.makedirs(d, exist_ok=True)
    arc = ptg_core.Archive(folder=ROOT, output_dir=d, progress_cb=lambda m: None,
                           extract_py_comments=True)
    arc.root_embedding = None
    arc.embedder.dim = dim
    return arc


def fingerprint(arc):
    branch_of = {nid: arc.nodes[nid].get("branch") for nid in arc.id_order}
    # ветки — по составу, а не по идентификатору: id ветки это id её корня,
    # он одинаков у обоих прогонов, но сверяем всё равно состав
    edges = sorted((e["from"], e["to"], e["type"]) for e in arc.edges)
    bodies = {b: sorted(v["atom_ids"]) for b, v in arc.branch_bodies.items()}
    # graft_fast подменяет и _path_coherence — значит сверять надо и её выход,
    # а не только структуру: матвек копит иначе, чем цепочка скалярных dot
    coh = {nid: (arc.nodes[nid].get("path_coherence"),
                 arc.nodes[nid].get("short_path_coherence"),
                 arc.nodes[nid].get("long_path_coherence")) for nid in arc.id_order}
    return branch_of, edges, bodies, coh


def run(atoms, dim, tag, fast, verify=0):
    arc = fresh(dim, tag)
    fg = graft_fast.install(arc, ptg_core, verify=verify) if fast else None
    t0 = time.time()
    marks = []
    for i, a in enumerate(atoms, 1):
        arc._add_atom(copy.deepcopy(a))
        if i % max(1, len(atoms) // 4) == 0:
            marks.append((i, time.time() - t0))
    dt = time.time() - t0
    return arc, dt, fg, marks


def main():
    print("загрузка атомов из store_text_cal ...")
    t0 = time.time()
    atoms, dim = load_atoms(N)
    print("атомов: %d, dim: %d, %.0f с" % (len(atoms), dim, time.time() - t0))

    print("\n--- движок как есть ---")
    a1, t1, _, m1 = run(atoms, dim, "orig", fast=False)
    print("время: %.1f с  (%s)" % (t1, ", ".join("%d:%.0fс" % x for x in m1)))
    print("узлов %d, веток %d, рёбер %d" % (len(a1.nodes), len(a1.branches), len(a1.edges)))

    print("\n--- graft_fast ---")
    a2, t2, fg, m2 = run(atoms, dim, "fast", fast=True, verify=VERIFY)
    print("время: %.1f с  (%s)" % (t2, ", ".join("%d:%.0fс" % x for x in m2)))
    print("узлов %d, веток %d, рёбер %d" % (len(a2.nodes), len(a2.branches), len(a2.edges)))
    print("счётчики: %s" % fg.stats())

    b1, e1, bo1, c1 = fingerprint(a1)
    b2, e2, bo2, c2 = fingerprint(a2)

    print("\n--- сверка ---")
    same_branch = sum(1 for k in b1 if b1[k] == b2.get(k))
    print("ветка совпала у %d из %d узлов (%.2f %%)"
          % (same_branch, len(b1), 100.0 * same_branch / max(1, len(b1))))
    print("рёбра: %s (%d против %d)"
          % ("совпали" if e1 == e2 else "РАЗОШЛИСЬ", len(e1), len(e2)))
    print("тела веток: %s" % ("совпали" if bo1 == bo2 else "РАЗОШЛИСЬ"))

    dev = 0.0
    diff_n = 0
    for k in c1:
        for x, y in zip(c1[k], c2.get(k, (None, None, None))):
            if x is None or y is None:
                continue
            if x != y:
                diff_n += 1
                dev = max(dev, abs(x - y))
    # значения хранятся округлёнными до 4 знаков, поэтому единица в последнем
    # знаке — это граница округления, а не расхождение счёта
    tol = 1.0001e-4
    print("path coherence: расходится у %d значений из %d, максимум %.6f (%s)"
          % (diff_n, 3 * len(c1), dev,
             "в пределах округления до 4 знаков" if dev <= tol else "БОЛЬШЕ округления"))

    ok = (b1 == b2) and (e1 == e2) and (bo1 == bo2) and fg.stats()["расхождений_ветка"] == 0 \
        and fg.stats()["расхождений_узел"] == 0 and fg.stats()["расхождений_состояние"] == 0 \
        and dev <= tol
    print("\nВЕРДИКТ: %s" % ("совпадение полное" if ok else "ЕСТЬ РАСХОЖДЕНИЯ"))
    if t2 > 0:
        print("ускорение фазы дерева: %.2fx (%.1f с -> %.1f с)" % (t1 / t2, t1, t2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
