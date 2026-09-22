# -*- coding: utf-8 -*-
r"""Прогон одного ядра по ПОЛНОМУ архиву и отчёт воротами.

Эмбеддер не нужен: атомы и векторы берутся из уже собранного хранилища и
переигрываются в том же порядке. Меняется только правило присоединения.

Два ядра гоняются РАЗНЫМИ процессами — иначе две копии по 27 тысяч векторов
по 2560 измерений лежали бы в памяти одновременно.

    python run_full.py base  <store> [N]
    python run_full.py v5    <store> [N] [margin]

Отчёт пишется в PTG_V5/out/gates_<режим>.json
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
V4 = os.path.join(ROOT, "PTG_V4")
SNAP = os.path.join(ROOT, "trace-probe", "ptg", "_snapshot_run")
PTG_TEXT = os.path.join(ROOT, "trace-probe", "ptg", "PTG+MCP1.3")
for p in (HERE, V4, SNAP, PTG_TEXT):
    sys.path.insert(0, p)

import ptg_core                                        # noqa: E402
import graft_fast                                      # noqa: E402
import budget_graft                                    # noqa: E402
import gates                                           # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "base"
STORE_NAME = sys.argv[2] if len(sys.argv) > 2 else "store_text_v3"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MARGIN = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
STORE = os.path.join(SNAP, STORE_NAME, ".ptg")

# калиброванные пороги — те же, которыми собраны store_text_cal и v3
ptg_core.BRANCH_THRESH = 0.47
ptg_core.CONTINUE_THRESH = 0.57
ptg_core.RETURN_THRESH = 0.65

DROP = ("vec", "index", "branch", "semantic_epoch", "placement_info",
        "path_coherence", "short_path_coherence", "long_path_coherence",
        "reinforcement_count", "reinforcement_links", "repetition_epochs")


def main():
    t0 = time.time()
    with io.open(os.path.join(STORE, "meta_tree.json"), encoding="utf-8") as f:
        order = json.load(f)["id_order"]
    vecs = np.load(os.path.join(STORE, "faiss.index.npy"))
    with io.open(os.path.join(STORE, "nodes.json"), encoding="utf-8") as f:
        src = json.load(f)
    print("%s: %s, узлов %d, dim %d, чтение %.0f с"
          % (MODE, STORE_NAME, len(order), vecs.shape[1], time.time() - t0), flush=True)

    out_dir = os.path.join(HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "_run_" + MODE)
    os.makedirs(work, exist_ok=True)
    arc = ptg_core.Archive(folder=ROOT, output_dir=work, progress_cb=lambda m: None,
                           extract_py_comments=True)
    arc.root_embedding = None
    arc.embedder.dim = int(vecs.shape[1])

    fg = graft_fast.install(arc, ptg_core, verify=0)
    bg = budget_graft.install(arc, ptg_core, fg, margin=MARGIN) if MODE == "v5" else None

    t0 = time.time()
    done = 0
    limit = N or len(order)
    for i, nid in enumerate(order):
        if done >= limit:
            break
        nd = src.get(nid)
        if nd is None:
            continue
        v = np.asarray(vecs[i], dtype="float32")
        nrm = float(np.linalg.norm(v))
        if nrm == 0:
            continue
        atom = {k: nd[k] for k in nd if k not in DROP}
        atom["vec"] = v / nrm
        arc._add_atom(atom)
        done += 1
        if done % 2500 == 0:
            print("  %d/%d, %.0f с, рёбер %d"
                  % (done, limit, time.time() - t0, len(arc.edges)), flush=True)
    dt = time.time() - t0
    print("%s: %d атомов за %.0f с (%.1f атом/с), узлов %d, ветвей %d, рёбер %d"
          % (MODE, done, dt, done / max(dt, 1e-9), len(arc.nodes),
             len(arc.branches), len(arc.edges)), flush=True)

    snap = {"nodes": arc.nodes, "edges": arc.edges, "branches": arc.branches,
            "path_memory": arc.path_memory}
    rep = {
        "режим": MODE, "хранилище": STORE_NAME, "атомов": done,
        "секунд": round(dt, 1), "margin": MARGIN,
        "связность": gates.connectivity(snap),
        "несхлопнутость": gates.no_coagulation(snap),
        "сходство_рёбер": gates.edge_similarity(snap),
        "путь": gates.path(snap),
        "ветка_атома": {nid: arc.nodes[nid].get("branch") for nid in arc.id_order},
    }
    if bg is not None:
        rep["счётчики"] = bg.stats()
    p = os.path.join(out_dir, "gates_%s.json" % MODE)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False)
    print("записано: %s" % p, flush=True)


if __name__ == "__main__":
    main()
