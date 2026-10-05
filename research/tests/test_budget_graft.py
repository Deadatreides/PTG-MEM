# -*- coding: utf-8 -*-
r"""Проверка ядра V5 на настоящих атомах.

ГЛАВНЫЙ ИНВАРИАНТ. Бюджет добавляет только отношения, поэтому назначение
ветви обязано совпасть с прогоном без V5 **атом в атом**, а число `continues`
— в точности. Если это не так, членство протекло, и все остальные числа
недействительны: связность получена схлопыванием.

Прогоняются два архива на ОДНОМ потоке атомов:
  * основа  — только `graft_fast` (движок как есть, ускоренный, сверенный);
  * V5      — `graft_fast` + `budget_graft`.

Пороги выставляются калиброванные (0.47/0.57/0.65) — те же, которыми собран
`store_text_cal`, откуда берутся атомы. С умолчательными 0.60/0.78/0.83 числа
были бы про другую шкалу.

Запуск:  python test_budget_graft.py [сколько_атомов] [margin]
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
V4 = os.path.join(ROOT, "PTG_V4")
SNAP = os.path.join(ROOT, "trace-probe", "ptg", "_snapshot_run")
PTG_TEXT = os.path.join(ROOT, "trace-probe", "ptg", "PTG+MCP1.3")
STORE = os.path.join(SNAP, "store_text_cal", ".ptg")
for p in (HERE, V4, SNAP, PTG_TEXT):
    sys.path.insert(0, p)

import ptg_core                                        # noqa: E402
import graft_fast                                      # noqa: E402
import budget_graft                                    # noqa: E402
import gates                                           # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
SCRATCH = os.path.join(tempfile.gettempdir(), "ptg_v5_test")

# калиброванные пороги — те же, что у store_text_cal
ptg_core.BRANCH_THRESH = 0.47
ptg_core.CONTINUE_THRESH = 0.57
ptg_core.RETURN_THRESH = 0.65


def load_atoms(n):
    with io.open(os.path.join(STORE, "meta_tree.json"), encoding="utf-8") as f:
        order = json.load(f)["id_order"]
    vecs = np.load(os.path.join(STORE, "faiss.index.npy"))
    with io.open(os.path.join(STORE, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    drop = ("vec", "index", "branch", "semantic_epoch", "placement_info",
            "path_coherence", "short_path_coherence", "long_path_coherence",
            "reinforcement_count", "reinforcement_links", "repetition_epochs")
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
        a = {k: nd[k] for k in nd if k not in drop}
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


def snapshot(arc):
    return {"nodes": arc.nodes, "edges": arc.edges, "branches": arc.branches,
            "path_memory": arc.path_memory}


def run(atoms, dim, tag, with_budget: bool):
    arc = fresh(dim, tag)
    fg = graft_fast.install(arc, ptg_core, verify=0)
    bg = budget_graft.install(arc, ptg_core, fg, margin=MARGIN) if with_budget else None
    t0 = time.time()
    for a in atoms:
        arc._add_atom(copy.deepcopy(a))
    return arc, time.time() - t0, bg


def main():
    print("загрузка атомов ...")
    atoms, dim = load_atoms(N)
    print("атомов %d, dim %d, пороги %.2f/%.2f/%.2f, margin %.1f"
          % (len(atoms), dim, ptg_core.BRANCH_THRESH, ptg_core.CONTINUE_THRESH,
             ptg_core.RETURN_THRESH, MARGIN))

    print("\n--- основа: движок как есть ---")
    a1, t1, _ = run(atoms, dim, "base", with_budget=False)
    print("время %.1f с, узлов %d, ветвей %d, рёбер %d"
          % (t1, len(a1.nodes), len(a1.branches), len(a1.edges)))

    print("\n--- V5: бюджетный графт ---")
    a2, t2, bg = run(atoms, dim, "v5", with_budget=True)
    print("время %.1f с, узлов %d, ветвей %d, рёбер %d"
          % (t2, len(a2.nodes), len(a2.branches), len(a2.edges)))
    print("счётчики: %s" % json.dumps(bg.stats(), ensure_ascii=False))

    # --- главный инвариант ---
    print("\n--- ИНВАРИАНТ: членство не тронуто ---")
    b1 = {nid: a1.nodes[nid].get("branch") for nid in a1.id_order}
    b2 = {nid: a2.nodes[nid].get("branch") for nid in a2.id_order}
    same = sum(1 for k in b1 if b1[k] == b2.get(k))
    cont1 = sum(1 for e in a1.edges if e["type"] == "continues")
    cont2 = sum(1 for e in a2.edges if e["type"] == "continues")
    ep1 = {nid: a1.nodes[nid].get("semantic_epoch") for nid in a1.id_order}
    ep2 = {nid: a2.nodes[nid].get("semantic_epoch") for nid in a2.id_order}
    ep_same = sum(1 for k in ep1 if ep1[k] == ep2.get(k))
    print("  ветка совпала у %d из %d (%.2f %%)" % (same, len(b1), 100.0 * same / max(1, len(b1))))
    print("  эпоха совпала у %d из %d" % (ep_same, len(ep1)))
    print("  continues: основа %d, V5 %d  %s"
          % (cont1, cont2, "совпало" if cont1 == cont2 else "РАЗОШЛОСЬ"))
    print("  ветвей: основа %d, V5 %d  %s"
          % (len(a1.branches), len(a2.branches),
             "совпало" if len(a1.branches) == len(a2.branches) else "РАЗОШЛОСЬ"))

    ok = (b1 == b2) and (cont1 == cont2) and (len(a1.branches) == len(a2.branches))

    # --- ворота ---
    for tag, arc in (("ОСНОВА", a1), ("V5", a2)):
        snap = snapshot(arc)
        rep = {"связность": gates.connectivity(snap),
               "несхлопнутость": gates.no_coagulation(snap),
               "сходство рёбер": gates.edge_similarity(snap),
               "путь": gates.path(snap)}
        print("\n" + "=" * 62)
        print(tag)
        print("=" * 62)
        for block, vals in rep.items():
            print("  [%s]" % block)
            for k, v in vals.items():
                print("    %-32s %s" % (k, v))
        if tag == "V5" and rep["несхлопнутость"]["НАРУШЕНИЕ_далёкое_продолжение"]:
            ok = False

    print("\nвремя: основа %.1f с, V5 %.1f с (%.2fx)" % (t1, t2, t2 / max(t1, 1e-9)))
    print("ВЕРДИКТ: %s" % ("инвариант держится" if ok else "ИНВАРИАНТ НАРУШЕН"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
