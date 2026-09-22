# -*- coding: utf-8 -*-
r"""PTG_V5/gates.py — ворота: связность И несхлопнутость И путь.

Связность — недостаточное условие. Её легко получить, схлопнув граф: если
поднять бюджет и позволить `continues` множиться, степень вырастет, сироты
исчезнут, а архив превратится в одну ветку, где всё продолжает всё. Числа
будут прекрасные, смысл — потерян.

Поэтому ворота состоят из трёх частей, и проходить надо все три.

  1. СВЯЗНОСТЬ. Степень против `ln n`, изолированные против `exp(−c)`,
     глубина обхода `ln n / ln c`, перекос (доля концов рёбер у 1 % самых
     связных).

  2. НЕСХЛОПНУТОСТЬ. Доля `continues` НЕ должна расти с бюджетом — членство
     однозначно, и бюджет добавляет только отношения. Средний размер ветви не
     растёт. Распределение эпох не сжимается. Доля межветвевых рёбер растёт —
     в этом и смысл. Доля рёбер через временной порог растёт, и ни одно из
     них не `continues`.

  3. ПУТЬ. Распределение `path_coherence` не сваливается к среднему; длина
     длинного пути не растёт быстрее прежнего.

Работает по хранилищу на диске, поэтому сравнивает архивы разных ядер:
    python gates.py <путь к .ptg> [<путь ко второму .ptg для сравнения>]
"""
from __future__ import annotations

import collections
import io
import json
import math
import os
import statistics
import sys


def load(ptg_dir: str) -> dict:
    def j(name):
        p = os.path.join(ptg_dir, name)
        if not os.path.exists(p):
            return None
        with io.open(p, encoding="utf-8") as f:
            return json.load(f)
    nodes = j("nodes.json") or {}
    edges = j("edges.json") or []
    meta = j("meta_tree.json") or {}
    return {"nodes": nodes, "edges": edges,
            "branches": meta.get("branches", {}),
            "path_memory": meta.get("path_memory", {})}


# ---------------------------------------------------------------------------
def connectivity(a: dict) -> dict:
    nodes, edges = a["nodes"], a["edges"]
    n = len(nodes)
    if n == 0:
        return {}
    simple = set()
    for e in edges:
        u, v = e["from"], e["to"]
        if u == v:
            continue
        simple.add((u, v) if u < v else (v, u))
    deg = collections.Counter()
    for u, v in simple:
        deg[u] += 1
        deg[v] += 1
    degs = [deg.get(k, 0) for k in nodes]
    c = 2.0 * len(simple) / n
    iso = sum(1 for d in degs if d == 0)
    ln_n = math.log(n) if n > 1 else 0.0
    top1 = max(1, n // 100)
    hot = sum(sorted(degs, reverse=True)[:top1])
    return {
        "узлов": n,
        "рёбер_простых": len(simple),
        "средняя_степень": round(c, 2),
        "медиана_степени": int(statistics.median(degs)),
        "максимум_степени": max(degs) if degs else 0,
        "ln_n": round(ln_n, 2),
        "нужный_бюджет": max(2, int(math.ceil(3 * ln_n))),
        "в_режиме": bool(c >= 3 * ln_n),
        "изолированных": iso,
        "изолированных_%": round(100.0 * iso / n, 2),
        "нулевая_модель_%": round(100.0 * math.exp(-c), 2),
        "глубина_обхода": (1 if c <= 1 else max(1, int(math.ceil(ln_n / math.log(c))))),
        "доля_рёбер_у_1%_самых_связных": round(100.0 * hot / max(1, 2 * len(simple)), 1),
    }


def no_coagulation(a: dict, temporal_days: float = 21.0) -> dict:
    nodes, edges, branches = a["nodes"], a["edges"], a["branches"]
    by_type = collections.Counter(e.get("type") for e in edges)
    total = sum(by_type.values()) or 1

    sizes = collections.Counter(nd.get("branch") for nd in nodes.values())
    size_list = [v for v in sizes.values() if v] or [0]

    epochs = collections.Counter(nd.get("semantic_epoch", 0) for nd in nodes.values())

    cross = same = 0
    far = far_continues = 0
    day = 86400.0
    for e in edges:
        a_n, b_n = nodes.get(e["from"]), nodes.get(e["to"])
        if not a_n or not b_n:
            continue
        if a_n.get("branch") == b_n.get("branch"):
            same += 1
        else:
            cross += 1
        ta, tb = a_n.get("timestamp"), b_n.get("timestamp")
        if ta and tb and abs(float(ta) - float(tb)) / day > temporal_days:
            far += 1
            if e.get("type") == "continues":
                far_continues += 1
    tot_pairs = max(1, cross + same)
    return {
        "рёбер_всего": sum(by_type.values()),
        "по_типам": dict(by_type.most_common()),
        "доля_continues_%": round(100.0 * by_type.get("continues", 0) / total, 2),
        "ветвей": len(branches),
        "размер_ветви_средний": round(statistics.mean(size_list), 2),
        "размер_ветви_максимум": max(size_list),
        "узлов_на_ветвь": round(len(nodes) / max(1, len(branches)), 2),
        "эпох_различных": len(epochs),
        "эпоха_максимум": max(epochs) if epochs else 0,
        "межветвевых_рёбер_%": round(100.0 * cross / tot_pairs, 2),
        "рёбер_через_%d_дней_%%" % int(temporal_days): round(100.0 * far / tot_pairs, 2),
        "из_них_continues": far_continues,
        "НАРУШЕНИЕ_далёкое_продолжение": far_continues > 0,
    }


def edge_similarity(a: dict, vecs=None, order=None) -> dict:
    """Распределение косинуса по типам рёбер.

    Ворота, которых не хватило 21.09.2026: у канала отношений не было нижнего
    порога, и все сироты «спаслись» рёбрами `reinforces` при сходстве заведомо
    ниже `BRANCH_THRESH` — молча, ложным утверждением о родстве. Структурные
    числа при этом выглядели прекрасно.

    Если `reinforces` появляется при низком косинусе — порог протёк.
    """
    nodes = a["nodes"]
    if vecs is None:
        vecs = {nid: nd.get("vec") for nid, nd in nodes.items()}
    out = {}
    buckets = collections.defaultdict(list)
    for e in a["edges"]:
        va, vb = vecs.get(e["from"]), vecs.get(e["to"])
        if va is None or vb is None:
            continue
        try:
            import numpy as np
            s = float(np.dot(np.asarray(va, dtype="float32"),
                             np.asarray(vb, dtype="float32")))
        except Exception:
            continue
        buckets[e.get("type")].append(s)
    for t, xs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        xs.sort()
        out[t] = {"рёбер": len(xs),
                  "мин": round(xs[0], 3),
                  "p10": round(xs[len(xs) // 10], 3),
                  "медиана": round(xs[len(xs) // 2], 3),
                  "макс": round(xs[-1], 3)}
    return out


def path(a: dict) -> dict:
    nodes = a["nodes"]
    pc = [nd.get("path_coherence") for nd in nodes.values()
          if isinstance(nd.get("path_coherence"), (int, float))]
    pm = a.get("path_memory") or {}
    out = {"длинный_путь": len(pm.get("long", []) or []),
           "короткий_путь": len(pm.get("short", []) or [])}
    if pc:
        pc.sort()
        out.update({
            "path_coherence_среднее": round(statistics.mean(pc), 4),
            "path_coherence_разброс": round(statistics.pstdev(pc), 4),
            "p10": round(pc[len(pc) // 10], 4),
            "p50": round(pc[len(pc) // 2], 4),
            "p90": round(pc[9 * len(pc) // 10], 4),
        })
    return out


def report(ptg_dir: str) -> dict:
    a = load(ptg_dir)
    return {"связность": connectivity(a),
            "несхлопнутость": no_coagulation(a),
            "путь": path(a)}


def _print(tag: str, rep: dict):
    print("\n" + "=" * 68)
    print(tag)
    print("=" * 68)
    for block, vals in rep.items():
        print("\n  [%s]" % block)
        for k, v in vals.items():
            print("    %-34s %s" % (k, v))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    _print(argv[1], report(argv[1]))
    if len(argv) > 2:
        _print(argv[2], report(argv[2]))
        print("\n" + "=" * 68)
        print("СРАВНЕНИЕ (второй против первого)")
        print("=" * 68)
        r1, r2 = report(argv[1]), report(argv[2])
        for block in r1:
            for k in r1[block]:
                v1, v2 = r1[block].get(k), r2[block].get(k)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1 != v2:
                    delta = "%+.2f" % (v2 - v1) if isinstance(v1, float) else "%+d" % (v2 - v1)
                    print("  %-34s %12s -> %-12s %s" % (k, v1, v2, delta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
