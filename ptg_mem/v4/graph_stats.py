# -*- coding: utf-8 -*-
r"""PTG_V4/graph_stats.py — диагностика графа архива против нулевой модели.

ЗАЧЕМ. Теорема о сэндвиче (Ким–Ву; Бихейг–Илькович–Монтгомери) говорит, что граф с
ограниченной степенью ведёт себя как биномиальный `G(n,p)` только при `d >> log n`.
Ниже этого порога переносить на архив интуицию из теории случайных графов нельзя.
Этот модуль считает, где архив стоит относительно порога, и сравнивает его с нулевой
моделью той же плотности. Ценность — в ОТКЛОНЕНИИ: доля изолированных узлов выше
`exp(-c)` означает не разреженность, а перекос раскладки бюджета связей.

Ничего не пишет в архив. Читает `nodes.json` и `edges.json` любого store.

Запуск:
    python graph_stats.py <путь к .ptg>            # весь граф
    python graph_stats.py <путь к .ptg> --proximity  # только рёбра близости
"""
from __future__ import annotations

import collections
import json
import math
import os
import statistics
import sys

# Рёбра близости: для них «больше рёбер — больше свойства», то есть монотонность,
# без которой перенос по сэндвичу незаконен.
PROXIMITY_TYPES = ("reinforces", "returns_to", "refines", "fixes", "continues")
# Рёбра порядка и отрицания. В граф близости не входят: supersedes задаёт замену,
# contradicts — несовместимость; монотонность на них ложна.
ORDER_TYPES = ("supersedes", "contradicts")


def load_graph(ptg_dir: str):
    with open(os.path.join(ptg_dir, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    with open(os.path.join(ptg_dir, "edges.json"), encoding="utf-8") as f:
        edges = json.load(f)
    if isinstance(edges, dict):
        edges = list(edges.values())
    node_ids = list(nodes.keys()) if isinstance(nodes, dict) else [n["id"] for n in nodes]
    return node_ids, edges


def degrees(node_ids, edges, types=None):
    """Степени по неориентированному следу графа. Кратные рёбра между одной парой
    считаются один раз: нас интересует достижимость, а не число отношений."""
    seen = set()
    deg = collections.Counter()
    for e in edges:
        if types is not None and e.get("type") not in types:
            continue
        a, b = e.get("from"), e.get("to")
        if a is None or b is None or a == b:
            continue
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        deg[a] += 1
        deg[b] += 1
    return deg, len(seen)


def components(node_ids, edges, types=None):
    """Размер наибольшей компоненты связности — через объединение непересекающихся множеств."""
    parent = {v: v for v in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        if types is not None and e.get("type") not in types:
            continue
        a, b = e.get("from"), e.get("to")
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    sizes = collections.Counter(find(v) for v in node_ids)
    return sizes.most_common(1)[0][1] if sizes else 0


def report(ptg_dir: str, proximity_only: bool = False) -> dict:
    node_ids, edges = load_graph(ptg_dir)
    n = len(node_ids)
    types = PROXIMITY_TYPES if proximity_only else None
    deg, m = degrees(node_ids, edges, types)
    d = sorted(deg.values())
    c = (2.0 * m / n) if n else 0.0
    iso = n - len(deg)
    ln_n = math.log(n) if n > 1 else 0.0
    iso_null = n * math.exp(-c) if c > 0 else n
    giant = components(node_ids, edges, types)

    by_type = collections.Counter(e.get("type") for e in edges)

    print("=" * 70)
    print("граф:", "только близость" if proximity_only else "все типы рёбер")
    print("  узлов                     %d" % n)
    print("  рёбер (без кратных)       %d" % m)
    print("  средняя степень c         %.2f" % c)
    if d:
        print("  медиана / p90 / p99 / max %d / %d / %d / %d"
              % (d[len(d) // 2], d[int(len(d) * .9)], d[int(len(d) * .99)], d[-1]))
    print("  наибольшая компонента     %d (%.1f %%)" % (giant, 100.0 * giant / n if n else 0))
    print()
    print("-- против нулевой модели G(n,p) той же плотности --")
    print("  изолированных фактически  %d (%.2f %%)" % (iso, 100.0 * iso / n if n else 0))
    print("  изолированных по exp(-c)  %.0f (%.2f %%)"
          % (iso_null, 100.0 * math.exp(-c) if c > 0 else 100.0))
    if iso_null > 0:
        print("  ОТКЛОНЕНИЕ                %.1f x" % (iso / iso_null))
    print()
    print("-- условие теоремы о сэндвиче --")
    print("  ln n                      %.2f   (= порог связности по средней степени)" % ln_n)
    print("  режим сэндвича требует    d >> ln n, практически d >= %d" % max(30, int(3 * ln_n)))
    print("  фактическая c             %.2f   -> %s" % (
        c, "В РЕЖИМЕ" if c >= 3 * ln_n else
           ("выше порога связности, но вне режима" if c >= ln_n else "НИЖЕ ПОРОГА СВЯЗНОСТИ")))
    if c > 1:
        print("  оценка глубины обхода     ln n / ln c = %.1f шага" % (ln_n / math.log(c)))
    print()
    if not proximity_only:
        print("-- типы рёбер --")
        for t, k in by_type.most_common():
            mark = "близость" if t in PROXIMITY_TYPES else ("порядок" if t in ORDER_TYPES else "?")
            print("  %-12s %6d   %s" % (t, k, mark))
    print("=" * 70)

    return {
        "n": n, "m": m, "c": c, "isolated": iso, "isolated_null": iso_null,
        "ln_n": ln_n, "giant": giant,
        "max_degree": d[-1] if d else 0,
        "median_degree": d[len(d) // 2] if d else 0,
        "in_sandwich_regime": bool(c >= 3 * ln_n),
        "by_type": dict(by_type),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    report(sys.argv[1], proximity_only="--proximity" in sys.argv)
