# -*- coding: utf-8 -*-
r"""PTG_V4/selftest.py — проверка правила присоединения, а не вера в него.

Три проверки, каждая отвечает на свой вопрос.

1. ВЛОЖЕНИЕ (сам сэндвич). Построить пороговый граф при двух порогах и проверить,
   что граф с бюджетом лежит между ними: G(tau_hi) ⊆ graft ⊆ G(tau_lo). Теорема
   обещает это для равномерно случайного регулярного графа; у нас граф семантический,
   поэтому вложение — величина ИЗМЕРЯЕМАЯ, а не данная.

2. ПРОТИВ ЖАДНОГО top-k. На одних и тех же кандидатах сравнить раскладку степеней у
   жадного отбора и у правила с бюджетом: максимум, медиана, доля изолированных
   против нулевой модели exp(-c).

3. РЕЖИМ. Проверить, что `suggest_budget` выводит бюджет в режим `d >> ln n`, а
   `traversal_depth` даёт не константу.

Запуск:  python selftest.py
Зависимости: только стандартная библиотека и numpy.
"""
from __future__ import annotations

import math
import random

import numpy as np

from sandwich_graft import graft, suggest_budget, traversal_depth, degree_summary


def make_semantic_cloud(n=3000, dim=64, clusters=40, seed=1):
    """Облако эмбеддингов с кластерами — грубая модель архива: не равномерный шум,
    а сгустки, как у настоящих атомов."""
    rnd = np.random.default_rng(seed)
    centers = rnd.normal(size=(clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    owner = rnd.integers(0, clusters, size=n)
    v = centers[owner] + 0.45 * rnd.normal(size=(n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def candidate_pairs(vecs, per_node=60, seed=2):
    """Кандидаты: для каждого узла — его ближайшие соседи. Возвращается список,
    отсортированный по убыванию близости (как ждёт graft)."""
    n = len(vecs)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -2.0)
    out = []
    idx = np.argpartition(-sims, kth=per_node, axis=1)[:, :per_node]
    for i in range(n):
        for j in idx[i]:
            if i < j:
                out.append((str(i), str(int(j)), float(sims[i, j])))
            elif j < i:
                out.append((str(int(j)), str(i), float(sims[i, j])))
    out = list({(u, v): (u, v, s) for u, v, s in out}.values())
    out.sort(key=lambda t: -t[2])
    return out


def threshold_graph(cands, tau):
    return {(u, v) for u, v, s in cands if s >= tau}


def greedy_topk(cands, k):
    """То, что делает обычный top-k: идём по убыванию близости, берём, пока у узла
    есть слоты. Именно здесь хаб забирает слоты первым."""
    deg = {}
    taken = []
    seen = set()
    for u, v, s in cands:
        key = (u, v)
        if key in seen:
            continue
        if deg.get(u, 0) >= k or deg.get(v, 0) >= k:
            continue
        seen.add(key)
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
        taken.append((u, v, s))
    return taken


def ptg_current_rule(vecs, seed=3):
    """Симуляция нынешнего правила ptg_core: атом цепляется К ДВУМ целям — к голове
    своей ветки и к лучшей точке возврата. Голова ветки при этом собирает всех детей.

    Нужна не для красоты: она показывает, что измеренная на архиве картина (медиана 3
    при максимуме в сотни и проценты изолированных) следует из САМОГО ПРАВИЛА, а не из
    разреженности данных."""
    rnd = random.Random(seed)
    n = len(vecs)
    heads = []           # головы веток
    taken, deg = [], {}
    # Пороги ptg_core (0.47/0.55) откалиброваны под настоящие эмбеддинги; на синтетическом
    # облаке они отсекли бы всё. Берём их квантилями фактического распределения близостей,
    # чтобы воспроизводилась МЕХАНИКА правила, а не чужая константа.
    _s = (vecs[:400] @ vecs[:400].T).ravel()
    BRANCH_THRESH = float(np.quantile(_s, 0.80))
    RETURN_THRESH = float(np.quantile(_s, 0.95))
    for i in range(n):
        v = vecs[i]
        if not heads:
            heads.append(i)
            continue
        sims = [(float(v @ vecs[h]), h) for h in heads]
        best_sim, parent = max(sims)
        if best_sim < BRANCH_THRESH:
            heads.append(i)          # новая ветка: атом ни к чему не привязан
            continue
        for _ in range(2):   # continues + refines/fixes на ту же цель
            taken.append((str(parent), str(i), best_sim))
            deg[parent] = deg.get(parent, 0) + 1
            deg[i] = deg.get(i, 0) + 1
        # точка возврата: изредка, к случайной старой голове выше порога
        if len(heads) > 2 and rnd.random() < 0.25:
            s, old = max((float(v @ vecs[h]), h) for h in heads if h != parent)
            if s > RETURN_THRESH:
                taken.append((str(old), str(i), s))
                deg[old] = deg.get(old, 0) + 1
                deg[i] = deg.get(i, 0) + 1
        heads[heads.index(parent)] = i if rnd.random() < 0.5 else parent
    return taken


def main():
    n = 3000
    vecs = make_semantic_cloud(n=n)
    node_ids = [str(i) for i in range(n)]
    cands = candidate_pairs(vecs)
    d = suggest_budget(n)
    print("узлов %d, кандидатов %d, ln n = %.2f, бюджет d = %d" % (n, len(cands), math.log(n), d))
    print()

    g = graft(cands, budget=d)
    gs = degree_summary(g, node_ids)
    tk = greedy_topk(cands, d)
    ts = degree_summary(tk, node_ids)

    cur = ptg_current_rule(vecs)
    cs = degree_summary(cur, node_ids)

    print("-- 2. три правила присоединения на одних данных --")
    row = "  %-22s %10s %8s %8s"
    print(row % ("", "ptg сейчас", "top-k", "graft"))
    print(row % ("рёбер", cs["m"], ts["m"], gs["m"]))
    print(row % ("средняя степень", "%.2f" % cs["c"], "%.2f" % ts["c"], "%.2f" % gs["c"]))
    print(row % ("медиана степени", cs["median"], ts["median"], gs["median"]))
    print(row % ("МАКСИМУМ степени", cs["max"], ts["max"], gs["max"]))
    print(row % ("изолированных", cs["isolated"], ts["isolated"], gs["isolated"]))
    print(row % ("их же по exp(-c)", "%.0f" % cs["isolated_null"],
                 "%.0f" % ts["isolated_null"], "%.0f" % gs["isolated_null"]))
    ok_iso = gs["isolated"] <= max(1.0, gs["isolated_null"]) * 1.5
    ok_hub = gs["max"] <= 2 * d                      # жёсткий потолок graft
    ok_vs_cur = gs["isolated"] <= cs["isolated"]     # сирот меньше, чем у нынешнего правила
    print("  изолированные в пределах нулевой модели  %s" % ("да" if ok_iso else "НЕТ"))
    print("  хаб не выше жёсткого потолка 2d = %-3d    %s" % (2 * d, "да" if ok_hub else "НЕТ"))
    print("  сирот меньше, чем у нынешнего правила    %s" % ("да" if ok_vs_cur else "НЕТ"))
    ok_hub = ok_hub and ok_vs_cur
    print()

    print("-- 1. вложение между двумя пороговыми графами --")
    taken = {(u, v) for u, v, _ in g}
    sims = sorted((s for _, _, s in cands), reverse=True)
    # пороги подбираются по плотности: верхний — по числу принятых рёбер, нижний — вдвое реже
    tau_hi = sims[min(len(sims) - 1, len(taken))]
    tau_lo = sims[min(len(sims) - 1, len(taken) * 3)]
    hi = threshold_graph(cands, tau_hi)
    lo = threshold_graph(cands, tau_lo)
    inner = len(hi & taken) / len(hi) if hi else 0.0
    outer = len(taken & lo) / len(taken) if taken else 0.0
    print("  нижний порог tau_hi=%.3f: рёбер %d, из них в graft — %.1f %%" % (tau_hi, len(hi), 100 * inner))
    print("  верхний порог tau_lo=%.3f: рёбер %d, graft внутри — %.1f %%" % (tau_lo, len(lo), 100 * outer))
    print("  (идеальный сэндвич — 100 %% и 100 %%; отклонение есть цена бюджета)")
    print()

    print("-- 3. режим и глубина обхода --")
    print("  d = %d против ln n = %.2f  ->  d/ln n = %.1f  (режим требует >> 1)"
          % (d, math.log(n), d / math.log(n)))
    for c in (4.91, 10.0, 30.0):
        print("  при c = %5.2f глубина обхода = %d шага" % (c, traversal_depth(n, c)))
    print()
    print("ИТОГ:", "правило работает" if (ok_iso and ok_hub) else "ПРОВЕРКА НЕ ПРОШЛА")
    return 0 if (ok_iso and ok_hub) else 1


if __name__ == "__main__":
    raise SystemExit(main())
