# -*- coding: utf-8 -*-
r"""PTG_V4/mcp_launch_v4.py — сервер MCP, поднятый через PTG_V4.

Не форк прежнего лаунчера, а надстройка: `mcp_launch` импортируется целиком (со всей его
настройкой путей, ленивой загрузкой архива и подъёмом эмбеддера), а поверх регистрируются
инструменты V4. Прежние инструменты работают как работали.

ЧТО ДОБАВЛЯЕТ V4:

* `ptg_v4_graph_health` — состояние графа против нулевой модели: средняя степень против
  порога связности `ln n`, изолированные против `exp(−c)`, глубина обхода `ln n / ln c` и
  ответ на вопрос, в режиме ли архив (`d ≫ ln n`). Раньше это спрашивать было не у кого.

* `ptg_v4_neighbors` — соседи узла в графе близости, пересобранном под бюджет: у него нет
  сирот (было 470) и нет хабов (максимум 60 против 386), поэтому обход по нему сходится за
  3 шага вместо 8.

* `ptg_v4_update_plan` — что надо переобработать при обновлении: добавленные, изменённые,
  удалённые и «тронутые, но те же». Отвечает без участия человека и без чтения дерева
  целиком.

ЕДИНИЦА СЧЁТА. Работа меряется в ФАЙЛАХ, бюджет связей — в УЗЛАХ: `ln n` в условии режима
`d ≫ ln n` берётся от размера графа. Здесь стояло число файлов манифеста, и на 2002 файлах
выходил бюджет 23 вместо 28 по 9745 узлам — ниже порога, в который граф и приводится.
Оба инструмента считают блок `бюджет_связей` одной функцией от одного `n` и обязаны
совпадать почленно; когда живой архив расходится с `out/report.json`, оба возвращают
`расхождение_архив_отчёт`.

Запуск — тот же, что у прежнего лаунчера; путь прописан в `.mcp.json`.
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SNAP = os.path.join(ROOT, "trace-probe", "ptg", "_snapshot_run")
OUT = os.path.join(HERE, "out")

sys.path.insert(0, HERE)
sys.path.insert(0, SNAP)

import mcp_launch                                    # noqa: E402  настраивает всё
import ptg_mcp_server as srv                         # noqa: E402
from sandwich_graft import nodes_until_next_budget, suggest_budget   # noqa: E402

_edges_cache = {"mtime": None, "adj": None}


# --- единица счёта: n — это УЗЛЫ ---------------------------------------------
# Бюджет связей считается по числу узлов графа (теорема о сэндвиче, d >> ln n), и
# только по нему. Подстановка числа файлов манифеста — не приближение, а другая
# величина: 2002 файла дают 23, те же файлы дают 9745 узлов и бюджет 28. Файл — это
# десятки атомов, и отношение между ними не константа.
#
# Источников числа узлов два, и они могут разойтись. Живой архив — это то, что MCP
# отдаёт сейчас (store в mcp_launch.STORE). report.json пишет rebalance.py по тому
# .ptg, на котором его гоняли. Если store переключили, а rebalance не перегнали, эти
# два числа разные, и тогда оба инструмента обязаны сказать об этом вслух, а не
# отвечать каждый своим n.


def _read_report() -> dict | None:
    path = os.path.join(OUT, "report.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _live_archive():
    """Архив, если он УЖЕ в памяти.

    Загрузку не форсируем. mcp_launch грузит архив фоновым потоком (17–54 с), и
    заставлять инструмент, которому нужно одно число, ждать эту загрузку незачем:
    пока её нет, отвечает report.json, а источник указывается в выдаче.
    """
    for mod in (srv, mcp_launch):
        arc = getattr(mod, "_archive", None)
        # Пустой архив — не источник n: ln 0 не определён, а bool({}) уже False,
        # поэтому проверка на None здесь не годится.
        if getattr(arc, "nodes", None):
            return arc
    return None


def _nodes_n(rep: dict | None = None) -> tuple[int | None, str]:
    """(число узлов, откуда взято). Живой архив точнее отчёта — он и первый."""
    arc = _live_archive()
    if arc is not None:
        return len(arc.nodes), "архив"
    if rep is None:
        rep = _read_report()
    if rep and rep.get("n"):
        return int(rep["n"]), "out/report.json"
    return None, "нет"


def _n_mismatch(rep: dict | None = None) -> dict | None:
    """Расхождение живого архива с тем, по которому снят report.json."""
    arc = _live_archive()
    if rep is None:
        rep = _read_report()
    if arc is None or not rep or not rep.get("n"):
        return None
    if len(arc.nodes) == rep["n"]:
        return None
    return {
        "узлов_в_живом_архиве": len(arc.nodes),
        "узлов_в_отчёте": rep["n"],
        "отчёт_снят_с": rep.get("source"),
        "замечание": "rebalance.py гонялся на другом store — граф в out/ описывает "
                     "не тот архив, который сейчас отдаёт MCP; бюджет считается по "
                     "живому архиву, статистика графа — по отчёту",
    }


def _budget_block(rep: dict | None = None) -> dict:
    """Бюджет связей по числу УЗЛОВ — общий для обоих инструментов."""
    n, src = _nodes_n(rep)
    if n is None:
        return {"ошибка": "число узлов неизвестно: архив не загружен и нет out/report.json"}
    headroom = nodes_until_next_budget(n)
    return {
        "узлов": n,
        "источник_n": src,
        "ln_n": round(math.log(n), 2),
        "бюджет_d": suggest_budget(n),
        "запас_узлов_до_следующей_ступени": headroom,
        "ступень_сдвинется_на_узле": n + headroom,
    }


def _load_v4_edges():
    path = os.path.join(OUT, "edges_proximity.json")
    if not os.path.exists(path):
        return None
    mt = os.path.getmtime(path)
    if _edges_cache["mtime"] == mt and _edges_cache["adj"] is not None:
        return _edges_cache["adj"]
    adj = {}
    with open(path, encoding="utf-8") as f:
        for e in json.load(f):
            adj.setdefault(e["from"], []).append((e["to"], e.get("sim", 0.0)))
            adj.setdefault(e["to"], []).append((e["from"], e.get("sim", 0.0)))
    for k in adj:
        adj[k].sort(key=lambda t: -t[1])
    _edges_cache.update({"mtime": mt, "adj": adj})
    return adj


@srv.mcp.tool()
def ptg_v4_graph_health() -> dict:
    """Состояние графа архива против нулевой модели «связей нет».

    Возвращает среднюю степень, порог связности ln n, долю изолированных против exp(-c),
    оценку глубины обхода и признак режима переноса d >> ln n."""
    r = _read_report()
    if r is None:
        return {"ошибка": "нет out/report.json — сначала rebalance.py"}
    n = r["n"]
    ln_n = math.log(n) if n > 1 else 0.0
    after = r["after"]
    c = after["c"]
    out = {
        "узлов": n,
        "ln_n_порог_связности": round(ln_n, 2),
        "средняя_степень": round(c, 2),
        "в_режиме_d_много_больше_ln_n": bool(c >= 3 * ln_n),
        "изолированных": after["isolated"],
        "изолированных_по_нулевой_модели": round(after["isolated_null"], 1),
        "максимум_степени": after["max"],
        "глубина_обхода": int(math.ceil(ln_n / math.log(c))) if c > 1 else 1,
        "было": {"средняя_степень": round(r["before"]["c"], 2),
                 "изолированных": r["before"]["isolated"],
                 "максимум_степени": r["before"]["max"]},
        "слой_порядка_рёбер": r.get("order_layer_edges"),
        # Тот же блок, что отдаёт ptg_v4_update_plan, и считается он той же
        # функцией от того же n — сверять два инструмента можно почленно.
        "бюджет_связей": _budget_block(r),
        "бюджет_в_сборке": r.get("budget"),
    }
    mism = _n_mismatch(r)
    if mism:
        out["расхождение_архив_отчёт"] = mism
    return out


@srv.mcp.tool()
def ptg_v4_neighbors(node_id: str, k: int = 12) -> dict:
    """Соседи узла в графе близости под бюджетом — без сирот и без хабов.

    Отличие от ptg_node_relations: тот ходит по типизированным рёбрам исходного архива,
    где у половины узлов степень 2, а у хабов 386. Здесь степень ограничена бюджетом."""
    adj = _load_v4_edges()
    if adj is None:
        return {"ошибка": "нет out/edges_proximity.json — сначала rebalance.py"}
    got = adj.get(node_id)
    if not got:
        return {"узел": node_id, "соседей": 0,
                "замечание": "узла нет в графе близости или он изолирован"}
    return {"узел": node_id, "степень": len(got),
            "соседи": [{"id": i, "близость": round(s, 4)} for i, s in got[:k]]}


@srv.mcp.tool()
def ptg_v4_update_plan() -> dict:
    """Что надо переобработать при обновлении архива — без чтения дерева целиком.

    Решение принимается по манифесту (размер и время), содержимое читается только у
    расходящихся файлов. Отвечает и на вопрос, сдвинулся ли порог режима: он логарифмичен
    по числу УЗЛОВ, поэтому дописывание обычно графа не пересматривает.

    Две величины здесь разной природы и не смешиваются: работа меряется в ФАЙЛАХ
    (их надо переэмбеддить), а бюджет связей — в УЗЛАХ (по ним считается ln n)."""
    import incremental
    mpath = os.path.join(OUT, "manifest.json")
    if not os.path.exists(mpath):
        return {"ошибка": "нет out/manifest.json — сначала incremental.py manifest"}
    with open(mpath, encoding="utf-8") as f:
        old = json.load(f)
    d = incremental.diff(old["root"], old)
    n_files = old.get("n_files", 0)
    out = {
        "добавлено": len(d["added"]), "изменено": len(d["changed"]),
        "удалено": len(d["removed"]),
        "тронуто_но_то_же": len(d["touched_but_same"]),
        "без_изменений": d["unchanged"],
        "файлов_к_обработке": len(d["added"]) + len(d["changed"]),
        "файлов_в_манифесте_было_станет":
            [n_files, n_files + len(d["added"]) - len(d["removed"])],
        # Новых узлов из этих файлов манифест не знает: файл даёт десятки атомов, и
        # сколько именно — решает фильтр и пороги ветвления. Поэтому «стало» по узлам
        # здесь не выдумывается, а вместо него даётся запас до следующей ступени.
        "бюджет_связей": _budget_block(),
        "примеры_добавленных": d["added"][:10],
        "примеры_изменённых": d["changed"][:10],
    }
    mism = _n_mismatch()
    if mism:
        out["расхождение_архив_отчёт"] = mism
    return out


def main() -> None:
    print("[ptg-mcp-v4] надстройка PTG_V4 над %s" % SNAP, file=sys.stderr)
    print("[ptg-mcp-v4] артефакты V4: %s" % OUT, file=sys.stderr)
    for f in ("edges_proximity.json", "centroids.npy", "manifest.json", "report.json"):
        p = os.path.join(OUT, f)
        print("[ptg-mcp-v4]   %-24s %s" % (f, "есть" if os.path.exists(p) else "НЕТ"),
              file=sys.stderr)
    mcp_launch.main()


if __name__ == "__main__":
    main()
