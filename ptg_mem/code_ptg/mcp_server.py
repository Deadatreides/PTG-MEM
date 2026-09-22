"""
code_ptg_mcp_server.py — MCP Toolset для Code PTG.

Прямое зеркало ptg_mcp_server.py: те же принципы (read-mostly слой поверх
уже построенного архива, ориентир -> целевой запрос -> сборка), тот же
паттерн имён (ptg_* -> code_ptg_*). Часть инструментов вызывает методы
НАСТОЯЩЕГО Archive из vendor_ptg/ptg_core.py напрямую (root_overview,
list_branches, branch_lineage, node_relations, edges_by_type) — они
домен-агностичны, Archive не знает и не обязан знать, что теперь в него
графтятся код-карточки, а не чат-атомы. Остальные — code_ptg-специфичны
(get_card/dependencies/dependents/risks/lineage) и используют CodeArchive
из code_ptg/archive.py.

Запуск (stdio, для Claude Desktop / claude mcp add):
    pip install "mcp[cli]"
    python code_ptg_mcp_server.py --project /path/to/code --store /path/to/store

Архив должен быть уже построен через `python cli.py index ...` — сервер
сам не индексирует и не пишет, только читает.
"""

import os
import sys
import argparse
import threading

sys.path.insert(0, ".")
from code_ptg.archive import CodeArchive, STRUCTURAL_EDGE_TYPES
from code_ptg.graph_engine import Embedder

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Нужен пакет mcp: pip install \"mcp[cli]\"", file=sys.stderr)
    raise

_archive: CodeArchive | None = None
_archive_lock = threading.Lock()

mcp = FastMCP("code-ptg")


def _require_archive() -> CodeArchive:
    if _archive is None:
        raise RuntimeError(
            "Архив не загружен. Запустите сервер с --project/--store, указывающими "
            "на уже проиндексированный код (python cli.py index ...)."
        )
    return _archive


# ---------------------------------------------------------------------------
# Форма выдачи: главная метрика этого сервера — токены на прыжок
# ---------------------------------------------------------------------------
# Замер 20.09.2026 на store_code (4723 узла): карточка целиком — 3346 символов
# медианой, из них по делу 438 (14.1 %). Съедали три вещи:
#   * `vec` — 384 нуля, сериализованных ТЕКСТОМ. В nodes.json его нет вовсе:
#     ptg_core вешает вектор на узел при загрузке индекса, а FastMCP
#     сериализует numpy-массив через str();
#   * `layer_2` — 1606 символов на семь полей «Unknown», у каждого полный
#     блок провенанса. Заполненных полей во всём архиве ноль;
#   * поля графтинга чат-атомов (path_coherence, reinforcement_*, off_root,
#     placement_case…), которые для карточки кода не значат ничего.
#
# И отдельно — обход. `dependencies`/`dependents` отдавали рёбра с
# идентификаторами-хешами, поэтому узнать, ЧТО на другом конце, можно было
# только вызвав get_card на каждого соседа. Один прыжок стоил десятка полных
# карточек. Теперь у ребра сразу есть имя, вид и файл другого конца.
#
# Ничего не удаляется: `full=True` возвращает узел как есть, `provenance=True`
# добавляет блоки происхождения.

_CARD_KEEP = ("id", "qualified_name", "kind", "subkind", "file", "span",
              "status", "domain", "content_hash", "signature_hash",
              "removed_from_source", "text")

_EMPTY = (None, "", [], {}, "Unknown")


def _rel(path: str | None) -> str | None:
    """Путь относительно корня проекта.

    Хранится абсолютный, а у модулей он ещё и продублирован в qualified_name.
    Наружу отдаём относительный: это и короче на сотню символов с кириллицей,
    и не выносит структуру дисков пользователя в контекст модели.

    qualified_name при этом НЕ трогаем ни при каких условиях — по нему
    адресуются все остальные инструменты, и подмена сломала бы переход
    «увидел соседа -> запросил его карточку».
    """
    if not path or _archive is None:
        return path
    try:
        return os.path.relpath(path, _archive.project_folder)
    except (ValueError, TypeError):
        return path


def _short_qname(qname: str | None, file_path: str | None) -> str | None:
    """Имя модуля наружу — относительным путём, а не абсолютным.

    Абсолютные пути с кириллицей стоят по сотне символов на строку и выносят
    структуру дисков пользователя в контекст модели. Обратный ход обеспечивает
    `_find_card`: он принимает и относительный путь, и абсолютный.
    """
    if qname and file_path and qname == file_path:
        return _rel(file_path)
    return qname


def _find_card(arc, qualified_name: str):
    """Карточка по имени. Для модулей принимает и относительный путь, и
    абсолютный — наружу отдаётся относительный, и он обязан работать на вход."""
    node = arc.get_active_card_by_qname(qualified_name)
    if node is not None:
        return node
    if _archive is not None and not os.path.isabs(qualified_name):
        cand = os.path.normpath(os.path.join(_archive.project_folder, qualified_name))
        node = arc.get_active_card_by_qname(cand)
        if node is not None:
            return node
    raise ValueError(f"Карточка {qualified_name!r} не найдена среди активных.")


def _name_of(card_id: str) -> str:
    """Имя карточки по идентификатору — для списков, где хранятся хеши."""
    if _archive is None:
        return card_id
    n = _archive.engine.archive.nodes.get(card_id)
    if n is None:
        return card_id
    return _short_qname(n.get("qualified_name"), n.get("file_path")) or card_id


def _slim_layer_1(l1: dict) -> dict:
    """Факты статического анализа без пустот: нулевая сложность и пустые
    списки вызовов/исключений не несут сведений, но занимают место.

    `children` хранятся идентификаторами-хешами — наружу отдаём именами,
    иначе «что лежит в этом модуле» снова требует вызова на каждого ребёнка,
    а это ровно та цена, ради снятия которой всё и затевалось."""
    out = {k: v for k, v in (l1 or {}).items() if v not in _EMPTY and v is not False}
    kids = out.get("children")
    if isinstance(kids, list) and kids:
        out["children"] = [_name_of(c) for c in kids]
    return out


def _slim_layer_2(l2: dict) -> dict | str:
    """Только заполненные поля. Если не заполнено ни одно — одна строка
    вместо полутора тысяч символов «Unknown» с провенансом."""
    filled = {}
    for name, field in (l2 or {}).items():
        val = (field or {}).get("value") if isinstance(field, dict) else field
        if val not in _EMPTY:
            filled[name] = val
    if not filled:
        return "не заполнен (нужен интерпретатор при индексации)"
    return filled


def slim_card(node: dict, provenance: bool = False, full: bool = False) -> dict:
    if full:
        return node
    qn, fp = node.get("qualified_name"), node.get("file_path")
    txt = node.get("text")
    if txt and _archive is not None:
        txt = txt.replace(_archive.project_folder + os.sep, "").replace(
            _archive.project_folder, "")
    node = dict(node, qualified_name=_short_qname(qn, fp), text=txt,
                file=(_rel(fp) if fp != qn else None))
    out = {k: node[k] for k in _CARD_KEEP if k in node and node[k] not in _EMPTY}
    l1 = _slim_layer_1(node.get("layer_1"))
    if l1:
        out["layer_1"] = l1
    out["layer_2"] = _slim_layer_2(node.get("layer_2"))
    if provenance:
        for k in ("layer_1_provenance", "layer_2"):
            if k in node:
                out.setdefault("provenance", {})[k] = node[k]
    return out


def _brief(node: dict | None, card_id: str) -> dict:
    """Кто на другом конце ребра — ровно столько, чтобы решить, идти туда или нет."""
    if node is None:
        return {"card_id": card_id, "qualified_name": None, "замечание": "узел не найден"}
    qn = node.get("qualified_name")
    fp = node.get("file_path")
    b = {"card_id": card_id, "qualified_name": _short_qname(qn, fp),
         "kind": node.get("kind"), "subkind": node.get("subkind")}
    # у модуля qualified_name и есть путь — вторым полем его не повторяем
    if fp != qn:
        b["file"] = _rel(fp)
    if node.get("span"):
        b["span"] = node["span"]
    return {k: v for k, v in b.items() if v not in _EMPTY}


def _edges_with_names(arc, edges: list, other_end: str) -> list:
    nodes = arc.engine.archive.nodes
    out = []
    for e in edges:
        cid = e[other_end]
        row = {"type": e["type"]}
        row.update(_brief(nodes.get(cid), cid))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# 1. Ориентир — домен-агностичные вызовы настоящего Archive
# ---------------------------------------------------------------------------
@mcp.tool()
def code_ptg_root_overview(top_branches: int = 12) -> dict:
    """Точка входа: сколько всего code-узлов/веток/файлов и top-N веток по
    накопленному весу. Вызывай ЭТО ПЕРВЫМ. Буквально Archive.root_overview()
    из настоящего PTG — здесь "ветки" это ветки concept-графтинга карточек,
    а не структурные contains/calls (для структуры см. code_ptg_dependents)."""
    with _archive_lock:
        return _require_archive().engine.archive.root_overview(top_branches=top_branches)


@mcp.tool()
def code_ptg_list_branches(include_dormant: bool = False, limit: int = 30) -> list:
    """Список веток concept-графа (не структурного графа calls/contains!) —
    ветки здесь означают "повторяющаяся инженерная идея/паттерн, встреченная
    в нескольких карточках", ровно то же понятие, что в Text PTG для мыслей."""
    with _archive_lock:
        return _require_archive().engine.archive.list_branches(
            include_dormant=include_dormant, limit=limit,
        )


@mcp.tool()
def code_ptg_branch_lineage(branch_id: str, max_atoms: int = 40, text_chars: int = 320) -> dict:
    """Карточки конкретной concept-ветки по порядку (root -> head), с
    инлайн-аннотациями supersedes/fixes/contradicts/refines — буквально
    Archive.branch_lineage()."""
    with _archive_lock:
        result = _require_archive().engine.archive.branch_lineage(
            branch_id, max_atoms=max_atoms, text_chars=text_chars
        )
        if result is None:
            raise ValueError(f"Ветка {branch_id} не найдена.")
        return result


# ---------------------------------------------------------------------------
# 2. Целевой запрос по конкретной карточке — code_ptg-специфичные вызовы
# ---------------------------------------------------------------------------
@mcp.tool()
def code_ptg_get_card(qualified_name: str, provenance: bool = False,
                      full: bool = False) -> dict:
    """Карточка по qualified_name, ТЕКУЩАЯ активная версия (status=active).

    Отдаёт факты статического анализа (layer_1: параметры, вызовы наружу,
    сложность, размер, span) без пустых полей и без вектора. Незаполненный
    layer_2 сворачивается в одну строку. `provenance=True` добавляет блоки
    происхождения, `full=True` — узел как есть. Для истории версий —
    code_ptg_lineage."""
    with _archive_lock:
        node = _find_card(_require_archive(), qualified_name)
        return slim_card(node, provenance=provenance, full=full)


@mcp.tool()
def code_ptg_dependencies(qualified_name: str) -> list:
    """Исходящие структурные связи карточки (calls/contains/implements/
    tests/imports) — детерминированные факты статического анализа, НЕ
    concept-графтинг. Что эта карточка использует.

    У каждой связи сразу стоит имя, вид и файл другого конца, поэтому за
    прыжок не надо дёргать get_card на каждого соседа."""
    with _archive_lock:
        arc = _require_archive()
        node = _find_card(arc, qualified_name)
        return _edges_with_names(arc, arc.get_dependencies(node["id"]), "to")


@mcp.tool()
def code_ptg_dependents(qualified_name: str) -> list:
    """Входящие структурные связи — кто использует эту карточку. Прямой
    (1-hop) срез; для полного каскада см. code_ptg_risks.

    У каждой связи сразу стоит имя, вид и файл другого конца."""
    with _archive_lock:
        arc = _require_archive()
        node = _find_card(arc, qualified_name)
        return _edges_with_names(arc, arc.get_dependents(node["id"]), "from")


@mcp.tool()
def code_ptg_risks(qualified_name: str, max_hops: int = 3) -> list:
    """Обратный обход ТОЛЬКО по структурным рёбрам (contains/calls/
    implements/tests/imports): что будет затронуто при изменении данной
    карточки. Вычисляется по запросу, никогда не хранится как факт в самой
    карточке (canonical spec §1.1: риски — производная операция)."""
    with _archive_lock:
        arc = _require_archive()
        node = _find_card(arc, qualified_name)
        return arc.get_risk_impact(node["id"], max_hops=max_hops)


@mcp.tool()
def code_ptg_lineage(qualified_name: str) -> list:
    """История версий карточки через supersedes/continues-рёбра — все
    версии, включая status=superseded, не удаляются никогда (append-only,
    как и в Text PTG)."""
    with _archive_lock:
        arc = _require_archive()
        node = _find_card(arc, qualified_name)
        return arc.get_lineage(node["id"])


@mcp.tool()
def code_ptg_edges_by_type(edge_type: str, node_id: str = None, limit: int = 50) -> list:
    """Все рёбра заданного типа (calls/contains/implements/tests/imports/
    supersedes/continues/reinforces/contradicts/fixes/refines/returns_to),
    опционально отфильтрованные по узлу. Буквально Archive.edges_by_type()."""
    with _archive_lock:
        return _require_archive().engine.archive.edges_by_type(
            edge_type, node_id=node_id, limit=limit,
        )


@mcp.tool()
def code_ptg_node_relations(node_id: str, depth: int = 1) -> dict:
    """Типизированные семантические связи узла (contradicts/fixes/refines/
    supersedes/returns_to) — буквально Archive.node_relations(). Для
    структурных связей (calls/contains) используй code_ptg_dependencies/
    code_ptg_dependents."""
    with _archive_lock:
        return _require_archive().engine.archive.node_relations(node_id, depth=depth)


# ---------------------------------------------------------------------------
# 3. Слой близости под бюджетом (PTG_V4/rebalance.py)
# ---------------------------------------------------------------------------
_PROX = {"dir": None, "mtime": None, "adj": None}


def _load_proximity():
    """Слой близости, разложенный под бюджет степеней.

    Зачем отдельно от структурных рёбер: `calls`/`contains` отвечают на вопрос
    «что вызывает что», а этот слой — на вопрос «что похоже по смыслу», и у
    него, в отличие от структурного, нет ни сирот, ни хабов. Замер 21.09.2026
    на store_code_v2 (5240 узлов): степень 3.16 -> 24.40, максимум 290 -> 52,
    изолированных 108 -> 0, глубина обхода 8 -> 3 шага.
    """
    d = _PROX["dir"]
    if not d:
        return None
    path = os.path.join(d, "edges_proximity.json")
    if not os.path.exists(path):
        return None
    mt = os.path.getmtime(path)
    if _PROX["mtime"] == mt and _PROX["adj"] is not None:
        return _PROX["adj"]
    import json
    adj: dict[str, list] = {}
    with open(path, encoding="utf-8") as f:
        for e in json.load(f):
            adj.setdefault(e["from"], []).append((e["to"], e.get("sim", 0.0)))
            adj.setdefault(e["to"], []).append((e["from"], e.get("sim", 0.0)))
    for k in adj:
        adj[k].sort(key=lambda t: -t[1])
    _PROX.update({"mtime": mt, "adj": adj})
    return adj


@mcp.tool()
def code_ptg_neighbors(qualified_name: str, k: int = 12) -> dict:
    """Смысловые соседи карточки — похожие по смыслу, а не связанные вызовом.

    Дополняет dependencies/dependents: те идут по структуре («кто кого
    вызывает»), этот — по близости интерпретаций, разложенной под бюджет
    степеней. Каждый сосед сразу назван, так что за прыжок не нужен get_card."""
    adj = _load_proximity()
    if adj is None:
        return {"ошибка": "слой близости не подключён: запустите "
                          "PTG_V4/rebalance.py на .code_ptg и передайте серверу "
                          "--proximity <каталог с edges_proximity.json>"}
    with _archive_lock:
        arc = _require_archive()
        node = _find_card(arc, qualified_name)
        got = adj.get(node["id"]) or []
        nodes = arc.engine.archive.nodes
        return {
            "карточка": _short_qname(node.get("qualified_name"), node.get("file_path")),
            "степень": len(got),
            "соседи": [dict(_brief(nodes.get(i), i), близость=round(s, 4))
                       for i, s in got[:k]],
        }


# ---------------------------------------------------------------------------
# 4. Здоровье эмбеддера
# ---------------------------------------------------------------------------
@mcp.tool()
def code_ptg_lm_studio_status() -> dict:
    """Какой эмбеддер настроен и жив ли он.

    Нужен только при ИНДЕКСАЦИИ (без него карточки не графтятся по смыслу) и
    для будущего code_ptg_search. Чтение графа — карточки, зависимости,
    риски, соседи — работает без эмбеддера всегда.

    Если задана переменная CODE_PTG_MODEL, проверяется внутрипроцессный
    эмбеддер, и никакого сервера на localhost не требуется вовсе."""
    if os.environ.get("CODE_PTG_MODEL"):
        from code_ptg.local_embedder import LocalEmbedder
        e = LocalEmbedder()
        ok = e.test_connection()
        return {"connected": ok, **e.status()}
    e = Embedder()
    ok = e.test_connection()
    return {"connected": ok, "backend": "lm-studio",
            "model": e.model if ok else None,
            "замечание": "LM Studio не обязателен: задайте CODE_PTG_MODEL, и "
                         "эмбеддер поднимется внутри процесса"}


def main():
    global _archive
    parser = argparse.ArgumentParser(description="Code PTG MCP Toolset")
    parser.add_argument("--project", required=True, help="Папка с проиндексированным кодом")
    parser.add_argument("--store", default=None, help="Папка с .code_ptg/ (если отдельно от --project)")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--proximity", default=None,
                        help="каталог с edges_proximity.json от PTG_V4/rebalance.py — "
                             "слой близости под бюджетом степеней (code_ptg_neighbors)")
    args = parser.parse_args()
    _PROX["dir"] = args.proximity

    _archive = CodeArchive(args.project, store_dir=args.store, use_embeddings=not args.no_embeddings)
    if not _archive.load():
        print(f"ВНИМАНИЕ: архив не найден в {args.store or args.project} — "
              f"запустите сначала `python cli.py index ...`", file=sys.stderr)

    mcp.run()


if __name__ == "__main__":
    main()
