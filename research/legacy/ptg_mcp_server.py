"""
ptg_mcp_server.py — MCP Toolset для Personal Thought Graph (PTG).

НОВЫЙ ФАЙЛ. Ничего в ptg_core.py / main.py / gui_panels.py не заменяется —
это отдельный процесс, который читает уже построенный архив (.ptg/) и
предоставляет большой модели набор инструментов для САМОСТОЯТЕЛЬНОЙ
навигации по графу вместо выгрузки всего архива в контекст.

Почему отдельный процесс, а не встройка в main.py:
    main.py — desktop GUI на CustomTkinter, работает с мутируемым Archive
    в отдельном потоке (build/save). MCP-сервер — read-mostly слой,
    которому не нужен tkinter mainloop и который должен быть доступен
    внешнему клиенту (большой модели) по stdio/HTTP, а не через GUI-поток.
    Смешивать их в одном процессе означало бы редизайн main.py — вместо
    этого MCP-сервер импортирует Archive из ptg_core.py как библиотеку.

Философия инструментов (в соответствии с root_project.txt):
    Инструменты НЕ отдают модели атомы "все подряд". Каждый инструмент —
    это либо (а) ориентир/обзор с жёстким лимитом объёма, либо
    (б) целевой doto-запрос по конкретному node_id/branch_id, либо
    (в) итоговая сборка CONTEXT SNAPSHOT с бюджетом символов. Модель сама
    решает, в каком порядке их вызывать (см. системный prompt клиента):
        root_overview → list_branches → branch_lineage / node_relations
        → find_unresolved / edges_by_type → resolve_supersession
        → assemble_context_snapshot

Запуск (stdio, для Claude Desktop / claude mcp add):
    pip install "mcp[cli]"
    python ptg_mcp_server.py --folder /path/to/archive/folder

Архив должен быть уже построен через GUI (main.py -> "Построить архив")
или через Archive(folder).build() — сервер сам архив не строит и не
пишет: build/save-логика остаётся эксклюзивно в ptg_core.Archive,
здесь только Archive.load_if_exists().
"""

import sys
import argparse
import threading

from ptg_core import Archive, Embedder
from ptg_snapshot_store import save_agent_snapshot, list_snapshots, load_snapshot, save_full_context
from ptg_logging import get_logger

_log = get_logger("mcp")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Нужен пакет mcp: pip install \"mcp[cli]\"", file=sys.stderr)
    raise

# ---------------------------------------------------------------------------
# Глобальное состояние процесса: один Archive на один запущенный сервер
# (один MCP-сервер = один граф = одна папка с логами). Загрузка read-only:
# load_if_exists() не пишет на диск и не требует LM Studio для навигации —
# LM Studio нужен только для ptg_search / ptg_seed_by_query (эмбеддинг
# текста запроса).
# ---------------------------------------------------------------------------
_archive: Archive | None = None
_archive_lock = threading.Lock()

mcp = FastMCP("ptg-thought-graph")


def _require_archive() -> Archive:
    if _archive is None:
        raise RuntimeError(
            "Архив не загружен. Запустите сервер с --folder, указывающим на "
            "папку с уже построенным .ptg/ (main.py -> «Построить архив»)."
        )
    return _archive


# ---------------------------------------------------------------------------
# 1. Ориентация
# ---------------------------------------------------------------------------
@mcp.tool()
def ptg_root_overview(top_branches: int = 12) -> dict:
    """Точка входа. Кто этот проект (root_project), сколько всего узлов/
    веток/файлов, и top-N веток по накопленному весу. Вызывай ЭТО ПЕРВЫМ,
    прежде чем запрашивать что-то ещё — это стоит O(веток), а не O(атомов)."""
    with _archive_lock:
        return _require_archive().root_overview(top_branches=top_branches)


@mcp.tool()
def ptg_file_manifest(sort_by: str = "path") -> list:
    """ПАТЧ 15 — обзор ВСЕЙ файловой структуры проекта одним вызовом: что
    где лежит (folder/filename), сколько атомов-мыслей в файле, когда он
    последний раз менялся, какие ключевые концепты доминируют, с какими
    другими файлами он семантически связан (даже если лежат в разных
    папках и называются по-разному — например несколько версий README),
    и is_isolated — файл не связан ни с чем и слабо соотносится с общей
    темой проекта (частый признак забытого черновика/устаревшей ветки,
    случайно оставшейся в папке).

    Вызывай ЭТО ВТОРЫМ, сразу после ptg_root_overview, ДО семантического
    поиска — если проект захламлён (разные даты, версии, забытые папки),
    именно этот вызов даёт карту "что где" прежде, чем нырять в контент.
    sort_by: "path" (по умолчанию) | "folder" | "last_modified" (новые сначала)."""
    with _archive_lock:
        return _require_archive().file_manifest(sort_by=sort_by)


@mcp.tool()
def ptg_full_project_context(max_atom_chars: int = 4000, save_to_disk: bool = True) -> dict:
    """ПАТЧ 16 — ГЛАВНЫЙ инструмент для возобновления работы над
    захламлённым проектом в новой сессии: собирает ВЕСЬ проект (структура
    папок, file_manifest, незакрытые линии мышления, противоречия, ПОЛНОЕ
    содержимое по каждому файлу с разрешением supersedes) в один
    структурированный текст — АВТОМАТИЧЕСКИ, без поиска и без выбора
    seed-узлов (в отличие от ptg_assemble_snapshot, который собирает
    контекст под конкретный запрос).

    Не требует LM Studio (чисто чтение уже построенного графа — semantic
    search внутри не используется). При save_to_disk=True (по умолчанию)
    результат также сохраняется как читаемый .md-файл в
    <output_root>/FULL_PROJECT_CONTEXT_{timestamp}.md — можно открыть и
    сразу вставить в новый чат с любым ИИ-клиентом."""
    with _archive_lock:
        arc = _require_archive()
        _log.debug(f"ptg_full_project_context: max_atom_chars={max_atom_chars}, save_to_disk={save_to_disk}")
        result = arc.export_full_context(max_atom_chars=max_atom_chars)
        if save_to_disk:
            result["saved_path"] = save_full_context(arc.output_root, result)
        _log.info(f"ptg_full_project_context: {result['file_count']} файлов, "
                 f"{result['atom_count']} атомов, {result['approx_tokens']} токенов")
        return result


# ---------------------------------------------------------------------------
# 2. Семантический вход по запросу (нужен LM Studio)
# ---------------------------------------------------------------------------
@mcp.tool()
def ptg_search(query: str, top_k: int = 10, top_branches: int = 6) -> list:
    """Branch-first семантический поиск (ПАТЧ 10): query -> оценка веток по
    центроиду (+ лёгкий temporal-bias) -> top_branches веток -> обход их
    реальной топологии (atom_ids) -> харвест атомов с ограничением на файл
    внутри ветки. Возвращает top_k атомов с branch_id и similarity —
    отправная точка для дальнейшего исследования (используй node_id/branch_id
    результатов в ptg_node_relations / ptg_branch_lineage / ptg_assemble_snapshot).
    Требует запущенный LM Studio (для эмбеддинга query)."""
    with _archive_lock:
        arc = _require_archive()
        _log.debug(f"ptg_search: query={query!r}, top_k={top_k}, top_branches={top_branches}")
        results = arc.search_branch_first(query, top_branches=top_branches, top_k=top_k)
        _log.debug(f"ptg_search: найдено {len(results)} результат(ов)")
        out = []
        for sim, nid in results:
            n = arc.nodes[nid]
            out.append({
                "node_id": nid,
                "similarity": round(sim, 4),
                "branch_id": n.get("branch"),
                "file": n.get("file"),
                "status": n.get("status", "active"),
                "text_preview": n["text"][:250],
            })
        return out


@mcp.tool()
def ptg_search_legacy(query: str, top_k: int = 10) -> list:
    """File-first поиск (L1 файлы -> L2 файловый граф -> L3 атомы) —
    прежняя реализация search(), оставлена для обратной совместимости и
    для случаев, когда важна файловая, а не траекторная локальность."""
    with _archive_lock:
        arc = _require_archive()
        results = arc.search(query, top_k=top_k)
        out = []
        for sim, nid in results:
            n = arc.nodes[nid]
            out.append({
                "node_id": nid,
                "similarity": round(sim, 4),
                "branch_id": n.get("branch"),
                "file": n.get("file"),
                "status": n.get("status", "active"),
                "text_preview": n["text"][:250],
            })
        return out


# ---------------------------------------------------------------------------
# 3. Список веток
# ---------------------------------------------------------------------------
@mcp.tool()
def ptg_list_branches(include_dormant: bool = False,
                       min_root_sim: float = None,
                       file: str = None,
                       limit: int = 30) -> list:
    """Список веток (не атомов!) с их состоянием: вес, активность,
    variance тела ветки, есть ли незакрытая голова. По умолчанию дремлющие
    ветки скрыты (include_dormant=False) — включай их явно, если ищешь
    что-то старое/заброшенное. Используй branch_id результатов в
    ptg_branch_lineage."""
    with _archive_lock:
        return _require_archive().list_branches(
            include_dormant=include_dormant, min_root_sim=min_root_sim,
            file=file, limit=limit,
        )


# ---------------------------------------------------------------------------
# 4. Развёрнутая линия ветки
# ---------------------------------------------------------------------------
@mcp.tool()
def ptg_branch_lineage(branch_id: str, max_atoms: int = 40,
                        text_chars: int = 320) -> dict:
    """Атомы конкретной ветки по порядку (root -> head) с инлайн-
    аннотациями исходящих relations (supersedes/fixes/contradicts/refines).
    max_atoms ограничивает объём — если атомов больше, берутся последние
    (самые актуальные). Для полной истории вызывай с большим max_atoms."""
    with _archive_lock:
        result = _require_archive().branch_lineage(
            branch_id, max_atoms=max_atoms, text_chars=text_chars
        )
        if result is None:
            raise ValueError(f"Ветка {branch_id} не найдена.")
        return result


# ---------------------------------------------------------------------------
# 5. Отношения узла
# ---------------------------------------------------------------------------
@mcp.tool()
def ptg_node_relations(node_id: str, depth: int = 1,
                        types: list = None) -> dict:
    """Типизированные связи узла: contradicts / fixes / refines /
    supersedes / returns_to. depth=1 — только прямые соседи; depth>1 —
    BFS-цепочка (например: A fixes B, B supersedes C). Используй ПЕРЕД
    тем, как утверждать что-то как решённое — проверь, нет ли contradicts
    или более новой supersedes-версии."""
    with _archive_lock:
        t = tuple(types) if types else ("contradicts", "fixes", "refines",
                                         "supersedes", "returns_to")
        return _require_archive().node_relations(node_id, types=t, depth=depth)


# ---------------------------------------------------------------------------
# 6. Незакрытые линии мышления
# ---------------------------------------------------------------------------
@mcp.tool()
def ptg_unresolved(limit: int = 30, min_root_sim: float = 0.0,
                    only_active_branches: bool = True) -> list:
    """Все головы веток без продолжения и без returns_to — то есть
    оборванные линии рассуждения, ещё не доведённые до конца. Полезно
    перед началом новой задачи: проверить, нет ли незакрытых вопросов
    в этой предметной области (используй min_root_sim, чтобы отфильтровать
    периферийный шум)."""
    with _archive_lock:
        return _require_archive().find_unresolved(
            limit=limit, min_root_sim=min_root_sim,
            only_active_branches=only_active_branches,
        )


# ---------------------------------------------------------------------------
# 7. Разрешение supersedes-цепочки
# ---------------------------------------------------------------------------
@mcp.tool()
def ptg_resolve_supersession(node_id: str) -> dict:
    """Дан любой узел — вернуть актуальную версию по цепочке supersedes
    (если узел не заменялся — current_node_id == node_id). ВСЕГДА вызывай
    это перед тем, как процитировать старый атом как истину — он мог быть
    заменён более новым решением."""
    with _archive_lock:
        return _require_archive().resolve_supersession(node_id)


# ---------------------------------------------------------------------------
# 8. Типизированные подмножества рёбер
# ---------------------------------------------------------------------------
@mcp.tool()
def ptg_edges_by_type(edge_type: str, node_id: str = None,
                       branch_id: str = None, file: str = None,
                       limit: int = 50) -> list:
    """Все рёбра заданного типа (contradicts / fixes / refines /
    supersedes / returns_to / continues) с опциональным фильтром по узлу,
    ветке или файлу. Например: ptg_edges_by_type("contradicts") без
    фильтров -> все известные противоречия во всём архиве."""
    with _archive_lock:
        return _require_archive().edges_by_type(
            edge_type, node_id=node_id, branch_id=branch_id,
            file=file, limit=limit,
        )


# ---------------------------------------------------------------------------
# 9. Сборка CONTEXT SNAPSHOT
# ---------------------------------------------------------------------------
@mcp.tool()
def ptg_assemble_snapshot(seed_node_ids: list, char_budget: int = 6000,
                           include_lineage: bool = True,
                           include_relations: bool = True,
                           user_query: str = "", system_prompt: str = "",
                           save_snapshot: bool = True) -> dict:
    """ФИНАЛЬНЫЙ шаг: собрать компактный контекст из найденных seed-узлов
    (из ptg_search / ptg_branch_lineage / ptg_unresolved). Дедуплицирует
    атомы, разрешает supersedes (не включает устаревшую версию как
    единственную правду), помечает contradicts/fixes/reinforces инлайн,
    укладывается в char_budget. Возвращает snapshot_text (контекст для
    вставки) + provenance map (откуда взят каждый кусок) + context_entropy.

    Если save_snapshot=True (по умолчанию) — ТОЧНЫЙ итоговый payload
    (system_prompt + snapshot_text + user_query, без сокращений и
    пересказа) сохраняется на диск в <folder>/snapshots/ — это главная
    поверхность для дебага: что именно ушло агенту и почему."""
    with _archive_lock:
        arc = _require_archive()
        _log.debug(f"ptg_assemble_snapshot: {len(seed_node_ids)} seed(ов), char_budget={char_budget}")
        snapshot = arc.assemble_context_snapshot(
            seed_node_ids, char_budget=char_budget,
            include_lineage=include_lineage,
            include_relations=include_relations,
        )
        _log.info(f"ptg_assemble_snapshot: {len(snapshot.get('included_node_ids', []))} атомов, "
                 f"{snapshot.get('token_total', 0)} токенов")
        if save_snapshot:
            path = save_agent_snapshot(arc.output_root, snapshot,
                                        user_query=user_query,
                                        system_prompt=system_prompt)
            snapshot["saved_snapshot_path"] = path
        return snapshot


@mcp.tool()
def ptg_list_agent_snapshots(limit: int = 20) -> list:
    """Список ранее сохранённых final-agent-input снапшотов (новые
    первыми) — для отладки и для GUI-панели «FINAL AGENT INPUT»."""
    with _archive_lock:
        return list_snapshots(_require_archive().output_root, limit=limit)


@mcp.tool()
def ptg_load_agent_snapshot(filename: str) -> dict:
    """Загрузить ранее сохранённый снапшот по имени файла (см.
    ptg_list_agent_snapshots) — точное содержимое без изменений."""
    with _archive_lock:
        return load_snapshot(_require_archive().output_root, filename)


# ---------------------------------------------------------------------------
# 10. Здоровье LM Studio (нужен только для ptg_search)
# ---------------------------------------------------------------------------
@mcp.tool()
def ptg_lm_studio_status() -> dict:
    """Проверить, доступен ли LM Studio (нужен только для ptg_search —
    остальные инструменты работают чисто по уже сохранённому графу)."""
    e = Embedder()
    ok = e.test_connection()
    return {"connected": ok, "model": e.model if ok else None}


def main():
    global _archive
    parser = argparse.ArgumentParser(description="PTG MCP Toolset")
    parser.add_argument("--folder", required=True,
                         help="Папка с исходными логами/документами")
    parser.add_argument("--output-dir", default=None,
                         help="ПАТЧ 14 (опционально): папка, где реально лежит "
                              "векторная база .ptg/ — если она хранится ОТДЕЛЬНО "
                              "от --folder (см. main.py: «Папка для векторной базы»). "
                              "Если не указано — .ptg/ ищется внутри --folder (старое "
                              "поведение, полностью обратно совместимо).")
    args = parser.parse_args()

    archive = Archive(args.folder, progress_cb=lambda msg: print(msg, file=sys.stderr),
                       output_dir=args.output_dir)
    loaded = archive.load_if_exists()
    if not loaded:
        print(
            f"ОШИБКА: в {archive.ptg_dir} нет построенного архива. "
            f"Сначала запустите main.py -> «Построить архив».",
            file=sys.stderr,
        )
        sys.exit(1)

    _archive = archive
    print(
        f"PTG MCP сервер: загружен архив «{archive.root_project.get('project_name','?')}» "
        f"— {len(archive.nodes)} узлов, {len(archive.branches)} веток.",
        file=sys.stderr,
    )
    _log.info(f"MCP-сервер запущен: folder={args.folder!r}, output_dir={args.output_dir!r}, "
             f"{len(archive.nodes)} узлов, {len(archive.branches)} веток")
    mcp.run()


if __name__ == "__main__":
    main()
