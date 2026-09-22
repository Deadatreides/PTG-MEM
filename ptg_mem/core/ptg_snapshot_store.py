"""
ptg_snapshot_store.py — точная фиксация финального agent-input payload.

НОВЫЙ ФАЙЛ (ПАТЧ 9, часть IX аудита). Не трогает ptg_core.py/ptg_mcp_server.py
логику построения графа — это чисто IO-слой поверх результата
Archive.assemble_context_snapshot().

Почему отдельный модуль, а не метод Archive:
    Archive ничего не знает про "агента", "system_prompt" или "user_query" —
    это понятия уровня MCP-клиента/сессии, а не графа. Смешивать их внутрь
    Archive означало бы, что граф-класс отвечает за то, что ему знать не
    нужно (нарушение границы ответственности). Поэтому эта функция берёт
    ГОТОВЫЙ dict от assemble_context_snapshot() и просто фиксирует его
    вместе с сессионным контекстом на диск — без интерпретации, без
    сокращений, "как есть" (требование части IX: "No abstraction. No
    summaries. No truncation.").
"""

import os
import re
import json
import time
import hashlib

from ptg_logging import get_logger

_log = get_logger("snapshot_store")


def _query_hash(query: str) -> str:
    return hashlib.sha1((query or "").encode("utf-8")).hexdigest()[:10]


def save_agent_snapshot(folder: str, snapshot: dict, user_query: str = "",
                         system_prompt: str = "") -> str:
    """Сохраняет ТОЧНЫЙ финальный payload на диск:
    <folder>/snapshots/{timestamp}_{query_hash}_agent_input.json

    final_agent_payload — конкатенация system_prompt + snapshot_text +
    user_query РОВНО в том виде, в каком она была бы отправлена агенту
    (без пересказа, без урезания сверх того, что уже сделал
    assemble_context_snapshot через свой char_budget).

    Возвращает путь к сохранённому файлу."""
    snap_dir = os.path.join(folder, "snapshots")
    try:
        os.makedirs(snap_dir, exist_ok=True)
    except OSError as ex:
        raise OSError(f"Не удалось создать папку снапшотов {snap_dir}: {ex}") from ex

    ts = time.time()
    qhash = _query_hash(user_query or "|".join(snapshot.get("seed_node_ids", [])))
    fname = f"{int(ts)}_{qhash}_agent_input.json"
    path = os.path.join(snap_dir, fname)

    final_agent_payload = "\n\n".join(
        part for part in (system_prompt, snapshot.get("snapshot_text", ""), user_query) if part
    )

    payload = {
        "system_prompt": system_prompt,
        "assembled_context": snapshot.get("snapshot_text", ""),
        "user_query": user_query,
        "final_agent_payload": final_agent_payload,
        "token_total": snapshot.get("token_total"),
        "context_entropy": snapshot.get("context_entropy"),
        # часть X — provenance map едет вместе со снапшотом: без него
        # "exact final payload" бесполезен для дебага (нельзя понять,
        # ПОЧЕМУ в payload оказался именно этот кусок).
        "provenance": snapshot.get("provenance", []),
        "seed_node_ids": snapshot.get("seed_node_ids", []),
        "included_node_ids": snapshot.get("included_node_ids", []),
        "char_budget": snapshot.get("char_budget"),
        "char_used": snapshot.get("char_used"),
        "truncated_by_budget": snapshot.get("truncated_by_budget"),
        "saved_at": ts,
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except OSError as ex:
        _log.error(f"не удалось сохранить снапшот в {snap_dir}: {ex}")
        raise OSError(f"Не удалось сохранить снапшот в {snap_dir}: {ex}") from ex
    _log.info(f"снапшот сохранён: {path} ({len(final_agent_payload)} символов, "
             f"{len(payload['included_node_ids'])} атомов)")
    return path


def save_full_context(folder: str, result: dict) -> str:
    """ПАТЧ 16 — сохранить результат Archive.export_full_context() как
    читаемый .md-файл: <folder>/FULL_PROJECT_CONTEXT_{timestamp}.md.
    В отличие от save_agent_snapshot() (JSON, машиночитаемый, seed-based),
    здесь простой markdown — специально, чтобы файл можно было открыть
    и сразу вставить в новый чат без дополнительной обработки."""
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as ex:
        raise OSError(f"Не удалось создать папку {folder}: {ex}") from ex

    ts = int(time.time())
    path = os.path.join(folder, f"FULL_PROJECT_CONTEXT_{ts}.md")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(result.get("full_text", ""))
    except OSError as ex:
        _log.error(f"не удалось сохранить полный контекст в {path}: {ex}")
        raise OSError(f"Не удалось сохранить полный контекст в {path}: {ex}") from ex
    _log.info(f"полный контекст проекта сохранён: {path} "
             f"({result.get('char_count', 0)} символов, {result.get('file_count', 0)} файлов)")
    return path


def list_snapshots(folder: str, limit: int = 50) -> list:
    """Список уже сохранённых снапшотов (для GUI-панели FINAL AGENT INPUT),
    новые сначала. Не парсит содержимое — только имена файлов + mtime.
    ПАТЧ 10, фикс бага #6: раньше отсутствующая/повреждённая папка snapshots
    приводила к необработанному исключению у вызывающего кода — теперь
    любая ошибка чтения директории даёт пустой список, а не падение."""
    snap_dir = os.path.join(folder, "snapshots")
    if not os.path.isdir(snap_dir):
        return []
    try:
        files = [f for f in os.listdir(snap_dir) if f.endswith("_agent_input.json")]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(snap_dir, f)), reverse=True)
    except OSError:
        return []
    return files[:limit]


def load_snapshot(folder: str, filename: str) -> dict:
    """Безопасная загрузка одного снапшота по имени файла (без выхода за
    пределы snapshots/ — filename нормализуется через basename).
    ПАТЧ 10, фикс бага #6: раньше отсутствующий файл или битый JSON давали
    сырое исключение (FileNotFoundError/JSONDecodeError) до самого вызывающего
    MCP-инструмента. Теперь — явная, читаемая ошибка с указанием причины."""
    snap_dir = os.path.join(folder, "snapshots")
    safe_name = os.path.basename(filename)
    path = os.path.join(snap_dir, safe_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Снапшот не найден: {safe_name} (искал в {snap_dir})")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as ex:
        raise ValueError(f"Снапшот {safe_name} повреждён (невалидный JSON): {ex}") from ex
