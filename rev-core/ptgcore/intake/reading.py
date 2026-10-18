"""Прочитать файл-источник: текст как есть, текст .py из докстрингов и комментариев, разобранный JSON экспорта чата и
пары «вопрос, ответ, время» из него.

node: leaf
effects: io
"""
import json
import os

import deal
import memshell
from ptg_mem.core import ptg_chat_exports, ptg_py_comments


@deal.raises(FileNotFoundError)
@deal.ensure(lambda path, result: chr(13) not in result)
@deal.has('io')
def plain_text(path: str) -> str:
    """ФТ-002. Текст файла: нет файла — FileNotFoundError(путь); иначе memshell.convert(путь) (MarkItDown); формат,
    который он не берёт, — ''.

    D: 2

    >>> plain_text('нет-такого.md')
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такого.md
    >>> plain_text('нет-такой-папки/план.docx')
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такой-папки/план.docx
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        text: str | None = memshell.convert(path)
    except Exception:
        return ''
    if not text:
        return ''
    return text.replace('\r\n', '\n').replace('\r', '\n')


@deal.pre(lambda path, lang: lang in ('ru', 'en'))
@deal.raises(FileNotFoundError)
@deal.ensure(lambda path, lang, result: chr(13) not in result)
@deal.has('io')
def py_text(path: str, lang: str) -> str:
    """ФТ-002. Текст .py из докстрингов и комментариев: нет файла — FileNotFoundError(путь); иначе модульная
    переменная LANG в ptg_py_comments ставится в lang и берётся extract_py_comments_text(путь); None (нет ни
    докстрингов, ни комментариев) — ''; переводы строк CRLF и CR — как LF.

    D: 3

    >>> py_text('нет-такого.py', 'ru')
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такого.py
    >>> py_text('нет-такого.py', 'en')
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такого.py
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ptg_py_comments.LANG = lang
    found: str | None = ptg_py_comments.extract_py_comments_text(path)
    if found is None:
        return ''
    return found.replace('\r\n', '\n').replace('\r', '\n')


@deal.raises(FileNotFoundError)
@deal.ensure(lambda path, result: result is None or isinstance(result, (dict, list)))
@deal.has('io')
def chat_data(path: str) -> dict[str, object] | list[object] | None:
    """ФТ-002. Разобранный JSON файла: нет файла — FileNotFoundError(путь); '.jsonl' и '.ndjson' (регистр не важен) —
    список значений непустых строк, строка, которую не разобрать, пропускается; прочие расширения — json.loads всего
    текста; не словарь и не список или разобрать не удалось — None.

    D: 5

    >>> chat_data('нет-такого.json')
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такого.json
    >>> chat_data('tests/fixtures/session.jsonl')
    [{'type': 'user', 'timestamp': '2025-10-09T08:00:00Z', 'cwd': 'H:/proj', 'message': {'role': 'user', 'content': 'Какой базой храним векторы памяти?'}}, {'type': 'assistant', 'timestamp': '2025-10-09T08:00:05Z', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Храним в LanceDB: вектор, полнотекст и SQL в одной таблице.'}, {'type': 'tool_use', 'name': 'Edit', 'input': {'file_path': 'H:/proj/memcore/filtering.py'}}]}}, {'type': 'user', 'timestamp': '2025-10-09T08:01:00Z', 'message': {'role': 'user', 'content': '<system-reminder>служебное</system-reminder>'}}, {'type': 'user', 'timestamp': '2025-10-09T08:02:00Z', 'message': {'role': 'user', 'content': 'А граф кода откуда берём?'}}, {'type': 'assistant', 'timestamp': '2025-10-09T08:02:09Z', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Граф кода строит codebase-memory-mcp целиком.'}]}}]
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(('.jsonl', '.ndjson')):

        def load(line: str) -> tuple[bool, object]:
            try:
                return True, json.loads(line)
            except (ValueError, RecursionError):
                return False, None

        rows: list[object] = []
        with open(path, encoding='utf-8-sig', errors='replace') as handle:
            for line in handle:
                if line.strip():
                    ok, value = load(line)
                    if ok:
                        rows.append(value)
        return rows
    with open(path, encoding='utf-8-sig', errors='replace') as handle:
        text = handle.read()
    try:
        data: object = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if isinstance(data, (dict, list)):
        return data
    return None


@deal.ensure(lambda data, result: all(t >= 0.0 for _, _, t in result))
@deal.has('io')
def chat_pairs(data: dict[str, object] | list[object] | None) -> tuple[tuple[str, str, float], ...]:
    """ФТ-002. Пары «вопрос, ответ, время» экспорта чата по порядку: ptg_chat_exports.extract_chat_pairs(data); формат не
    распознан или data = None — (); время пары, если оно известно и больше нуля, иначе 0.0.

    D: 3

    >>> chat_pairs([{'role': 'user', 'content': 'Какой базой?', 'timestamp': 1760000000.0}, {'role': 'assistant', 'content': 'LanceDB.', 'timestamp': 1760000005.0}])
    (('Какой базой?', 'LanceDB.', 1760000000.0),)
    >>> chat_pairs([{'role': 'user', 'content': 'Привет'}, {'role': 'assistant', 'content': 'Здравствуйте.'}])
    (('Привет', 'Здравствуйте.', 0.0),)
    >>> chat_pairs({'name': 'package'})
    ()
    >>> chat_pairs(None)
    ()
    """
    if data is None:
        return ()
    pairs: list[dict[str, object]] | None = ptg_chat_exports.extract_chat_pairs(data)
    out: list[tuple[str, str, float]] = []
    for pair in pairs or []:
        stamp = pair.get('timestamp')
        when = float(stamp) if isinstance(stamp, (int, float)) and stamp > 0 else 0.0
        out.append((str(pair['question']), str(pair['answer']), when))
    return tuple(out)
