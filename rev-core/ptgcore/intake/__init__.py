"""Приём источников: файл папки, сеанс Claude Code, сеанс Codex → черновики атомов; какие источники у проекта есть;
время правки; время последнего сжатия контекста сеанса; текст заметки.

Готовое:
* ptg_mem.transcripts — parse_claude, parse_codex (атом — вопрос и ответ хода, строки инструментов «Tools: [Имя] …»,
  служебные реплики отброшены), claude_transcripts, codex_transcripts, CompactTracker;
* ptg_mem.filters.FileFilter — какие файлы папки брать;
* ptg_mem.core.ptg_py_comments.extract_py_comments_text — у .py только докстринги и комментарии; язык меток —
  модульная переменная LANG этого модуля, её ставят перед вызовом;
* ptg_mem.core.ptg_chat_exports.extract_chat_pairs — пары экспорта чата из разобранного JSON (.json, .jsonl, .ndjson);
* memshell.convert (MarkItDown), memshell.atomize (атом — абзац), memcore.scrub (секреты → <secret>).

node: composite
effects: io
"""
import json
import os

import deal
import memcore
import memshell
from ptg_mem import filters, transcripts
from ptg_mem.core import ptg_chat_exports, ptg_py_comments

from ptgcore.intake.clean import note_text, scrub_drafts
from ptgcore.intake.listing import claude_files, codex_files, file_time, folder_files, gone, stamped
from ptgcore.intake.reading import chat_data, chat_pairs, plain_text, py_text
from ptgcore.intake.sessions import compact_at, session_drafts
from ptgcore.intake.shaping import chat_drafts, file_kind, text_drafts
from ptgcore.types import Draft, Settings, Source

KINDS = ("file", "claude", "codex")
MIN_NOTE = 8


@deal.pre(lambda source, lang, limit: source.kind in KINDS and lang in ("ru", "en") and limit >= 200)
@deal.ensure(lambda source, lang, limit, result: result[1] >= 0
             and [d.position for d in result[0]] == list(range(len(result[0])))
             and all(d.source in KINDS for d in result[0]))
@deal.raises(FileNotFoundError)
@deal.has("io")
def drafts(source: Source, lang: str, limit: int) -> tuple[tuple[Draft, ...], int]:
    """ФТ-001, ФТ-002, ФТ-003. Черновики источника и сколько секретов вычищено (memcore.scrub по text, question,
    answer — сумма found). Нет источника — FileNotFoundError(путь).
    * 'claude' / 'codex' — parse_claude / parse_codex: position = order, at = timestamp, session = session_id,
      source = вид;
    * 'file' — текст: .py — extract_py_comments_text (LANG = lang; None — нет черновиков); .json, .jsonl, .ndjson —
      extract_chat_pairs(разобранный JSON) → по паре черновик (question, answer, text = question + "\\n\\n" + answer,
      at — время пары или правки файла), если пар нет — как прочие; прочие — memshell.convert. Затем
      memshell.atomize(path, text, limit): question = '', answer = text, at — время правки файла, session = '',
      source = 'file'.

    D: 7

    >>> drafts(Source('tests/fixtures/session.jsonl', 'claude'), 'ru', 1200)
    ((Draft(position=0, text='Какой базой храним векторы памяти?\\n\\nХраним в LanceDB: вектор, полнотекст и SQL в одной таблице.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py', question='Какой базой храним векторы памяти?', answer='Храним в LanceDB: вектор, полнотекст и SQL в одной таблице.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py', at=1759996800.0, session='session', source='claude'), Draft(position=1, text='А граф кода откуда берём?\\n\\nГраф кода строит codebase-memory-mcp целиком.', question='А граф кода откуда берём?', answer='Граф кода строит codebase-memory-mcp целиком.', at=1759996920.0, session='session', source='claude')), 0)
    >>> drafts(Source('нет-такого.md', 'file'), 'ru', 1200)
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такого.md
    """
    if source.kind != "file":
        return scrub_drafts(session_drafts(source))
    at = file_time(source.path)
    if at <= 0.0:
        raise FileNotFoundError(source.path)
    kind = file_kind(source.path)
    pairs: tuple[tuple[str, str, float], ...] = chat_pairs(chat_data(source.path)) if kind == "chat" else ()
    if pairs:
        return scrub_drafts(chat_drafts(pairs, at))
    text = py_text(source.path, lang) if kind == "py" else plain_text(source.path)
    return scrub_drafts(text_drafts(source.path, text, limit, at))


@deal.ensure(lambda text, result: result == "" or len(result) >= MIN_NOTE)
@deal.has()
def note(text: str) -> str:
    """Текст заметки агента: memcore.scrub(text.strip()).text; короче MIN_NOTE знаков — ''.

    D: 1

    >>> note('  Решение: векторы храним в LanceDB.  ')
    'Решение: векторы храним в LanceDB.'
    >>> note('ок')
    ''
    """
    return note_text(text, MIN_NOTE)


@deal.ensure(lambda settings, result: len({s.path for s, _ in result}) == len(result)
             and all(s.kind in KINDS and m > 0 for s, m in result))
@deal.has("io")
def sources(settings: Settings) -> tuple[tuple[Source, float], ...]:
    """Источники проекта сейчас и время их правки: при folder — FileFilter(root, exclude_dirs, include_exts или None,
    store).walk() → 'file'; при claude — claude_transcripts(root) → 'claude'; при codex — codex_transcripts(root) →
    'codex'. Исчезнувшие при чтении времени пропускаются.

    D: 4

    >>> sources(Settings('нет-такой-папки', 'нет-базы', True, False, False, (), (), 'ru'))
    ()
    """
    found: list[tuple[str, str]] = []
    if settings.folder:
        found += [(p, "file") for p in folder_files(settings.root, settings.store, settings.exclude_dirs, settings.include_exts)]
    if settings.claude:
        found += [(p, "claude") for p in claude_files(settings.root)]
    if settings.codex:
        found += [(p, "codex") for p in codex_files(settings.root)]
    return stamped(tuple((Source(p, k), file_time(p)) for p, k in found))


@deal.ensure(lambda paths, result: set(result) <= set(paths) and len(set(result)) == len(result))
@deal.has("io")
def missing(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Пути, которых больше нет на диске, по порядку, без повторов.

    D: 2

    >>> missing(('нет-такого.md', 'tests/fixtures/session.jsonl', 'нет-такого.md'))
    ('нет-такого.md',)
    """
    return gone(paths)


@deal.ensure(lambda path, result: result >= 0.0)
@deal.has("io")
def mtime(path: str) -> float:
    """Время правки файла; нет файла — 0.0.

    D: 1

    >>> mtime('нет-такого.md')
    0.0
    """
    return file_time(path)


@deal.ensure(lambda path, result: result >= 0.0)
@deal.has("io")
def compact_time(path: str) -> float:
    """Время последнего сжатия контекста сеанса Claude Code (CompactTracker().last_compact); нет — 0.0.

    D: 1

    >>> compact_time('tests/fixtures/session.jsonl')
    0.0
    >>> compact_time('нет-такого.jsonl')
    0.0
    """
    return compact_at(path)
