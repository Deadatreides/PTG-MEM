"""Сеансы Claude Code и Codex: ходы сеанса → черновики атомов; время последнего сжатия контекста сеанса.

node: leaf
effects: io
"""
import os

import deal
from ptg_mem import transcripts

from ptgcore.types import Draft, Source


@deal.pre(lambda source: source.kind in ('claude', 'codex'))
@deal.ensure(lambda source, result: [d.position for d in result] == list(range(len(result)))
             and all(d.source == source.kind for d in result))
@deal.raises(FileNotFoundError)
@deal.has('io')
def session_drafts(source: Source) -> tuple[Draft, ...]:
    """ФТ-001. Черновики сеанса, по одному на атом разбора: нет файла — FileNotFoundError(путь); вид 'claude' —
    transcripts.parse_claude, вид 'codex' — transcripts.parse_codex; Draft(position = order, text, question, answer,
    at = timestamp, session = session_id, source = вид).

    D: 3

    >>> session_drafts(Source('tests/fixtures/session.jsonl', 'claude'))
    (Draft(position=0, text='Какой базой храним векторы памяти?\\n\\nХраним в LanceDB: вектор, полнотекст и SQL в одной таблице.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py', question='Какой базой храним векторы памяти?', answer='Храним в LanceDB: вектор, полнотекст и SQL в одной таблице.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py', at=1759996800.0, session='session', source='claude'), Draft(position=1, text='А граф кода откуда берём?\\n\\nГраф кода строит codebase-memory-mcp целиком.', question='А граф кода откуда берём?', answer='Граф кода строит codebase-memory-mcp целиком.', at=1759996920.0, session='session', source='claude'))
    >>> session_drafts(Source('нет-такого.jsonl', 'claude'))
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такого.jsonl
    >>> session_drafts(Source('нет-такого.jsonl', 'codex'))
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такого.jsonl
    """
    if not os.path.isfile(source.path):
        raise FileNotFoundError(source.path)
    parse = transcripts.parse_claude if source.kind == 'claude' else transcripts.parse_codex
    return tuple(
        Draft(
            position=atom['order'],
            text=atom['text'],
            question=atom['question'],
            answer=atom['answer'],
            at=float(atom['timestamp']),
            session=atom['session_id'],
            source=source.kind,
        )
        for atom in parse(source.path)
    )


@deal.ensure(lambda path, result: result >= 0.0)
@deal.has('io')
def compact_at(path: str) -> float:
    """Время последнего сжатия контекста сеанса Claude Code: transcripts.CompactTracker().last_compact(path); нет файла
    или сжатий не было — 0.0.

    D: 2

    >>> compact_at('tests/fixtures/session.jsonl')
    0.0
    >>> compact_at('нет-такого.jsonl')
    0.0
    """
    try:
        last = transcripts.CompactTracker().last_compact(path)
    except (OSError, ValueError):
        return 0.0
    if last is None or last <= 0.0:
        return 0.0
    return float(last)
