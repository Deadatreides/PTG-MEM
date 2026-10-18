"""Правила сеанса для подсказки: какие файлы — текущий сеанс и какие атомы допустимы (не служебный шум, не реплика
текущего сеанса, ещё не ушедшая из окна модели при сжатии контекста).

Готовое: memshell.is_noise и memshell.types.AUTOMATED, memshell.session_ok.

node: leaf
"""
import math
import os

import deal
import memshell
from memshell.types import AUTOMATED

from ptgcore.types import Entry


@deal.ensure(lambda files, session_id, transcript_path, result: list(result) == sorted(set(result))
             and set(result) <= set(files) | {transcript_path} - {''})
@deal.ensure(lambda files, session_id, transcript_path, result: all(
    f == transcript_path or (session_id != '' and session_id[:8] in os.path.basename(f)) for f in result))
@deal.has()
def session_files(files: tuple[str, ...], session_id: str, transcript_path: str) -> tuple[str, ...]:
    """Файлы текущего сеанса: transcript_path (если дан) и файлы, в имени которых (os.path.basename) есть первые 8
    знаков session_id (если дан); по алфавиту, без повторов.

    D: 3

    >>> session_files(('s/2a9ebf22-f8b0.jsonl', 'n.md'), '2a9ebf22-f8b0-4b20', '')
    ('s/2a9ebf22-f8b0.jsonl',)
    >>> session_files(('n.md',), '', 's/x.jsonl')
    ('s/x.jsonl',)
    >>> session_files(('n.md',), '', '')
    ()
    >>> session_files(('s/b-2a9ebf22.jsonl', 's/a-2a9ebf22.jsonl'), '2a9ebf22', 's/a-2a9ebf22.jsonl')
    ('s/a-2a9ebf22.jsonl', 's/b-2a9ebf22.jsonl')
    """
    picked: set[str] = set()
    if transcript_path != '':
        picked.add(transcript_path)
    if session_id != '':
        prefix = session_id[:8]
        picked.update(f for f in files if prefix in os.path.basename(f))
    return tuple(sorted(picked))


@deal.pre(lambda entries, files, cut: math.isfinite(cut) and all(math.isfinite(e.at) for e in entries))
@deal.ensure(lambda entries, files, cut, result: set(result) <= set(entries) and list(result) == [e for e in entries if e in result])
@deal.has()
def eligible(entries: tuple[Entry, ...], files: tuple[str, ...], cut: float) -> tuple[Entry, ...]:
    """ФТ-008. Допустимые в подсказку: не шум (memshell.is_noise(первые 200 знаков вопроса или текста, AUTOMATED)) и
    memshell.session_ok(file, at, files, cut); по порядку entries.

    D: 3

    >>> eligible((Entry('a1', 's/2a9ebf22.jsonl', 0, 'd1', 'свежая реплика', 'свежая реплика', '', 100.0, 'active', 0.0, '2a9ebf22', 'claude'), Entry('a2', 'n.md', 0, 'd2', 'LanceDB выбран.', '', 'LanceDB выбран.', 100.0, 'active', 0.0, '', 'file'), Entry('a3', 'n.md', 1, 'd3', '<task-notification> готово', '<task-notification> готово', '', 100.0, 'active', 0.0, '', 'file')), ('s/2a9ebf22.jsonl',), 50.0)
    (Entry(id='a2', file='n.md', position=0, digest='d2', text='LanceDB выбран.', question='', answer='LanceDB выбран.', at=100.0, status='active', retracted_at=0.0, session='', source='file'),)
    >>> eligible((), (), 0.0)
    ()
    >>> eligible((Entry('a1', 's/2a9ebf22.jsonl', 0, 'd1', 'старая реплика', 'старая реплика', '', 10.0, 'active', 0.0, '2a9ebf22', 'claude'),), ('s/2a9ebf22.jsonl',), 50.0)
    (Entry(id='a1', file='s/2a9ebf22.jsonl', position=0, digest='d1', text='старая реплика', question='старая реплика', answer='', at=10.0, status='active', retracted_at=0.0, session='2a9ebf22', source='claude'),)
    >>> eligible((Entry('a4', 'n.md', 0, 'd4', 'Решение принято.', '<system-reminder> напоминание', '', 100.0, 'active', 0.0, '', 'file'),), (), 0.0)
    ()
    """
    def noisy(e: Entry) -> bool:
        return memshell.is_noise((e.question or e.text)[:200], AUTOMATED)

    return tuple(
        e for e in entries
        if not noisy(e) and memshell.session_ok(e.file, e.at, files, cut)
    )
