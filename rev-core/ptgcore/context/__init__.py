"""Подсказка и сводка: какие атомы допустимы в подсказку (служебный шум, текущий сеанс после сжатия), окно от
лучшего и бюджет, блок <ptg-memory> с пометками, сводка начала сеанса.

Готовое: memshell.is_noise и memshell.types.AUTOMATED, memshell.session_ok, memshell.in_window, memcore.select_context
(memcore.types.Hit, Budget), memshell.render_recall и render_brief (memshell.types.Entry(date, file, score, text,
notes)).

node: composite
"""
import math

import deal
import memcore
import memshell

from ptgcore import relations
from ptgcore.context import admission, block, digest, marks, selection, sessions
from ptgcore.types import Card, Entry

DAY = 86400.0


@deal.ensure(lambda files, session_id, transcript_path, result: list(result) == sorted(set(result))
             and set(result) <= set(files) | {transcript_path} - {""})
@deal.has()
def session_files(files: tuple[str, ...], session_id: str, transcript_path: str) -> tuple[str, ...]:
    """Файлы текущего сеанса: transcript_path (если дан) и файлы, в имени которых (os.path.basename) есть первые 8
    знаков session_id (если дан); по алфавиту, без повторов.

    D: 3

    >>> session_files(('s/2a9ebf22-f8b0.jsonl', 'n.md'), '2a9ebf22-f8b0-4b20', '')
    ('s/2a9ebf22-f8b0.jsonl',)
    >>> session_files(('n.md',), '', 's/x.jsonl')
    ('s/x.jsonl',)
    """
    return sessions.session_files(files, session_id, transcript_path)


@deal.ensure(lambda entries, files, cut, result: set(result) <= set(entries))
@deal.has()
def eligible(entries: tuple[Entry, ...], files: tuple[str, ...], cut: float) -> tuple[Entry, ...]:
    """ФТ-008. Допустимые в подсказку: не шум (memshell.is_noise(первые 200 знаков вопроса или текста, AUTOMATED)) и
    memshell.session_ok(file, at, files, cut); по порядку entries.

    D: 2

    >>> eligible((Entry('a1', 's/2a9ebf22.jsonl', 0, 'd1', 'свежая реплика', 'свежая реплика', '', 100.0, 'active', 0.0, '2a9ebf22', 'claude'), Entry('a2', 'n.md', 0, 'd2', 'LanceDB выбран.', '', 'LanceDB выбран.', 100.0, 'active', 0.0, '', 'file'), Entry('a3', 'n.md', 1, 'd3', '<task-notification> готово', '<task-notification> готово', '', 100.0, 'active', 0.0, '', 'file')), ('s/2a9ebf22.jsonl',), 50.0)
    (Entry(id='a2', file='n.md', position=0, digest='d2', text='LanceDB выбран.', question='', answer='LanceDB выбран.', at=100.0, status='active', retracted_at=0.0, session='', source='file'),)
    """
    if not math.isfinite(cut):
        return ()
    return sessions.eligible(tuple(e for e in entries if math.isfinite(e.at)), files, cut)


@deal.pre(lambda cards, near, words, min_score, window, max_items: window >= 0.0 and max_items >= 0
          and math.isfinite(min_score) and math.isfinite(window) and all(math.isfinite(s) for _, s in near))
@deal.ensure(lambda cards, near, words, min_score, window, max_items, result: len(result) <= max_items
             and len({c.id for c in result}) == len(result) and {c.id for c in result} <= {c.id for c in cards})
@deal.ensure(lambda cards, near, words, min_score, window, max_items, result:
             result == () or (near != () and max(s for _, s in near) >= min_score))
@deal.has()
def admit(cards: tuple[Card, ...], near: tuple[tuple[str, float], ...], words: tuple[str, ...], min_score: float, window: float,
          max_items: int) -> tuple[Card, ...]:
    """ФТ-008. Отбор в подсказку при порядке RRF (замер 13.10, правило B2: из 47 своих запросов нашла 25 против 15 у
    прежнего отбора по косинусу; на 24 посторонних молчит так же — 16):
    * лучший косинус запроса — наибольшая оценка в near (пары id, косинус); near пуст или лучший ниже min_score — ();
    * карточка проходит, если её косинус (оценка её id в near) не ниже max(min_score, лучший − window) или её id
      среди первых max_items в words (id полнотекста по порядку);
    * по порядку cards, без повторов id, не больше max_items;
    * оценка выданной карточки — её косинус из near, нет его — None.

    D: 3

    >>> admit((Card(id='a1', score=0.0325, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''), Card(id='a2', score=0.0317, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB заменил SQLite.', chars=23, links=(), current='', session=''), Card(id='a3', score=0.0161, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='кэш эмбеддингов', chars=15, links=(), current='', session='')), (('a2', 0.71), ('a1', 0.66), ('a3', 0.4)), ('a3',), 0.5, 0.12, 5)
    (Card(id='a1', score=0.66, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''), Card(id='a2', score=0.71, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB заменил SQLite.', chars=23, links=(), current='', session=''), Card(id='a3', score=0.4, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='кэш эмбеддингов', chars=15, links=(), current='', session=''))
    >>> admit((Card(id='a1', score=0.0325, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''), Card(id='a2', score=0.0317, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB заменил SQLite.', chars=23, links=(), current='', session='')), (('a1', 0.45),), ('a1',), 0.5, 0.12, 5)
    ()
    >>> admit((Card(id='a1', score=0.0325, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''),), (('a2', 0.7),), ('a1',), 0.5, 0.12, 5)
    (Card(id='a1', score=None, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''),)
    >>> admit((Card(id='a1', score=0.0325, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''), Card(id='a2', score=0.0317, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB заменил SQLite.', chars=23, links=(), current='', session=''), Card(id='a3', score=0.0161, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='кэш эмбеддингов', chars=15, links=(), current='', session='')), (('a2', 0.71), ('a1', 0.66), ('a3', 0.4)), ('a3',), 0.5, 0.12, 1)
    (Card(id='a1', score=0.66, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''),)
    >>> admit((), (), (), 0.5, 0.12, 5)
    ()
    """
    return admission.admit(cards, near, words, min_score, window, max_items)


@deal.pre(lambda cards, later, code, budget: budget >= 0)
@deal.ensure(lambda cards, later, code, budget, result: len(result[0]) <= budget and 0 <= result[1] <= len(cards)
             and (result[1] == 0) == (result[0] == ""))
@deal.has()
def render(cards: tuple[Card, ...], later: tuple[Entry, ...], code: tuple[tuple[str, str], ...], budget: int) -> tuple[str, int]:
    """ФТ-008. Блок подсказки memshell.render_recall(записи, budget) и сколько карточек в него вошло (номера «\\nN. [»).
    Запись карточки — memshell.types.Entry(date, rel_file, оценка или 0.0, text, пометки); пометки по порядку:
    * текущая версия есть в later — 'replaced later (<day(at)>): <первые 160 знаков>';
    * иначе первая связь out 'supersedes' — 'replaces an earlier version (<date>): <первые 160 знаков>';
    * первая связь 'contradicts' — 'contradicts (<date>): …' (out) или 'contradicted by (<date>): …' (in), 200 знаков;
    * каждая строка code с id карточки — 'code: <строка>'.

    D: 6

    >>> render((Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), '', ''),), (), (), 3600)
    ("<ptg-memory>\\nAuto-recalled from this project's memory (PTG-MEM). Older notes may be outdated — check the date and the markers before relying on them.\\n1. [2025-10-09 · n.md · 0.90] LanceDB выбран.\\n</ptg-memory>", 1)
    >>> render((Card('a1', 0.9, 'superseded', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), 'a2', ''),), (Entry('a2', 'n.md', 0, 'd2', 'LanceDB заменил SQLite.', '', 'LanceDB заменил SQLite.', 1760003600.0, 'active', 0.0, '', 'file'),), (('a1', 'edits file:PTG-MEM/ptg_mem/engine.py'),), 3600)
    ("<ptg-memory>\\nAuto-recalled from this project's memory (PTG-MEM). Older notes may be outdated — check the date and the markers before relying on them.\\n1. [2025-10-09 · n.md · 0.90] LanceDB выбран.\\n   ↳ replaced later (2025-10-09): LanceDB заменил SQLite.\\n   ↳ code: edits file:PTG-MEM/ptg_mem/engine.py\\n</ptg-memory>", 1)
    >>> render((Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), '', ''),), (), (), 40)
    ('', 0)
    """
    kept = tuple(c for c in cards if c.score is None or math.isfinite(c.score))
    notes = tuple(marks.notes_of(c, later, code) for c in kept)
    text = block.render_block(kept, notes, budget)
    return text, min(block.shown(text), len(kept))


@deal.pre(lambda notes, now, days, limit: days >= 0 and limit >= 0)
@deal.ensure(lambda notes, now, days, limit, result: len(result) <= limit and set(result) <= set(notes)
             and all(a.at >= b.at for a, b in zip(result, result[1:], strict=False)))
@deal.has()
def recent_notes(notes: tuple[Entry, ...], now: float, days: int, limit: int) -> tuple[Entry, ...]:
    """Живые заметки не старше days дней (at >= now − days·DAY), новые первыми, не больше limit.

    D: 1

    >>> recent_notes((Entry('n1', 'memory://notes', 0, 'd1', 'старая', '', 'старая', 1.0, 'active', 0.0, '', 'agent'), Entry('n2', 'memory://notes', 1, 'd2', 'Решение: векторы храним в LanceDB.', 'Note (agent)', 'Решение: векторы храним в LanceDB.', 1760000000.0, 'active', 0.0, '', 'agent')), 1760003600.0, 14, 5)
    (Entry(id='n2', file='memory://notes', position=1, digest='d2', text='Решение: векторы храним в LanceDB.', question='Note (agent)', answer='Решение: векторы храним в LanceDB.', at=1760000000.0, status='active', retracted_at=0.0, session='', source='agent'),)
    """
    return digest.recent_notes(notes, now, days, limit)


@deal.pre(lambda live, notes, max_chars: live >= 0 and max_chars >= 0)
@deal.ensure(lambda live, notes, max_chars, result: len(result) <= max_chars or result == "")
@deal.has()
def brief(live: int, notes: tuple[Entry, ...], max_chars: int) -> str:
    """ФТ-009. Сводка начала сеанса: memshell.render_brief(live, записи заметок (day(at), file, 0.0, text, ()),
    max_chars).

    D: 1

    >>> brief(3, (Entry('n2', 'memory://notes', 1, 'd2', 'Решение: векторы храним в LanceDB.', 'Note (agent)', 'Решение: векторы храним в LanceDB.', 1760000000.0, 'active', 0.0, '', 'agent'),), 1600)
    '<ptg-memory>\\nPTG-MEM is active for this project: 3 memory atoms. Relevant memory is attached to each prompt automatically. Tools: ptg_search, ptg_node, ptg_decisions, ptg_remember (store a decision the moment it is made).\\nRecent notes and decisions (last two weeks):\\n- [2025-10-09 · memory://notes] Решение: векторы храним в LanceDB.\\n</ptg-memory>'
    """
    return digest.brief(live, notes, max_chars)
