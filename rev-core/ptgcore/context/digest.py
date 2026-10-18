"""Заметки агента для сводки начала сеанса: живые за последние дни, новые первыми, и сама сводка.

Готовое: memshell.render_brief (memshell.types.Entry(date, file, score, text, notes)), relations.day.

node: leaf
"""
import deal
import memshell
from memshell.types import Entry as Line

from ptgcore import relations
from ptgcore.types import Entry

DAY = 86400.0


@deal.pre(lambda notes, now, days, limit: days >= 0 and limit >= 0)
@deal.ensure(lambda notes, now, days, limit, result: len(result) <= limit and set(result) <= set(notes)
             and all(a.at >= b.at for a, b in zip(result, result[1:], strict=False)))
@deal.ensure(lambda notes, now, days, limit, result: all(e.status == 'active' and e.at >= now - days * DAY for e in result))
@deal.has()
def recent_notes(notes: tuple[Entry, ...], now: float, days: int, limit: int) -> tuple[Entry, ...]:
    """Живые (status 'active') заметки не старше days дней (at >= now − days·DAY), новые первыми (при равном времени —
    в порядке notes), не больше limit.

    D: 3

    >>> recent_notes((Entry('n1', 'memory://notes', 0, 'd1', 'старая', '', 'старая', 1.0, 'active', 0.0, '', 'agent'), Entry('n2', 'memory://notes', 1, 'd2', 'Решение: векторы храним в LanceDB.', 'Note (agent)', 'Решение: векторы храним в LanceDB.', 1760000000.0, 'active', 0.0, '', 'agent')), 1760003600.0, 14, 5)
    (Entry(id='n2', file='memory://notes', position=1, digest='d2', text='Решение: векторы храним в LanceDB.', question='Note (agent)', answer='Решение: векторы храним в LanceDB.', at=1760000000.0, status='active', retracted_at=0.0, session='', source='agent'),)
    >>> recent_notes((), 1760003600.0, 14, 5)
    ()
    >>> recent_notes((Entry('n1', 'memory://notes', 0, 'd1', 'раз', '', 'раз', 1760000000.0, 'active', 0.0, '', 'agent'), Entry('n2', 'memory://notes', 1, 'd2', 'два', '', 'два', 1760003000.0, 'active', 0.0, '', 'agent'), Entry('n3', 'memory://notes', 2, 'd3', 'три', '', 'три', 1760001000.0, 'active', 0.0, '', 'agent')), 1760003600.0, 14, 2)
    (Entry(id='n2', file='memory://notes', position=1, digest='d2', text='два', question='', answer='два', at=1760003000.0, status='active', retracted_at=0.0, session='', source='agent'), Entry(id='n3', file='memory://notes', position=2, digest='d3', text='три', question='', answer='три', at=1760001000.0, status='active', retracted_at=0.0, session='', source='agent'))
    >>> recent_notes((Entry('n4', 'memory://notes', 0, 'd4', 'снята', '', 'снята', 1760003000.0, 'retracted', 1760003500.0, '', 'agent'),), 1760003600.0, 14, 5)
    ()
    """
    live = [e for e in notes if e.status == 'active' and e.at >= now - days * DAY]
    live.sort(key=lambda e: e.at, reverse=True)
    return tuple(live[:limit])


@deal.pre(lambda live, notes, max_chars: live >= 0 and max_chars >= 0)
@deal.ensure(lambda live, notes, max_chars, result: len(result) <= max_chars)
@deal.ensure(lambda live, notes, max_chars, result: result == '' or (result.startswith('<ptg-memory>') and result.endswith('</ptg-memory>')))
@deal.has()
def brief(live: int, notes: tuple[Entry, ...], max_chars: int) -> str:
    """ФТ-009. Сводка начала сеанса: memshell.render_brief(live, записи заметок (relations.day(at), file, 0.0, text, ()),
    max_chars).

    D: 2

    >>> brief(3, (Entry('n2', 'memory://notes', 1, 'd2', 'Решение: векторы храним в LanceDB.', 'Note (agent)', 'Решение: векторы храним в LanceDB.', 1760000000.0, 'active', 0.0, '', 'agent'),), 1600)
    '<ptg-memory>\\nPTG-MEM is active for this project: 3 memory atoms. Relevant memory is attached to each prompt automatically. Tools: ptg_search, ptg_node, ptg_decisions, ptg_remember (store a decision the moment it is made).\\nRecent notes and decisions (last two weeks):\\n- [2025-10-09 · memory://notes] Решение: векторы храним в LanceDB.\\n</ptg-memory>'
    >>> brief(0, (), 1000)
    '<ptg-memory>\\nPTG-MEM is active for this project: 0 memory atoms. Relevant memory is attached to each prompt automatically. Tools: ptg_search, ptg_node, ptg_decisions, ptg_remember (store a decision the moment it is made).\\n</ptg-memory>'
    >>> brief(5, (), 50)
    ''
    """
    lines = [Line(relations.day(e.at), e.file, 0.0, e.text, ()) for e in notes]
    return memshell.render_brief(live, lines, max_chars)
