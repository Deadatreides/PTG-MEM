"""Пометки карточки в подсказке: заменена позже, заменяет прежнюю версию, противоречит, швы с кодом.

Готовое: relations.day (дата для показа).

node: leaf
"""
import deal

from ptgcore import relations
from ptgcore.types import Card, Entry, Link


@deal.ensure(lambda card, later, result: (result == '') == (not (card.current != '' and any(e.id == card.current for e in later))
                                                           and not any(k.dir == 'out' and k.kind == 'supersedes' for k in card.links)))
@deal.ensure(lambda card, later, result: result.startswith('replaced later (') == (card.current != ''
                                                                                   and any(e.id == card.current for e in later)))
@deal.ensure(lambda card, later, result: result == '' or result.startswith(('replaced later (', 'replaces an earlier version (')))
@deal.has()
def version_note(card: Card, later: tuple[Entry, ...]) -> str:
    """Пометка о версии: текущая версия карточки (card.current не пуст) есть в later — 'replaced later (<day(at)>):
    <первые 160 знаков её текста>' (relations.day); иначе первая связь out 'supersedes' — 'replaces an earlier version
    (<date связи>): <первые 160 знаков текста связи>'; иначе ''.

    D: 5

    >>> version_note(Card('a1', 0.9, 'superseded', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), 'a2', ''), (Entry('a2', 'n.md', 0, 'd2', 'LanceDB заменил SQLite.', '', 'LanceDB заменил SQLite.', 1760003600.0, 'active', 0.0, '', 'file'),))
    'replaced later (2025-10-09): LanceDB заменил SQLite.'
    >>> version_note(Card('a2', 0.8, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB заменил SQLite.', 23, (Link('out', 'supersedes', 'a1', 'edit', 'retracted', '2025-10-08', 'Векторы храним в SQLite.'),), '', ''), ())
    'replaces an earlier version (2025-10-08): Векторы храним в SQLite.'
    >>> version_note(Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), '', ''), ())
    ''
    >>> version_note(Card('a1', 0.9, 'superseded', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), 'a2', ''), ())
    ''
    """
    for e in later:
        if card.current != '' and e.id == card.current:
            return f'replaced later ({relations.day(e.at)}): {e.text[:160]}'
    for k in card.links:
        if k.dir == 'out' and k.kind == 'supersedes':
            return f'replaces an earlier version ({k.date}): {k.text[:160]}'
    return ''


@deal.ensure(lambda card, result: (result == '') == (not any(k.kind == 'contradicts' for k in card.links)))
@deal.ensure(lambda card, result: result == '' or (result.startswith('contradicts (')
                                                   == (next(k.dir for k in card.links if k.kind == 'contradicts') == 'out')))
@deal.ensure(lambda card, result: result == '' or result.startswith(('contradicts (', 'contradicted by (')))
@deal.has()
def conflict_note(card: Card) -> str:
    """Пометка о противоречии: первая связь вида 'contradicts' — 'contradicts (<date связи>): <первые 200 знаков текста
    связи>', если связь out, и 'contradicted by (<date связи>): <первые 200 знаков>', если in; связи нет — ''.

    D: 3

    >>> conflict_note(Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (Link('out', 'contradicts', 'a2', 'agent', 'active', '2025-10-09', 'Векторы храним в SQLite.'),), '', ''))
    'contradicts (2025-10-09): Векторы храним в SQLite.'
    >>> conflict_note(Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (Link('in', 'contradicts', 'a3', 'agent', 'active', '2025-10-10', 'Нет, в LanceDB.'),), '', ''))
    'contradicted by (2025-10-10): Нет, в LanceDB.'
    >>> conflict_note(Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), '', ''))
    ''
    >>> conflict_note(Card('a2', 0.8, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB заменил SQLite.', 23, (Link('out', 'supersedes', 'a1', 'edit', 'retracted', '2025-10-08', 'Векторы храним в SQLite.'),), '', ''))
    ''
    """
    for k in card.links:
        if k.kind == 'contradicts':
            if k.dir == 'out':
                return f'contradicts ({k.date}): {k.text[:200]}'
            return f'contradicted by ({k.date}): {k.text[:200]}'
    return ''


@deal.ensure(lambda card_id, code, result: len(result) == sum(1 for i, _ in code if i == card_id)
             and all(r.startswith('code: ') for r in result))
@deal.has()
def code_notes(card_id: str, code: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """Швы карточки с кодом: каждая строка code с id карточки (card_id) — 'code: <строка>', по порядку code.

    D: 2

    >>> code_notes('a1', (('a1', 'edits file:PTG-MEM/ptg_mem/engine.py'), ('a2', 'reads file:x.py'), ('a1', 'mentions file:y.py')))
    ('code: edits file:PTG-MEM/ptg_mem/engine.py', 'code: mentions file:y.py')
    >>> code_notes('a1', ())
    ()
    """
    return tuple('code: ' + line for i, line in code if i == card_id)


@deal.ensure(lambda card, later, code, result: all(n != '' for n in result)
             and result[len(result) - sum(1 for i, _ in code if i == card.id):] == tuple('code: ' + line for i, line in code if i == card.id))
@deal.has()
def notes_of(card: Card, later: tuple[Entry, ...], code: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """Все пометки карточки по порядку: version_note, conflict_note (пустые пропускаются), затем code_notes по строкам
    code с id карточки.

    D: 2

    >>> notes_of(Card('a1', 0.9, 'superseded', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), 'a2', ''), (Entry('a2', 'n.md', 0, 'd2', 'LanceDB заменил SQLite.', '', 'LanceDB заменил SQLite.', 1760003600.0, 'active', 0.0, '', 'file'),), (('a1', 'edits file:PTG-MEM/ptg_mem/engine.py'),))
    ('replaced later (2025-10-09): LanceDB заменил SQLite.', 'code: edits file:PTG-MEM/ptg_mem/engine.py')
    >>> notes_of(Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), '', ''), (), ())
    ()
    >>> notes_of(Card('a2', 0.8, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB заменил SQLite.', 23, (Link('out', 'supersedes', 'a1', 'edit', 'retracted', '2025-10-08', 'Векторы храним в SQLite.'), Link('in', 'contradicts', 'a3', 'agent', 'active', '2025-10-10', 'Нет, в LanceDB.')), '', ''), (), ())
    ('replaces an earlier version (2025-10-08): Векторы храним в SQLite.', 'contradicted by (2025-10-10): Нет, в LanceDB.')
    """
    notes = [version_note(card, later), conflict_note(card)]
    return tuple(n for n in notes if n != '') + code_notes(card.id, code)
