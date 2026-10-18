"""Порядок и состояние атомов в памяти: один атом на id, порядок выдачи, порядок по месту в файле, снятие.

node: leaf
"""
import math
from dataclasses import replace

import deal

from ptgcore.types import Entry


@deal.ensure(lambda entries, result: [e.id for e in result] == list(dict.fromkeys(e.id for e in entries))
             and all(e == [x for x in entries if x.id == e.id][-1] for e in result))
@deal.has()
def distinct(entries: tuple[Entry, ...]) -> tuple[Entry, ...]:
    """Один атом на id: из повторов — последний; порядок — по первому появлению id.

    D: 2

    >>> distinct((Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'два', '', 'два', 2.0, 'active', 0.0, '', 'file'), Entry('a1', 'n.md', 0, 'd4', 'четыре', '', 'четыре', 3.0, 'active', 0.0, '', 'file')))
    (Entry(id='a1', file='n.md', position=0, digest='d4', text='четыре', question='', answer='четыре', at=3.0, status='active', retracted_at=0.0, session='', source='file'), Entry(id='a2', file='n.md', position=1, digest='d2', text='два', question='', answer='два', at=2.0, status='active', retracted_at=0.0, session='', source='file'))
    >>> distinct(())
    ()
    """
    last: dict[str, Entry] = {}
    for e in entries:
        last[e.id] = e
    return tuple(last.values())


@deal.pre(lambda entries, ids: all(not math.isnan(e.at) for e in entries))
@deal.ensure(lambda entries, ids, result: len({e.id for e in result}) == len(result) and all(e in entries for e in result)
             and (not ids or [e.id for e in result] == [i for i in dict.fromkeys(ids) if i in {e.id for e in entries}])
             and (bool(ids) or {e.id for e in result} == {e.id for e in entries})
             and (bool(ids) or [(e.file, e.position, e.at, e.id) for e in result] == sorted((e.file, e.position, e.at, e.id) for e in result)))
@deal.has()
def arrange(entries: tuple[Entry, ...], ids: tuple[str, ...]) -> tuple[Entry, ...]:
    """Атомы без повторов id (при повторе — последний). ids непуст — только они, в порядке ids без повторов, отсутствующие
    пропускаются; ids = () — все атомы по (file, position, at, id).

    D: 3

    >>> arrange((Entry('a2', 'n.md', 1, 'd2', 'два', '', 'два', 2.0, 'active', 0.0, '', 'file'), Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file')), ())
    (Entry(id='a1', file='n.md', position=0, digest='d1', text='раз', question='', answer='раз', at=1.0, status='active', retracted_at=0.0, session='', source='file'), Entry(id='a2', file='n.md', position=1, digest='d2', text='два', question='', answer='два', at=2.0, status='active', retracted_at=0.0, session='', source='file'))
    >>> arrange((Entry('a2', 'n.md', 1, 'd2', 'два', '', 'два', 2.0, 'active', 0.0, '', 'file'), Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file')), ('a2', 'x', 'a2', 'a1'))
    (Entry(id='a2', file='n.md', position=1, digest='d2', text='два', question='', answer='два', at=2.0, status='active', retracted_at=0.0, session='', source='file'), Entry(id='a1', file='n.md', position=0, digest='d1', text='раз', question='', answer='раз', at=1.0, status='active', retracted_at=0.0, session='', source='file'))
    >>> arrange((), ('a1',))
    ()
    """
    atoms = distinct(entries)
    if not ids:
        return tuple(sorted(atoms, key=lambda e: (e.file, e.position, e.at, e.id)))
    by_id = {e.id: e for e in atoms}
    return tuple(by_id[i] for i in dict.fromkeys(ids) if i in by_id)


@deal.pre(lambda entries: all(not math.isnan(e.at) for e in entries))
@deal.ensure(lambda entries, result: len(result) == len(entries) and set(result) == set(entries)
             and [(e.position, e.at, e.id) for e in result] == sorted((e.position, e.at, e.id) for e in entries))
@deal.has()
def in_place(entries: tuple[Entry, ...]) -> tuple[Entry, ...]:
    """Все атомы (и снятые) по месту в файле, затем по времени; равные — по id.

    D: 1

    >>> in_place((Entry('a2', 'n.md', 1, 'd2', 'два', '', 'два', 2.0, 'active', 0.0, '', 'file'), Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file')))
    (Entry(id='a1', file='n.md', position=0, digest='d1', text='раз', question='', answer='раз', at=1.0, status='active', retracted_at=0.0, session='', source='file'), Entry(id='a2', file='n.md', position=1, digest='d2', text='два', question='', answer='два', at=2.0, status='active', retracted_at=0.0, session='', source='file'))
    >>> in_place((Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file'), Entry('a3', 'n.md', 0, 'd3', 'три', '', 'три', 0.5, 'retracted', 4.0, '', 'file')))
    (Entry(id='a3', file='n.md', position=0, digest='d3', text='три', question='', answer='три', at=0.5, status='retracted', retracted_at=4.0, session='', source='file'), Entry(id='a1', file='n.md', position=0, digest='d1', text='раз', question='', answer='раз', at=1.0, status='active', retracted_at=0.0, session='', source='file'))
    """
    return tuple(sorted(entries, key=lambda e: (e.position, e.at, e.id)))


@deal.pre(lambda entries, at: at >= 0.0)
@deal.ensure(lambda entries, at, result: [r.id for r in result] == [e.id for e in entries if e.status == 'active']
             and all(r.status == 'retracted' and r.retracted_at == at for r in result))
@deal.has()
def retire(entries: tuple[Entry, ...], at: float) -> tuple[Entry, ...]:
    """Живые из entries (status == 'active') — снятыми: dataclasses.replace(e, status='retracted', retracted_at=at); порядок
    entries; уже снятые и прочие пропускаются.

    D: 2

    >>> retire((Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file'), Entry('a3', 'n.md', 0, 'd3', 'три', '', 'три', 0.5, 'retracted', 4.0, '', 'file')), 5.0)
    (Entry(id='a1', file='n.md', position=0, digest='d1', text='раз', question='', answer='раз', at=1.0, status='retracted', retracted_at=5.0, session='', source='file'),)
    >>> retire((), 5.0)
    ()
    """
    return tuple(replace(e, status='retracted', retracted_at=at) for e in entries if e.status == 'active')
