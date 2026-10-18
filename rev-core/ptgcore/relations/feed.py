"""Показ связей: связи у карточки, соседи по файлу и лента решений — Link, Side и Decision из связей и атомов.

node: leaf
"""
import math

import deal
import memshell
from memcore.types import Edge

from ptgcore.relations import show, state
from ptgcore.types import Decision, Entry, Link, Side

TYPE_ORDER = ('supersedes', 'contradicts', 'fixes', 'refines', 'continues', 'returns_to', 'reinforces')


@deal.ensure(lambda atom, edges, others, result: all(x.dir in ('in', 'out') and x.kind in TYPE_ORDER and len(x.text) <= 240 for x in result)
             and [TYPE_ORDER.index(x.kind) for x in result] == sorted(TYPE_ORDER.index(x.kind) for x in result))
@deal.has()
def links(atom: str, edges: tuple[Edge, ...], others: tuple[Entry, ...]) -> tuple[Link, ...]:
    """Действующие связи атома (invalid_at == 0.0, вид из TYPE_ORDER, atom — src или dst), другой конец которых есть среди
    others (по id): Link(dir — 'out', если atom — src, иначе 'in'; kind; id другого конца; source связи; status —
    state.status(другой конец, edges); date — show.day(его at); text — первые 240 знаков его text). Порядок: по месту вида в
    TYPE_ORDER, внутри вида — по valid_at (при равном — как в edges).

    D: 7

    >>> links('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), (Entry('a2', 'n.md', 0, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file'),))
    (Link(dir='in', kind='supersedes', id='a2', source='edit', status='active', date='2025-10-09', text='новое'),)
    >>> links('a1', (), ())
    ()
    >>> links('a1', (Edge('a1', 'a3', 'refines', 6.0, 0.0, 'agent'), Edge('a2', 'a1', 'contradicts', 7.0, 0.0, 'model'), Edge('a4', 'a1', 'supersedes', 8.0, 0.0, 'edit'), Edge('a5', 'a1', 'fixes', 9.0, 3.0, 'agent')), (Entry('a2', 'n.md', 0, 'd2', 'раз', '', 'раз', 1760003600.0, 'active', 0.0, '', 'file'), Entry('a3', 'n.md', 1, 'd3', 'два', '', 'два', 1760000000.0, 'retracted', 1760003600.0, '', 'file')))
    (Link(dir='in', kind='contradicts', id='a2', source='model', status='active', date='2025-10-09', text='раз'), Link(dir='out', kind='refines', id='a3', source='agent', status='retracted', date='2025-10-09', text='два'))
    >>> links('a1', (Edge('a3', 'a1', 'refines', 9.0, 0.0, 'agent'), Edge('a2', 'a1', 'refines', 4.0, 0.0, 'agent')), (Entry('a2', 'n.md', 0, 'd2', 'раз', '', 'раз', 1760003600.0, 'active', 0.0, '', 'file'), Entry('a3', 'n.md', 1, 'd3', 'два', '', 'два', 1760000000.0, 'active', 0.0, '', 'file')))
    (Link(dir='in', kind='refines', id='a2', source='agent', status='active', date='2025-10-09', text='раз'), Link(dir='in', kind='refines', id='a3', source='agent', status='active', date='2025-10-09', text='два'))
    """
    by_id: dict[str, Entry] = {}
    for other in others:
        by_id.setdefault(other.id, other)
    rows: list[tuple[int, float, Link]] = []
    for e in edges:
        if e.invalid_at != 0.0 or e.kind not in TYPE_ORDER or atom not in (e.src, e.dst):
            continue
        out = e.src == atom
        end = by_id.get(e.dst if out else e.src)
        if end is None:
            continue
        link = Link(
            dir='out' if out else 'in',
            kind=e.kind,
            id=end.id,
            source=e.source,
            status=state.status(end, edges),
            date=show.day(end.at),
            text=end.text[:240],
        )
        rows.append((TYPE_ORDER.index(e.kind), e.valid_at, link))
    rows.sort(key=lambda row: (row[0], row[1]))
    return tuple(row[2] for row in rows)


@deal.ensure(lambda entry, siblings, edges, root, result: len(result) <= 9 and (not result or entry.id in {s.id for s in result}))
@deal.has()
def lineage(entry: Entry, siblings: tuple[Entry, ...], edges: tuple[Edge, ...], root: str) -> tuple[Side, ...]:
    """Соседи по файлу: позиция pos — первый атом в siblings с id, равным entry.id; берутся siblings[max(0, pos - 4):pos + 5]
    (до четырёх до и до четырёх после, сам атом внутри, порядок siblings); Side(id, первые 160 знаков text,
    show.rel(file, root), state.status(атом, edges), show.day(at)). entry нет в siblings — ().

    D: 3

    >>> lineage(Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1760000000.0, 'active', 0.0, '', 'file'), (Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1760000000.0, 'active', 0.0, '', 'file'),), (), '.')
    (Side(id='a1', text='раз', file='n.md', status='active', date='2025-10-09'),)
    >>> lineage(Entry('x2', 'n.md', 1, 'd2', 'два', '', 'два', 1760000000.0, 'active', 0.0, '', 'file'), (Entry('x1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1760000000.0, 'active', 0.0, '', 'file'), Entry('x2', 'n.md', 1, 'd2', 'два', '', 'два', 1760000000.0, 'active', 0.0, '', 'file'), Entry('x3', 'n.md', 2, 'd3', 'три', '', 'три', 1760000000.0, 'active', 0.0, '', 'file')), (Edge('x3', 'x1', 'supersedes', 5.0, 0.0, 'edit'),), '.')
    (Side(id='x1', text='раз', file='n.md', status='superseded', date='2025-10-09'), Side(id='x2', text='два', file='n.md', status='active', date='2025-10-09'), Side(id='x3', text='три', file='n.md', status='active', date='2025-10-09'))
    >>> lineage(Entry('z1', 'n.md', 5, 'd9', 'нет', '', 'нет', 1760000000.0, 'active', 0.0, '', 'file'), (Entry('x1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1760000000.0, 'active', 0.0, '', 'file'),), (), '.')
    ()
    """
    pos = next((i for i, s in enumerate(siblings) if s.id == entry.id), -1)
    if pos < 0:
        return ()
    window = siblings[max(0, pos - 4):pos + 5]
    return tuple(
        Side(
            id=s.id,
            text=s.text[:160],
            file=show.rel(s.file, root),
            status=state.status(s, edges),
            date=show.day(s.at),
        )
        for s in window
    )


@deal.pre(lambda edges, entries, limit: limit >= 0)
@deal.ensure(lambda edges, entries, limit, result: len(result) <= limit and all(e in edges for e in result)
             and {i for e in result for i in (e.src, e.dst)} <= {x.id for x in entries}
             and all(a.valid_at >= b.valid_at for a, b in zip(result, result[1:], strict=False)))
@deal.has()
def picked(edges: tuple[Edge, ...], entries: tuple[Entry, ...], limit: int) -> tuple[Edge, ...]:
    """Связи ленты решений: memshell.decision_pairs(список связей с конечным valid_at, limit) — supersedes и contradicts, и
    действующие, и закрытые, новые первыми, не больше limit; из них остаются те, у которых и src, и dst есть среди entries
    (по id), порядок тот же. Отбор по известным концам идёт после ограничения limit.

    D: 3

    >>> picked((Edge('a2', 'a1', 'supersedes', 1760003600.0, 0.0, 'edit'),), (Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1760000000.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file')), 50)
    (Edge(src='a2', dst='a1', kind='supersedes', valid_at=1760003600.0, invalid_at=0.0, source='edit'),)
    >>> picked((), (), 50)
    ()
    >>> picked((Edge('a2', 'a1', 'supersedes', 1760003600.0, 0.0, 'edit'),), (Entry('a2', 'n.md', 1, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file'),), 50)
    ()
    >>> picked((Edge('x', 'a1', 'supersedes', 1760006400.0, 0.0, 'edit'), Edge('a2', 'a1', 'supersedes', 1760003600.0, 0.0, 'edit')), (Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1760000000.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file')), 1)
    ()
    >>> picked((Edge('a2', 'a1', 'supersedes', 1760000000.0, 1760003600.0, 'edit'), Edge('a3', 'a2', 'contradicts', 1760003600.0, 0.0, 'model')), (Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1760000000.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file'), Entry('a3', 'n.md', 2, 'd3', 'третье', '', 'третье', 1760007200.0, 'active', 0.0, '', 'file')), 50)
    (Edge(src='a3', dst='a2', kind='contradicts', valid_at=1760003600.0, invalid_at=0.0, source='model'), Edge(src='a2', dst='a1', kind='supersedes', valid_at=1760000000.0, invalid_at=1760003600.0, source='edit'))
    """
    ids = {x.id for x in entries}
    finite = [e for e in edges if math.isfinite(e.valid_at)]
    top = memshell.decision_pairs(finite, limit)
    return tuple(e for e in top if e.src in ids and e.dst in ids)


@deal.pre(lambda picks, near, entries, root: {i for e in picks for i in (e.src, e.dst)} <= {x.id for x in entries})
@deal.ensure(lambda picks, near, entries, root, result: len(result) == len(picks))
@deal.has()
def decide(picks: tuple[Edge, ...], near: tuple[Edge, ...], entries: tuple[Entry, ...], root: str) -> tuple[Decision, ...]:
    """Решения по связям picks (концы всех есть среди entries), по порядку picks: Decision(kind, show.day(valid_at),
    valid_at, source, closed — invalid_at != 0.0, new — Side для src, old — Side для dst); Side(id, первые 280 знаков
    text, show.rel(file, root), state.status(атом, near), show.day(at)). near — связи, по которым считаются состояния
    сторон (не все связи базы, а их часть, достаточная для состояний этих атомов).

    D: 2

    >>> decide((Edge('a2', 'a1', 'supersedes', 1760003600.0, 0.0, 'edit'),), (Edge('a2', 'a1', 'supersedes', 1760003600.0, 0.0, 'edit'),), (Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1760000000.0, 'retracted', 1760003600.0, '', 'file'), Entry('a2', 'n.md', 0, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file')), '.')
    (Decision(kind='supersedes', date='2025-10-09', at=1760003600.0, source='edit', closed=False, new=Side(id='a2', text='новое', file='n.md', status='active', date='2025-10-09'), old=Side(id='a1', text='старое', file='n.md', status='retracted', date='2025-10-09')),)
    >>> decide((), (), (), '.')
    ()
    >>> decide((Edge('a2', 'a1', 'supersedes', 1760000000.0, 1760003600.0, 'edit'),), (Edge('a2', 'a1', 'supersedes', 1760000000.0, 1760003600.0, 'edit'),), (Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1760000000.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file')), '.')
    (Decision(kind='supersedes', date='2025-10-09', at=1760000000.0, source='edit', closed=True, new=Side(id='a2', text='новое', file='n.md', status='active', date='2025-10-09'), old=Side(id='a1', text='старое', file='n.md', status='active', date='2025-10-09')),)
    """
    by_id: dict[str, Entry] = {}
    for x in entries:
        by_id.setdefault(x.id, x)

    def side(x: Entry) -> Side:
        return Side(
            id=x.id,
            text=x.text[:280],
            file=show.rel(x.file, root),
            status=state.status(x, near),
            date=show.day(x.at),
        )

    return tuple(
        Decision(
            kind=e.kind,
            date=show.day(e.valid_at),
            at=e.valid_at,
            source=e.source,
            closed=e.invalid_at != 0.0,
            new=side(by_id[e.src]),
            old=side(by_id[e.dst]),
        )
        for e in picks
    )
