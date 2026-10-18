"""Связи формата 6: какие из edges.json переносятся, как они становятся Edge и как собираются пачками для
relations.record_pairs. Решения чистые.

node: leaf
"""
import deal
from memcore.types import Edge

from ptgcore import relations
from ptgcore.migration import scalars
from ptgcore.types import Entry


@deal.ensure(lambda raw, entries, kinds, result: all(s != d and k in kinds and t > 0.0 for s, d, k, t in result)
             and {s for s, _, _, _ in result} | {d for _, d, _, _ in result} <= {e.id for e in entries})
@deal.has()
def known(raw: object, entries: tuple[Entry, ...], kinds: tuple[str, ...]) -> tuple[tuple[str, str, str, float], ...]:
    """Аргументы v6_edge (src, dst, kind, at) для связей edges.json (raw — разобранный JSON, список объектов
    {'from', 'to', 'type'}; не список — ()), по порядку raw. Берутся связи, у которых оба конца среди entries, концы
    разные, вид (как записан в raw) среди kinds, а время атома from (at) больше 0.0; at — это время.

    D: 7

    >>> known([{'from': 'n2', 'to': 'n1', 'type': 'supersedes'}], (Entry('n1', 'docs/a.md', 0, '0470878cd8043cf6', 'Векторы храним в SQLite.', '', 'Векторы храним в SQLite.', 1760000000.0, 'retracted', 1760000000.0, '', 'file'), Entry('n2', 'docs/a.md', 0, '1d91240b07a4dac5', 'Векторы храним в LanceDB.', '', 'Векторы храним в LanceDB.', 1760003600.0, 'active', 0.0, '', 'file')), ('supersedes', 'revises'))
    (('n2', 'n1', 'supersedes', 1760003600.0),)
    >>> known([{'from': 'n2', 'to': 'n1', 'type': 'revises'}], (Entry('n1', 'a.md', 0, 'd1', 't', '', 't', 5.0, 'active', 0.0, '', 'file'), Entry('n2', 'a.md', 1, 'd2', 't2', '', 't2', 7.0, 'active', 0.0, '', 'file')), ('supersedes', 'revises'))
    (('n2', 'n1', 'revises', 7.0),)
    >>> known([{'from': 'n2', 'to': 'zz', 'type': 'supersedes'}, {'from': 'n2', 'to': 'n1', 'type': 'similar'}, {'from': 'n2', 'to': 'n2', 'type': 'refines'}], (Entry('n1', 'a.md', 0, 'd1', 't', '', 't', 5.0, 'active', 0.0, '', 'file'), Entry('n2', 'a.md', 1, 'd2', 't2', '', 't2', 7.0, 'active', 0.0, '', 'file')), ('supersedes', 'refines'))
    ()
    >>> known([{'from': 'n3', 'to': 'n1', 'type': 'refines'}], (Entry('n1', 'a.md', 0, 'd1', 't', '', 't', 5.0, 'active', 0.0, '', 'file'), Entry('n3', 'a.md', 2, 'd3', 't3', '', 't3', 0.0, 'active', 0.0, '', 'file')), ('refines',))
    ()
    >>> known({}, (), ('supersedes',))
    ()
    """
    if not isinstance(raw, list):
        return ()
    by_id: dict[str, Entry] = {e.id: e for e in entries}
    found: list[tuple[str, str, str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        src = item.get('from')
        dst = item.get('to')
        kind = item.get('type')
        if not (isinstance(src, str) and isinstance(dst, str) and isinstance(kind, str)):
            continue
        if src == dst or src not in by_id or dst not in by_id or kind not in kinds:
            continue
        at = float(by_id[src].at)
        if at > 0.0:
            found.append((src, dst, kind, at))
    return tuple(found)


@deal.pre(lambda src, dst, kind, at: src != dst and at > 0.0)
@deal.ensure(lambda src, dst, kind, at, result: result.invalid_at == 0.0 and result.source == 'v6')
@deal.has()
def build(src: str, dst: str, kind: str, at: float) -> Edge:
    """Связь формата 6: Edge(src, dst, kind, at, 0.0, 'v6').

    D: 1

    >>> build('n2', 'n1', 'supersedes', 1760003600.0)
    Edge(src='n2', dst='n1', kind='supersedes', valid_at=1760003600.0, invalid_at=0.0, source='v6')
    >>> build('b', 'a', 'refines', 5.0)
    Edge(src='b', dst='a', kind='refines', valid_at=5.0, invalid_at=0.0, source='v6')
    """
    return Edge(src, dst, kind, at, 0.0, 'v6')


@deal.pre(lambda edges: all(e.src != e.dst and e.valid_at > 0.0 and e.kind in relations.TYPE_ORDER for e in edges))
@deal.ensure(lambda edges, result: sum(len(p) for _, _, p in result) == len(edges)
             and all(a != b for _, _, p in result for a, b in p))
@deal.has()
def batches(edges: tuple[Edge, ...]) -> tuple[tuple[str, float, tuple[tuple[str, str], ...]], ...]:
    """Связи пачками для relations.record_pairs: (вид, valid_at, пары (src, dst)); пачка на каждую пару (вид,
    valid_at) по порядку её первого появления, пары внутри — как у edges.

    D: 3

    >>> batches((Edge('n2', 'n1', 'supersedes', 1760003600.0, 0.0, 'v6'),))
    (('supersedes', 1760003600.0, (('n2', 'n1'),)),)
    >>> batches((Edge('b', 'a', 'refines', 5.0, 0.0, 'v6'), Edge('c', 'a', 'supersedes', 5.0, 0.0, 'v6'), Edge('d', 'a', 'refines', 5.0, 0.0, 'v6')))
    (('refines', 5.0, (('b', 'a'), ('d', 'a'))), ('supersedes', 5.0, (('c', 'a'),)))
    >>> batches(())
    ()
    """
    groups: dict[tuple[str, float], list[tuple[str, str]]] = {}
    for e in edges:
        groups.setdefault((e.kind, e.valid_at), []).append((e.src, e.dst))
    return tuple((kind, at, tuple(pairs)) for (kind, at), pairs in groups.items())
