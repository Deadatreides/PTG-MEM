"""Заявки на связи: что становится со связями после заявки — чистые решения поверх memcore.record_relation, без базы.

node: leaf
"""
import math
import sys

import deal
import memcore
from memcore.types import Claim, Edge

TYPE_ORDER = ('supersedes', 'contradicts', 'fixes', 'refines', 'continues', 'returns_to', 'reinforces')


@deal.pre(lambda edges, src, dst, kind, at, source: src != dst and kind in TYPE_ORDER and at > 0.0 and source != ''
          and all(math.isfinite(e.valid_at) and math.isfinite(e.invalid_at) for e in edges)
          and len({(e.src, e.dst) for e in edges if e.invalid_at == 0.0}) == sum(e.invalid_at == 0.0 for e in edges))
@deal.ensure(lambda edges, src, dst, kind, at, source, result: len(result) - len(edges) in (0, 1)
             and sum(e.invalid_at == 0.0 for e in result if e.src == src and e.dst == dst) == 1)
@deal.has()
def apply(edges: tuple[Edge, ...], src: str, dst: str, kind: str, at: float, source: str) -> tuple[Edge, ...]:
    """Связи после одной заявки: memcore.record_relation(list(edges), Claim(src, dst, kind, min(at, sys.float_info.max),
    source)) (бесконечное время заменяется наибольшим конечным); итог — кортеж. Действующая связь пары (src, dst) другого
    вида закрывается (invalid_at = at), новая встаёт в конец; такая же действующая — без изменений.

    D: 1

    >>> apply((), 'a2', 'a1', 'supersedes', 5.0, 'edit')
    (Edge(src='a2', dst='a1', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='edit'),)
    >>> apply((Edge('b', 'a', 'refines', 5.0, 0.0, 'agent'),), 'b', 'a', 'supersedes', 10.0, 'model')
    (Edge(src='b', dst='a', kind='refines', valid_at=5.0, invalid_at=10.0, source='agent'), Edge(src='b', dst='a', kind='supersedes', valid_at=10.0, invalid_at=0.0, source='model'))
    >>> apply((Edge('b', 'a', 'supersedes', 5.0, 0.0, 'agent'),), 'b', 'a', 'supersedes', 10.0, 'model')
    (Edge(src='b', dst='a', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='agent'),)
    >>> apply((Edge('c', 'a', 'refines', 3.0, 0.0, 'agent'),), 'b', 'a', 'supersedes', 10.0, 'model')
    (Edge(src='c', dst='a', kind='refines', valid_at=3.0, invalid_at=0.0, source='agent'), Edge(src='b', dst='a', kind='supersedes', valid_at=10.0, invalid_at=0.0, source='model'))
    """
    return tuple(memcore.record_relation(list(edges), Claim(src, dst, kind, min(at, sys.float_info.max), source)))


@deal.pre(lambda edges, pairs, kind, at, source: kind in TYPE_ORDER and at > 0.0 and source != ''
          and all(a != b for a, b in pairs)
          and all(math.isfinite(e.valid_at) and math.isfinite(e.invalid_at) for e in edges)
          and len({(e.src, e.dst) for e in edges if e.invalid_at == 0.0}) == sum(e.invalid_at == 0.0 for e in edges))
@deal.ensure(lambda edges, pairs, kind, at, source, result: 0 <= len(result) - len(edges) <= len(pairs)
             and all(any((e.src, e.dst, e.kind, e.invalid_at) == (a, b, kind, 0.0) for e in result) for a, b in pairs))
@deal.has()
def apply_all(edges: tuple[Edge, ...], pairs: tuple[tuple[str, str], ...], kind: str, at: float, source: str) -> tuple[Edge, ...]:
    """Связи после заявок (src, dst) одного вида: по порядку pairs каждая заявка — как в apply (вход следующей заявки — итог
    предыдущей); итог — кортеж. pairs = () — edges без изменений.

    D: 2

    >>> apply_all((), (('b', 'a'), ('d', 'a')), 'refines', 5.0, 'v6')
    (Edge(src='b', dst='a', kind='refines', valid_at=5.0, invalid_at=0.0, source='v6'), Edge(src='d', dst='a', kind='refines', valid_at=5.0, invalid_at=0.0, source='v6'))
    >>> apply_all((Edge('b', 'a', 'supersedes', 5.0, 0.0, 'edit'),), (), 'refines', 6.0, 'v6')
    (Edge(src='b', dst='a', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='edit'),)
    >>> apply_all((Edge('b', 'a', 'refines', 5.0, 0.0, 'v6'),), (('b', 'a'),), 'supersedes', 9.0, 'edit')
    (Edge(src='b', dst='a', kind='refines', valid_at=5.0, invalid_at=9.0, source='v6'), Edge(src='b', dst='a', kind='supersedes', valid_at=9.0, invalid_at=0.0, source='edit'))
    """
    result = edges
    for src, dst in pairs:
        result = apply(result, src, dst, kind, at, source)
    return result


@deal.ensure(lambda edges, src, dst, result: all(e.src == src and e.dst == dst and e in edges for e in result))
@deal.has()
def of_pair(edges: tuple[Edge, ...], src: str, dst: str) -> tuple[Edge, ...]:
    """Связи пары в направлении src → dst (действующие и закрытые) в порядке edges; связи dst → src не берутся.

    D: 2

    >>> of_pair((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), 'a2', 'a1')
    (Edge(src='a2', dst='a1', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='edit'),)
    >>> of_pair((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a1', 'refines', 6.0, 0.0, 'agent'), Edge('a1', 'a2', 'refines', 7.0, 0.0, 'agent')), 'a2', 'a1')
    (Edge(src='a2', dst='a1', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='edit'),)
    >>> of_pair((), 'a2', 'a1')
    ()
    """
    return tuple(e for e in edges if e.src == src and e.dst == dst)
