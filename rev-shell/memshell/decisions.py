"""Лента решений: что чем заменено и что чему противоречит (связи memcore, действующие и закрытые).

node: leaf
"""
import math

import deal
from memcore.types import Edge

from memshell.types import DECISION_KINDS


@deal.pre(lambda edges, limit: limit >= 0 and all(math.isfinite(e.valid_at) for e in edges))
@deal.ensure(lambda edges, limit, result: len(result) <= limit and all(e in edges and e.kind in DECISION_KINDS for e in result))
@deal.ensure(lambda edges, limit, result: all(a.valid_at >= b.valid_at for a, b in zip(result, result[1:], strict=False)))
@deal.has()
def decision_pairs(edges: list[Edge], limit: int) -> list[Edge]:
    """Связи видов DECISION_KINDS (supersedes, contradicts) — и действующие, и закрытые: история не теряется. Новые
    первыми — по убыванию valid_at, при равном — в порядке входа. Не больше limit.

    D: 2

    >>> decision_pairs([Edge("b", "a", "supersedes", 5.0, 0.0, "edit"), Edge("c", "b", "refines", 7.0, 0.0, "agent"), Edge("d", "c", "contradicts", 9.0, 0.0, "model")], 10)
    [Edge(src='d', dst='c', kind='contradicts', valid_at=9.0, invalid_at=0.0, source='model'), Edge(src='b', dst='a', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='edit')]
    >>> decision_pairs([Edge("b", "a", "supersedes", 5.0, 8.0, "edit"), Edge("c", "a", "supersedes", 5.0, 0.0, "agent")], 1)
    [Edge(src='b', dst='a', kind='supersedes', valid_at=5.0, invalid_at=8.0, source='edit')]
    >>> decision_pairs([], 10)
    []
    """
    picked = [e for e in edges if e.kind in DECISION_KINDS]
    picked = sorted(picked, key=lambda e: e.valid_at, reverse=True)
    return picked[:limit]
