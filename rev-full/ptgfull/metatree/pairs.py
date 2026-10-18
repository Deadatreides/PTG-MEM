"""Одно ребро на пару (источник, цель): повторы пары отбрасываются.

node: leaf
"""
import deal

from ptgfull.types import Edge


@deal.ensure(lambda edges, result: len({(e.src, e.dst) for e in result}) == len(result)
             and {(e.src, e.dst) for e in result} == {(e.src, e.dst) for e in edges}
             and all(e in edges for e in result))
@deal.has()
def first_per_pair(edges: tuple[Edge, ...]) -> tuple[Edge, ...]:
    """ФТ-007, ФТ-009. Из рёбер с одной парой (src, dst) остаётся первое, порядок прежний; старший вид шва выигрывает за
    счёт порядка вызывающего: edits, reads, mentions.

    D: 2

    >>> first_per_pair((Edge('a1', 'file:x.py', 'edits'), Edge('a1', 'file:x.py', 'mentions'), Edge('a1', 'x.f', 'mentions')))
    (Edge(src='a1', dst='file:x.py', kind='edits'), Edge(src='a1', dst='x.f', kind='mentions'))
    >>> first_per_pair((Edge('a1', 'file:x.py', 'reads'), Edge('a2', 'file:x.py', 'reads'), Edge('a1', 'file:x.py', 'reads')))
    (Edge(src='a1', dst='file:x.py', kind='reads'), Edge(src='a2', dst='file:x.py', kind='reads'))
    >>> first_per_pair(())
    ()
    """
    seen: set[tuple[str, str]] = set()
    out: list[Edge] = []
    for e in edges:
        key = (e.src, e.dst)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return tuple(out)
