"""Повторы id: остаётся первая находка по порядку входа, остальные с тем же id отбрасываются.

node: leaf
"""
import deal

from memcore.types import Hit


@deal.ensure(lambda hits, result: len({h.id for h in result}) == len(result))
@deal.ensure(lambda hits, result: [h.id for h in result] == list(dict.fromkeys(h.id for h in hits)))
@deal.ensure(lambda hits, result: all(next(h for h in hits if h.id == r.id) == r for r in result))
@deal.has()
def first_per_id(hits: list[Hit]) -> list[Hit]:
    """Находки без повторов id: от каждого id остаётся первая по порядку входа (счёт не сравнивается), порядок
    остальных сохраняется.

    D: 2

    >>> first_per_id([Hit("a", "f.md", "x", 0.9, ""), Hit("a", "g.md", "y", 0.5, ""), Hit("b", "f.md", "z", 0.4, "")])
    [Hit(id='a', file='f.md', text='x', score=0.9, successor=''), Hit(id='b', file='f.md', text='z', score=0.4, successor='')]
    >>> first_per_id([Hit("a", "f.md", "старое", 0.5, ""), Hit("a", "g.md", "новое", 0.9, "")])
    [Hit(id='a', file='f.md', text='старое', score=0.5, successor='')]
    >>> first_per_id([Hit("b", "f.md", "x", 0.9, ""), Hit("a", "g.md", "y", 0.8, "")])
    [Hit(id='b', file='f.md', text='x', score=0.9, successor=''), Hit(id='a', file='g.md', text='y', score=0.8, successor='')]
    >>> first_per_id([])
    []
    """
    seen: set[str] = set()
    result: list[Hit] = []
    for h in hits:
        if h.id not in seen:
            seen.add(h.id)
            result.append(h)
    return result
