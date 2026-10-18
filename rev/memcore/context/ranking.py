"""Порядок находок: порог сходства, затем убывание score, при равном — по id.

node: leaf
"""
import deal

from memcore.types import Hit


@deal.ensure(lambda hits, min_score, result: len(result) == sum(1 for h in hits if h.score >= min_score))
@deal.ensure(lambda hits, min_score, result: all(h.score >= min_score for h in result))
@deal.ensure(lambda hits, min_score, result: all(r in hits for r in result))
@deal.ensure(
    lambda hits, min_score, result: all(
        (-a.score, a.id) <= (-b.score, b.id) for a, b in zip(result, result[1:], strict=False)
    )
)
@deal.has()
def rank_hits(hits: list[Hit], min_score: float) -> list[Hit]:
    """Находки со сходством не ниже min_score — по убыванию score, при равном — по id. Повторы id не трогаются.

    D: 2

    >>> rank_hits([Hit("a", "f.md", "x", 0.4, ""), Hit("b", "f.md", "y", 0.9, ""), Hit("c", "g.md", "z", 0.5, "")], 0.5)
    [Hit(id='b', file='f.md', text='y', score=0.9, successor=''), Hit(id='c', file='g.md', text='z', score=0.5, successor='')]
    >>> rank_hits([Hit("b", "f.md", "x", 0.5, ""), Hit("a", "g.md", "y", 0.5, ""), Hit("c", "f.md", "z", 0.7, "")], 0.0)
    [Hit(id='c', file='f.md', text='z', score=0.7, successor=''), Hit(id='a', file='g.md', text='y', score=0.5, successor=''), Hit(id='b', file='f.md', text='x', score=0.5, successor='')]
    >>> rank_hits([Hit("a", "f.md", "x", 0.1, "")], 0.5)
    []
    >>> rank_hits([], 0.5)
    []
    """
    kept = [h for h in hits if h.score >= min_score]
    return sorted(kept, key=lambda h: (-h.score, h.id))
