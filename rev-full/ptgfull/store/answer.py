"""Ответ хранилища: порядок атомов и находки с оценками из сырых пар, которые отдают таблицы.

node: leaf
"""
import math

import deal

from ptgfull.types import Atom, Hit


@deal.ensure(lambda atoms, ids, result: len({a.id for a in result}) == len(result)
             and all(a in atoms for a in result)
             and (not ids or [a.id for a in result] == [i for i in dict.fromkeys(ids) if i in {a.id for a in atoms}])
             and (bool(ids) or {a.id for a in result} == {a.id for a in atoms})
             and (bool(ids) or [(a.file, a.order, a.id) for a in result] == sorted((a.file, a.order, a.id) for a in result)))
@deal.has()
def arrange_atoms(atoms: tuple[Atom, ...], ids: tuple[str, ...]) -> tuple[Atom, ...]:
    """Атомы без повторов id (при повторе — последний). ids непуст — только они, в порядке ids без повторов, отсутствующие
    пропускаются; ids = () — все атомы по (file, order, id).

    D: 3

    >>> arrange_atoms((Atom('a2', 'n.md', 'два', '', 1), Atom('a1', 'n.md', 'раз', '', 0)), ())
    (Atom(id='a1', file='n.md', text='раз', session='', order=0), Atom(id='a2', file='n.md', text='два', session='', order=1))
    >>> arrange_atoms((Atom('a2', 'n.md', 'два', '', 1), Atom('a1', 'n.md', 'раз', '', 0)), ('a2', 'x', 'a2', 'a1'))
    (Atom(id='a2', file='n.md', text='два', session='', order=1), Atom(id='a1', file='n.md', text='раз', session='', order=0))
    >>> arrange_atoms((), ('a1',))
    ()
    """
    last: dict[str, Atom] = {}
    for atom in atoms:
        last[atom.id] = atom
    if ids:
        return tuple(last[i] for i in dict.fromkeys(ids) if i in last)
    return tuple(sorted(last.values(), key=lambda a: (a.file, a.order, a.id)))


@deal.pre(lambda pairs, via, k: k >= 0)
@deal.ensure(lambda pairs, via, k, result: len(result) <= k and all(h.via == via for h in result)
             and all(math.isfinite(h.score) for h in result) and {h.id for h in result} <= {i for i, _ in pairs}
             and all(a.score >= b.score for a, b in zip(result, result[1:], strict=False)))
@deal.has()
def nearest_hits(pairs: tuple[tuple[str, float], ...], via: str, k: int) -> tuple[Hit, ...]:
    """Находки по парам (id, косинусное расстояние): оценка round(1 − расстояние, 6), via = via; пары с нечисловой оценкой
    (nan, бесконечность) отбрасываются; лучшие первыми, при равной оценке — в порядке пар; не больше k.

    D: 2

    >>> nearest_hits((('a1', 0.25), ('a2', 0.0), ('a3', 0.5)), 'h', 2)
    (Hit(id='a2', score=1.0, via='h'), Hit(id='a1', score=0.75, via='h'))
    >>> nearest_hits((('a1', 0.1234567),), 'h', 5)
    (Hit(id='a1', score=0.876543, via='h'),)
    >>> nearest_hits((), 'h', 3)
    ()
    """
    hits: list[Hit] = []
    for ident, distance in pairs:
        score = round(1 - distance, 6)
        if math.isfinite(score):
            hits.append(Hit(ident, score, via))
    hits.sort(key=lambda h: -h.score)
    return tuple(hits[:k])


@deal.pre(lambda pairs, k: k >= 0)
@deal.ensure(lambda pairs, k, result: len(result) <= k and all(h.via == "lexical" for h in result)
             and all(math.isfinite(h.score) for h in result) and {h.id for h in result} <= {i for i, _ in pairs}
             and all(a.score >= b.score for a, b in zip(result, result[1:], strict=False)))
@deal.has()
def word_hits(pairs: tuple[tuple[str, float], ...], k: int) -> tuple[Hit, ...]:
    """Находки по парам (id, оценка _score): оценка round(_score, 6), via = 'lexical'; пары с нечисловой оценкой (nan,
    бесконечность) отбрасываются; лучшие первыми, при равной оценке — в порядке пар; не больше k.

    D: 2

    >>> word_hits((('a1', 0.5), ('a2', 1.25)), 5)
    (Hit(id='a2', score=1.25, via='lexical'), Hit(id='a1', score=0.5, via='lexical'))
    >>> word_hits((('a1', 1.23456789),), 5)
    (Hit(id='a1', score=1.234568, via='lexical'),)
    >>> word_hits((('a1', 2.0), ('a2', 1.0)), 1)
    (Hit(id='a1', score=2.0, via='lexical'),)
    >>> word_hits((), 3)
    ()
    """
    hits: list[Hit] = []
    for ident, value in pairs:
        score = round(value, 6)
        if math.isfinite(score):
            hits.append(Hit(ident, score, 'lexical'))
    hits.sort(key=lambda h: -h.score)
    return tuple(hits[:k])
