"""Связи между атомами по схеме Graphiti: у связи есть срок действия, новая заявка другого вида закрывает прежнюю
сроком, а не удаляет её; действующая версия атома — по цепочке действующих связей supersedes.

node: leaf
"""
import math

import deal

from memcore.types import RELATION_KINDS, Claim, Edge


@deal.pre(lambda edges, claim: math.isfinite(claim.at) and claim.at > 0 and all(math.isfinite(e.valid_at) and math.isfinite(e.invalid_at) for e in edges))
@deal.pre(lambda edges, claim: len({(e.src, e.dst) for e in edges if e.invalid_at == 0.0}) == sum(1 for e in edges if e.invalid_at == 0.0))
@deal.ensure(lambda edges, claim, result: len({(e.src, e.dst) for e in result if e.invalid_at == 0.0}) == sum(1 for e in result if e.invalid_at == 0.0))
@deal.ensure(lambda edges, claim, result: len(result) - len(edges) in (0, 1))
@deal.ensure(lambda edges, claim, result: all((r.src, r.dst, r.kind, r.valid_at, r.source) == (e.src, e.dst, e.kind, e.valid_at, e.source) for r, e in zip(result, edges, strict=False)))
@deal.ensure(lambda edges, claim, result: claim.src == claim.dst or claim.kind not in RELATION_KINDS
             or any((e.src, e.dst, e.kind, e.invalid_at) == (claim.src, claim.dst, claim.kind, 0.0) for e in result))
@deal.has()
def record_relation(edges: list[Edge], claim: Claim) -> list[Edge]:
    """Связи после заявки claim. Связь действует, пока invalid_at == 0.0.

    Связь атома с самим собой и вид не из RELATION_KINDS не записываются — связи без изменений. Если у пары
    (src, dst) уже действует связь того же вида — без изменений. Если действует связь другого вида — она закрывается
    (invalid_at = claim.at, остальное как было, на своём месте), а в конец добавляется новая. Если действующей нет —
    новая в конец. Новая связь: Edge(src, dst, kind, valid_at=claim.at, invalid_at=0.0, source=claim.source).

    D: 5

    >>> record_relation([], Claim("b", "a", "supersedes", 10.0, "agent"))
    [Edge(src='b', dst='a', kind='supersedes', valid_at=10.0, invalid_at=0.0, source='agent')]
    >>> record_relation([Edge("b", "a", "refines", 5.0, 0.0, "agent")], Claim("b", "a", "supersedes", 10.0, "model"))
    [Edge(src='b', dst='a', kind='refines', valid_at=5.0, invalid_at=10.0, source='agent'), Edge(src='b', dst='a', kind='supersedes', valid_at=10.0, invalid_at=0.0, source='model')]
    >>> record_relation([Edge("b", "a", "supersedes", 5.0, 0.0, "agent")], Claim("b", "a", "supersedes", 10.0, "model"))
    [Edge(src='b', dst='a', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='agent')]
    >>> record_relation([], Claim("a", "a", "supersedes", 10.0, "agent"))
    []
    >>> record_relation([], Claim("b", "a", "похоже", 10.0, "agent"))
    []
    """
    if claim.src == claim.dst or claim.kind not in RELATION_KINDS:
        return list(edges)
    result: list[Edge] = []
    for e in edges:
        if (e.src, e.dst) == (claim.src, claim.dst) and e.invalid_at == 0.0:
            if e.kind == claim.kind:
                return list(edges)
            result.append(Edge(e.src, e.dst, e.kind, e.valid_at, claim.at, e.source))
        else:
            result.append(e)
    result.append(Edge(claim.src, claim.dst, claim.kind, claim.at, 0.0, claim.source))
    return result


@deal.pre(lambda atom, edges: all(math.isfinite(e.valid_at) and math.isfinite(e.invalid_at) for e in edges))
@deal.ensure(lambda atom, edges, result: result == atom or any(e.src == result and e.kind == "supersedes" and e.invalid_at == 0.0 for e in edges))
@deal.has()
def current_version(atom: str, edges: list[Edge]) -> str:
    """Действующая версия атома: идти по действующим связям supersedes от атома к тому, кто его заменил (связь
    src → dst значит «src заменяет dst»). Если заменивших несколько — с наибольшим valid_at, при равном — с меньшим
    src. Цикл: остановиться на последнем атоме перед повтором.

    D: 4

    >>> current_version("a", [Edge("b", "a", "supersedes", 5.0, 0.0, "agent"), Edge("c", "b", "supersedes", 7.0, 0.0, "agent")])
    'c'
    >>> current_version("a", [Edge("b", "a", "supersedes", 5.0, 9.0, "agent")])
    'a'
    >>> current_version("a", [Edge("b", "a", "supersedes", 5.0, 0.0, "agent"), Edge("a", "b", "supersedes", 6.0, 0.0, "agent")])
    'b'
    >>> current_version("a", [Edge("b", "a", "refines", 5.0, 0.0, "agent")])
    'a'
    """
    current = atom
    seen = {atom}
    while True:
        replacers = [
            e
            for e in edges
            if e.dst == current and e.kind == "supersedes" and e.invalid_at == 0.0
        ]
        if not replacers:
            return current
        best = min(replacers, key=lambda e: (-e.valid_at, e.src))
        if best.src in seen:
            return current
        seen.add(best.src)
        current = best.src
