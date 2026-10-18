"""Цепочки замен по индексу связей: номера связей, касающихся атомов, связи цепочек supersedes от атомов, ближайший
заменивший каждого атома и сколько атомов заменено — каждое за один проход по связям (14.10: поиск по всем 16 702
связям для каждого из тысяч атомов — минуты).

node: leaf
"""
import math

import deal
from memcore.types import Edge


@deal.ensure(lambda edges, ids, result: result == tuple(sorted(set(result))) and all(0 <= i < len(edges) for i in result)
             and all((i in result) == (e.src in ids or e.dst in ids) for i, e in enumerate(edges)))
@deal.has()
def touching(edges: tuple[Edge, ...], ids: tuple[str, ...]) -> tuple[int, ...]:
    """Номера связей edges (с нуля, по возрастанию), у которых src или dst в ids (закрытые тоже); ids берётся
    множеством, один проход по edges. ids = () — ().

    D: 2

    >>> touching((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('b2', 'b1', 'supersedes', 7.0, 0.0, 'edit')), ('a1',))
    (0,)
    >>> touching((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ())
    ()
    >>> touching((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'supersedes', 6.0, 0.0, 'edit'), Edge('b2', 'b1', 'supersedes', 7.0, 0.0, 'edit')), ('a2', 'b1'))
    (0, 1, 2)
    """
    wanted = set(ids)
    return tuple(i for i, e in enumerate(edges) if e.src in wanted or e.dst in wanted)


@deal.ensure(lambda edges, result: len(result) == len({a for a, _ in result})
             and {a for a, _ in result} == {e.dst for e in edges if e.kind == 'supersedes' and e.invalid_at == 0.0 and math.isfinite(e.valid_at)}
             and all(any(e.dst == a and e.src == s and e.kind == 'supersedes' and e.invalid_at == 0.0 for e in edges) for a, s in result))
@deal.has()
def heads(edges: tuple[Edge, ...]) -> tuple[tuple[str, str], ...]:
    """Ближайший заменивший каждого заменённого атома: среди действующих связей supersedes (invalid_at == 0.0, valid_at
    конечно) с dst = атом берётся связь с наибольшим valid_at, при равном — с меньшим src (как в memcore.current_version);
    пары (атом, его src) по первому появлению атома среди dst связей edges; один проход.

    D: 4

    >>> heads((Edge('b', 'a', 'supersedes', 5.0, 0.0, 'edit'), Edge('c', 'a', 'supersedes', 7.0, 0.0, 'edit')))
    (('a', 'c'),)
    >>> heads((Edge('c', 'a', 'supersedes', 5.0, 0.0, 'edit'), Edge('b', 'a', 'supersedes', 5.0, 0.0, 'edit')))
    (('a', 'b'),)
    >>> heads((Edge('b', 'a', 'supersedes', 5.0, 9.0, 'edit'), Edge('c', 'a', 'refines', 6.0, 0.0, 'agent')))
    ()
    """
    best: dict[str, Edge] = {}
    for e in edges:
        if e.kind != 'supersedes' or e.invalid_at != 0.0 or not math.isfinite(e.valid_at):
            continue
        cur = best.get(e.dst)
        if cur is None or e.valid_at > cur.valid_at or (e.valid_at == cur.valid_at and e.src < cur.src):
            best[e.dst] = e
    return tuple((a, e.src) for a, e in best.items())


@deal.ensure(lambda edges, atoms, result: result == tuple(sorted(set(result))) and all(0 <= i < len(edges) for i in result)
             and all(edges[i].kind == 'supersedes' and edges[i].invalid_at == 0.0 and math.isfinite(edges[i].valid_at) for i in result)
             and all(i in result for a in atoms for i, e in enumerate(edges)
                     if e.dst == a and e.kind == 'supersedes' and e.invalid_at == 0.0 and math.isfinite(e.valid_at)))
@deal.has()
def walk(edges: tuple[Edge, ...], atoms: tuple[str, ...]) -> tuple[int, ...]:
    """Номера (по возрастанию) действующих связей supersedes (invalid_at == 0.0, valid_at конечно), по которым
    memcore.current_version находит текущую версию каждого из atoms: для атома и каждого атома его цепочки — все такие
    связи с dst = этот атом. Цепочка идёт к ближайшему заменившему (как в heads) и кончается, когда заменивших нет или
    он уже был в цепочке этого атома; атом, чьи заменившие уже учтены, второй раз не разбирается — один проход.

    D: 6

    >>> walk((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'supersedes', 6.0, 0.0, 'edit'), Edge('b2', 'b1', 'supersedes', 7.0, 0.0, 'edit')), ('a1',))
    (0, 1)
    >>> walk((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ())
    ()
    >>> walk((Edge('b', 'a', 'supersedes', 5.0, 0.0, 'agent'), Edge('a', 'b', 'supersedes', 6.0, 0.0, 'agent')), ('a',))
    (0, 1)
    >>> walk((Edge('a2', 'a1', 'supersedes', 5.0, 9.0, 'edit'), Edge('a3', 'a1', 'refines', 6.0, 0.0, 'agent')), ('a1',))
    ()
    >>> walk((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'supersedes', 6.0, 0.0, 'edit'), Edge('b2', 'b1', 'supersedes', 7.0, 0.0, 'edit'), Edge('a4', 'a2', 'supersedes', 3.0, 0.0, 'edit')), ('a1',))
    (0, 1, 3)
    """
    def live(e: Edge) -> bool:
        return e.kind == 'supersedes' and e.invalid_at == 0.0 and math.isfinite(e.valid_at)

    by_dst: dict[str, list[int]] = {}
    for i, e in enumerate(edges):
        if live(e):
            by_dst.setdefault(e.dst, []).append(i)
    nearest = dict(heads(edges))
    picked: set[int] = set()
    done: set[str] = set()
    for a in atoms:
        cur = a
        while cur not in done:
            done.add(cur)
            picked.update(by_dst.get(cur, []))
            if cur not in nearest:
                break
            cur = nearest[cur]
    return tuple(sorted(picked))


@deal.ensure(lambda edges, picked, result: len(result) <= len(edges) and all(e in edges for e in result)
             and len(result) == sum(i in set(picked) for i in range(len(edges))))
@deal.has()
def pick(edges: tuple[Edge, ...], picked: tuple[int, ...]) -> tuple[Edge, ...]:
    """Связи edges с номерами из picked, по порядку edges, каждая один раз (повторы в picked и номера вне edges не
    мешают).

    D: 2

    >>> pick((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'supersedes', 6.0, 0.0, 'edit'), Edge('b2', 'b1', 'supersedes', 7.0, 0.0, 'edit')), (2, 0, 2))
    (Edge(src='a2', dst='a1', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='edit'), Edge(src='b2', dst='b1', kind='supersedes', valid_at=7.0, invalid_at=0.0, source='edit'))
    >>> pick((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ())
    ()
    >>> pick((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), (5,))
    ()
    """
    chosen = set(picked)
    return tuple(e for i, e in enumerate(edges) if i in chosen)


@deal.ensure(lambda edges, ids, result: 0 <= result <= len(set(ids)))
@deal.has()
def replaced(edges: tuple[Edge, ...], ids: tuple[str, ...]) -> int:
    """Сколько разных атомов из ids заменены, как у state.successor (результат не ''): у атома есть ближайший
    заменивший по heads и он не сам атом; пустой атом '' (это не id) считается, как у successor (его результат ' ').
    Индекс heads строится один раз, дальше — поиск по нему для каждого атома.

    D: 3

    >>> replaced((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ('a1', 'a2'))
    1
    >>> replaced((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ('a1', 'a1'))
    1
    >>> replaced((), ('a1',))
    0
    >>> replaced((Edge('a1', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ('a1',))
    0
    """
    nearest = dict(heads(edges))
    return sum(1 for a in set(ids) if a == '' or (a in nearest and nearest[a] != a))
