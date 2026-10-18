"""Швы атомов с кодом для подсказки и рёбра метадерева у одного атома.

node: leaf
"""
import deal
from ptgfull.types import CodeNode, Edge

from ptgcore.types import GEdge

SEAMS = ('edits', 'reads', 'mentions')


@deal.ensure(lambda dst, nodes, result: (result == dst) if dst.startswith('file:') else (result == '' or dst in result))
@deal.has()
def target(dst: str, nodes: tuple[CodeNode, ...]) -> str:
    """Цель шва для показа: 'file:<путь>' — как есть; символ — '<label> <qn> — <file>:<start>-<end>' по символу с таким
    полным именем; символа нет в графе кода — ''.

    D: 3

    >>> target('file:memcore/filtering.py', ())
    'file:memcore/filtering.py'
    >>> target('memcore.filtering.admit', (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74),))
    'Function memcore.filtering.admit — memcore/filtering.py:50-74'
    >>> target('memcore.filtering.gone', ())
    ''
    """
    if dst.startswith('file:'):
        return dst
    for node in nodes:
        if node.qn == dst:
            return f'{node.label} {node.qn} — {node.file}:{node.start}-{node.end}'
    return ''


@deal.ensure(lambda ids, edges, nodes, result: {i for i, _ in result} <= set(ids)
             and all(line.split(' ', 1)[0] in SEAMS for _, line in result))
@deal.has()
def seam_lines(ids: tuple[str, ...], edges: tuple[Edge, ...], nodes: tuple[CodeNode, ...]) -> tuple[tuple[str, str], ...]:
    """Швы атомов ids с кодом по порядку ids (без повторов), внутри атома — по порядку edges: (id, '<вид> <цель>'); только
    виды 'edits', 'reads', 'mentions' и только цели, которые знает target.

    D: 5

    >>> seam_lines(('a1', 'a2'), (Edge('a2', 'file:x.py', 'reads'), Edge('a1', 'file:memcore/filtering.py', 'edits'), Edge('a1', 'a2', 'next'), Edge('a1', 'memcore.filtering.admit', 'mentions')), (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74),))
    (('a1', 'edits file:memcore/filtering.py'), ('a1', 'mentions Function memcore.filtering.admit — memcore/filtering.py:50-74'), ('a2', 'reads file:x.py'))
    >>> seam_lines(('a1',), (Edge('a1', 'memcore.filtering.gone', 'mentions'),), ())
    ()
    >>> seam_lines(('a1',), (), ())
    ()
    """
    out: list[tuple[str, str]] = []
    for atom in dict.fromkeys(ids):
        for edge in edges:
            if edge.src != atom or edge.kind not in SEAMS:
                continue
            goal = target(edge.dst, nodes)
            if goal:
                out.append((atom, f'{edge.kind} {goal}'))
    return tuple(out)


@deal.pre(lambda edges, atom, limit: limit >= 0)
@deal.ensure(lambda edges, atom, limit, result: len(result) <= limit and all(atom in (e.src, e.dst) for e in result))
@deal.has()
def touching(edges: tuple[Edge, ...], atom: str, limit: int) -> tuple[GEdge, ...]:
    """Рёбра, у которых атом — начало или конец, по порядку edges, не больше limit; GEdge(src, dst, kind).

    D: 1

    >>> touching((Edge('src:n.md', 'a1', 'contains'), Edge('a1', 'a2', 'next'), Edge('a2', 'a3', 'next'), Edge('a1', 'file:x.py', 'edits')), 'a1', 2)
    (GEdge(src='src:n.md', dst='a1', kind='contains'), GEdge(src='a1', dst='a2', kind='next'))
    >>> touching((Edge('a1', 'a2', 'next'),), 'a1', 0)
    ()
    >>> touching((), 'a1', 5)
    ()
    """
    found = tuple(GEdge(edge.src, edge.dst, edge.kind) for edge in edges if atom in (edge.src, edge.dst))
    return found[:limit]
