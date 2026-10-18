"""Обход метадерева личным PageRank от находок (как HippoRAG) и сборка подсказки в бюджете.

node: leaf
"""
import math

import deal
import networkx

from ptgfull.types import Atom, CodeNode, Edge, Hit, Piece

HEAD = "<ptg-context>\n"
TAIL = "</ptg-context>"


@deal.pre(lambda edges, seeds, limit, alpha: limit >= 0 and 0 < alpha < 1
          and all(math.isfinite(h.score) and h.score > 0 for h in seeds))
@deal.ensure(lambda edges, seeds, limit, alpha, result: len(result) <= limit and len(set(result)) == len(result)
             and not set(result) & {h.id for h in seeds}
             and set(result) <= {e.src for e in edges} | {e.dst for e in edges}
             and not any(n.startswith(("dir:", "src:")) for n in result))
@deal.has()
def walk(edges: tuple[Edge, ...], seeds: tuple[Hit, ...], limit: int, alpha: float) -> tuple[str, ...]:
    """ФТ-010. Соседи находок: networkx.pagerank на неориентированном графе рёбер (networkx.Graph) с alpha и
    затравкой {id: score} по находкам, которые есть в графе; без затравки, без узлов 'dir:' и 'src:'; по убыванию
    round(оценки, 9), при равенстве — по id; не больше limit. Ни одной находки в графе — ().

    D: 6

    >>> walk((Edge('s', 'a', 'next'), Edge('a', 'b', 'next')), (Hit('s', 1.0, 'h'),), 5, 0.85)
    ('a', 'b')
    >>> walk((Edge('s', 'b', 'next'), Edge('s', 'a', 'next')), (Hit('s', 1.0, 'h'),), 5, 0.85)
    ('a', 'b')
    >>> walk((Edge('dir:', 'file:x.py', 'contains'), Edge('file:x.py', 'x.f', 'contains')), (Hit('x.f', 0.5, 'h'),), 5, 0.85)
    ('file:x.py',)
    >>> walk((Edge('s', 'a', 'next'),), (Hit('z', 1.0, 'h'),), 5, 0.85)
    ()
    """
    graph = networkx.Graph()
    graph.add_edges_from((e.src, e.dst) for e in edges)
    seed = {h.id: h.score for h in seeds if h.id in graph}
    if not seed:
        return ()
    top = max(seed.values())
    personalization = {n: seed.get(n, 0.0) / top for n in graph}
    try:
        scores = networkx.pagerank(
            graph, alpha=alpha, personalization=personalization, max_iter=1000
        )
    except networkx.PowerIterationFailedConvergence:
        return ()
    seeded = {h.id for h in seeds}
    ranked = sorted(
        (n for n in scores if n not in seeded and not n.startswith(("dir:", "src:"))),
        key=lambda n: (-round(scores[n], 9), n),
    )
    return tuple(ranked[:limit])


@deal.pre(lambda pieces, budget: budget >= 0)
@deal.ensure(lambda pieces, budget, result: len(result) <= budget
             and (result == "" or (result.startswith(HEAD) and result.endswith(TAIL))))
@deal.has()
def assemble(pieces: tuple[Piece, ...], budget: int) -> str:
    """ФТ-011. Блок HEAD + строки «[<kind> <ref>] <text>\\n» + TAIL; куски по порядку и целиком, пока блок не длиннее
    budget; не влез первый или кусков нет — ''.

    D: 3

    >>> assemble((Piece('atom', 'a1', 'LanceDB выбран.', 1.0), Piece('code', 'm.f', 'def f(): ...', 0.5)), 200)
    '<ptg-context>\\n[atom a1] LanceDB выбран.\\n[code m.f] def f(): ...\\n</ptg-context>'
    >>> assemble((Piece('atom', 'a1', 'LanceDB выбран.', 1.0), Piece('code', 'm.f', 'def f(): ...', 0.5)), 60)
    '<ptg-context>\\n[atom a1] LanceDB выбран.\\n</ptg-context>'
    >>> assemble((Piece('atom', 'a1', 'LanceDB выбран.', 1.0),), 10)
    ''
    >>> assemble((), 100)
    ''
    """
    body = ""
    for piece in pieces:
        candidate = body + f"[{piece.kind} {piece.ref}] {piece.text}\n"
        if len(HEAD + candidate + TAIL) > budget:
            break
        body = candidate
    return HEAD + body + TAIL if body else ""


@deal.ensure(lambda order, atoms, nodes, result: [p.ref for p in result]
             == [r for r in dict.fromkeys(order) if r in {a.id for a in atoms} or r in {n.qn for n in nodes}]
             and all(p.kind in ("atom", "code") for p in result))
@deal.has()
def pieces(order: tuple[str, ...], atoms: tuple[Atom, ...], nodes: tuple[CodeNode, ...]) -> tuple[Piece, ...]:
    """Куски подсказки по order без повторов: id атома — Piece('atom', id, текст, оценка); полное имя символа —
    Piece('code', qn, '<label> <qn> — <file>:<start>-<end>', оценка); прочее пропускается; при совпадении id атом
    важнее символа; оценка = round(1 / место, 6), место — с 1 среди взятых.

    D: 4

    >>> pieces(('a1', 'm.f', 'x', 'a1'), (Atom('a1', 'n.md', 'LanceDB выбран.', '', 0),), (CodeNode('m.f', 'f', 'Function', 'm.py', 1, 2),))
    (Piece(kind='atom', ref='a1', text='LanceDB выбран.', score=1.0), Piece(kind='code', ref='m.f', text='Function m.f — m.py:1-2', score=0.5))
    >>> pieces((), (), ())
    ()
    """
    by_atom = {a.id: a for a in atoms}
    by_node = {n.qn: n for n in nodes}
    refs = [r for r in dict.fromkeys(order) if r in by_atom or r in by_node]
    out: list[Piece] = []
    for place, ref in enumerate(refs, start=1):
        score = round(1 / place, 6)
        if ref in by_atom:
            out.append(Piece("atom", ref, by_atom[ref].text, score))
        else:
            node = by_node[ref]
            text = f"{node.label} {node.qn} — {node.file}:{node.start}-{node.end}"
            out.append(Piece("code", ref, text, score))
    return tuple(out)
