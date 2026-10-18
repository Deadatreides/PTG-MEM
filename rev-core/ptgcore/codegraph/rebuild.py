"""Пересборка метадерева: атомы ptgfull из записей базы и полный набор рёбер.

node: leaf
"""
import deal
import ptgfull.metatree
from ptgfull.types import Atom, CodeNode, Edge

from ptgcore.codegraph.marks import mark_text
from ptgcore.types import Entry


@deal.ensure(lambda entries, result: [a.id for a in result] == [e.id for e in entries if e.status == 'active'])
@deal.has()
def to_atoms(entries: tuple[Entry, ...]) -> tuple[Atom, ...]:
    """ФТ-013. Живые записи (status == 'active') — в атомы ptgfull: Atom(id, file, mark_text(text), session, position)
    по порядку записей; снятые пропускаются.

    D: 2

    >>> to_atoms((Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'снято', '', 'снято', 1760000000.0, 'retracted', 1760003600.0, '', 'file')))
    (Atom(id='a1', file='n.md', text='LanceDB выбран.', session='', order=0),)
    >>> to_atoms((Entry('a3', 's.jsonl', 2, 'd3', 'Tools: [Edit] x.py', '', 'Tools: [Edit] x.py', 1760000000.0, 'active', 0.0, 's1', 'claude'),))
    (Atom(id='a3', file='s.jsonl', text='Tools: [инструмент Edit: x.py]', session='s1', order=2),)
    >>> to_atoms(())
    ()
    """
    return tuple(
        Atom(e.id, e.file, mark_text(e.text), e.session, e.position)
        for e in entries
        if e.status == 'active'
    )


@deal.ensure(lambda atoms, nodes, links, result: result[len(result) - len(links):] == links)
@deal.has()
def assemble(atoms: tuple[Atom, ...], nodes: tuple[CodeNode, ...], links: tuple[Edge, ...]) -> tuple[Edge, ...]:
    """ФТ-013. Все рёбра метадерева подряд: дерево кода (ptgfull.metatree.tree_edges по символам), источники текста
    (source_edges по атомам), швы (stitch по атомам и символам), затем рёбра кода links как были.

    D: 1

    >>> assemble((Atom('a1', 'n.md', 'LanceDB выбран.', '', 0),), (), ())
    (Edge(src='src:n.md', dst='a1', kind='contains'),)
    >>> assemble((Atom('a1', 'n.md', 'правка [инструмент Edit: H:/p/m/x.py]', '', 0),), (CodeNode('m.f', 'f', 'Function', 'm/x.py', 1, 2),), (Edge('m.f', 'm.f', 'CALLS'),))
    (Edge(src='dir:', dst='dir:m', kind='contains'), Edge(src='dir:m', dst='file:m/x.py', kind='contains'), Edge(src='file:m/x.py', dst='m.f', kind='contains'), Edge(src='src:n.md', dst='a1', kind='contains'), Edge(src='a1', dst='file:m/x.py', kind='edits'), Edge(src='m.f', dst='m.f', kind='CALLS'))
    >>> assemble((), (), ())
    ()
    """
    verbs = {'Edit': 'edits', 'Write': 'edits', 'MultiEdit': 'edits', 'Read': 'reads'}

    def chain(n: CodeNode) -> list[Edge]:
        dirs = n.file.split('/')[:-1]
        steps = ['dir:' + '/'.join(dirs[:i]) for i in range(len(dirs) + 1)] + ['file:' + n.file]
        return [Edge(p, q, 'contains') for p, q in zip(steps, steps[1:], strict=False)] + [
            Edge('file:' + n.file, n.qn, 'contains')
        ]

    def seams(a: Atom) -> list[Edge]:
        found = [Edge('src:' + a.file, a.id, 'contains')]
        for part in a.text.replace('\\', '/').split('[инструмент ')[1:]:
            name, _, path = part.split(']')[0].split('\n')[0].partition(':')
            found += [
                Edge(a.id, 'file:' + n.file, verbs[name.strip()])
                for n in nodes
                if name.strip() in verbs and ('/' + path.strip()).endswith('/' + n.file)
            ]
        found += [Edge(a.id, n.qn, 'mentions') for n in nodes if n.qn in a.text]
        return found

    edges = [e for n in nodes for e in chain(n)] + [e for a in atoms for e in seams(a)]
    unique = list({(e.src, e.dst, e.kind): e for e in edges}.values())
    return tuple(unique) + links
