"""Порядок атомов текста по источникам и рёбра 'src:<файл>' → атом, атом → следующий атом.

node: leaf
"""
import deal

from ptgfull.types import Atom, Edge


@deal.ensure(lambda atoms, result: len({a.id for a in result}) == len(result)
             and {a.id for a in result} == {a.id for a in atoms}
             and all(a in atoms for a in result)
             and (not result or sum(a.file != b.file for a, b in zip(result, result[1:], strict=False)) == len({a.file for a in result}) - 1)
             and all(a.order <= b.order for a, b in zip(result, result[1:], strict=False) if a.file == b.file))
@deal.has()
def arrange(atoms: tuple[Atom, ...]) -> tuple[Atom, ...]:
    """ФТ-007. Атомы по источникам: на каждый id — первый встреченный атом; файлы по первому появлению, внутри файла — по
    order (при равных order — как в atoms).

    D: 5

    >>> arrange((Atom('a2', 'n.md', 'два', '', 1), Atom('a1', 'n.md', 'раз', '', 0)))
    (Atom(id='a1', file='n.md', text='раз', session='', order=0), Atom(id='a2', file='n.md', text='два', session='', order=1))
    >>> arrange((Atom('a1', 'n.md', 'раз', '', 0), Atom('b1', 'm.md', 'три', 's1', 0)))
    (Atom(id='a1', file='n.md', text='раз', session='', order=0), Atom(id='b1', file='m.md', text='три', session='s1', order=0))
    >>> arrange((Atom('a1', 'n.md', 'раз', '', 0), Atom('b1', 'm.md', 'три', 's1', 0), Atom('a2', 'n.md', 'два', '', 1)))
    (Atom(id='a1', file='n.md', text='раз', session='', order=0), Atom(id='a2', file='n.md', text='два', session='', order=1), Atom(id='b1', file='m.md', text='три', session='s1', order=0))
    >>> arrange((Atom('a1', 'n.md', 'раз', '', 0), Atom('a1', 'm.md', 'раз', '', 0)))
    (Atom(id='a1', file='n.md', text='раз', session='', order=0),)
    >>> arrange(())
    ()
    """
    seen: set[str] = set()
    groups: dict[str, list[Atom]] = {}
    for atom in atoms:
        if atom.id in seen:
            continue
        seen.add(atom.id)
        groups.setdefault(atom.file, []).append(atom)
    result: list[Atom] = []
    for file_atoms in groups.values():
        result.extend(sorted(file_atoms, key=lambda item: item.order))
    return tuple(result)


@deal.pre(lambda atoms: len({a.id for a in atoms}) == len(atoms))
@deal.ensure(lambda atoms, result: all(e.kind in ('contains', 'next') for e in result) and len(set(result)) == len(result)
             and sum(e.kind == 'contains' for e in result) == len(atoms))
@deal.has()
def chain_edges(atoms: tuple[Atom, ...]) -> tuple[Edge, ...]:
    """ФТ-007. Для атомов, уже разложенных по источникам (arrange): у каждого сначала 'contains' от 'src:<файл>', затем,
    если предыдущий атом из того же файла, — 'next' от него.

    D: 3

    >>> chain_edges((Atom('a1', 'n.md', 'раз', '', 0), Atom('a2', 'n.md', 'два', '', 1)))
    (Edge(src='src:n.md', dst='a1', kind='contains'), Edge(src='src:n.md', dst='a2', kind='contains'), Edge(src='a1', dst='a2', kind='next'))
    >>> chain_edges((Atom('a1', 'n.md', 'раз', '', 0), Atom('b1', 'm.md', 'три', 's1', 0)))
    (Edge(src='src:n.md', dst='a1', kind='contains'), Edge(src='src:m.md', dst='b1', kind='contains'))
    >>> chain_edges(())
    ()
    """
    edges: list[Edge] = []
    for index, atom in enumerate(atoms):
        edges.append(Edge(src=f'src:{atom.file}', dst=atom.id, kind='contains'))
        if index > 0 and atoms[index - 1].file == atom.file:
            edges.append(Edge(src=atoms[index - 1].id, dst=atom.id, kind='next'))
    return tuple(edges)
