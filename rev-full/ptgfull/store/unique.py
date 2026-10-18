"""Единственность по ключу: повторы ключа в пачке записи и в прочитанных строках сводятся к одному.

node: leaf
"""
import deal

from ptgfull.types import Atom, CodeNode


@deal.ensure(lambda atoms, result: [a.id for a in result] == list(dict.fromkeys(a.id for a in atoms))
             and all(a in atoms for a in result)
             and all(a == [x for x in atoms if x.id == a.id][-1] for a in result))
@deal.has()
def distinct_atoms(atoms: tuple[Atom, ...]) -> tuple[Atom, ...]:
    """Один атом на id: из повторов — последний; порядок — по первому появлению id.

    D: 2

    >>> distinct_atoms((Atom('a1', 'n.md', 'раз', '', 0), Atom('a1', 'n.md', 'два', '', 1)))
    (Atom(id='a1', file='n.md', text='два', session='', order=1),)
    >>> distinct_atoms((Atom('b', 'n.md', 'x', '', 0), Atom('a', 'n.md', 'y', '', 1), Atom('b', 'n.md', 'z', '', 2)))
    (Atom(id='b', file='n.md', text='z', session='', order=2), Atom(id='a', file='n.md', text='y', session='', order=1))
    >>> distinct_atoms(())
    ()
    """
    last: dict[str, Atom] = {}
    for atom in atoms:
        last[atom.id] = atom
    return tuple(last.values())


@deal.ensure(lambda pairs, result: [i for i, _ in result] == list(dict.fromkeys(i for i, _ in pairs))
             and all(v == [w for j, w in pairs if j == i][-1] for i, v in result))
@deal.has()
def distinct_vectors(pairs: tuple[tuple[str, tuple[float, ...]], ...]) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Одна пара (id, вектор) на id: из повторов — последняя; порядок — по первому появлению id.

    D: 1

    >>> distinct_vectors((('a1', (1.0, 0.0)), ('a2', (0.0, 1.0)), ('a1', (0.5, 0.5))))
    (('a1', (0.5, 0.5)), ('a2', (0.0, 1.0)))
    >>> distinct_vectors((('a1', (1.0, 0.0)),))
    (('a1', (1.0, 0.0)),)
    >>> distinct_vectors(())
    ()
    """
    last: dict[str, tuple[float, ...]] = {}
    for key, vector in pairs:
        last[key] = vector
    return tuple(last.items())


@deal.ensure(lambda nodes, result: [n.qn for n in result] == sorted({n.qn for n in nodes})
             and all(n == min((m for m in nodes if m.qn == n.qn), key=lambda m: (m.file, m.start, m.end, m.name, m.label))
                     for n in result))
@deal.has()
def distinct_nodes(nodes: tuple[CodeNode, ...]) -> tuple[CodeNode, ...]:
    """Символы по qn без повторов, по возрастанию qn: из повторов — наименьший по (file, start, end, name, label).

    D: 2

    >>> distinct_nodes((CodeNode('b.f', 'f', 'Function', 'b.py', 1, 2), CodeNode('a.f', 'f', 'Function', 'a.py', 3, 4)))
    (CodeNode(qn='a.f', name='f', label='Function', file='a.py', start=3, end=4), CodeNode(qn='b.f', name='f', label='Function', file='b.py', start=1, end=2))
    >>> distinct_nodes((CodeNode('a.f', 'f', 'Function', 'z.py', 1, 2), CodeNode('a.f', 'f', 'Function', 'a.py', 5, 6)))
    (CodeNode(qn='a.f', name='f', label='Function', file='a.py', start=5, end=6),)
    >>> distinct_nodes(())
    ()
    """
    def order_key(node: CodeNode) -> tuple[str, int, int, str, str]:
        return (node.file, node.start, node.end, node.name, node.label)

    best: dict[str, CodeNode] = {}
    for node in nodes:
        current = best.get(node.qn)
        if current is None or order_key(node) < order_key(current):
            best[node.qn] = node
    return tuple(best[qn] for qn in sorted(best))
