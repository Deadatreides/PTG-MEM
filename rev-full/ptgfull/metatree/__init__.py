"""Метадерево: дерево «папка → файл → символ» для кода, «источник → атом» для текста и швы «атом ↔ код» по
происхождению — правка, чтение, упоминание файла или символа.

Узлы: атом — его id; символ — полное имя; файл кода — 'file:<путь>'; папка — 'dir:<путь>' (корень — 'dir:');
источник текста — 'src:<файл>'.

node: composite
"""
import deal

from ptgfull.metatree.folders import file_chain
from ptgfull.metatree.marks import tool_paths, word_paths
from ptgfull.metatree.pairs import first_per_pair
from ptgfull.metatree.paths import file_edges
from ptgfull.metatree.sources import arrange, chain_edges
from ptgfull.metatree.symbols import symbol_edges
from ptgfull.types import Atom, CodeNode, Edge

EDITS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
TREE = ("contains", "next")
SEAMS = ("edits", "reads", "mentions")


@deal.ensure(lambda nodes, result: all(e.kind == "contains" for e in result) and len(set(result)) == len(result)
             and {n.qn for n in nodes} <= {e.dst for e in result})
@deal.has()
def tree_edges(nodes: tuple[CodeNode, ...]) -> tuple[Edge, ...]:
    """ФТ-007. Рёбра 'contains': 'dir:' → 'dir:a' → 'dir:a/b' → 'file:a/b/x.py' → символ; без повторов, в порядке
    первого появления по nodes.

    D: 3

    >>> tree_edges((CodeNode('m.f', 'f', 'Function', 'm/x.py', 1, 2),))
    (Edge(src='dir:', dst='dir:m', kind='contains'), Edge(src='dir:m', dst='file:m/x.py', kind='contains'), Edge(src='file:m/x.py', dst='m.f', kind='contains'))
    >>> tree_edges((CodeNode('x.f', 'f', 'Function', 'x.py', 1, 2), CodeNode('x.g', 'g', 'Function', 'x.py', 3, 4)))
    (Edge(src='dir:', dst='file:x.py', kind='contains'), Edge(src='file:x.py', dst='x.f', kind='contains'), Edge(src='file:x.py', dst='x.g', kind='contains'))
    >>> tree_edges(())
    ()
    """
    return first_per_pair(tuple(
        e for n in nodes for e in (*file_chain(n.file), Edge("file:" + n.file, n.qn, "contains"))))


@deal.ensure(lambda atoms, result: all(e.kind in TREE for e in result) and len(set(result)) == len(result)
             and sum(e.kind == "contains" for e in result) == len({a.id for a in atoms}))
@deal.has()
def source_edges(atoms: tuple[Atom, ...]) -> tuple[Edge, ...]:
    """ФТ-007. Для текста: 'src:<файл>' → атом ('contains', по разу на id) и 'next' между соседними по order атомами
    одного файла. Порядок: файлы по первому появлению, внутри — по order, у каждого атома сначала 'contains', затем
    'next' от предыдущего.

    D: 4

    >>> source_edges((Atom('a2', 'n.md', 'два', '', 1), Atom('a1', 'n.md', 'раз', '', 0)))
    (Edge(src='src:n.md', dst='a1', kind='contains'), Edge(src='src:n.md', dst='a2', kind='contains'), Edge(src='a1', dst='a2', kind='next'))
    >>> source_edges((Atom('a1', 'n.md', 'раз', '', 0), Atom('b1', 'm.md', 'три', 's1', 0)))
    (Edge(src='src:n.md', dst='a1', kind='contains'), Edge(src='src:m.md', dst='b1', kind='contains'))
    >>> source_edges(())
    ()
    """
    return chain_edges(arrange(atoms))


@deal.ensure(lambda atoms, nodes, result: all(e.kind in SEAMS for e in result)
             and len({(e.src, e.dst) for e in result}) == len(result)
             and {e.src for e in result} <= {a.id for a in atoms}
             and {e.dst for e in result} <= {n.qn for n in nodes} | {"file:" + n.file for n in nodes})
@deal.has()
def stitch(atoms: tuple[Atom, ...], nodes: tuple[CodeNode, ...]) -> tuple[Edge, ...]:
    """ФТ-008, ФТ-009. Швы атомов с кодом:
    * 'edits' — `[инструмент <вид из EDITS>: путь]`, 'reads' — `[инструмент Read: путь]` → 'file:<путь файла кода>';
    * 'mentions' — путь файла в тексте (слово с точкой и расширением) → 'file:<…>';
    * 'mentions' — символ: полное имя или хвост полного имени с точкой (`filtering.admit`), либо короткое имя в
      обратных кавычках или перед «(», если это имя ровно у одного символа.
    Путь сшивается по ФТ-008: после замены «\\» на «/» оканчивается на «/<путь файла>» или равен ему; из подходящих —
    самый длинный, при равной длине — шва нет. Хвост с точкой сшивается, только если подходит ровно одному символу.
    На пару (атом, цель) — одно ребро, старший вид: edits, затем reads, затем mentions. Порядок — атомы по порядку,
    внутри — edits, reads, mentions файла, mentions символа, каждое по первому появлению в тексте.

    D: 6

    >>> stitch((Atom('a1', 's.jsonl', 'Claude: [инструмент Edit: H:\\\\p\\\\memcore\\\\filtering.py]', 's1', 0),), (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74),))
    (Edge(src='a1', dst='file:memcore/filtering.py', kind='edits'),)
    >>> stitch((Atom('a1', 'n.md', 'вызов filtering.admit и `run`', '', 0),), (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74), CodeNode('a.run', 'run', 'Function', 'a.py', 1, 2), CodeNode('b.run', 'run', 'Function', 'b.py', 1, 2)))
    (Edge(src='a1', dst='memcore.filtering.admit', kind='mentions'),)
    >>> stitch((Atom('a1', 'n.md', 'смотри memcore/filtering.py и admit(x)', '', 0),), (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74),))
    (Edge(src='a1', dst='file:memcore/filtering.py', kind='mentions'), Edge(src='a1', dst='memcore.filtering.admit', kind='mentions'))
    >>> stitch((Atom('a1', 'n.md', 'просто текст', '', 0),), ())
    ()
    """
    files = tuple(n.file for n in nodes)
    return first_per_pair(tuple(
        e
        for a in atoms
        for e in (
            *file_edges(a.id, tool_paths(a.text, EDITS), files, "edits"),
            *file_edges(a.id, tool_paths(a.text, ("Read",)), files, "reads"),
            *file_edges(a.id, word_paths(a.text), files, "mentions"),
            *symbol_edges(a.id, a.text, nodes),
        )
    ))
