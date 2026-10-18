"""Строки таблицы запроса к графу кода — в символы и рёбра: имена без префикса проекта; чужие и неполные строки
пропускаются.

node: leaf
"""
import deal

from ptgfull.types import CodeNode, Edge


@deal.ensure(lambda row, label, prefix, result: len(result) <= 1 and all(
    len(row) == 5 and prefix + n.qn == row[0] and n.qn != '' and n.label == label and n.name == row[1] and n.file == row[2]
    and n.file != '' and not n.file.startswith('<') and n.start >= 0 and n.end >= 0 for n in result))
@deal.has()
def node_of(row: tuple[str, ...], label: str, prefix: str) -> tuple[CodeNode, ...]:
    """ФТ-006. Символ из строки (полное имя, имя, файл, начало, конец): ровно пять значений; полное имя начинается с
    prefix и после него не пусто; файл не пуст и не в «<…>»; начало и конец — десятичные числа (str.isdecimal).
    Иначе — (). Итог — один символ (полное имя без prefix) или ().

    D: 7

    >>> node_of(('P.memcore.filtering.admit', 'admit', 'memcore/filtering.py', '50', '74'), 'Function', 'P.')
    (CodeNode(qn='memcore.filtering.admit', name='admit', label='Function', file='memcore/filtering.py', start=50, end=74),)
    >>> node_of(('builtins.len', 'len', '<python-builtins>', '1', '1'), 'Function', 'P.')
    ()
    >>> node_of(('P.a.B', 'B', 'a.py', 'x', '9'), 'Class', 'P.')
    ()
    >>> node_of(('P.a.f', 'f'), 'Function', 'P.')
    ()
    >>> node_of(('P.', 'f', 'a.py', '1', '2'), 'Function', 'P.')
    ()
    """
    if len(row) != 5:
        return ()
    qn_full = row[0]
    name = row[1]
    path = row[2]
    start = row[3]
    end = row[4]
    if not qn_full.startswith(prefix) or qn_full == prefix:
        return ()
    if path == '' or path.startswith('<'):
        return ()
    if not (start.isdecimal() and end.isdecimal()):
        return ()
    return (CodeNode(qn=qn_full[len(prefix):], name=name, label=label, file=path, start=int(start), end=int(end)),)


@deal.ensure(lambda row, kind, prefix, result: len(result) <= 1 and all(
    len(row) == 2 and prefix + e.src == row[0] and prefix + e.dst == row[1] and e.src != '' and e.dst != '' and e.kind == kind
    for e in result))
@deal.has()
def edge_of(row: tuple[str, ...], kind: str, prefix: str) -> tuple[Edge, ...]:
    """ФТ-006. Ребро из строки (от, к): ровно два значения; оба полных имени начинаются с prefix и после него не пусты.
    Итог — одно ребро (имена без prefix) или ().

    D: 4

    >>> edge_of(('P.a.f', 'P.a.g'), 'CALLS', 'P.')
    (Edge(src='a.f', dst='a.g', kind='CALLS'),)
    >>> edge_of(('P.a.f', 'builtins.len'), 'CALLS', 'P.')
    ()
    >>> edge_of(('P.a.f',), 'USAGE', 'P.')
    ()
    >>> edge_of(('P.', 'P.a.g'), 'CALLS', 'P.')
    ()
    """
    if len(row) != 2:
        return ()
    src_full = row[0]
    dst_full = row[1]
    if not src_full.startswith(prefix) or not dst_full.startswith(prefix):
        return ()
    if src_full == prefix or dst_full == prefix:
        return ()
    return (Edge(src=src_full[len(prefix):], dst=dst_full[len(prefix):], kind=kind),)


@deal.ensure(lambda edges, nodes, result: len(result) <= len(edges) and all(e in edges for e in result)
             and {e.src for e in result} | {e.dst for e in result} <= {n.qn for n in nodes})
@deal.has()
def inner_edges(edges: tuple[Edge, ...], nodes: tuple[CodeNode, ...]) -> tuple[Edge, ...]:
    """ФТ-006. Рёбра, у которых оба конца — полные имена символов из nodes, в прежнем порядке.

    D: 2

    >>> inner_edges((Edge('a.f', 'a.g', 'CALLS'), Edge('a.f', 'b.h', 'CALLS')), (CodeNode('a.f', 'f', 'Function', 'a.py', 1, 2), CodeNode('a.g', 'g', 'Function', 'a.py', 3, 4)))
    (Edge(src='a.f', dst='a.g', kind='CALLS'),)
    >>> inner_edges((Edge('a.f', 'a.f', 'USAGE'),), ())
    ()
    >>> inner_edges((), ())
    ()
    """
    qns = {n.qn for n in nodes}
    return tuple(e for e in edges if e.src in qns and e.dst in qns)
