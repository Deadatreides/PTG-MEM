"""Цепочка папок файла кода: 'dir:' → 'dir:a' → 'dir:a/b' → 'file:a/b/x.py'.

node: leaf
"""
import deal

from ptgfull.types import Edge


@deal.ensure(lambda file, result: len(result) == file.count('/') + 1 and all(e.kind == 'contains' for e in result)
             and result[0].src == 'dir:' and result[-1].dst == 'file:' + file
             and all(a.dst == b.src for a, b in zip(result, result[1:], strict=False)))
@deal.has()
def file_chain(file: str) -> tuple[Edge, ...]:
    """ФТ-007. Рёбра 'contains' от корня 'dir:' через каждую папку пути (разделитель «/») к 'file:<файл>'.

    D: 2

    >>> file_chain('m/x.py')
    (Edge(src='dir:', dst='dir:m', kind='contains'), Edge(src='dir:m', dst='file:m/x.py', kind='contains'))
    >>> file_chain('x.py')
    (Edge(src='dir:', dst='file:x.py', kind='contains'),)
    >>> file_chain('a/b/x.py')
    (Edge(src='dir:', dst='dir:a', kind='contains'), Edge(src='dir:a', dst='dir:a/b', kind='contains'), Edge(src='dir:a/b', dst='file:a/b/x.py', kind='contains'))
    """
    parts = file.split('/')
    folders = ['dir:' + '/'.join(parts[:i]) for i in range(1, len(parts))]
    nodes = ['dir:', *folders, 'file:' + file]
    return tuple(Edge(src, dst, 'contains') for src, dst in zip(nodes, nodes[1:], strict=False))
