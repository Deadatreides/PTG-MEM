"""Сшивка путей из текста с файлами кода (ФТ-008).

node: leaf
"""
import deal

from ptgfull.types import Edge


@deal.ensure(lambda path, files, result: (result != '') == any(f != '' and ('/' + path.replace(chr(92), '/')).endswith('/' + f) for f in files)
             and (result == '' or (result in files and ('/' + path.replace(chr(92), '/')).endswith('/' + result)
                                   and all(len(f) <= len(result) for f in files if f != '' and ('/' + path.replace(chr(92), '/')).endswith('/' + f)))))
@deal.has()
def resolve(path: str, files: tuple[str, ...]) -> str:
    """ФТ-008. Файл кода для пути из текста: после замены обратной косой черты на «/» путь оканчивается на «/<файл>» или
    равен файлу; из подходящих — самый длинный (при равной длине шва нет). Нет подходящего — ''.

    D: 3

    >>> resolve('p\\\\x.py', ('a/x.py', 'x.py'))
    'x.py'
    >>> resolve('/home/u/a/x.py', ('a/x.py', 'x.py'))
    'a/x.py'
    >>> resolve('a/x.py', ('a/x.py',))
    'a/x.py'
    >>> resolve('x.py', ('a/x.py',))
    ''
    >>> resolve('bx.py', ('x.py',))
    ''
    """
    normalized = '/' + path.replace('\\', '/')
    best = ''
    for name in files:
        if name != '' and normalized.endswith('/' + name) and len(name) > len(best):
            best = name
    return best


@deal.ensure(lambda src, paths, files, kind, result: all(e.src == src and e.kind == kind and e.dst.startswith('file:') and e.dst[5:] in files
                                                         for e in result) and len({e.dst for e in result}) == len(result))
@deal.has()
def file_edges(src: str, paths: tuple[str, ...], files: tuple[str, ...], kind: str) -> tuple[Edge, ...]:
    """ФТ-008. Рёбра (src → 'file:<файл>', kind) для путей, сшитых с файлами кода по resolve; без повторов, по порядку путей.

    D: 3

    >>> file_edges('a1', ('/home/u/memcore/filtering.py', 'memcore/filtering.py', 'x.py'), ('memcore/filtering.py', 'a.py'), 'edits')
    (Edge(src='a1', dst='file:memcore/filtering.py', kind='edits'),)
    >>> file_edges('a1', ('b.py', 'a.py'), ('a.py', 'b.py'), 'reads')
    (Edge(src='a1', dst='file:b.py', kind='reads'), Edge(src='a1', dst='file:a.py', kind='reads'))
    >>> file_edges('a1', (), ('a.py',), 'mentions')
    ()
    """
    edges: list[Edge] = []
    seen: set[str] = set()
    for path in paths:
        name = resolve(path, files)
        if name == '':
            continue
        dst = 'file:' + name
        if dst in seen:
            continue
        seen.add(dst)
        edges.append(Edge(src=src, dst=dst, kind=kind))
    return tuple(edges)
