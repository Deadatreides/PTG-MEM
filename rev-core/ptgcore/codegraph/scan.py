"""Граф кода нескольких репозиториев: ptgfull.code.code_graph по каждому, имя папки репозитория перед путём файла.

node: leaf
effects: io
"""
import os

import deal
import ptgfull.code
from ptgfull.types import CodeNode, Edge


@deal.ensure(lambda repo, result: '/' not in result)
@deal.has('io')
def repo_name(repo: str) -> str:
    """ФТ-012. Имя папки репозитория: os.path.basename(os.path.normpath(repo)).

    D: 1

    >>> repo_name('H:/work/PTG-MEM')
    'PTG-MEM'
    >>> repo_name('PTG-MEM/')
    'PTG-MEM'
    >>> repo_name('.')
    '.'
    """
    return os.path.basename(os.path.normpath(repo))


@deal.ensure(lambda name, nodes, result: len(result) == len(nodes)
             and all(r.qn == n.qn and r.file == name + '/' + n.file for r, n in zip(result, nodes, strict=True)))
@deal.has()
def prefixed(name: str, nodes: tuple[CodeNode, ...]) -> tuple[CodeNode, ...]:
    """ФТ-012. Символы с путём файла, где впереди имя папки репозитория и «/»; остальное как было.

    D: 1

    >>> prefixed('PTG-MEM', (CodeNode('ptg_mem.engine.run', 'run', 'Function', 'ptg_mem/engine.py', 10, 20),))
    (CodeNode(qn='ptg_mem.engine.run', name='run', label='Function', file='PTG-MEM/ptg_mem/engine.py', start=10, end=20),)
    >>> prefixed('r', ())
    ()
    """
    return tuple(
        CodeNode(
            n.qn, n.name, n.label, name + '/' + n.file, n.start, n.end,
        )
        for n in nodes
    )


@deal.pre(lambda binary, repos: len(set(repos)) == len(repos))
@deal.ensure(lambda binary, repos, result: (len(repos) > 0 or result == ((), ()))
             and {e.src for e in result[1]} | {e.dst for e in result[1]} <= {n.qn for n in result[0]})
@deal.raises(FileNotFoundError, RuntimeError)
@deal.has('io')
def collect(binary: str, repos: tuple[str, ...]) -> tuple[tuple[CodeNode, ...], tuple[Edge, ...]]:
    """ФТ-012. Граф кода каждого репозитория (ptgfull.code.code_graph(binary, repo)) одним списком: символы с путём файла
    с именем папки репозитория впереди (repo_name, prefixed), рёбра как есть; репозитории по порядку. Нет программы —
    FileNotFoundError(binary); сбой запуска — RuntimeError.

    D: 2

    >>> collect('нет-такой-программы.exe', ())
    ((), ())
    >>> collect('нет-такой-программы.exe', ('.',))
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такой-программы.exe
    """
    nodes: list[CodeNode] = []
    edges: list[Edge] = []
    for repo in repos:
        repo_nodes, repo_edges = ptgfull.code.code_graph(binary, repo)
        nodes.extend(prefixed(repo_name(repo), repo_nodes))
        edges.extend(repo_edges)
    return tuple(nodes), tuple(edges)
