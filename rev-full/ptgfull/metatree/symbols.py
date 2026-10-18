"""Упоминания символов кода в тексте (ФТ-009): полное имя, хвост с точкой, короткое имя в обратных кавычках или перед «(».

node: leaf
"""
import deal

from ptgfull.types import CodeNode, Edge


@deal.ensure(lambda text, result: len(set(result)) == len(result) and all(n != '' and n in text for n in result))
@deal.has()
def names(text: str) -> tuple[str, ...]:
    """ФТ-009. Имена-кандидаты из текста по первому появлению, без повторов: цепочки слов через точку (`filtering.admit`),
    слово в обратных кавычках, слово прямо перед «(».

    D: 5

    >>> names('вызов filtering.admit и `run`, потом admit(x)')
    ('filtering.admit', 'run', 'admit')
    >>> names('memcore.filtering.admit и memcore.filtering.admit')
    ('memcore.filtering.admit',)
    >>> names('просто текст без имён')
    ()
    >>> names('смотри a.py и f(x)')
    ('a.py', 'f')
    """
    def is_word(ch: str) -> bool:
        return ch.isalnum() or ch == '_'

    def chains() -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        size = len(text)
        i = 0
        while i < size:
            if not is_word(text[i]) or (i > 0 and is_word(text[i - 1])):
                i += 1
                continue
            end = i
            while end < size and is_word(text[end]):
                end += 1
            stop = end
            while stop + 1 < size and text[stop] == '.' and is_word(text[stop + 1]):
                stop += 1
                while stop < size and is_word(text[stop]):
                    stop += 1
            if stop > end:
                out.append((i, text[i:stop]))
            i = stop
        return out

    def quoted() -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        size = len(text)
        j = 0
        while j < size:
            if text[j] == '`':
                k = j + 1
                while k < size and is_word(text[k]):
                    k += 1
                if k > j + 1 and k < size and text[k] == '`':
                    out.append((j + 1, text[j + 1:k]))
                    j = k + 1
                    continue
            j += 1
        return out

    def before_paren() -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for p in range(1, len(text)):
            if text[p] == '(' and is_word(text[p - 1]):
                s = p - 1
                while s > 0 and is_word(text[s - 1]):
                    s -= 1
                out.append((s, text[s:p]))
        return out

    found = chains() + quoted() + before_paren()
    result: list[str] = []
    for _, name in sorted(found, key=lambda item: item[0]):
        if name not in result:
            result.append(name)
    return tuple(result)


@deal.pre(lambda name, nodes: name != '')
@deal.ensure(lambda name, nodes, result: (result == '' or result in {n.qn for n in nodes})
             and (result == '' or result == name or result.endswith('.' + name) or any(n.qn == result and n.name == name for n in nodes))
             and (name not in {n.qn for n in nodes} or '.' not in name or result == name))
@deal.has()
def lookup(name: str, nodes: tuple[CodeNode, ...]) -> str:
    """ФТ-009. Полное имя символа для имени из текста: с точкой — равное полному имени, иначе хвост полного имени после
    точки, если он подходит ровно одному символу; без точки — короткое имя, если оно ровно у одного символа (символы
    считаются по полному имени). Нет такого — ''.

    D: 5

    >>> lookup('memcore.filtering.admit', (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74), CodeNode('a.run', 'run', 'Function', 'a.py', 1, 2), CodeNode('b.run', 'run', 'Function', 'b.py', 1, 2)))
    'memcore.filtering.admit'
    >>> lookup('filtering.admit', (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74), CodeNode('a.run', 'run', 'Function', 'a.py', 1, 2), CodeNode('b.run', 'run', 'Function', 'b.py', 1, 2)))
    'memcore.filtering.admit'
    >>> lookup('admit', (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74), CodeNode('a.run', 'run', 'Function', 'a.py', 1, 2), CodeNode('b.run', 'run', 'Function', 'b.py', 1, 2)))
    'memcore.filtering.admit'
    >>> lookup('run', (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74), CodeNode('a.run', 'run', 'Function', 'a.py', 1, 2), CodeNode('b.run', 'run', 'Function', 'b.py', 1, 2)))
    ''
    >>> lookup('m.f', (CodeNode('a.m.f', 'f', 'Function', 'a.py', 1, 2), CodeNode('b.m.f', 'f', 'Function', 'b.py', 1, 2)))
    ''
    """
    if '.' in name and any(n.qn == name for n in nodes):
        return name
    if '.' in name:
        found = {n.qn for n in nodes if n.qn.endswith('.' + name)}
    else:
        found = {n.qn for n in nodes if n.name == name}
    if len(found) == 1:
        return next(iter(found))
    return ''


@deal.ensure(lambda src, text, nodes, result: all(e.src == src and e.kind == 'mentions' and e.dst in {n.qn for n in nodes} for e in result)
             and len({e.dst for e in result}) == len(result))
@deal.has()
def symbol_edges(src: str, text: str, nodes: tuple[CodeNode, ...]) -> tuple[Edge, ...]:
    """ФТ-009. Рёбра (src → полное имя, 'mentions') для символов, упомянутых в тексте (names и lookup); по порядку первого
    появления в тексте, без повторов.

    D: 3

    >>> symbol_edges('a1', 'вызов filtering.admit и `run`', (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74), CodeNode('a.run', 'run', 'Function', 'a.py', 1, 2), CodeNode('b.run', 'run', 'Function', 'b.py', 1, 2)))
    (Edge(src='a1', dst='memcore.filtering.admit', kind='mentions'),)
    >>> symbol_edges('a1', 'смотри memcore/filtering.py и admit(x)', (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74),))
    (Edge(src='a1', dst='memcore.filtering.admit', kind='mentions'),)
    >>> symbol_edges('a1', 'memcore.filtering.admit и admit(x)', (CodeNode('memcore.filtering.admit', 'admit', 'Function', 'memcore/filtering.py', 50, 74),))
    (Edge(src='a1', dst='memcore.filtering.admit', kind='mentions'),)
    >>> symbol_edges('a1', 'просто текст', ())
    ()
    """
    edges: list[Edge] = []
    seen: set[str] = set()
    for name in names(text):
        qn = lookup(name, nodes)
        if qn != '' and qn not in seen:
            seen.add(qn)
            edges.append(Edge(src, qn, 'mentions'))
    return tuple(edges)
