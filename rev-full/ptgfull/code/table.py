"""Текст ответа codebase-memory-mcp как таблица: строки блока под заголовком и значения одной строки.

node: leaf
"""
import deal


@deal.ensure(lambda text, head, result: all(len(line.splitlines()) <= 1 for line in result))
@deal.ensure(lambda text, head, result: not result or any(line.startswith(head) for line in text.splitlines()))
@deal.has()
def block(text: str, head: str) -> tuple[str, ...]:
    """ФТ-006. Строки блока: сразу после первой строки текста (строки — как в str.splitlines()), начинающейся с head,
    идут строки с отступом в два пробела; блок кончается первой строкой без такого отступа или концом текста; отступ
    снят. Нет строки с head — ().

    D: 5

    >>> block('a: 1\\n  x\\n  y z\\nb: 2\\n  w\\n', 'a:')
    ('x', 'y z')
    >>> block('rows_refs: 1\\n  0 P.\\nrows: 1\\n  @0+a "1"\\nreturned: 1\\n', 'rows:')
    ('@0+a "1"',)
    >>> block('total: 0\\nhint: "no results"\\n', 'rows:')
    ()
    >>> block('rows: 0\\n', 'rows:')
    ()
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(head):
            block_lines: list[str] = []
            for follower in lines[index + 1 :]:
                if not follower.startswith('  '):
                    break
                block_lines.append(follower[2:])
            return tuple(block_lines)
    return ()


@deal.ensure(lambda line, result: len(result) <= len(line))
@deal.has()
def cells(line: str) -> tuple[str, ...]:
    """ФТ-006. Значения строки таблицы: слова, разделённые пробелами; кавычки — как в shlex.split. Строка не
    разбирается (например, незакрытая кавычка) — ().

    D: 2

    >>> cells('@0+filtering.admit memcore/filtering.py "50"')
    ('@0+filtering.admit', 'memcore/filtering.py', '50')
    >>> cells('"x y" z')
    ('x y', 'z')
    >>> cells('say "unclosed')
    ()
    >>> cells('   ')
    ()
    """
    table = (
        ((0, 0, ''),) * 4 + ((2, 0, ''), (3, 0, ''), (5, 0, ''), (1, 0, '{}')),
        ((0, 1, ''),) * 4 + ((2, 0, ''), (3, 0, ''), (5, 0, ''), (1, 0, '{}')),
        ((2, 0, '{}'),) * 4 + ((1, 0, ''), (2, 0, '{}'), (4, 0, ''), (2, 0, '{}')),
        ((3, 0, '{}'),) * 4 + ((3, 0, '{}'), (1, 0, ''), (3, 0, '{}'), (3, 0, '{}')),
        ((2, 0, '\\{}'),) * 4 + ((2, 0, '{}'), (2, 0, '\\{}'), (2, 0, '{}'), (2, 0, '\\{}')),
        ((1, 0, '{}'),) * 8,
    )
    words: list[str] = []
    chars: list[str] = []
    state = 0
    for ch in line:
        state, emit, template = table[state][' \t\r\n"\'\\'.find(ch)]
        if emit:
            words.append(''.join(chars))
            chars.clear()
        chars.append(template.format(ch))
    if state in (2, 3, 4, 5):
        return ()
    if state == 1:
        words.append(''.join(chars))
    return tuple(words)
