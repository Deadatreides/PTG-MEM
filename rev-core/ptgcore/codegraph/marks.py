"""Метки инструментов сеанса: строки «[Имя] путь» и «Tools: [Имя] путь» в разметку метадерева «[инструмент Имя: путь]».

node: leaf
"""
import deal

TOOLS = ('Edit', 'Write', 'MultiEdit', 'NotebookEdit', 'Read')
HEAD = 'Tools: '


@deal.pre(lambda line: '\n' not in line)
@deal.ensure(lambda line, result: '\n' not in result and (result == line or result.endswith(']')))
@deal.has()
def mark_line(line: str) -> str:
    """ФТ-013. Одна строка «[Имя] путь» или «Tools: [Имя] путь» (приставка HEAD) с Имя из TOOLS — в «[инструмент Имя: путь]»
    (приставка «Tools: » сохраняется, путь без пробелов по краям); другое имя, пустой путь и любая иная строка — как были.

    D: 5

    >>> mark_line('Tools: [Edit] H:/proj/memcore/filtering.py')
    'Tools: [инструмент Edit: H:/proj/memcore/filtering.py]'
    >>> mark_line('[Read] H:/proj/spec.md')
    '[инструмент Read: H:/proj/spec.md]'
    >>> mark_line('[Bash] ls')
    '[Bash] ls'
    >>> mark_line('[Edit] ')
    '[Edit] '
    >>> mark_line('просто текст')
    'просто текст'
    """
    prefix = HEAD if line.startswith(HEAD) else ''
    rest = line[len(prefix):]
    if not rest.startswith('['):
        return line
    close = rest.find(']')
    if close < 0:
        return line
    name = rest[1:close]
    if name not in TOOLS:
        return line
    path = rest[close + 1:].strip()
    if not path:
        return line
    return f'{prefix}[инструмент {name}: {path}]'


@deal.ensure(lambda text, result: result.count('\n') == text.count('\n'))
@deal.has()
def mark_text(text: str) -> str:
    """ФТ-013. Каждая строка текста (разделитель — перевод строки) через mark_line; число строк не меняется.

    D: 1

    >>> mark_text('Ответ.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py\\n[Read] H:/proj/spec.md\\n[Bash] ls')
    'Ответ.\\n\\nTools: [инструмент Edit: H:/proj/memcore/filtering.py]\\n[инструмент Read: H:/proj/spec.md]\\n[Bash] ls'
    >>> mark_text('просто текст')
    'просто текст'
    >>> mark_text('')
    ''
    """
    return '\n'.join(mark_line(part) for part in text.split('\n'))
