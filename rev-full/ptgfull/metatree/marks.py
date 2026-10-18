"""Пути в тексте атома: метки инструментов и слова, похожие на путь к файлу.

node: leaf
"""
import deal


@deal.ensure(lambda text, kinds, result: all(p != '' and ']' not in p and p == p.strip() and p in text for p in result)
             and len(result) <= text.count('[инструмент ') and (bool(kinds) or result == ()))
@deal.has()
def tool_paths(text: str, kinds: tuple[str, ...]) -> tuple[str, ...]:
    """ФТ-008. Пути из меток `[инструмент <вид>: путь]`, где вид входит в kinds, по порядку в тексте; путь — всё до «]»
    без пробелов по краям, пустой путь пропускается.

    D: 3

    >>> tool_paths('Claude: [инструмент Edit: a/x.py] и [инструмент Read: b/y.py]', ('Edit', 'Write', 'MultiEdit', 'NotebookEdit'))
    ('a/x.py',)
    >>> tool_paths('Claude: [инструмент Edit: a/x.py] и [инструмент Read: b/y.py]', ('Read',))
    ('b/y.py',)
    >>> tool_paths('[инструмент Bash: ls]', ('Read',))
    ()
    >>> tool_paths('[инструмент Read: my dir/a b.py ]', ('Read',))
    ('my dir/a b.py',)
    >>> tool_paths('[инструмент Read: ]', ('Read',))
    ()
    """
    marker = '[инструмент '
    out: list[str] = []
    pos = 0
    while True:
        start = text.find(marker, pos)
        if start == -1:
            break
        body_start = start + len(marker)
        end = text.find(']', body_start)
        if end == -1:
            break
        kind, _, path = text[body_start:end].partition(':')
        path = path.strip()
        if kind in kinds and path:
            out.append(path)
        pos = end + 1
    return tuple(out)


@deal.ensure(lambda text, result: all('.' in p and (p[-1].isalnum() or p[-1] == '_') and not any(c.isspace() for c in p) and p in text
                                       for p in result))
@deal.has()
def word_paths(text: str) -> tuple[str, ...]:
    """ФТ-008. Слова-пути по порядку в тексте: подряд идущие буквы, цифры и знаки «_», «.», «/», «-», «~», а также обратная
    косая черта, оканчивающиеся на точку с буквами или цифрами (расширение); точка в конце фразы в слово не входит.

    D: 2

    >>> word_paths('смотри memcore/filtering.py и admit(x)')
    ('memcore/filtering.py',)
    >>> word_paths('правка a/b.py, потом ./c.md.')
    ('a/b.py', './c.md')
    >>> word_paths('просто текст. Конец.')
    ()
    >>> word_paths('вызов filtering.admit и `run`')
    ('filtering.admit',)
    """
    punct = '_./-~' + chr(92)

    def is_run_char(c: str) -> bool:
        return c.isalnum() or c in punct

    def is_path(word: str) -> bool:
        return '.' in word and word.rpartition('.')[2].isalnum()

    out: list[str] = []
    i = 0
    while i < len(text):
        if not is_run_char(text[i]):
            i += 1
            continue
        j = i
        while j < len(text) and is_run_char(text[j]):
            j += 1
        word = text[i:j].rstrip('.')
        if is_path(word):
            out.append(word)
        i = j
    return tuple(out)
