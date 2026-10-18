"""Форма текста для показа: схлопывание пробелов и обрезка по длине.

node: leaf
"""
import deal


@deal.ensure(lambda text, result: result.split() == text.split())
@deal.ensure(lambda text, result: result == result.strip() and "  " not in result)
@deal.ensure(lambda text, result: all(not c.isspace() or c == " " for c in result))
@deal.has()
def squash(text: str) -> str:
    """Текст с пробелами, схлопнутыми в один: любая череда пробельных знаков, в том числе переводов строк, —
    один пробел; по краям пробелов нет.

    D: 1

    >>> squash("Решено:   хранилище — LanceDB.")
    'Решено: хранилище — LanceDB.'
    >>> squash("  первая\\n  вторая\\t третья ")
    'первая вторая третья'
    >>> squash(" \\n ")
    ''
    """
    return " ".join(text.split())


@deal.pre(lambda text, limit: limit >= 1)
@deal.ensure(lambda text, limit, result: len(text) > limit or result == text)
@deal.ensure(lambda text, limit, result: len(text) <= limit or (result.endswith(" …") and len(result) <= limit + 2))
@deal.ensure(lambda text, limit, result: len(text) <= limit or text.startswith(result[:-2]))
@deal.has()
def clip(text: str, limit: int) -> str:
    """Текст не длиннее limit — как есть. Длиннее — первые limit знаков, справа без пробелов, и « …».

    D: 2

    >>> clip("коротко", 300)
    'коротко'
    >>> clip("один два три", 7)
    'один дв …'
    >>> clip("один два три", 5)
    'один …'
    """
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


@deal.pre(lambda text, limit: limit >= 0)
@deal.ensure(lambda text, limit, result: len(result) == min(len(text), limit) and text.startswith(result))
@deal.has()
def cut(text: str, limit: int) -> str:
    """Первые limit знаков текста (весь текст, если он короче).

    D: 1

    >>> cut("Решение.", 160)
    'Решение.'
    >>> cut("абвгд", 3)
    'абв'
    >>> cut("абвгд", 0)
    ''
    """
    return text[:limit]
