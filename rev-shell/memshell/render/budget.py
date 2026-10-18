"""Бюджет символов блока памяти: доля на запись и посадка записей в блок.

node: leaf
"""
import deal


@deal.pre(lambda budget_chars, head_len, count: budget_chars >= 0 and head_len >= 0 and count >= 1)
@deal.ensure(lambda budget_chars, head_len, count, result: 300 <= result <= max(300, budget_chars))
@deal.ensure(
    lambda budget_chars, head_len, count, result: result == 300 or (result + 120) * count <= budget_chars - head_len
)
@deal.ensure(
    lambda budget_chars, head_len, count, result: result == 300 or (result + 121) * count > budget_chars - head_len
)
@deal.has()
def item_limit(budget_chars: int, head_len: int, count: int) -> int:
    """Знаков на текст одной записи: max(300, (budget_chars - head_len) // count - 120), где head_len — длина шапки
    блока, count — число записей.

    D: 1

    >>> item_limit(1000, 149, 1)
    731
    >>> item_limit(200, 149, 2)
    300
    >>> item_limit(5000, 149, 4)
    1092
    >>> item_limit(0, 0, 3)
    300
    """
    return max(300, (budget_chars - head_len) // count - 120)


@deal.pre(lambda top, items, bottom, max_chars: max_chars >= 0 and len(top) > 0)
@deal.ensure(lambda top, items, bottom, max_chars, result: len(result) <= max_chars)
@deal.ensure(
    lambda top, items, bottom, max_chars, result: result != ""
    or len("\n".join([*top, *items[:1], bottom])) > max_chars
)
@deal.ensure(
    lambda top, items, bottom, max_chars, result: result == ""
    or any(
        result == "\n".join([*top, *items[:n], bottom])
        and (n == len(items) or len("\n".join([*top, *items[: n + 1], bottom])) > max_chars)
        for n in range(min(1, len(items)), len(items) + 1)
    )
)
@deal.has()
def pack_block(top: list[str], items: list[str], bottom: str, max_chars: int) -> str:
    """Блок из строк top, записей items и строки bottom, склеенных переводами строк. Берётся наибольшее число первых
    записей, при котором блок не длиннее max_chars; если записи есть, то хотя бы одна. Не влезает — пустая строка.
    Записей нет — блок из top и bottom.

    D: 3

    >>> pack_block(["<a>", "шапка"], ["один", "два"], "</a>", 100)
    '<a>\\nшапка\\nодин\\nдва\\n</a>'
    >>> pack_block(["<a>", "шапка"], ["один", "два"], "</a>", 19)
    '<a>\\nшапка\\nодин\\n</a>'
    >>> pack_block(["<a>", "шапка"], ["один", "два"], "</a>", 18)
    ''
    >>> pack_block(["<a>", "шапка"], [], "</a>", 14)
    '<a>\\nшапка\\n</a>'
    >>> pack_block(["<a>", "шапка"], [], "</a>", 13)
    ''
    """
    best = ""
    for n in range(min(1, len(items)), len(items) + 1):
        text = "\n".join([*top, *items[:n], bottom])
        if len(text) > max_chars:
            break
        best = text
    return best
