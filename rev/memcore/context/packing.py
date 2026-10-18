"""Набор записей по порядку в бюджет: число записей, символы и не больше per_file записей из одного файла.

node: leaf
"""
import deal

from memcore.types import Budget, Item


@deal.pre(lambda entries, budget: budget.chars >= 0 and budget.items >= 0 and budget.per_file >= 0)
@deal.pre(lambda entries, budget: len({i.id for _, i in entries}) == len(entries))
@deal.ensure(lambda entries, budget, result: len(result) <= budget.items)
@deal.ensure(lambda entries, budget, result: sum(len(i.text) + len(i.note) for i in result) <= budget.chars)
@deal.ensure(lambda entries, budget, result: {i.id for i in result} <= {i.id for _, i in entries})
@deal.ensure(
    lambda entries, budget, result: all(
        sum(1 for f, i in entries if f == g and i in result) <= budget.per_file for g, _ in entries
    )
)
@deal.has()
def take_within(entries: list[tuple[str, Item]], budget: Budget) -> list[Item]:
    """Записи из пар (файл, запись) по порядку входа. Запись берётся, если из её файла взято меньше budget.per_file
    записей и её цена len(text) + len(note) помещается в остаток budget.chars; иначе пропускается и набор идёт
    дальше. Набор кончается, когда взято budget.items записей.

    D: 7

    >>> take_within([("f.md", Item("a", "старое", "заменено: b")), ("f.md", Item("b", "новое", ""))], Budget(100, 5, 0.5, 2))
    [Item(id='a', text='старое', note='заменено: b'), Item(id='b', text='новое', note='')]
    >>> take_within([("f.md", Item("a", "длинный текст абзаца", "")), ("f.md", Item("b", "раз", "")), ("f.md", Item("c", "два", "")), ("f.md", Item("e", "еще", "")), ("g.md", Item("d", "три", ""))], Budget(10, 5, 0.0, 2))
    [Item(id='b', text='раз', note=''), Item(id='c', text='два', note=''), Item(id='d', text='три', note='')]
    >>> take_within([("f.md", Item("a", "один", "")), ("g.md", Item("b", "два", ""))], Budget(100, 1, 0.0, 5))
    [Item(id='a', text='один', note='')]
    >>> take_within([("f.md", Item("a", "один", ""))], Budget(100, 5, 0.0, 0))
    []
    >>> take_within([("f.md", Item("a", "один", ""))], Budget(4, 5, 0.0, 2))
    [Item(id='a', text='один', note='')]
    """
    taken: list[Item] = []
    used = 0
    per: dict[str, int] = {}
    for f, item in entries:
        if len(taken) >= budget.items:
            break
        price = len(item.text) + len(item.note)
        if per.get(f, 0) >= budget.per_file or used + price > budget.chars:
            continue
        taken.append(item)
        used += price
        per[f] = per.get(f, 0) + 1
    return taken
