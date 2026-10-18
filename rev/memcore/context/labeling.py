"""Пометка преемника: из находки — пара (файл, запись подсказки) с note «заменено: <successor>» или пустой.

node: leaf
"""
import deal

from memcore.types import Hit, Item


@deal.ensure(lambda hits, result: len(result) == len(hits))
@deal.ensure(
    lambda hits, result: all(
        f == h.file and i.id == h.id and i.text == h.text for (f, i), h in zip(result, hits, strict=False)
    )
)
@deal.ensure(
    lambda hits, result: all(
        i.note == ('заменено: ' + h.successor if h.successor else '') for (f, i), h in zip(result, hits, strict=False)
    )
)
@deal.has()
def label_hits(hits: list[Hit]) -> list[tuple[str, Item]]:
    """Для каждой находки по порядку — пара (file, Item(id, text, note)); note = «заменено: <successor>», если successor
    не пуст, иначе пустая строка.

    D: 2

    >>> label_hits([Hit("a", "f.md", "старое", 0.9, "b")])
    [('f.md', Item(id='a', text='старое', note='заменено: b'))]
    >>> label_hits([Hit("b", "f.md", "новое", 0.8, ""), Hit("c", "g.md", "шум", 0.3, "")])
    [('f.md', Item(id='b', text='новое', note='')), ('g.md', Item(id='c', text='шум', note=''))]
    >>> label_hits([])
    []
    """
    return [(h.file, Item(h.id, h.text, ('заменено: ' + h.successor) if h.successor else '')) for h in hits]
