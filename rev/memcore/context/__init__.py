"""Отбор находок в подсказку модели: порог сходства, без повторов, порядок по сходству, не больше N записей из
одного файла, бюджет записей и символов; у заменённых — пометка преемника.

node: composite
"""
import deal

from memcore.context.dedup import first_per_id
from memcore.context.labeling import label_hits
from memcore.context.packing import take_within
from memcore.context.ranking import rank_hits
from memcore.types import Budget, Hit, Item


@deal.pre(lambda hits, budget: budget.chars >= 0 and budget.items >= 0 and budget.per_file >= 0)
@deal.ensure(lambda hits, budget, result: len(result) <= budget.items)
@deal.ensure(lambda hits, budget, result: sum(len(i.text) + len(i.note) for i in result) <= budget.chars)
@deal.ensure(lambda hits, budget, result: len({i.id for i in result}) == len(result))
@deal.ensure(lambda hits, budget, result: {i.id for i in result} <= {h.id for h in hits if h.score >= budget.min_score})
@deal.has()
def select_context(hits: list[Hit], budget: Budget) -> list[Item]:
    """Записи подсказки из находок поиска.

    Правила по порядку:
    1) находки со сходством ниже budget.min_score отбрасываются;
    2) порядок — по убыванию score, при равном — по id;
    3) повторы id: остаётся первая по этому порядку, остальные отбрасываются, даже если первая потом не войдёт;
    4) пометка: у находки с преемником note = «заменено: <successor>», иначе пустая;
    5) набор по порядку: запись берётся, если из её файла взято меньше budget.per_file записей и её цена
       len(text) + len(note) помещается в остаток budget.chars; иначе пропускается и набор идёт дальше;
    6) набор кончается, когда взято budget.items записей.

    D: 7

    >>> select_context([Hit("a", "f.md", "старое", 0.9, "b"), Hit("b", "f.md", "новое", 0.8, ""), Hit("c", "g.md", "шум", 0.3, "")], Budget(100, 5, 0.5, 2))
    [Item(id='a', text='старое', note='заменено: b'), Item(id='b', text='новое', note='')]
    >>> select_context([Hit("a", "f.md", "длинный текст абзаца", 0.9, ""), Hit("b", "f.md", "раз", 0.8, ""), Hit("c", "f.md", "два", 0.7, ""), Hit("e", "f.md", "еще", 0.65, ""), Hit("d", "g.md", "три", 0.6, "")], Budget(10, 5, 0.0, 2))
    [Item(id='b', text='раз', note=''), Item(id='c', text='два', note=''), Item(id='d', text='три', note='')]
    >>> select_context([Hit("b", "f.md", "два", 0.5, ""), Hit("a", "g.md", "один", 0.5, ""), Hit("a", "g.md", "один", 0.5, "")], Budget(100, 5, 0.0, 3))
    [Item(id='a', text='один', note=''), Item(id='b', text='два', note='')]
    >>> select_context([], Budget(100, 5, 0.5, 2))
    []
    """
    ranked = rank_hits(hits, budget.min_score)
    unique = first_per_id(ranked)
    return take_within(label_hits(unique), budget)
