"""Условия запроса к таблицам: SQL-фильтр по списку значений и проверка, есть ли в запросе слова.

node: leaf
"""
import re

import deal


@deal.pre(lambda column, values: column != "" and "'" not in column)
@deal.ensure(lambda column, values, result: (result == "") == (len(values) == 0)
             and (len(values) == 0 or (result.startswith(column + " IN (") and result.endswith(")")
                                       and result.count("'") == sum(2 + 2 * v.count("'") for v in values))))
@deal.has()
def where_in(column: str, values: tuple[str, ...]) -> str:
    """SQL-условие «<column> IN ('v1', 'v2')»: значения в одинарных кавычках, кавычка внутри значения удваивается; значений
    нет — '' (без условия).

    D: 2

    >>> where_in('id', ('a1', "o'k"))
    "id IN ('a1', 'o''k')"
    >>> where_in('id', ('a1',))
    "id IN ('a1')"
    >>> where_in('id', ())
    ''
    """
    if not values:
        return ""
    items = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
    return column + " IN (" + items + ")"


@deal.ensure(lambda query, result: result == (len(re.findall(r"\w+", query)) > 0))
@deal.has()
def has_words(query: str) -> bool:
    """Есть ли в запросе хотя бы одно слово — совпадение re.search(r"\\w", query).

    D: 1

    >>> has_words('кэш и LanceDB')
    True
    >>> has_words(' ?! ')
    False
    >>> has_words('')
    False
    """
    return re.search(r"\w", query) is not None
