"""Условия запросов к таблицам LanceDB: SQL-фильтры «колонка = значение» и «колонка IN (значения)». Значения — в
одинарных кавычках, кавычка внутри значения удваивается (так читает строки SQL-разбор LanceDB).

node: leaf
"""
import deal


@deal.pre(lambda column, value: column != '' and "'" not in column)
@deal.ensure(lambda column, value, result: result.startswith(column + " = '") and result.endswith("'")
             and result.count("'") == 2 + 2 * value.count("'"))
@deal.has()
def equals(column: str, value: str) -> str:
    """SQL-условие «<column> = '<значение>'»: значение в одинарных кавычках, кавычка внутри значения удваивается.

    D: 1

    >>> equals('file', 'n.md')
    "file = 'n.md'"
    >>> equals('file', "o'k.md")
    "file = 'o''k.md'"
    """
    return column + " = '" + value.replace("'", "''") + "'"


@deal.pre(lambda column, values: column != '' and "'" not in column)
@deal.ensure(lambda column, values, result: (result == '') == (len(values) == 0)
             and (len(values) == 0 or (result.startswith(column + ' IN (') and result.endswith(')')
                                       and result.count("'") == sum(2 + 2 * v.count("'") for v in values))))
@deal.has()
def one_of(column: str, values: tuple[str, ...]) -> str:
    """SQL-условие «<column> IN ('v1', 'v2')»: значения в одинарных кавычках через запятую, кавычка внутри значения
    удваивается; значений нет — '' (без условия: «все»).

    D: 2

    >>> one_of('id', ('a1', "o'k"))
    "id IN ('a1', 'o''k')"
    >>> one_of('id', ())
    ''
    """
    if not values:
        return ''

    def quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    return column + ' IN (' + ', '.join(quote(v) for v in values) + ')'
