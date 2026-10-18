"""Ссылки `@N+хвост` в таблице ответа: пары (номер, префикс) из блока rows_refs и раскрытие значений строки.

node: leaf
"""
import deal


@deal.ensure(lambda rows, result: len(result) <= len(rows) and all(any(r == (n, p) for r in rows) for n, p in result))
@deal.has()
def ref_map(rows: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, str], ...]:
    """ФТ-006. Пары (номер, префикс) из строк блока rows_refs: только строки ровно из двух значений, в порядке строк.

    D: 2

    >>> ref_map((('0', 'P.memcore.'), ('1', 'P.other.')))
    (('0', 'P.memcore.'), ('1', 'P.other.'))
    >>> ref_map((('0',), ('1', 'P.', 'x'), ()))
    ()
    >>> ref_map(())
    ()
    """
    return tuple((row[0], row[1]) for row in rows if len(row) == 2)


@deal.ensure(lambda cells, refs, result: len(result) == len(cells) and all(
    r == c or ('+' in c and r.endswith(c.split('+', 1)[1])) for c, r in zip(cells, result, strict=False)))
@deal.has()
def expand(cells: tuple[str, ...], refs: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """ФТ-006. Значения строки, где `@N+хвост` заменено на префикс + хвост: N — всё между «@» и первым «+», префикс —
    из первой пары refs с номером N. Значение без «@» в начале или без «+», а также с номером, которого нет в refs,
    остаётся как есть.

    D: 4

    >>> expand(('@0+filtering.admit', '50'), (('0', 'P.memcore.'),))
    ('P.memcore.filtering.admit', '50')
    >>> expand(('@1+x', 'a@b+c', 'plain'), (('0', 'P.'),))
    ('@1+x', 'a@b+c', 'plain')
    >>> expand(('@0+a',), (('0', 'P.'), ('0', 'Q.')))
    ('P.a',)
    >>> expand((), ())
    ()
    """
    def one(cell: str) -> str:
        if not cell.startswith('@') or '+' not in cell:
            return cell
        number, tail = cell[1:].split('+', 1)
        for n, p in refs:
            if n == number:
                return p + tail
        return cell

    return tuple(one(cell) for cell in cells)
