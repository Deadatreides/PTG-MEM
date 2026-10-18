"""Слияние кусков текста: пересекающиеся, вложенные и смежные куски (конец одного = начало другого) сливаются в один.

node: leaf
"""
import deal


@deal.pre(lambda spans: all(0 <= s < e for s, e in spans))
@deal.ensure(lambda spans, result: all(a[1] < b[0] for a, b in zip(result, result[1:], strict=False)))
@deal.ensure(lambda spans, result: (result == ()) == (spans == ()))
@deal.ensure(lambda spans, result: all(any(r[0] <= s and e <= r[1] for r in result) for s, e in spans))
@deal.ensure(lambda spans, result: all(any(r[0] == s for s, _ in spans) and any(r[1] == e for _, e in spans) and all(any(s <= x < e2 for s, e2 in spans) for _, x in spans if r[0] <= x < r[1]) for r in result))
@deal.has()
def merge_spans(spans: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Куски (начало, конец) без наложений, по возрастанию начала. Вход — в любом порядке; куски с общим местом или
    касающиеся друг друга (конец == начало следующего) становятся одним, вложенный растворяется в объемлющем.

    D: 4

    >>> merge_spans(((6, 46),))
    ((6, 46),)
    >>> merge_spans(((0, 3), (3, 5), (8, 9)))
    ((0, 5), (8, 9))
    >>> merge_spans(((8, 9), (2, 4), (3, 6)))
    ((2, 6), (8, 9))
    >>> merge_spans(((0, 10), (2, 3)))
    ((0, 10),)
    >>> merge_spans(())
    ()
    """
    result: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            if end > result[-1][1]:
                result[-1] = (result[-1][0], end)
        else:
            result.append((start, end))
    return tuple(result)
