"""Векторы запроса: какой вектор нести в пространство пути и как перевести его адаптером в другую модель.

node: leaf
"""
import math

import deal
import ptgfull.adapter
from ptgfull.types import Route


@deal.pre(lambda route, vectors: len({s for s, _ in vectors}) == len(vectors))
@deal.ensure(lambda route, vectors, result: all(math.isfinite(x) for x in result)
             and (result == () or (route.source, result) in vectors))
@deal.has()
def source_vector(route: Route, vectors: tuple[tuple[str, tuple[float, ...]], ...]) -> tuple[float, ...]:
    """Вектор модели-источника пути (route.source) среди пар (пространство модели, вектор) из vectors; нет такой пары
    или в векторе есть nan или бесконечность — ().

    D: 4

    >>> source_vector(Route('h', 'direct', 'h'), (('h', (1.0, 0.0)), ('g', (0.0, 1.0))))
    (1.0, 0.0)
    >>> source_vector(Route('b', 'adapter', 'g'), (('h', (1.0, 0.0)), ('g', (0.0, 1.0))))
    (0.0, 1.0)
    >>> source_vector(Route('b', 'adapter', 'x'), (('h', (1.0, 0.0)),))
    ()
    """
    for space, vector in vectors:
        if space == route.source:
            if all(math.isfinite(x) for x in vector):
                return vector
            return ()
    return ()


@deal.ensure(lambda vector, matrix, result: result == () or len(result) == len(matrix[0]))
@deal.has()
def carry(vector: tuple[float, ...], matrix: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    """Вектор запроса в пространстве цели адаптера: ptgfull.adapter.adapt(vector, matrix), когда они подходят друг другу —
    условия те же, что у adapt: len(matrix) == len(vector) > 0, строки матрицы одной длины > 0, числа конечные и по
    модулю не больше 1e6; иначе ().

    D: 6

    >>> carry((1.0, 0.0), ((0.0, 1.0), (-1.0, 0.0)))
    (0.0, 1.0)
    >>> carry((3.0,), ((2.0, 0.0),))
    (1.0, 0.0)
    >>> carry((1.0, 0.0), ())
    ()
    >>> carry((1.0, 0.0, 0.0), ((0.0, 1.0), (-1.0, 0.0)))
    ()
    """
    def fits(x: float) -> bool:
        return math.isfinite(x) and abs(x) <= 1e6

    if not vector or not matrix or len(matrix) != len(vector):
        return ()
    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        return ()
    if not all(fits(x) for x in vector) or not all(fits(x) for row in matrix for x in row):
        return ()
    return ptgfull.adapter.adapt(vector, matrix)
