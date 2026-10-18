"""Перенос запроса между моделями (Drift-Adapter): матрица по парам векторов и её применение.

node: leaf
"""
import math

import deal
import numpy
import scipy.linalg


@deal.pre(lambda src, dst: len(src) == len(dst) > 0
          and len({len(v) for v in src}) == 1 and len({len(v) for v in dst}) == 1
          and len(src[0]) > 0 and len(dst[0]) > 0
          and all(math.isfinite(x) and abs(x) <= 1e6 for v in (*src, *dst) for x in v))
@deal.ensure(lambda src, dst, result: len(result) == len(src[0]) and all(len(r) == len(dst[0]) for r in result))
@deal.has()
def fit(src: tuple[tuple[float, ...], ...], dst: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """ФТ-005. Матрица W (размерность src × размерность dst), чтобы src @ W ≈ dst: при равных размерностях —
    scipy.linalg.orthogonal_procrustes(src, dst)[0], иначе — numpy.linalg.lstsq(src, dst, rcond=None)[0];
    числа round(x, 6) + 0.0.

    D: 2

    >>> fit(((1.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (-1.0, 0.0)))
    ((0.0, 1.0), (-1.0, 0.0))
    >>> fit(((1.0,), (2.0,)), ((2.0, 0.0), (4.0, 0.0)))
    ((2.0, 0.0),)
    """
    a = numpy.array(src, dtype=float)
    b = numpy.array(dst, dtype=float)
    if a.shape[1] == b.shape[1]:
        w = scipy.linalg.orthogonal_procrustes(a, b)[0]
    else:
        w = numpy.linalg.lstsq(a, b, rcond=None)[0]
    return tuple(tuple(float(round(float(x), 6)) + 0.0 for x in row) for row in w)


@deal.pre(lambda vec, matrix: len(matrix) == len(vec) > 0 and len({len(r) for r in matrix}) == 1 and len(matrix[0]) > 0
          and all(math.isfinite(x) and abs(x) <= 1e6 for x in vec)
          and all(math.isfinite(x) and abs(x) <= 1e6 for r in matrix for x in r))
@deal.ensure(lambda vec, matrix, result: len(result) == len(matrix[0])
             and (sum(x * x for x in result) == 0 or abs(sum(x * x for x in result) - 1) < 1e-4))
@deal.has()
def adapt(vec: tuple[float, ...], matrix: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    """ФТ-005. Вектор запроса в пространство цели: vec @ matrix, нормировка L2 (нулевой — нули), round(x, 6) + 0.0.

    D: 2

    >>> adapt((1.0, 0.0), ((0.0, 1.0), (-1.0, 0.0)))
    (0.0, 1.0)
    >>> adapt((3.0,), ((2.0, 0.0),))
    (1.0, 0.0)
    >>> adapt((0.0, 0.0), ((1.0,), (1.0,)))
    (0.0,)
    """
    y = [sum(v * matrix[i][j] for i, v in enumerate(vec)) for j in range(len(matrix[0]))]
    norm = math.hypot(*y)
    if norm == 0:
        return tuple(0.0 for _ in y)
    return tuple(round(x / norm, 6) + 0.0 for x in y)
