"""Поля формата 6 так, как они лежат в JSON: числа бывают строками, отсутствие — 'None' или нет ключа. Здесь они
приводятся к числу и тексту; решений о переносе тут нет.

node: leaf
"""
import math

import deal


@deal.ensure(lambda value, result: math.isfinite(result) and result >= 0.0)
@deal.has()
def number(value: object) -> float:
    """Неотрицательное число поля: int и float — как есть, строка — разобранное число; 'None', пустое, не число,
    отрицательное, nan и бесконечность — 0.0.

    D: 6

    >>> number('1760003600.0')
    1760003600.0
    >>> number(5)
    5.0
    >>> number('None')
    0.0
    >>> number(None)
    0.0
    >>> number('-3')
    0.0
    """
    try:
        result = float(value) if isinstance(value, (int, float, str)) else 0.0
    except (ValueError, OverflowError):
        return 0.0
    return result if math.isfinite(result) and result >= 0.0 else 0.0


@deal.ensure(lambda value, result: result != 'None' and (value is not None or result == ''))
@deal.has()
def text(value: object) -> str:
    """Текст поля: строка — как есть, кроме 'None' (так записано отсутствие) — ''; нет значения (None) — ''; прочее —
    str(value) (если вышло 'None' — '').

    D: 3

    >>> text('file')
    'file'
    >>> text('None')
    ''
    >>> text(None)
    ''
    >>> text(7)
    '7'
    """
    if value is None:
        return ''
    result = value if isinstance(value, str) else str(value)
    return '' if result == 'None' else result
