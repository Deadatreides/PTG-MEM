"""Раскладка базы формата 6: что сказано в meta.json и подходят ли векторы. Решения чистые; чтение файлов — в
migration.disk.

node: leaf
"""
import deal

from ptgcore.migration import scalars


@deal.ensure(lambda meta, result: all(p != '' and t > 0.0 for p, t in result[1]))
@deal.raises(ValueError)
@deal.has()
def parse_meta(meta: object) -> tuple[tuple[str, ...], tuple[tuple[str, float], ...]]:
    """meta.json формата 6 (meta — разобранный JSON): (id_order — id атомов текстом по порядку строк vectors.npy, без
    пропусков; processed_files — пары (путь, время правки) по порядку записи, без пустых путей и неположительных
    времён). Нет поля id_order или processed_files — пусто. Не объект JSON или format (число или строка с числом) не
    6 — ValueError('формат <значение поля format>, нужен 6').

    D: 6

    >>> parse_meta({'format': 6, 'id_order': ['n1', 'n2'], 'processed_files': {'docs/a.md': 1760000000.0}, 'embedder_model': 'Qwen3-Embedding-4B-Q4_K_M', 'embedder_dim': 4})
    (('n1', 'n2'), (('docs/a.md', 1760000000.0),))
    >>> parse_meta({'format': 6, 'id_order': ['a'], 'processed_files': {'b.md': 'None', '': 5.0, 'c.md': '7'}})
    (('a',), (('c.md', 7.0),))
    >>> parse_meta({'format': 6})
    ((), ())
    >>> parse_meta({'format': 5, 'id_order': []})
    Traceback (most recent call last):
    ...
    ValueError: формат 5, нужен 6
    >>> parse_meta('мусор')
    Traceback (most recent call last):
    ...
    ValueError: формат None, нужен 6
    """
    def is_six(value: object) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return value == 6
        if isinstance(value, str):
            try:
                return float(value) == 6.0
            except ValueError:
                return False
        return False

    if not isinstance(meta, dict):
        raise ValueError('формат None, нужен 6')
    fmt = meta.get('format')
    if not is_six(fmt):
        raise ValueError(f'формат {fmt}, нужен 6')
    ids = meta.get('id_order')
    done = meta.get('processed_files')
    order = tuple(scalars.text(x) for x in ids) if isinstance(ids, list) else ()
    files = tuple((scalars.text(p), scalars.number(t)) for p, t in done.items()) if isinstance(done, dict) else ()
    return order, tuple((p, t) for p, t in files if p != '' and t > 0.0)


@deal.pre(lambda shape, count, dim: count >= 0 and dim >= 1)
@deal.ensure(lambda shape, count, dim, result: result == count)
@deal.raises(ValueError)
@deal.has()
def fit(shape: tuple[int, ...], count: int, dim: int) -> int:
    """Подходят ли векторы: shape — форма vectors.npy (() — файла нет), count — сколько строк нужно (по id_order), dim —
    размерность пространства. count = 0 — подходит любая форма. Иначе строк rows = shape[0] при двух измерениях, иначе
    0: rows != count — ValueError('векторов <rows>, нужно <count>'); затем shape[1] != dim —
    ValueError('размерность векторов <shape[1]>, нужна <dim>'). Итог — count.

    D: 4

    >>> fit((2, 4), 2, 4)
    2
    >>> fit((), 0, 4)
    0
    >>> fit((2, 3), 2, 4)
    Traceback (most recent call last):
    ...
    ValueError: размерность векторов 3, нужна 4
    >>> fit((1, 4), 2, 4)
    Traceback (most recent call last):
    ...
    ValueError: векторов 1, нужно 2
    >>> fit((), 2, 4)
    Traceback (most recent call last):
    ...
    ValueError: векторов 0, нужно 2
    """
    if count == 0:
        return count
    rows = shape[0] if len(shape) == 2 else 0
    if rows != count:
        raise ValueError(f'векторов {rows}, нужно {count}')
    if shape[1] != dim:
        raise ValueError(f'размерность векторов {shape[1]}, нужна {dim}')
    return count
