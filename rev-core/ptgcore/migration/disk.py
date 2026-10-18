"""Чтение папки data/ базы формата 6 с диска: тонкий слой без решений — файлы JSON и векторы vectors.npy. Решения
принимают layout, atoms и links.

node: leaf
effects: io
"""
import json
import os

import deal
import numpy


@deal.pre(lambda store_dir, name: name != '')
@deal.ensure(lambda store_dir, name, result: isinstance(result, bool))
@deal.has('io')
def has_file(store_dir: str, name: str) -> bool:
    """Есть ли файл store_dir/data/name.

    D: 1

    >>> has_file('tests/fixtures/store_v6', 'edges.json')
    True
    >>> has_file('tests/fixtures/store_v6', 'нет-такого.json')
    False
    >>> has_file('нет-такой-базы', 'meta.json')
    False
    """
    return os.path.isfile(os.path.join(store_dir, 'data', name))


@deal.pre(lambda store_dir, name: name != '')
@deal.ensure(lambda store_dir, name, result: result is None or isinstance(result, (dict, list, str, int, float, bool)))
@deal.raises(FileNotFoundError, ValueError)
@deal.has('io')
def read_json(store_dir: str, name: str) -> object:
    """Разобранный JSON файла store_dir/data/name (utf-8, метка BOM допустима). Нет файла — FileNotFoundError(store_dir);
    не JSON — ValueError.

    D: 2

    >>> read_json('tests/fixtures/store_v6', 'meta.json')
    {'format': 6, 'id_order': ['n1', 'n2'], 'processed_files': {'docs/a.md': 1760000000.0}, 'embedder_model': 'Qwen3-Embedding-4B-Q4_K_M', 'embedder_dim': 4}
    >>> read_json('tests/fixtures/store_v6', 'nodes.json')
    {'n1': {'id': 'n1', 'file': 'docs/a.md', 'order': '0', 'question': '', 'answer': 'Векторы храним в SQLite.', 'text': 'Векторы храним в SQLite.', 'timestamp': '1760000000.0', 'status': 'retracted', 'session_id': '', 'source': 'file'}, 'n2': {'id': 'n2', 'file': 'docs/a.md', 'order': '0', 'question': '', 'answer': 'Векторы храним в LanceDB.', 'text': 'Векторы храним в LanceDB.', 'timestamp': '1760003600.0', 'status': 'active', 'session_id': '', 'source': 'file'}}
    >>> read_json('tests/fixtures/store_v6', 'edges.json')
    [{'from': 'n2', 'to': 'n1', 'type': 'supersedes'}]
    >>> read_json('нет-такой-базы', 'meta.json')
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такой-базы
    """
    try:
        with open(os.path.join(store_dir, 'data', name), encoding='utf-8-sig') as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(store_dir) from e


@deal.ensure(lambda store_dir, result: all(n >= 0 for n in result))
@deal.raises(ValueError)
@deal.has('io')
def vector_shape(store_dir: str) -> tuple[int, ...]:
    """Форма store_dir/data/vectors.npy без чтения самих векторов (numpy.load с mmap_mode='r'); нет файла — (). Не
    массив — ValueError.

    D: 2

    >>> vector_shape('tests/fixtures/store_v6')
    (2, 4)
    >>> vector_shape('нет-такой-базы')
    ()
    """
    path = os.path.join(store_dir, 'data', 'vectors.npy')
    if not os.path.isfile(path):
        return ()
    arr = numpy.load(path, mmap_mode='r')
    if not isinstance(arr, numpy.ndarray):
        raise ValueError(f'не массив: {path}')
    return tuple(int(n) for n in arr.shape)


@deal.pre(lambda store_dir, rows: all(r >= 0 for r in rows))
@deal.ensure(lambda store_dir, rows, result: len(result) == len(rows) and len({len(v) for v in result}) <= 1)
@deal.raises(FileNotFoundError, IndexError)
@deal.has('io')
def read_vectors(store_dir: str, rows: tuple[int, ...]) -> tuple[tuple[float, ...], ...]:
    """Векторы из store_dir/data/vectors.npy по номерам строк rows, в порядке rows, числами float. rows = () — (). Нет
    файла — FileNotFoundError(store_dir); номер за пределами — IndexError.

    D: 3

    >>> read_vectors('tests/fixtures/store_v6', (0, 1))
    ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
    >>> read_vectors('tests/fixtures/store_v6', (1, 0))
    ((0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    >>> read_vectors('tests/fixtures/store_v6', ())
    ()
    >>> read_vectors('нет-такой-базы', (0,))
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такой-базы
    """
    path = os.path.join(store_dir, 'data', 'vectors.npy')
    if not os.path.isfile(path):
        raise FileNotFoundError(store_dir)
    arr = numpy.load(path, mmap_mode='r')
    return tuple(tuple(float(x) for x in arr[r]) for r in rows)
