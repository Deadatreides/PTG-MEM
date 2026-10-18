"""Поиск LanceDB: полнотекстовый индекс колонки и два поиска — по словам и по косинусу.

Нет папки (`os.path.isdir(home)`) или таблицы (`name in db`, где `db = lancedb.connect(home)`) — пусто, и папка не
создаётся. Индекс — `from lancedb.index import FTS`; `db.open_table(name).create_index(column, config=FTS(), replace=True)`.
Слова — `table.search(query, query_type="fts").select(["id"]).limit(k).to_list()`, оценка — поле `_score`; любая ошибка
движка (нет индекса, в запросе нет слов) — (). Векторы — `table.search(list(vector)).metric("cosine").select(["id"]).limit(k).to_list()`,
расстояние — поле `_distance`; размерность вектора другая — ошибка движка наружу.

node: leaf
effects: io
"""
import math
import os

import deal
import lancedb
from lancedb.index import FTS


@deal.pre(lambda home, name, column: home != "" and name != "" and column != "")
@deal.ensure(lambda home, name, column, result: result >= 0)
@deal.has("io")
def index_words(home: str, name: str, column: str) -> int:
    """Построить заново полнотекстовый индекс колонки column таблицы; итог — строк в таблице. Нет папки или таблицы — 0.

    D: 3

    >>> index_words('нет-базы', 'atoms', 'text')
    0
    >>> index_words('нет-базы', 'notes', 'body')
    0
    """
    if not os.path.isdir(home):
        return 0
    db = lancedb.connect(home)
    if name not in db.table_names():
        return 0
    table = db.open_table(name)
    table.create_index(column, config=FTS(), replace=True)
    return int(table.count_rows())


@deal.pre(lambda home, name, query, k: home != "" and name != "" and k >= 1)
@deal.ensure(lambda home, name, query, k, result: len(result) <= k)
@deal.has("io")
def find_words(home: str, name: str, query: str, k: int) -> tuple[tuple[str, float], ...]:
    """Пары (id, оценка _score) лучших по словам запроса строк таблицы, лучшие первыми, не больше k. Нет папки, таблицы,
    индекса или слов в запросе — ().

    D: 4

    >>> find_words('нет-базы', 'atoms', 'кэш', 3)
    ()
    >>> find_words('нет-базы', 'atoms', '', 1)
    ()
    """
    if not os.path.isdir(home):
        return ()
    if not any(c.isalnum() or c == "_" for c in query):
        return ()
    try:
        db = lancedb.connect(home)
        if name not in db.table_names():
            return ()
        rows = db.open_table(name).search(query, query_type="fts").select(["id"]).limit(k).to_list()
    except Exception:
        return ()
    return tuple((str(row["id"]), float(row["_score"])) for row in rows[:k])


@deal.pre(lambda home, name, vector, k: home != "" and name != "" and k >= 1 and all(math.isfinite(x) for x in vector))
@deal.ensure(lambda home, name, vector, k, result: len(result) <= k)
@deal.has("io")
def find_nearest(home: str, name: str, vector: tuple[float, ...], k: int) -> tuple[tuple[str, float], ...]:
    """Пары (id, косинусное расстояние _distance) ближайших строк таблицы, ближайшие первыми, не больше k. Нет папки или
    таблицы — ().

    D: 3

    >>> find_nearest('нет-базы', 'vec_h', (1.0, 0.0), 3)
    ()
    >>> find_nearest('нет-базы', 'vec_x', (0.5,), 1)
    ()
    """
    if not os.path.isdir(home):
        return ()
    db = lancedb.connect(home)
    if name not in db.table_names():
        return ()
    builder = db.open_table(name).search(list(vector))
    attr = "metric"
    rows = getattr(builder, attr)("cosine").select(["id"]).limit(k).to_list()
    return tuple((str(row["id"]), float(row["_distance"])) for row in rows[:k])
