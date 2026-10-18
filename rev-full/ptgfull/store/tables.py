"""Таблицы LanceDB без смысла данных: число строк, запись upsert'ом, замена целиком, чтение строк.

База — папка home, таблица — name. Движок: `db = lancedb.connect(home)`; connect сам создаёт папку, поэтому читающие функции
сначала проверяют `os.path.isdir(home)`, а таблицу — `name in db`: нет папки или таблицы — пусто, и ничего не создаётся.
Число строк — `db.open_table(name).count_rows()` (обернуть в int ради mypy). Запись — `db.create_table(name, data=list(rows))`,
если таблицы нет, иначе
`db.open_table(name).merge_insert(list(keys)).when_matched_update_all().when_not_matched_insert_all().execute(list(rows))`;
итог — count_rows() после записи. Замена — `db.create_table(name, data=list(rows), mode="overwrite")`; пустая замена —
`db.drop_table(name, ignore_missing=True)` и только если папка есть. Чтение — `q = db.open_table(name).search()`, затем
`q = q.select(list(columns))` при непустых columns и `q = q.where(where)` при непустом where, итог — `q.limit(None).to_list()`.

node: leaf
effects: io
"""
import os
from typing import Any

import deal
import lancedb


@deal.pre(lambda home, name: home != "" and name != "")
@deal.ensure(lambda home, name, result: result >= 0)
@deal.has("io")
def count_rows(home: str, name: str) -> int:
    """Строк в таблице; нет папки или таблицы — 0, ничего не создаётся.

    D: 3

    >>> count_rows('нет-базы', 'atoms')
    0
    >>> count_rows('нет-базы', 'vec_h')
    0
    """
    if not os.path.isdir(home):
        return 0
    db = lancedb.connect(home)
    if name not in db.table_names():
        return 0
    return int(db.open_table(name).count_rows())


@deal.pre(lambda home, name, keys, rows: home != "" and name != "" and len(keys) >= 1 and all(k != "" for k in keys)
          and all(all(k in r for k in keys) for r in rows))
@deal.ensure(lambda home, name, keys, rows, result: result >= (1 if rows else 0))
@deal.has("io")
def put_rows(home: str, name: str, keys: tuple[str, ...], rows: tuple[dict[str, Any], ...]) -> int:
    """Записать строки upsert'ом по колонкам keys (нет таблицы — создаётся вместе с папкой); итог — строк в таблице. Ключи
    в rows не повторяются (повторы убирает вызывающий). Пустой rows ничего не создаёт: итог — прежнее число строк (0,
    если таблицы нет).

    D: 3

    >>> put_rows('tb_put', 'items', ('id',), ({'id': 'a', 'n': 1}, {'id': 'b', 'n': 2}))
    2
    >>> put_rows('tb_put_pair', 'pairs', ('x', 'y'), ({'x': 'p', 'y': 'q', 'n': 1}, {'x': 'p', 'y': 'r', 'n': 2}))
    2
    >>> put_rows('нет-базы-put', 'items', ('id',), ())
    0
    """
    if not rows:
        if not os.path.isdir(home):
            return 0
        db = lancedb.connect(home)
        if name not in db.table_names():
            return 0
        return int(db.open_table(name).count_rows())
    db = lancedb.connect(home)
    if name not in db.table_names():
        db.create_table(name, data=list(rows))
    else:
        db.open_table(name).merge_insert(list(keys)).when_matched_update_all().when_not_matched_insert_all().execute(list(rows))
    return int(db.open_table(name).count_rows())


@deal.pre(lambda home, name, rows: home != "" and name != "")
@deal.ensure(lambda home, name, rows, result: result == len(rows))
@deal.has("io")
def replace_rows(home: str, name: str, rows: tuple[dict[str, Any], ...]) -> int:
    """Заменить таблицу целиком строками rows в их порядке; итог — число строк. Пустой rows удаляет таблицу, если она
    есть, и ничего не создаёт (нет папки — тоже).

    D: 3

    >>> replace_rows('tb_replace', 'items', ({'id': 'a'}, {'id': 'b'}))
    2
    >>> replace_rows('tb_replace_one', 'items', ({'id': 'c', 'n': 1},))
    1
    >>> replace_rows('нет-базы-replace', 'items', ())
    0
    """
    if not rows:
        if os.path.isdir(home):
            db = lancedb.connect(home)
            if name in db.table_names():
                db.drop_table(name)
        return 0
    db = lancedb.connect(home)
    db.create_table(name, data=list(rows), mode="overwrite")
    return int(db.open_table(name).count_rows())


@deal.pre(lambda home, name, columns, where: home != "" and name != "")
@deal.ensure(lambda home, name, columns, where, result: not columns or all(set(r) <= set(columns) for r in result))
@deal.has("io")
def table_rows(home: str, name: str, columns: tuple[str, ...], where: str) -> tuple[dict[str, Any], ...]:
    """Строки таблицы (порядок не гарантирован): только колонки columns (пусто — все) и только подходящие под SQL-условие
    where ('' — все). Нет папки или таблицы — (), папка не создаётся.

    D: 5

    >>> table_rows('нет-базы', 'atoms', (), '')
    ()
    >>> table_rows('нет-базы', 'atoms', ('id',), "id IN ('a1')")
    ()
    """
    if not os.path.isdir(home):
        return ()
    db = lancedb.connect(home)
    if name not in db.table_names():
        return ()
    q = db.open_table(name).search()
    if columns:
        q = q.select(list(columns))
    if where:
        q = q.where(where)
    return tuple(q.limit(None).to_list())
