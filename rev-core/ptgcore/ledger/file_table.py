"""Таблица files в LanceDB папки home — путь и время правки обработанных источников (ключ — path) — и сжатие всех таблиц
базы.

База — папка home. lancedb.connect(home) сама создаёт папку, поэтому каждая функция сначала проверяет os.path.isdir(home), а
таблицу — `TABLE in db` (не table_names(): он отдаёт лишь первые 10 имён). Нет папки или таблицы — чтение отдаёт пустое,
удаление и сжатие ничего не делают, и ничего не создаётся. Строка таблицы — {'path': str, 'mtime': float}. Запись — upsert:
нет таблицы — `db.create_table(TABLE, data=[row])`, иначе
`db.open_table(TABLE).merge_insert(['path']).when_matched_update_all().when_not_matched_insert_all().execute([row])`; итог —
`int(db.open_table(TABLE).count_rows())`. Запись пачки (write_files) — то же одним вызовом со списком строк
`[{'path': p, 'mtime': m} for p, m in dict(marks).items()]` (у повторного пути — последнее время); пустая пачка — не
обращаясь к базе, итог `len(read_files(home))`. Чтение — `db.open_table(TABLE).search().limit(None).to_list()`. Удаление —
`table.delete(where)`, итог — `int(table.count_rows())`. Сжатие: имена всех таблиц — страницы `db.list_tables(page_token=token)`
(в ответе `.tables` и `.page_token`, пока токен не пуст), у каждой `db.open_table(name).optimize(cleanup_older_than=KEEP_VERSIONS,
delete_unverified=True)`: старые версии старше KEEP_VERSIONS удаляются (по умолчанию lancedb держит их 7 дней, и база
распухала на полную копию векторов при каждом сохранении демона).

node: leaf
effects: io
"""
import os
from datetime import timedelta

import deal
import lancedb

TABLE = 'files'
KEEP_VERSIONS = timedelta(minutes=10)   # запас для чтений, начатых до сохранения


@deal.pre(lambda home: home != '')
@deal.ensure(lambda home, result: [p for p, _ in result] == sorted({p for p, _ in result}))
@deal.has('io')
def read_files(home: str) -> tuple[tuple[str, float], ...]:
    """Обработанные источники — пары (str(path), float(mtime)) таблицы files по возрастанию пути. Нет папки или таблицы —
    (), ничего не создаётся.

    D: 3

    >>> read_files('нет-базы')
    ()
    """
    def has(conn: lancedb.DBConnection) -> bool:
        return TABLE in conn  # type: ignore[operator]

    if not os.path.isdir(home):
        return ()
    db = lancedb.connect(home)
    if not has(db):
        return ()
    rows = db.open_table(TABLE).search().limit(None).to_list()
    return tuple(sorted((str(r['path']), float(r['mtime'])) for r in rows))


@deal.pre(lambda home, path, mtime: home != '' and path != '' and mtime > 0.0)
@deal.ensure(lambda home, path, mtime, result: result >= 1)
@deal.has('io')
def write_file(home: str, path: str, mtime: float) -> int:
    """Отметить источник обработанным при времени правки mtime: строка {'path': path, 'mtime': mtime}, upsert по path (нет
    таблицы — создаётся вместе с папкой); итог — строк в таблице files.

    D: 2

    >>> write_file('ld_files', 'n.md', 1760000000.0)
    1
    """
    def has(conn: lancedb.DBConnection) -> bool:
        return TABLE in conn  # type: ignore[operator]

    db = lancedb.connect(home)
    row = {'path': path, 'mtime': mtime}
    if not has(db):
        db.create_table(TABLE, data=[row])
    else:
        db.open_table(TABLE).merge_insert(['path']).when_matched_update_all().when_not_matched_insert_all().execute([row])
    return int(db.open_table(TABLE).count_rows())


@deal.pre(lambda home, marks: home != '' and all(p != '' and m > 0.0 for p, m in marks))
@deal.ensure(lambda home, marks, result: result >= len({p for p, _ in marks}))
@deal.has('io')
def write_files(home: str, marks: tuple[tuple[str, float], ...]) -> int:
    """Отметить источники (путь, время правки) обработанными одним upsert'ом по path (нет таблицы — создаётся вместе с
    папкой); у повторного пути берётся последнее время; итог — строк в таблице files. Пустой marks ничего не создаёт:
    итог — число строк, как у read_files.

    D: 3

    >>> write_files('ld_files_many', (('a.md', 5.0), ('b.md', 6.0), ('a.md', 7.0)))
    2
    >>> write_files('нет-базы-отметок', ())
    0
    """
    def has(conn: lancedb.DBConnection) -> bool:
        return TABLE in conn  # type: ignore[operator]

    if not marks:
        return len(read_files(home))
    db = lancedb.connect(home)
    rows = [{'path': p, 'mtime': m} for p, m in dict(marks).items()]
    if not has(db):
        db.create_table(TABLE, data=rows)
    else:
        db.open_table(TABLE).merge_insert(['path']).when_matched_update_all().when_not_matched_insert_all().execute(rows)
    return int(db.open_table(TABLE).count_rows())


@deal.pre(lambda home, where: home != '' and where != '')
@deal.ensure(lambda home, where, result: result >= 0)
@deal.has('io')
def delete_files(home: str, where: str) -> int:
    """Убрать из таблицы files строки под SQL-условием where; итог — строк в таблице files после удаления. Нет папки или
    таблицы — 0, ничего не создаётся.

    D: 3

    >>> delete_files('нет-базы', "path = 'n.md'")
    0
    """
    def has(conn: lancedb.DBConnection) -> bool:
        return TABLE in conn  # type: ignore[operator]

    if not os.path.isdir(home):
        return 0
    db = lancedb.connect(home)
    if not has(db):
        return 0
    table = db.open_table(TABLE)
    table.delete(where)
    return int(table.count_rows())


@deal.pre(lambda home: home != '')
@deal.ensure(lambda home, result: result >= 0)
@deal.has('io')
def optimize_all(home: str) -> int:
    """Сжать все таблицы базы (optimize у каждой); итог — сколько таблиц. Нет папки — 0, папка не создаётся.

    D: 4

    >>> optimize_all('нет-базы')
    0
    """
    if not os.path.isdir(home):
        return 0
    db = lancedb.connect(home)
    count = 0
    token: str | None = None
    while True:
        page = db.list_tables(page_token=token)
        for name in page.tables:
            db.open_table(name).optimize(cleanup_older_than=KEEP_VERSIONS, delete_unverified=True)
            count += 1
        token = page.page_token
        if not token:
            break
    return count
