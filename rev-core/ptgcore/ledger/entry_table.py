"""Таблица entries в LanceDB папки home: все поля Entry, кроме вектора; ключ — id.

База — папка home. lancedb.connect(home) сама создаёт папку, поэтому каждая функция сначала проверяет os.path.isdir(home), а
таблицу — `TABLE in db` (не table_names(): он отдаёт лишь первые 10 имён). Нет папки или таблицы — чтение отдаёт пустое, счёт
0, и ничего не создаётся. Строка таблицы — словарь с ключами в порядке id, file, position, digest, text, question, answer, at,
status, retracted_at, session, source; position — int, at и retracted_at — float(...), остальное — str. Чтение:
`q = db.open_table(TABLE).search()`, затем `q = q.where(where)` при непустом where и (для read_ids) `q = q.select(['id'])`,
итог — `q.limit(None).to_list()`. Запись — upsert: нет таблицы — `db.create_table(TABLE, data=rows)`, иначе
`db.open_table(TABLE).merge_insert(['id']).when_matched_update_all().when_not_matched_insert_all().execute(rows)`; итог —
`int(db.open_table(TABLE).count_rows())`. Счёт — `int(db.open_table(TABLE).count_rows(where or None))`.

node: leaf
effects: io
"""
import os

import deal
import lancedb

from ptgcore.types import Entry

TABLE = 'entries'


@deal.pre(lambda home, where: home != '')
@deal.ensure(lambda home, where, result: len({e.id for e in result}) == len(result))
@deal.has('io')
def read_entries(home: str, where: str) -> tuple[Entry, ...]:
    """Атомы таблицы entries, подходящие под SQL-условие where ('' — все); порядок не гарантирован. Нет папки или таблицы —
    (), ничего не создаётся. Строка превращается в Entry(str(id), str(file), int(position), str(digest), str(text),
    str(question), str(answer), float(at), str(status), float(retracted_at), str(session), str(source)).

    D: 4

    >>> read_entries('нет-базы', '')
    ()
    >>> read_entries('нет-базы', "id IN ('a1')")
    ()
    """
    def has_table() -> bool:
        try:
            lancedb.connect(home).open_table(TABLE)
        except Exception:
            return False
        return True

    if not os.path.isdir(home):
        return ()
    db = lancedb.connect(home)
    if not has_table():
        return ()
    q = db.open_table(TABLE).search()
    if where:
        q = q.where(where)
    rows = q.limit(None).to_list()
    return tuple(
        Entry(
            str(r['id']), str(r['file']), int(r['position']), str(r['digest']), str(r['text']),
            str(r['question']), str(r['answer']), float(r['at']), str(r['status']),
            float(r['retracted_at']), str(r['session']), str(r['source']),
        )
        for r in rows
    )


@deal.pre(lambda home, where: home != '')
@deal.ensure(lambda home, where, result: list(result) == sorted(set(result)))
@deal.has('io')
def read_ids(home: str, where: str) -> tuple[str, ...]:
    """id атомов таблицы entries под SQL-условием where ('' — все): по возрастанию, без повторов; читается одна колонка
    (q.select(['id'])). Нет папки или таблицы — (), ничего не создаётся.

    D: 4

    >>> read_ids('нет-базы', '')
    ()
    >>> read_ids('нет-базы', "status = 'active'")
    ()
    """
    def has_table() -> bool:
        try:
            lancedb.connect(home).open_table(TABLE)
        except Exception:
            return False
        return True

    if not os.path.isdir(home):
        return ()
    db = lancedb.connect(home)
    if not has_table():
        return ()
    q = db.open_table(TABLE).search()
    if where:
        q = q.where(where)
    q = q.select(['id'])
    rows = q.limit(None).to_list()
    return tuple(sorted({str(r['id']) for r in rows}))


@deal.pre(lambda home, where: home != '')
@deal.ensure(lambda home, where, result: result >= 0)
@deal.has('io')
def count_entries(home: str, where: str) -> int:
    """Атомов в таблице entries под SQL-условием where ('' — всех). Нет папки или таблицы — 0, ничего не создаётся.

    D: 3

    >>> count_entries('нет-базы', '')
    0
    >>> count_entries('нет-базы', "status = 'active'")
    0
    """
    def has_table() -> bool:
        try:
            lancedb.connect(home).open_table(TABLE)
        except Exception:
            return False
        return True

    if not os.path.isdir(home):
        return 0
    db = lancedb.connect(home)
    if not has_table():
        return 0
    return int(db.open_table(TABLE).count_rows(where or None))


@deal.pre(lambda home, entries: home != '' and len({e.id for e in entries}) == len(entries))
@deal.ensure(lambda home, entries, result: result >= len(entries))
@deal.has('io')
def write_entries(home: str, entries: tuple[Entry, ...]) -> int:
    """Записать атомы upsert'ом по id (id в entries не повторяются; нет таблицы — она создаётся вместе с папкой); итог —
    атомов в таблице. Пустой entries ничего не создаёт: итог — count_entries(home, ''). Строка каждого атома — словарь
    id, file, position, digest, text, question, answer, at (float), status, retracted_at (float), session, source.

    D: 3

    >>> write_entries('ld_entries', (Entry('a1', 'n.md', 0, '0b00d70ea64ce933', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'),))
    1
    >>> write_entries('нет-базы-записи', ())
    0
    """
    def has_table() -> bool:
        try:
            lancedb.connect(home).open_table(TABLE)
        except Exception:
            return False
        return True

    if not entries:
        return count_entries(home, '')
    db = lancedb.connect(home)
    rows = [
        {
            'id': e.id,
            'file': e.file,
            'position': e.position,
            'digest': e.digest,
            'text': e.text,
            'question': e.question,
            'answer': e.answer,
            'at': e.at,
            'status': e.status,
            'retracted_at': e.retracted_at,
            'session': e.session,
            'source': e.source,
        }
        for e in entries
    ]
    if has_table():
        db.open_table(TABLE).merge_insert(['id']).when_matched_update_all().when_not_matched_insert_all().execute(rows)
    else:
        db.create_table(TABLE, data=rows)
    return int(db.open_table(TABLE).count_rows())
