"""Таблица relations в LanceDB папки home: связи memcore.types.Edge; таблица хранится целиком.

База — папка home. lancedb.connect(home) сама создаёт папку, поэтому каждая функция сначала проверяет os.path.isdir(home), а
таблицу — `TABLE in db` (не table_names(): он отдаёт лишь первые 10 имён). Нет папки или таблицы — чтение отдаёт пустое, и
ничего не создаётся. Строка таблицы — словарь с ключами в порядке src, dst, kind, valid_at, invalid_at, source; src, dst, kind
и source — str, valid_at и invalid_at — float(...). Чтение: `db.open_table(TABLE).search().limit(None).to_list()`. Запись
заменяет всё содержимое: `db.create_table(TABLE, data=rows, mode='overwrite')`; итог — `int(db.open_table(TABLE).count_rows())`.

node: leaf
effects: io
"""
import os

import deal
import lancedb
from memcore.types import Edge

TABLE = 'relations'


@deal.pre(lambda home: home != '')
@deal.ensure(lambda home, result: all(a.valid_at <= b.valid_at for a, b in zip(result, result[1:], strict=False)))
@deal.has('io')
def read_edges(home: str) -> tuple[Edge, ...]:
    """Все связи таблицы relations по возрастанию valid_at (при равном — в порядке хранения). Нет папки или таблицы — (),
    ничего не создаётся. Строка превращается в Edge(str(src), str(dst), str(kind), float(valid_at), float(invalid_at),
    str(source)).

    D: 3

    >>> read_edges('нет-базы')
    ()
    >>> read_edges('нет-базы-связей')
    ()
    """
    if not os.path.isdir(home):
        return ()
    db = lancedb.connect(home)
    try:
        table = db.open_table(TABLE)
    except ValueError:
        return ()
    rows = table.search().limit(None).to_list()
    edges = [
        Edge(
            str(row['src']),
            str(row['dst']),
            str(row['kind']),
            float(row['valid_at']),
            float(row['invalid_at']),
            str(row['source']),
        )
        for row in rows
    ]
    edges.sort(key=lambda edge: edge.valid_at)
    return tuple(edges)


@deal.pre(lambda home, edges: home != '')
@deal.ensure(lambda home, edges, result: result >= 0 and (not edges or result == len(edges)))
@deal.has('io')
def write_edges(home: str, edges: tuple[Edge, ...]) -> int:
    """Заменить содержимое таблицы relations связями edges (строки — в порядке edges); итог — строк в таблице. Нет таблицы —
    она создаётся вместе с папкой. Пустой edges ничего не создаёт и не стирает: итог — строк в таблице (нет папки или
    таблицы — 0).

    D: 4

    >>> write_edges('rel_table', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),))
    1
    >>> write_edges('нет-базы-связей', ())
    0
    """
    if not edges:
        if not os.path.isdir(home):
            return 0
        db = lancedb.connect(home)
        try:
            return int(db.open_table(TABLE).count_rows())
        except ValueError:
            return 0
    rows = [
        {
            'src': edge.src,
            'dst': edge.dst,
            'kind': edge.kind,
            'valid_at': float(edge.valid_at),
            'invalid_at': float(edge.invalid_at),
            'source': edge.source,
        }
        for edge in edges
    ]
    db = lancedb.connect(home)
    db.create_table(TABLE, data=rows, mode='overwrite')
    return int(db.open_table(TABLE).count_rows())
