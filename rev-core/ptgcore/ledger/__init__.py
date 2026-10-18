"""Учёт атомов: таблица entries (всё об атоме, кроме вектора) и таблица files (путь и время правки обработанных
файлов) в LanceDB папки home (lancedb.connect(home)); сверка источника memcore.plan_sync; id и отпечатки
(memshell.digest). Запись entries — upsert по id (merge_insert); нет папки или таблицы — чтение отдаёт пустое.

node: composite
effects: io
"""
import math
import deal
import lancedb
import memcore
import memshell
import pyarrow

from ptgcore.ledger import atoms, clause, entry_table, file_table, files, order
from ptgcore.types import Batch, Draft, Entry, Source

STATUSES = ("active", "retracted")


@deal.ensure(lambda file, draft, result: result.file == file and result.position == draft.position
             and result.status == "active" and result.retracted_at == 0.0 and len(result.id) == 16)
@deal.has()
def entry(file: str, draft: Draft) -> Entry:
    """ФТ-004. Атом из черновика: id = memshell.digest(f"{file}\\n{position}\\n{at!r}\\n{text}"), отпечаток =
    memshell.digest(text), состояние 'active', не снят.

    D: 1

    >>> entry('n.md', Draft(0, 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, '', 'file'))
    Entry(id='2295c768ade7f62e', file='n.md', position=0, digest='0b00d70ea64ce933', text='LanceDB выбран.', question='', answer='LanceDB выбран.', at=1760000000.0, status='active', retracted_at=0.0, session='', source='file')
    """
    return atoms.entry(file, draft)


@deal.pre(lambda file, stored, drafts, secrets: secrets >= 0 and all(e.file == file for e in stored))
@deal.ensure(lambda file, stored, drafts, secrets, result: result.file == file and result.secrets == secrets
             and set(result.retract) <= {e.id for e in stored if e.status == "active"}
             and all(n in {f.id for f in result.fresh} and o in result.retract for n, o in result.revises)
             and result.kept + len(result.retract) == sum(e.status == "active" for e in stored))
@deal.has()
def plan_batch(file: str, stored: tuple[Entry, ...], drafts: tuple[Draft, ...], secrets: int) -> Batch:
    """ФТ-005. План приёма: атомы черновиков (entry), memcore.plan_sync(Stored(id, file, position, digest, живой) по
    stored, Atom(id, file, position, digest) по атомам) → fresh — атомы из add по порядку, retract, revises, kept =
    len(keep).

    D: 2

    >>> plan_batch('n.md', (), (Draft(0, 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, '', 'file'),), 0)
    Batch(file='n.md', fresh=(Entry(id='2295c768ade7f62e', file='n.md', position=0, digest='0b00d70ea64ce933', text='LanceDB выбран.', question='', answer='LanceDB выбран.', at=1760000000.0, status='active', retracted_at=0.0, session='', source='file'),), retract=(), revises=(), kept=0, secrets=0)
    >>> plan_batch('n.md', (Entry('o1', 'n.md', 0, '0b00d70ea64ce933', 'LanceDB выбран.', '', 'LanceDB выбран.', 1.0, 'active', 0.0, '', 'file'),), (Draft(0, 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, '', 'file'),), 0)
    Batch(file='n.md', fresh=(), retract=(), revises=(), kept=1, secrets=0)
    >>> plan_batch('n.md', (Entry('o1', 'n.md', 0, '0b00d70ea64ce933', 'LanceDB выбран.', '', 'LanceDB выбран.', 1.0, 'active', 0.0, '', 'file'),), (Draft(0, 'LanceDB заменил SQLite.', '', 'LanceDB заменил SQLite.', 1760003600.0, '', 'file'),), 0)
    Batch(file='n.md', fresh=(Entry(id='b9b7a96df021571e', file='n.md', position=0, digest='3e24c9c9b6794eb2', text='LanceDB заменил SQLite.', question='', answer='LanceDB заменил SQLite.', at=1760003600.0, status='active', retracted_at=0.0, session='', source='file'),), retract=('o1',), revises=(('b9b7a96df021571e', 'o1'),), kept=0, secrets=0)
    >>> plan_batch('n.md', (Entry('o1', 'n.md', 0, '0b00d70ea64ce933', 'LanceDB выбран.', '', 'LanceDB выбран.', 1.0, 'active', 0.0, '', 'file'),), (), 0)
    Batch(file='n.md', fresh=(), retract=('o1',), revises=(), kept=0, secrets=0)
    """
    return atoms.plan_batch(file, stored, drafts, secrets)


@deal.pre(lambda home, file: home != "")
@deal.ensure(lambda home, file, result: all(e.file == file for e in result)
             and [(e.position, e.at) for e in result] == sorted((e.position, e.at) for e in result))
@deal.has("io")
def stored(home: str, file: str) -> tuple[Entry, ...]:
    """Атомы источника (и снятые) по месту, затем по времени.

    D: 1

    >>> stored('нет-базы', 'n.md')
    ()
    """
    return order.in_place(entry_table.read_entries(home, clause.equals("file", file)))


@deal.pre(lambda home, entries: home != "" and all(e.status in STATUSES for e in entries))
@deal.ensure(lambda home, entries, result: result >= len({e.id for e in entries}))
@deal.has("io")
def write(home: str, entries: tuple[Entry, ...]) -> int:
    """Записать атомы (upsert по id); итог — атомов в таблице. Пустой entries ничего не создаёт.

    D: 2

    >>> write('led_write', (Entry('a1', 'n.md', 0, '0b00d70ea64ce933', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'),))
    1
    >>> write('нет-базы-записи', ())
    0
    """
    return entry_table.write_entries(home, order.distinct(entries))


@deal.pre(lambda home, ids, at: home != "" and at >= 0.0)
@deal.ensure(lambda home, ids, at, result: 0 <= result <= len(set(ids)))
@deal.has("io")
def retract(home: str, ids: tuple[str, ...], at: float) -> int:
    """Снять атомы: состояние 'retracted', retracted_at = at; итог — сколько живых снято (снятые и неизвестные не
    считаются).

    D: 2

    >>> retract('нет-базы', ('a1',), 5.0)
    0
    """
    if not ids:
        return 0
    gone = order.retire(entry_table.read_entries(home, clause.one_of("id", ids)), at)
    entry_table.write_entries(home, gone)
    return len(gone)


@deal.pre(lambda home, ids: home != "")
@deal.ensure(lambda home, ids, result: len({e.id for e in result}) == len(result)
             and [e.id for e in result] == [i for i in dict.fromkeys(ids) if i in {e.id for e in result}])
@deal.ensure(lambda home, ids, result: ids or result == ())
@deal.has("io")
def entries(home: str, ids: tuple[str, ...]) -> tuple[Entry, ...]:
    """Атомы по ids в их порядке, без повторов, неизвестные пропускаются; ids = () — () без чтения таблицы (14.10:
    прежний смысл «() — все атомы» в recall читал и сортировал все 27 816 атомов на каждый пустой вычисленный список —
    1,1 с из 2 с подсказки). Выборка — запросом по id, не чтением всей таблицы.

    D: 2

    >>> entries('нет-базы', ('a1',))
    ()
    >>> entries('нет-базы', ())
    ()
    """
    if not ids:
        return ()
    return order.arrange(entry_table.read_entries(home, clause.one_of("id", ids)), ids)


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: len(set(result)) == len(result))
@deal.has("io")
def live_ids(home: str) -> tuple[str, ...]:
    """id живых (не снятых) атомов.

    D: 1

    >>> live_ids('нет-базы')
    ()
    """
    return entry_table.read_ids(home, clause.equals("status", "active"))


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: result[0] == result[1] + result[2] and min(result) >= 0)
@deal.has("io")
def counts(home: str) -> tuple[int, int, int]:
    """Атомов всего, живых, снятых.

    D: 1

    >>> counts('нет-базы')
    (0, 0, 0)
    """
    total = entry_table.count_entries(home, "")
    live = entry_table.count_entries(home, clause.equals("status", "active"))
    return total, live, total - live


@deal.pre(lambda home, path, mtime: home != "" and path != "" and mtime >= 0.0)
@deal.ensure(lambda home, path, mtime, result: result >= (1 if mtime > 0.0 else 0))
@deal.has("io")
def mark(home: str, path: str, mtime: float) -> int:
    """Источник обработан при времени правки mtime (upsert по пути); mtime = 0.0 — источника нет, строка удаляется;
    итог — строк в таблице files.

    D: 2

    >>> mark('led_mark', 'n.md', 1760000000.0)
    1
    >>> mark('нет-базы-отметок', 'n.md', 0.0)
    0
    """
    if mtime > 0.0:
        return file_table.write_file(home, path, mtime)
    return file_table.delete_files(home, clause.equals("path", path))


@deal.pre(lambda home, marks: home != "" and all(p != "" and math.isfinite(m) and m > 0.0 for p, m in marks))
@deal.ensure(lambda home, marks, result: result >= len({p for p, _ in marks}))
@deal.has("io")
def mark_all(home: str, marks: tuple[tuple[str, float], ...]) -> int:
    """Отметить источники (путь, время правки) обработанными разом — один upsert по path (перенос базы: 3077 отметок
    по одной — 3077 версий таблицы); у повторного пути берётся последнее время; итог — строк в таблице files.

    D: 2

    >>> mark_all('led_mark_all', (('a.md', 5.0), ('b.md', 6.0), ('a.md', 7.0)))
    2
    >>> mark_all('нет-базы-отметок', ())
    0
    """
    return file_table.write_files(home, marks)


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: len({p for p, _ in result}) == len(result))
@deal.has("io")
def processed(home: str) -> tuple[tuple[str, float], ...]:
    """Обработанные источники и их время правки, по пути.

    D: 1

    >>> processed('нет-базы')
    ()
    """
    return file_table.read_files(home)


@deal.ensure(lambda current, done, result: set(s.path for s in result[0]) <= {s.path for s, _ in current}
             and not set(result[1]) & {s.path for s, _ in current}
             and not any(p.startswith("memory://") for p in result[1]))
@deal.has()
def outdated(current: tuple[tuple[Source, float], ...], done: tuple[tuple[str, float], ...]) -> tuple[tuple[Source, ...], tuple[str, ...]]:
    """Что устарело: источники, чьё время правки не равно записанному (или записи нет), по порядку current; и
    обработанные пути, которых нет среди current (кроме 'memory://…'), по порядку done.

    D: 3

    >>> outdated(((Source('a.md', 'file'), 5.0), (Source('c.md', 'file'), 2.0)), (('a.md', 5.0), ('b.md', 1.0), ('memory://notes', 0.0)))
    ((Source(path='c.md', kind='file'),), ('b.md',))
    >>> outdated((), ())
    ((), ())
    """
    return files.outdated(current, done)


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: result >= 0)
@deal.has("io")
def compact(home: str) -> int:
    """Сжать все таблицы базы (optimize у каждой); итог — сколько таблиц; нет базы — 0.

    D: 2

    >>> compact('нет-базы')
    0
    """
    return file_table.optimize_all(home)
