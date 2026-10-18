"""Учёт обработанных файлов: что из источников проекта устарело по времени правки.

node: leaf
"""
import deal

from ptgcore.types import Source


@deal.ensure(lambda current, done, result: set(s.path for s in result[0]) <= {s.path for s, _ in current}
             and not set(result[1]) & {s.path for s, _ in current}
             and not any(p.startswith('memory://') for p in result[1]))
@deal.has()
def outdated(current: tuple[tuple[Source, float], ...], done: tuple[tuple[str, float], ...]) -> tuple[tuple[Source, ...], tuple[str, ...]]:
    """Что устарело: источники, чьё время правки не равно записанному (или записи нет), по порядку current; и
    обработанные пути, которых нет среди current (кроме 'memory://…'), по порядку done. Записанное время — dict(done).

    D: 3

    >>> outdated(((Source('a.md', 'file'), 5.0), (Source('c.md', 'file'), 2.0)), (('a.md', 5.0), ('b.md', 1.0), ('memory://notes', 0.0)))
    ((Source(path='c.md', kind='file'),), ('b.md',))
    >>> outdated(((Source('a.md', 'file'), 6.0),), (('a.md', 5.0),))
    ((Source(path='a.md', kind='file'),), ())
    >>> outdated((), ())
    ((), ())
    """
    recorded = dict(done)
    names = {s.path for s, _ in current}
    todo = tuple(s for s, mtime in current if recorded.get(s.path) != mtime)
    gone = tuple(p for p, _ in done if p not in names and not p.startswith('memory://'))
    return todo, gone
