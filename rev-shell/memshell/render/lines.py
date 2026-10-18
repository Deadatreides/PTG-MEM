"""Строки блока памяти: запись подсказки с пометками и строка сводки.

node: leaf
"""
import deal

from memshell.render.text import clip, cut, squash
from memshell.types import Entry


@deal.pre(lambda note, limit: limit >= 0)
@deal.ensure(lambda note, limit, result: result.startswith("   ↳ ") and len(result) <= 5 + limit)
@deal.ensure(lambda note, limit, result: "\n" not in result and " ".join(note.split()).startswith(result[5:]))
@deal.has()
def note_line(note: str, limit: int) -> str:
    """Строка пометки: «   ↳ » (три пробела, стрелка, пробел) и пометка — пробелы схлопнуты, не длиннее limit знаков.

    D: 1

    >>> note_line("заменено (2026-10-12):   уже DuckDB", 200)
    '   ↳ заменено (2026-10-12): уже DuckDB'
    >>> note_line("", 200)
    '   ↳ '
    >>> note_line("абвгд", 3)
    '   ↳ абв'
    """
    return "   ↳ " + cut(squash(note), limit)


@deal.pre(lambda index, entry, text_limit, note_limit: index >= 1 and text_limit >= 1 and note_limit >= 0)
@deal.ensure(
    lambda index, entry, text_limit, note_limit, result: result.startswith(
        f"{index}. [{entry.date} · {entry.file} · {entry.score:.2f}] "
    )
)
@deal.ensure(lambda index, entry, text_limit, note_limit, result: result.count("\n   ↳ ") >= len(entry.notes))
@deal.ensure(
    lambda index, entry, text_limit, note_limit, result: not entry.notes
    or (
        result.rsplit("\n", 1)[-1].startswith("   ↳ ")
        and len(result.rsplit("\n", 1)[-1]) <= 5 + note_limit
    )
)
@deal.ensure(
    lambda index, entry, text_limit, note_limit, result: bool(entry.notes)
    or len(result) <= len(f"{index}. [{entry.date} · {entry.file} · {entry.score:.2f}] ") + text_limit + 2
)
@deal.has()
def recall_item(index: int, entry: Entry, text_limit: int, note_limit: int) -> str:
    """Запись подсказки: строка «<index>. [<date> · <file> · <score с 2 знаками>] <текст>» — текст со схлопнутыми
    пробелами, длиннее text_limit обрезан с « …» — и под ней по строке на каждую пометку (до note_limit знаков).

    D: 2

    >>> recall_item(1, Entry("2026-10-11", "plan.md", 0.812, "Решено:   хранилище — LanceDB.", ("заменено (2026-10-12): уже DuckDB",)), 731, 200)
    '1. [2026-10-11 · plan.md · 0.81] Решено: хранилище — LanceDB.\\n   ↳ заменено (2026-10-12): уже DuckDB'
    >>> recall_item(2, Entry("2026-10-11", "b.md", 0.6, "один два три", ()), 7, 200)
    '2. [2026-10-11 · b.md · 0.60] один дв …'
    >>> recall_item(1, Entry("2026-10-11", "a.md", 0.7, "x", ()), 300, 200)
    '1. [2026-10-11 · a.md · 0.70] x'
    >>> recall_item(3, Entry("2026-10-11", "c.md", 0.5, "z", ("абвгд", "ежзи")), 300, 3)
    '3. [2026-10-11 · c.md · 0.50] z\\n   ↳ абв\\n   ↳ ежз'
    """
    head = f"{index}. [{entry.date} · {entry.file} · {entry.score:.2f}] "
    lines = [head + clip(squash(entry.text), text_limit)]
    lines.extend(note_line(note, note_limit) for note in entry.notes)
    return "\n".join(lines)


@deal.pre(lambda entry, limit: limit >= 0)
@deal.ensure(lambda entry, limit, result: result.startswith(f"- [{entry.date} · {entry.file}] "))
@deal.ensure(lambda entry, limit, result: len(result) <= len(f"- [{entry.date} · {entry.file}] ") + limit)
@deal.ensure(
    lambda entry, limit, result: " ".join(entry.text.split()).startswith(
        result[len(f"- [{entry.date} · {entry.file}] "):]
    )
)
@deal.has()
def recent_line(entry: Entry, limit: int) -> str:
    """Строка сводки: «- [<date> · <file>] <текст>» — текст со схлопнутыми пробелами, не длиннее limit знаков.

    D: 1

    >>> recent_line(Entry("2026-10-11", "memory://notes", 0.0, "Решение: ядро — memcore.", ()), 160)
    '- [2026-10-11 · memory://notes] Решение: ядро — memcore.'
    >>> recent_line(Entry("2026-10-11", "notes.md", 0.0, "Ночью   правил\\nконфиг", ()), 160)
    '- [2026-10-11 · notes.md] Ночью правил конфиг'
    >>> recent_line(Entry("2026-10-11", "a.md", 0.0, "абвгдеж", ()), 5)
    '- [2026-10-11 · a.md] абвгд'
    """
    return f"- [{entry.date} · {entry.file}] " + cut(squash(entry.text), limit)
