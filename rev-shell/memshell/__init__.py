"""memshell — оболочка памяти PTG-MEM 0.3 между источниками и ядром memcore: файл → Markdown → атомы-абзацы;
находки поиска → окно, текущий сеанс, обвязка; записи → блок памяти в подсказке и сводка сеанса; связи → лента
решений. Хранилище LanceDB, эмбеддер и демон — переходник PTG-MEM (ptg_mem/engine3.py): состояние и многошаговая
работа с хранилищем в карточки не ложатся. Спецификация — spec.md.

node: composite
"""
import math

import deal
from memcore.types import Edge, Hit

from memshell import atoms, decisions, recall, render, sources
from memshell.types import CLOSE, DECISION_KINDS, OPEN, Draft, Entry


@deal.pre(lambda text, limit: limit >= 64)
@deal.ensure(lambda text, limit, result: all(0 < len(p) <= limit and p == p.strip() for p in result))
@deal.ensure(lambda text, limit, result: "".join("".join(result).split()) == "".join(text.split()))
@deal.has()
def split_text(text: str, limit: int) -> list[str]:
    """ФТ-001. Атом — абзац; абзац длиннее limit режет Chonkie. Текст не теряется: все знаки, кроме пробелов, — в
    атомах по порядку.

    D: 1

    >>> split_text("Вопрос: какая база?\\n\\nОтвет: LanceDB.\\n\\nВопрос: как кодируем?\\n\\nОтвет: по одному тексту.", 1200)
    ['Вопрос: какая база?', 'Ответ: LanceDB.', 'Вопрос: как кодируем?', 'Ответ: по одному тексту.']
    """
    return atoms.split_text(text, limit)


@deal.ensure(lambda text, result: len(result) == 16)
@deal.has()
def digest(text: str) -> str:
    """ФТ-002. Отпечаток атома не меняется от правки одних пробелов.

    D: 1

    >>> digest("Ответ: LanceDB.")
    '47e75f7326185bcb'
    >>> digest("Ответ:   LanceDB.\\n")
    '47e75f7326185bcb'
    """
    return atoms.digest(text)


@deal.pre(lambda file, text, limit: limit >= 64)
@deal.ensure(lambda file, text, limit, result: [d.position for d in result] == list(range(len(result))))
@deal.ensure(lambda file, text, limit, result: all(d.file == file for d in result))
@deal.has()
def atomize(file: str, text: str, limit: int) -> list[Draft]:
    """ФТ-001, ФТ-002. Черновики атомов файла: место по порядку, отпечаток.

    D: 1

    >>> atomize("notes.md", "Ответ: LanceDB.", 1200)
    [Draft(file='notes.md', position=0, digest='47e75f7326185bcb', text='Ответ: LanceDB.')]
    """
    return atoms.atomize(file, text, limit)


@deal.pre(lambda hits, min_score, window: window >= 0 and math.isfinite(min_score) and math.isfinite(window)
          and all(math.isfinite(h.score) for h in hits))
@deal.ensure(lambda hits, min_score, window, result: all(h in hits and h.score >= min_score for h in result))
@deal.has()
def in_window(hits: list[Hit], min_score: float, window: float) -> list[Hit]:
    """ФТ-003. В подсказку — только находки не ниже порога и не дальше окна от лучшей.

    D: 1

    >>> in_window([Hit("a", "f.md", "x", 0.9, ""), Hit("b", "f.md", "y", 0.7, "")], 0.5, 0.12)
    [Hit(id='a', file='f.md', text='x', score=0.9, successor='')]
    """
    return recall.in_window(hits, min_score, window)


@deal.pre(lambda file, turn_at, session_files, cut: math.isfinite(turn_at) and math.isfinite(cut))
@deal.ensure(lambda file, turn_at, session_files, cut, result: file in session_files or result)
@deal.has()
def session_ok(file: str, turn_at: float, session_files: tuple[str, ...], cut: float) -> bool:
    """ФТ-004. Из текущего сеанса в подсказку — только то, что ушло из окна модели при сжатии контекста.

    D: 1

    >>> session_ok("cur.jsonl", 100.0, ("cur.jsonl",), 200.0)
    True
    >>> session_ok("cur.jsonl", 300.0, ("cur.jsonl",), 200.0)
    False
    """
    return recall.session_ok(file, turn_at, session_files, cut)


@deal.ensure(lambda head, automated, result: not result or len(automated) > 0)
@deal.has()
def is_noise(head: str, automated: tuple[str, ...]) -> bool:
    """ФТ-005. Сообщения обвязки в подсказку не идут.

    D: 1

    >>> is_noise("<system-reminder>…</system-reminder>", ("<system-reminder",))
    True
    """
    return recall.is_noise(head, automated)


@deal.pre(lambda entries, budget_chars: budget_chars >= 0 and all(math.isfinite(e.score) for e in entries))
@deal.ensure(lambda entries, budget_chars, result: len(result) <= budget_chars)
@deal.ensure(lambda entries, budget_chars, result: result == "" or (result.startswith(OPEN) and result.endswith(CLOSE)))
@deal.has()
def render_recall(entries: list[Entry], budget_chars: int) -> str:
    """ФТ-006. Блок памяти к запросу — не длиннее бюджета, с датой, файлом и пометками; пусто, если показать нечего.

    D: 1

    >>> render_recall([Entry("2026-10-11", "plan.md", 0.9, "Хранилище — LanceDB.", ())], 1000)
    "<ptg-memory>\\nAuto-recalled from this project's memory (PTG-MEM). Older notes may be outdated — check the date and the markers before relying on them.\\n1. [2026-10-11 · plan.md · 0.90] Хранилище — LanceDB.\\n</ptg-memory>"
    >>> render_recall([], 1000)
    ''
    """
    return render.render_recall(entries, budget_chars)


@deal.pre(lambda live, recent, max_chars: live >= 0 and max_chars >= 0)
@deal.ensure(lambda live, recent, max_chars, result: len(result) <= max_chars)
@deal.has()
def render_brief(live: int, recent: list[Entry], max_chars: int) -> str:
    """ФТ-007. Сводка в начале сеанса: сколько атомов и последние решения — не длиннее предела.

    D: 1

    >>> render_brief(3, [], 1000)
    '<ptg-memory>\\nPTG-MEM is active for this project: 3 memory atoms. Relevant memory is attached to each prompt automatically. Tools: ptg_search, ptg_node, ptg_decisions, ptg_remember (store a decision the moment it is made).\\n</ptg-memory>'
    """
    return render.render_brief(live, recent, max_chars)


@deal.raises(FileNotFoundError)
@deal.ensure(lambda path, result: "\r" not in result)
@deal.has("read")
def convert(path: str) -> str:
    """ФТ-008. Документ любого поддержанного формата — в Markdown; неподдержанный — пусто.

    D: 1

    >>> convert("tests/fixtures/note.md")
    '# План\\n\\nПервый абзац.\\n\\nВторой абзац.\\n'
    >>> convert("tests/fixtures/blob.bin")
    ''
    """
    return sources.convert(path)


@deal.pre(lambda edges, limit: limit >= 0 and all(math.isfinite(e.valid_at) for e in edges))
@deal.ensure(lambda edges, limit, result: len(result) <= limit and all(e.kind in DECISION_KINDS for e in result))
@deal.has()
def decision_pairs(edges: list[Edge], limit: int) -> list[Edge]:
    """ФТ-009. Лента решений — новые первыми, закрытые связи тоже видны.

    D: 1

    >>> decision_pairs([Edge("b", "a", "supersedes", 5.0, 0.0, "edit"), Edge("c", "b", "refines", 7.0, 0.0, "agent")], 10)
    [Edge(src='b', dst='a', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='edit')]
    """
    return decisions.decision_pairs(edges, limit)
