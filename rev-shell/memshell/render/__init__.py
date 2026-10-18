"""Тексты памяти для модели: блок подсказки к запросу и сводка в начале сеанса — строго в бюджете символов.

node: composite
"""
import deal

from memshell.render.budget import item_limit, pack_block
from memshell.render.lines import recall_item, recent_line
from memshell.types import BRIEF_HEAD, CLOSE, OPEN, RECALL_HEAD, RECENT_HEAD, Entry


@deal.pre(lambda entries, budget_chars: budget_chars >= 0 and all(e.score == e.score for e in entries))
@deal.ensure(lambda entries, budget_chars, result: len(result) <= budget_chars)
@deal.ensure(lambda entries, budget_chars, result: result == "" or (result.startswith(OPEN + "\n" + RECALL_HEAD + "\n") and result.endswith("\n" + CLOSE)))
@deal.ensure(lambda entries, budget_chars, result: result == "" or len(entries) > 0)
@deal.has()
def render_recall(entries: list[Entry], budget_chars: int) -> str:
    """Блок памяти к запросу — строки через "\\n":
        <ptg-memory>
        <RECALL_HEAD>
        1. [<date> · <file> · <score с 2 знаками>] <текст>
           ↳ <пометка>
        </ptg-memory>
    Текст записи — пробелы схлопнуты; если длиннее per_item = max(300, (budget_chars − len(шапки)) // len(entries) − 120),
    где шапка — первые две строки с "\\n" между ними, — обрезан до per_item знаков, справа убраны пробелы и добавлено « …».
    Пометка — пробелы схлопнуты, не длиннее 200 знаков, строка «   ↳ » + пометка, по одной на пометку. Записи
    берутся по порядку и нумеруются с 1, пока весь блок с закрывающей строкой не длиннее budget_chars; первая не
    влезшая и все после неё не входят. Ни одной записи — пустая строка.

    D: 6

    >>> render_recall([Entry("2026-10-11", "plan.md", 0.812, "Решено:   хранилище — LanceDB.", ("заменено (2026-10-12): уже DuckDB",))], 1000)
    "<ptg-memory>\\nAuto-recalled from this project's memory (PTG-MEM). Older notes may be outdated — check the date and the markers before relying on them.\\n1. [2026-10-11 · plan.md · 0.81] Решено: хранилище — LanceDB.\\n   ↳ заменено (2026-10-12): уже DuckDB\\n</ptg-memory>"
    >>> render_recall([Entry("2026-10-11", "a.md", 0.7, "x", ()), Entry("2026-10-11", "b.md", 0.6, "y", ())], 200)
    "<ptg-memory>\\nAuto-recalled from this project's memory (PTG-MEM). Older notes may be outdated — check the date and the markers before relying on them.\\n1. [2026-10-11 · a.md · 0.70] x\\n</ptg-memory>"
    >>> render_recall([Entry("2026-10-11", "a.md", 0.7, "x", ())], 100)
    ''
    >>> render_recall([], 1000)
    ''
    """
    if not entries:
        return ""
    top = [OPEN, RECALL_HEAD]
    limit = item_limit(budget_chars, len("\n".join(top)), len(entries))
    items = [recall_item(number, entry, limit, 200) for number, entry in enumerate(entries, 1)]
    return pack_block(top, items, CLOSE, budget_chars)


@deal.pre(lambda live, recent, max_chars: live >= 0 and max_chars >= 0)
@deal.ensure(lambda live, recent, max_chars, result: len(result) <= max_chars)
@deal.ensure(lambda live, recent, max_chars, result: result == "" or (result.startswith(OPEN + "\n") and result.endswith("\n" + CLOSE)))
@deal.has()
def render_brief(live: int, recent: list[Entry], max_chars: int) -> str:
    """Сводка в начале сеанса — строки через "\\n": <ptg-memory>, BRIEF_HEAD с числом атомов live; если recent не пуст —
    RECENT_HEAD и до 5 строк «- [<date> · <file>] <текст>» (пробелы схлопнуты, не длиннее 160 знаков); </ptg-memory>.
    Не влезает в max_chars — строки recent убираются с конца (без них и RECENT_HEAD); не влезает и без них — "".

    D: 4

    >>> render_brief(120, [Entry("2026-10-11", "memory://notes", 0.0, "Решение: ядро — memcore.", ())], 1000)
    '<ptg-memory>\\nPTG-MEM is active for this project: 120 memory atoms. Relevant memory is attached to each prompt automatically. Tools: ptg_search, ptg_node, ptg_decisions, ptg_remember (store a decision the moment it is made).\\nRecent notes and decisions (last two weeks):\\n- [2026-10-11 · memory://notes] Решение: ядро — memcore.\\n</ptg-memory>'
    >>> render_brief(0, [], 1000)
    '<ptg-memory>\\nPTG-MEM is active for this project: 0 memory atoms. Relevant memory is attached to each prompt automatically. Tools: ptg_search, ptg_node, ptg_decisions, ptg_remember (store a decision the moment it is made).\\n</ptg-memory>'
    >>> render_brief(5, [Entry("2026-10-11", "memory://notes", 0.0, "Решение.", ())], 300)
    '<ptg-memory>\\nPTG-MEM is active for this project: 5 memory atoms. Relevant memory is attached to each prompt automatically. Tools: ptg_search, ptg_node, ptg_decisions, ptg_remember (store a decision the moment it is made).\\n</ptg-memory>'
    >>> render_brief(5, [], 50)
    ''
    """
    top = [OPEN, BRIEF_HEAD.format(live=live)]
    lines = [recent_line(entry, 160) for entry in recent[:5]]
    if lines:
        shown = pack_block([*top, RECENT_HEAD], lines, CLOSE, max_chars)
        if shown:
            return shown
    return pack_block(top, [], CLOSE, max_chars)
