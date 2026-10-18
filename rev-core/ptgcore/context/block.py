"""Блок <ptg-memory> подсказки из карточек и их пометок в бюджете символов; сколько карточек в него вошло.

Готовое: memshell.render_recall (memshell.types.Entry(date, file, score, text, notes)).

node: leaf
"""
import math

import deal
import memshell
from memshell.types import Entry as Line

from ptgcore.types import Card, Link


@deal.pre(lambda cards, notes, budget: budget >= 0 and len(cards) == len(notes)
          and all(c.score is None or math.isfinite(c.score) for c in cards))
@deal.ensure(lambda cards, notes, budget, result: len(result) <= budget)
@deal.ensure(lambda cards, notes, budget, result: result == '' or (result.startswith('<ptg-memory>') and result.endswith('</ptg-memory>')))
@deal.ensure(lambda cards, notes, budget, result: len(cards) > 0 or result == '')
@deal.has()
def render_block(cards: tuple[Card, ...], notes: tuple[tuple[str, ...], ...], budget: int) -> str:
    """Блок подсказки memshell.render_recall([memshell.types.Entry(date, rel_file, оценка или 0.0, text, пометки) по
    каждой карточке и её пометкам], budget); карточек нет или в бюджет не вошла ни одна — ''.
    Оценки карточек — конечные или None (таково условие memshell.render_recall).

    D: 2

    >>> render_block((Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), '', ''),), ((),), 3600)
    "<ptg-memory>\\nAuto-recalled from this project's memory (PTG-MEM). Older notes may be outdated — check the date and the markers before relying on them.\\n1. [2025-10-09 · n.md · 0.90] LanceDB выбран.\\n</ptg-memory>"
    >>> render_block((Card('a1', 0.9, 'superseded', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), 'a2', ''),), (('replaced later (2025-10-09): LanceDB заменил SQLite.', 'code: edits file:PTG-MEM/ptg_mem/engine.py'),), 3600)
    "<ptg-memory>\\nAuto-recalled from this project's memory (PTG-MEM). Older notes may be outdated — check the date and the markers before relying on them.\\n1. [2025-10-09 · n.md · 0.90] LanceDB выбран.\\n   ↳ replaced later (2025-10-09): LanceDB заменил SQLite.\\n   ↳ code: edits file:PTG-MEM/ptg_mem/engine.py\\n</ptg-memory>"
    >>> render_block((Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), '', ''),), ((),), 40)
    ''
    >>> render_block((), (), 3600)
    ''
    """
    if not cards:
        return ''
    lines = [Line(c.date, c.rel_file, c.score if c.score is not None else 0.0, c.text, n)
             for c, n in zip(cards, notes, strict=True)]
    text = memshell.render_recall(lines, budget)
    return text if shown(text) > 0 else ''


@deal.ensure(lambda text, result: 0 <= result <= text.count('\n'))
@deal.has()
def shown(text: str) -> int:
    """Сколько карточек вошло в блок: наибольшее N, при котором подряд есть номера «\\n1. [», «\\n2. [», …, «\\nN. [»;
    ни одного — 0.

    D: 2

    >>> shown("<ptg-memory>\\nAuto-recalled from this project's memory (PTG-MEM). Older notes may be outdated — check the date and the markers before relying on them.\\n1. [2025-10-09 · n.md · 0.90] LanceDB выбран.\\n</ptg-memory>")
    1
    >>> shown("<ptg-memory>\\nAuto-recalled from this project's memory (PTG-MEM). Older notes may be outdated — check the date and the markers before relying on them.\\n1. [2025-10-09 · n.md · 0.90] LanceDB выбран.\\n   ↳ replaced later (2025-10-09): LanceDB заменил SQLite.\\n   ↳ code: edits file:PTG-MEM/ptg_mem/engine.py\\n</ptg-memory>")
    1
    >>> shown('<ptg-memory>\\nшапка\\n1. [d · f · 0.90] a\\n2. [d · g · 0.80] b\\n</ptg-memory>')
    2
    >>> shown('<ptg-memory>\\n1. [d · f · 0.90] a\\n3. [d · g · 0.80] b\\n</ptg-memory>')
    1
    >>> shown('')
    0
    """
    count = 0
    while f'\n{count + 1}. [' in text:
        count += 1
    return count
