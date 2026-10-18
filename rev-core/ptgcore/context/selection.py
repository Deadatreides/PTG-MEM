"""Отбор находок в подсказку: порог сходства, окно от лучшей находки и число записей.

Готовое: memshell.in_window, memcore.select_context (memcore.types.Hit, Budget).

node: leaf
"""
import math

import deal
import memcore
import memshell
from memcore.types import Budget, Hit

from ptgcore.types import Card


@deal.pre(lambda cards, min_score, window, max_items: window >= 0.0 and max_items >= 0 and math.isfinite(min_score)
          and math.isfinite(window) and all(c.score is None or math.isfinite(c.score) for c in cards))
@deal.ensure(lambda cards, min_score, window, max_items, result: len(result) <= max_items and set(result) <= set(cards))
@deal.ensure(lambda cards, min_score, window, max_items, result: len({c.id for c in result}) == len(result)
             and all((c.score or 0.0) >= min_score for c in result))
@deal.has()
def choose(cards: tuple[Card, ...], min_score: float, window: float, max_items: int) -> tuple[Card, ...]:
    """Отбор в подсказку: memshell.in_window(Hit(id, file, text, оценка или 0.0, current) по cards, min_score,
    window), затем memcore.select_context(…, Budget(10 ** 9, max_items, min_score, max_items)); карточки по порядку
    отобранного.

    D: 3

    >>> choose((Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), '', ''), Card('a2', 0.85, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB заменил SQLite.', 23, (), '', ''), Card('a3', 0.5, 'active', '2025-10-09', 'n.md', 'n.md', '', 'шум', 3, (), '', '')), 0.5, 0.12, 5)
    (Card(id='a1', score=0.9, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''), Card(id='a2', score=0.85, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB заменил SQLite.', chars=23, links=(), current='', session=''))
    >>> choose((), 0.5, 0.12, 5)
    ()
    >>> choose((Card('a1', 0.9, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB выбран.', 15, (), '', ''), Card('a2', 0.85, 'active', '2025-10-09', 'n.md', 'n.md', '', 'LanceDB заменил SQLite.', 23, (), '', '')), 0.5, 0.12, 1)
    (Card(id='a1', score=0.9, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''),)
    >>> choose((Card('a3', 0.4, 'active', '2025-10-09', 'n.md', 'n.md', '', 'шум', 3, (), '', ''),), 0.5, 0.12, 5)
    ()
    """
    best: dict[str, Card] = {}
    for c in cards:
        if c.id not in best or (c.score or 0.0) > (best[c.id].score or 0.0):
            best[c.id] = c
    hits = [
        Hit(id=c.id, file=c.file, text=c.text, score=c.score or 0.0, successor=c.current)
        for c in cards
    ]
    budget = Budget(10 ** 9, max_items, min_score, max_items)
    picked = memcore.select_context(memshell.in_window(hits, min_score, window), budget)
    return tuple(best[item.id] for item in picked)
