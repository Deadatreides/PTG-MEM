"""Отбор в подсказку при порядке RRF: лучший косинус запроса, окно от него, id полнотекста и число записей.

Готовое: dataclasses.replace для карточки с новой оценкой.

node: leaf
"""
import math
from dataclasses import replace

import deal

from ptgcore.types import Card


@deal.pre(lambda cards, near, words, min_score, window, max_items: window >= 0.0 and max_items >= 0
          and math.isfinite(min_score) and math.isfinite(window) and all(math.isfinite(s) for _, s in near))
@deal.ensure(lambda cards, near, words, min_score, window, max_items, result: len(result) <= max_items
             and len({c.id for c in result}) == len(result) and {c.id for c in result} <= {c.id for c in cards})
@deal.ensure(lambda cards, near, words, min_score, window, max_items, result:
             result == () or (near != () and max(s for _, s in near) >= min_score))
@deal.has()
def admit(cards: tuple[Card, ...], near: tuple[tuple[str, float], ...], words: tuple[str, ...], min_score: float, window: float,
          max_items: int) -> tuple[Card, ...]:
    """ФТ-008. Отбор в подсказку при порядке RRF (замер 13.10, правило B2: из 47 своих запросов нашла 25 против 15 у
    прежнего отбора по косинусу; на 24 посторонних молчит так же — 16):
    * лучший косинус запроса — наибольшая оценка в near (пары id, косинус); near пуст или лучший ниже min_score — ();
    * карточка проходит, если её косинус (оценка её id в near) не ниже max(min_score, лучший − window) или её id
      среди первых max_items в words (id полнотекста по порядку);
    * по порядку cards, без повторов id, не больше max_items;
    * оценка выданной карточки — её косинус из near, нет его — None.

    D: 6

    >>> admit((Card(id='a1', score=0.0325, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''), Card(id='a2', score=0.0317, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB заменил SQLite.', chars=23, links=(), current='', session=''), Card(id='a3', score=0.0161, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='кэш эмбеддингов', chars=15, links=(), current='', session='')), (('a2', 0.71), ('a1', 0.66), ('a3', 0.4)), ('a3',), 0.5, 0.12, 5)
    (Card(id='a1', score=0.66, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''), Card(id='a2', score=0.71, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB заменил SQLite.', chars=23, links=(), current='', session=''), Card(id='a3', score=0.4, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='кэш эмбеддингов', chars=15, links=(), current='', session=''))
    >>> admit((Card(id='a1', score=0.0325, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''), Card(id='a2', score=0.0317, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB заменил SQLite.', chars=23, links=(), current='', session='')), (('a1', 0.45),), ('a1',), 0.5, 0.12, 5)
    ()
    >>> admit((Card(id='a1', score=0.0325, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''),), (('a2', 0.7),), ('a1',), 0.5, 0.12, 5)
    (Card(id='a1', score=None, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''),)
    >>> admit((Card(id='a1', score=0.0325, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''), Card(id='a2', score=0.0317, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB заменил SQLite.', chars=23, links=(), current='', session=''), Card(id='a3', score=0.0161, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='кэш эмбеддингов', chars=15, links=(), current='', session='')), (('a2', 0.71), ('a1', 0.66), ('a3', 0.4)), ('a3',), 0.5, 0.12, 1)
    (Card(id='a1', score=0.66, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''),)
    >>> admit((), (), (), 0.5, 0.12, 5)
    ()
    """
    best = max((s for _, s in near), default=None)
    if best is None or best < min_score:
        return ()
    scores: dict[str, float] = {}
    for cid, s in near:
        scores[cid] = max(s, scores.get(cid, s))
    threshold = max(min_score, best - window)
    top = set(words[:max_items])
    out: list[Card] = []
    seen: set[str] = set()
    for card in cards:
        if len(out) >= max_items:
            break
        if card.id in seen:
            continue
        cos = scores.get(card.id)
        if (cos is not None and cos >= threshold) or card.id in top:
            seen.add(card.id)
            out.append(replace(card, score=cos))
    return tuple(out)
