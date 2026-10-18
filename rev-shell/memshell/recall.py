"""Отбор находок для подсказки: окно сходства, текущий сеанс, сообщения обвязки.

node: leaf
"""
import math
import re

import deal
from memcore.types import Hit

from memshell.types import TURN_PREFIX


@deal.pre(lambda hits, min_score, window: window >= 0 and math.isfinite(min_score) and math.isfinite(window)
          and all(math.isfinite(h.score) for h in hits))
@deal.ensure(lambda hits, min_score, window, result: all(h in hits for h in result) and len(result) <= len(hits))
@deal.ensure(lambda hits, min_score, window, result: all(h.score >= min_score for h in result))
@deal.ensure(lambda hits, min_score, window, result: not hits or all(h.score >= max(x.score for x in hits) - window for h in result))
@deal.has()
def in_window(hits: list[Hit], min_score: float, window: float) -> list[Hit]:
    """Находки, достойные подсказки: score не ниже min_score и не ниже (лучший score среди hits) − window. Порядок —
    как во входе. Окно отсекает хвост, далеко отставший от лучшей находки (PTG-MEM 0.2: window = 0.12).

    D: 2

    >>> in_window([Hit("a", "f.md", "x", 0.81, ""), Hit("b", "f.md", "y", 0.75, ""), Hit("c", "g.md", "z", 0.6, "")], 0.5, 0.12)
    [Hit(id='a', file='f.md', text='x', score=0.81, successor=''), Hit(id='b', file='f.md', text='y', score=0.75, successor='')]
    >>> in_window([Hit("a", "f.md", "x", 0.4, "")], 0.5, 0.12)
    []
    >>> in_window([], 0.5, 0.12)
    []
    """
    if not hits:
        return []
    floor = max(min_score, max(h.score for h in hits) - window)
    return [h for h in hits if h.score >= floor]


@deal.pre(lambda file, turn_at, session_files, cut: math.isfinite(turn_at) and math.isfinite(cut))
@deal.ensure(lambda file, turn_at, session_files, cut, result: file in session_files or result)
@deal.ensure(lambda file, turn_at, session_files, cut, result: not result or file not in session_files or (cut > 0 and turn_at < cut))
@deal.has()
def session_ok(file: str, turn_at: float, session_files: tuple[str, ...], cut: float) -> bool:
    """Нужна ли подсказке находка из файла file, сказанная в момент turn_at. Файл не из текущего сеанса — да. Файл
    текущего сеанса — только сказанное до последнего сжатия контекста (cut > 0 и turn_at < cut): сказанное позже ещё
    в окне модели, повтор — шум; cut = 0 — сжатия не было.

    D: 1

    >>> session_ok("other.jsonl", 100.0, ("cur.jsonl",), 0.0)
    True
    >>> session_ok("cur.jsonl", 100.0, ("cur.jsonl",), 0.0)
    False
    >>> session_ok("cur.jsonl", 100.0, ("cur.jsonl",), 200.0)
    True
    >>> session_ok("cur.jsonl", 300.0, ("cur.jsonl",), 200.0)
    False
    """
    if file not in session_files:
        return True
    return cut > 0 and turn_at < cut


@deal.ensure(lambda head, automated, result: not result or len(automated) > 0)
@deal.has()
def is_noise(head: str, automated: tuple[str, ...]) -> bool:
    """Сообщение обвязки, а не человека: текст без метки хода (TURN_PREFIX, «Пользователь: [ГГГГ-ММ-ДД ЧЧ:ММ]») и
    без пробелов в начале начинается с одного из automated.

    D: 1

    >>> is_noise("<task-notification>done</task-notification>", ("<task-notification",))
    True
    >>> is_noise("Пользователь: [2026-09-22 14:57]  <task-notification>готово", ("<task-notification",))
    True
    >>> is_noise("Почему сборка медленная?", ("<task-notification",))
    False
    >>> is_noise("<task-notification>", ())
    False
    """
    body = re.sub(TURN_PREFIX, "", head.lstrip(), count=1).lstrip()
    return body.startswith(automated)
