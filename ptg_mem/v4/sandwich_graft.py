# -*- coding: utf-8 -*-
r"""PTG_V4/sandwich_graft.py — присоединение с бюджетом по конструкции из доказательства.

ЧТО ЗАМЕНЯЕТ. В ptg_core каждый новый атом получает рёбра только к ДВУМ целям: к голове
ветки (`parent`) и к точке возврата (`best_old_id`). Это преференциальное присоединение:
голова ветки собирает всех детей, периферия не получает ничего. Измеренное следствие на
действующем архиве — медиана степени 3 при максимуме 500 и 4.82 % изолированных узлов
против 0.73 %, которые даёт нулевая модель той же плотности.

ЧТО ВМЕСТО. Конструкция из доказательства гипотезы Кима–Ву: рёбра предлагаются в порядке
убывания близости (это «биномиальное предложение»), а когда бюджет степеней мешает —
решение принимается с вероятностью, ЗАВИСЯЩЕЙ ОТ ТЕКУЩЕГО СОСТОЯНИЯ СТЕПЕНЕЙ и
сохраняющей достижимость остатка.

    q(u,v) = min(1, rem_u * rem_v / d^2)

При полных бюджетах q = 1, то есть сильнейшие связи берутся всегда и порядок близости не
искажается. По мере заполнения бюджета приём плавно падает — хаб перестаёт выигрывать
конкуренцию за слоты, и они достаются периферии.

ЧЕСТНАЯ ГРАНИЦА. Это практический аналог конструкции, а не сама конструкция: доказательство
строит связку для РАВНОМЕРНО случайного регулярного графа, а здесь рёбра порождены
семантикой и граф не случаен ни в каком смысле. Сохранено то свойство, ради которого
конструкция нужна: вероятность приёма зависит только от остатков бюджета и не разрушает
достижимость. Проверяется это не верой, а `selftest.py` — эмпирической проверкой вложения
между двумя пороговыми графами.
"""
from __future__ import annotations

import collections
import math
import random
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

Edge = Tuple[str, str, float]


def graft(candidates: Iterable[Edge],
          budget: int = 30,
          hard_cap: int | None = None,
          rescue: bool = True,
          seed: int = 0) -> List[Edge]:
    """Отобрать рёбра под бюджет степеней.

    candidates — пары (u, v, sim). Порядок ВАЖЕН: ожидается убывание sim. Модуль сам не
                 сортирует, чтобы не тянуть весь список в память на больших архивах;
                 при необходимости сортировать на стороне вызова.
    budget     — целевая степень d. Ставить по условию теоремы: d >> ln n.
    hard_cap   — жёсткий потолок степени (по умолчанию 2*budget). Ограничивает хаб даже
                 тогда, когда вероятностное правило его пропустило бы.
    rescue     — добить узлы, оставшиеся без единого ребра (см. ниже).

    Возвращает список принятых рёбер.
    """
    if hard_cap is None:
        hard_cap = 2 * budget
    rnd = random.Random(seed)
    deg: Dict[str, int] = collections.defaultdict(int)
    taken: List[Edge] = []
    seen = set()
    # лучший отвергнутый кандидат на узел — для спасательного прохода
    best_rejected: Dict[str, Edge] = {}

    for u, v, sim in candidates:
        if u == v:
            continue
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        du, dv = deg[u], deg[v]
        if du >= hard_cap or dv >= hard_cap:
            _remember(best_rejected, u, v, sim)
            continue
        rem_u = max(0, budget - du)
        rem_v = max(0, budget - dv)
        if rem_u > 0 and rem_v > 0:
            # Порог вдвое: пока у обоих концов остаётся больше половины бюджета, ребро
            # берётся БЕЗУСЛОВНО. Это сделано намеренно — чистое произведение остатков
            # отбрасывало бы и сильнейшие связи (замер: терялось 42 % верхних рёбер), а
            # для выдачи важны именно они. Вероятностным приём становится только в
            # конкуренции за последние слоты, где и рождается хаб.
            half = max(1.0, budget / 2.0)
            q = min(1.0, (rem_u * rem_v) / (half * half))
        else:
            # бюджет исчерпан хотя бы с одной стороны: ребро берётся лишь изредка,
            # чтобы не запирать узел наглухо, но и не растить хаб
            q = 1.0 / float(budget)
        if rnd.random() < q:
            seen.add(key)
            deg[u] += 1
            deg[v] += 1
            taken.append((u, v, sim))
        else:
            _remember(best_rejected, u, v, sim)

    if rescue:
        # Спасательный проход. Узел без единого ребра недостижим никаким обходом —
        # его находит только плоский косинус. Это ровно измеренные 4.82 % архива.
        # Каждому такому узлу форсируется его лучший отвергнутый кандидат.
        for node, (u, v, sim) in list(best_rejected.items()):
            if deg[node] > 0:
                continue
            key = (u, v) if u <= v else (v, u)
            if key in seen:
                continue
            seen.add(key)
            deg[u] += 1
            deg[v] += 1
            taken.append((u, v, sim))

    return taken


def _remember(store: Dict[str, Edge], u: str, v: str, sim: float) -> None:
    for node in (u, v):
        old = store.get(node)
        if old is None or sim > old[2]:
            store[node] = (u, v, sim)


def suggest_budget(n: int, margin: float = 3.0) -> int:
    """Бюджет степеней по условию теоремы: d >> ln n.

    n      — число УЗЛОВ графа. Не файлов, не атомов на входе фильтра: ln n в условии
             теоремы берётся от размера графа, и подстановка другой величины даёт не
             менее точный ответ, а ответ на другой вопрос (2002 файла -> 23, те же
             файлы как 9745 узлов -> 28).
    margin — во сколько раз перекрыть ln n. 1.0 это ровно порог связности (там ещё есть
             изолированные), 3.0 — практический вход в режим сэндвича.
    """
    if n <= 2:
        return 1
    return max(2, int(math.ceil(margin * math.log(n))))


def nodes_until_next_budget(n: int, margin: float = 3.0) -> int:
    """На сколько узлов надо вырасти, чтобы бюджет поднялся на единицу.

    Это и есть ответ на вопрос «сдвинулся ли порог режима» при дописывании. Сравнивать
    для этого два числа файлов бессмысленно вдвойне: порог логарифмичен и считается по
    узлам, а сколько узлов дадут новые файлы, манифест не знает.

    Ступень d = ceil(margin * ln n) переступается при n > exp(d / margin). Граница
    берётся оценкой, но подтверждается самой suggest_budget — иначе ответ зависел бы от
    округления exp/log в последнем разряде.
    """
    if n < 2:
        n = 2
    d = suggest_budget(n, margin)
    nxt = max(n + 1, int(math.exp(d / margin)) + 1)
    while nxt > n + 1 and suggest_budget(nxt - 1, margin) > d:
        nxt -= 1
    while suggest_budget(nxt, margin) <= d:
        nxt += 1
    return nxt - n


def traversal_depth(n: int, c: float) -> int:
    """Глубина обхода по оценке диаметра ln n / ln c.

    Константа здесь вредна: при c = 5 нужно шесть шагов, при c = 30 — три. Обход,
    заданный константой 2, покрывает окрестность, а не архив.
    """
    if n <= 1 or c <= 1.0:
        return 1
    return max(1, int(math.ceil(math.log(n) / math.log(c))))


def degree_summary(edges: Sequence[Edge], node_ids: Sequence[str]) -> dict:
    deg = collections.Counter()
    for u, v, _ in edges:
        deg[u] += 1
        deg[v] += 1
    d = sorted(deg.values())
    n = len(node_ids)
    c = 2.0 * len(edges) / n if n else 0.0
    return {
        "n": n,
        "m": len(edges),
        "c": c,
        "isolated": n - len(deg),
        "isolated_null": n * math.exp(-c) if c > 0 else n,
        "median": d[len(d) // 2] if d else 0,
        "max": d[-1] if d else 0,
    }
