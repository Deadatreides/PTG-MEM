# -*- coding: utf-8 -*-
r"""PTG_V5/budget_graft.py — ядро: бюджет связей в момент вставки атома.

ЧЕМ V5 ОТЛИЧАЕТСЯ ОТ V4. В V4 бюджет применялся глобальной пересборкой
(`rebalance.py`): брались готовые векторы, считался top-k по косинусу, слой
раскладывался под бюджет. Получался второй граф рядом — **путенезависимый и
слепой ко времени**, и навигация ходила по нему. Это индекс похожести, а не
траекторный граф; для PTG это чужая вещь.

Статья о сэндвиче говорила, куда бюджет относится, прямым текстом:

    «задача построить граф с ограниченной степенью, статистически
     эквивалентный пороговому — это дословно задача graft-scoring с бюджетом»

Graft-scoring. Здесь бюджет и стоит — в присоединении, в момент вставки, на
том состоянии графа, которое существовало к этому моменту.

ГЛАВНОЕ РЕШЕНИЕ: ЧТО ЕДИНСТВЕННО, А ЧТО МНОЖЕСТВЕННО.

  * **Принадлежность ветви — единственна.** Её как выбирал движок, так и
    выбирает: argmax по graft-score, та же логика CASE A–E, тот же
    `continues` только при CASE D. V5 в это не вмешивается ВООБЩЕ — оригинал
    `_add_atom` отрабатывает целиком и первым.
  * **Отношения — множественны.** Уже ПОСЛЕ оригинала V5 добирает до `e−1`
    дополнительных рёбер под бюджет.

Отсюда гарантия несхлопывания, и это не надежда, а следствие:

    бюджет добавляет только НЕ-`continues` рёбра; `continues` по-прежнему не
    более одного на атом и по-прежнему требует CASE D (тот же файл И тот же
    локальный кластер). Значит рост бюджета НЕ МОЖЕТ слить две линии в одну:
    коагуляция шла через членство, а членство осталось однозначным.

Проверяется это не верой: `test_budget_graft.py` сверяет назначение ветви
атом в атом с прогоном без V5.

АРИФМЕТИКА. Степень узла = что он испустил сам + что пришло позже, значит
`c = 2E/n = 2·e`. Чтобы средняя степень вышла на `d`, атом испускает ПОЛОВИНУ
бюджета: `e = ceil(d/2)`. Вторую половину принесут будущие атомы.

    d(n) = ceil(margin · ln n),  margin = 3   — условие режима d >> ln n
    e    = ceil(d/2)
    k    = oversample·d (по умолчанию 8d)      — сколько кандидатов перебрать
    потолок = 2d                              — жёсткий, поверх вероятностного

ПРАВИЛО ПРИЁМА (из sandwich_graft, без изменений по сути):

    q(a,u) = min(1, rem_a · rem_u / d²),   rem_x = max(0, d − deg x)

При вставке `rem_a` полон, поэтому первое ребро принимается с вероятностью
`rem_u/d` — пропорционально остатку бюджета ЦЕЛИ. Голова, набравшая d детей,
получает ноль; периферия — почти единицу. Это и лечит измеренное «медиана 2
при максимуме 386». Дальше множитель `rem_a/d` падает с 1 до 0.5, и атом
становится разборчивее к концу вставки.

ВРЕМЯ И ПУТЬ.

  * Порядок кандидатов — тот же многоосевой graft-score движка, где ПУТЬ
    весит 0.25, второй после семантики. Кандидаты считаются по состоянию ДО
    вставки этого атома — то есть решение зависит от того, что было раньше,
    как и должно быть в path-dependent графе.
  * Тип ребра: отношения к головам ветвей — `reinforces`, отношения к
    глубоким узлам выше `RETURN_THRESH` — `returns_to`. `continues` не
    выдаётся никогда (он занят членством), поэтому далёкое по времени
    физически не может стать продолжением. Время не подавляет связь, а
    запрещает ей быть продолжением — ровно замысел.
  * **Предел на возвраты: не больше двух на атом.** `returns_to`
    реактивирует ветвь и дописывает её в `path_memory["long"]`; при e=16
    длинный путь пух бы в разы, `long_coh` (среднее по нему) размывался бы, и
    бюджет убил бы ту самую путезависимость, ради которой всё.

СПАСЕНИЕ СИРОТ. Атом ниже `BRANCH_THRESH` не получает от движка ни одного
ребра — отсюда измеренные 4.82 % изолированных. V5 цепляет такому атому одно
ребро к лучшему кандидату, но ОТДЕЛЬНЫМ типом `rescue`. Спасательная связь
обязана быть отличима от смысловой, иначе она станет ложным свидетельством:
обход по ней уводит, а по выдаче этого не видно.

ПОРОГИ НЕ ЗАШИТЫ. Читаются из модуля движка в момент вызова. В самом
`ptg_core` они 0.60/0.78/0.83, а рантайм сборки (`run_text_ptg_v3.py`)
выставляет калиброванные 0.47/0.57/0.65. Зашить любое из двух значило бы
получить архив, собранный по одной шкале и надстроенный по другой.

ЗАВИСИМОСТЬ. Требуется `PTG_V4/graft_fast.py`: он держит матрицы ветвей и
узлов и считает оба скана матвеком. Без него перебор кандидатов был бы
питоновским циклом по всем ветвям на каждый атом, и бюджетный графт стал бы
неподъёмным.
"""
from __future__ import annotations

import math
import random

import numpy as np

REL_TYPE = "reinforces"          # отношение к голове чужой ветви
RETURN_TYPE = "returns_to"       # отношение к глубокому узлу
RESCUE_TYPE = "rescue"           # спасение сироты — намеренно отдельный тип

MAX_CONTINUES_PER_ATOM = 1       # членство, его выдаёт движок
MAX_RETURNS_PER_ATOM = 2         # см. «предел на возвраты» в шапке


def budget_for(n: int, margin: float = 3.0) -> int:
    """d = ceil(margin · ln n) — условие режима d >> ln n."""
    if n <= 2:
        return 2
    return max(2, int(math.ceil(margin * math.log(n))))


class BudgetGraft:
    def __init__(self, arc, core, fg, margin: float = 3.0, seed: int = 0,
                 max_returns: int = MAX_RETURNS_PER_ATOM, rescue: bool = True,
                 oversample: int = 8):
        self.arc, self.core, self.fg = arc, core, fg
        self.margin = float(margin)
        self.oversample = int(oversample)
        self.max_returns = int(max_returns)
        self.rescue = bool(rescue)
        self.rnd = random.Random(seed)
        self.deg: dict[str, int] = {}
        self.stat = {
            "атомов": 0, "рёбер_движка": 0, "рёбер_отношений": 0,
            "возвратов_добавлено": 0, "спасено_сирот": 0,
            "отказов_по_бюджету": 0, "отказов_по_порогу": 0,
            "кандидатов_просмотрено": 0, "бюджет_последний": 0,
            "эмиссия_заполнена_%": 0.0, "_emit_want": 0, "_emit_got": 0,
            # диагностика второго канала: он не добавил ни одного ребра,
            # и надо знать, где именно он глохнет
            "возвр_канал_вызван": 0, "возвр_кэш_промах": 0,
            "возвр_кандидатов": 0, "возвр_выше_порога": 0,
            "возвр_уже_занято": 0, "возвр_отказ_бюджет": 0,
        }
        # степени уже загруженного архива
        for e in arc.edges:
            self._bump(e["from"])
            self._bump(e["to"])

    # --- учёт степеней ----------------------------------------------------
    def _bump(self, nid, k: int = 1):
        self.deg[nid] = self.deg.get(nid, 0) + k

    def _rem(self, nid, d: int) -> int:
        return max(0, d - self.deg.get(nid, 0))

    # --- бюджет на текущий момент ----------------------------------------
    def current_budget(self) -> int:
        return budget_for(len(self.arc.id_order), self.margin)

    # --- перебор кандидатов ----------------------------------------------
    def _branch_candidates(self, vec, k: int):
        """Головы чужих ветвей в порядке graft-score. Score берётся из кэша
        `graft_fast`, то есть считается по состоянию ДО вставки атома."""
        fg, core = self.fg, self.core
        n = len(fg.bid_list)
        if n == 0:
            return []
        # Кэш снят ДО вставки атома, поэтому он короче текущего списка: атом
        # мог создать ветвь. Сверять длину с текущей — ошибка, из-за которой
        # кэш промахивался у 60 % атомов и самый дорогой матвек считался
        # второй раз. Берём длину самого кэша: его записи и есть состояние
        # «до атома», а именно оно и нужно путезависимому решению.
        c = fg._sem_cache
        if c is not None and c[0] is vec:
            sem = c[1]
            m = int(sem.shape[0])
        else:
            m = n
            sem = fg.C[:m] @ vec
        score = (core.W_SEM * sem
                 + core.W_MOM * np.tanh(fg.mom[:m] / 5.0)
                 + core.W_ACT * fg.act[:m]
                 - core.W_ENT * fg.ent[:m])
        kk = min(k, m)
        if kk <= 0:
            return []
        idx = np.argpartition(-score, kk - 1)[:kk]
        idx = idx[np.argsort(-score[idx])]
        return [(fg.bid_list[int(i)], float(sem[int(i)])) for i in idx]

    def _deep_candidates(self, vec, k: int):
        """Глубокие узлы в порядке косинуса — второй канал движка, тот же,
        из которого берётся `returns_to`."""
        fg = self.fg
        if not fg.nid_list:
            return []
        # Та же поправка, что в _branch_candidates: длина кэша — это состояние
        # ДО вставки атома, и сверять её с текущей нельзя. Раньше сверял, и
        # канал глох целиком: 887 промахов из 887 вызовов, ноль кандидатов.
        c = fg._old_cache
        if c is None or c[0] is not vec:
            return []
        sims = c[1]
        m = int(sims.shape[0])
        kk = min(k, m)
        if kk <= 0:
            return []
        idx = np.argpartition(-sims, kk - 1)[:kk]
        idx = idx[np.argsort(-sims[idx])]
        return [(fg.nid_list[int(i)], float(sims[int(i)])) for i in idx
                if np.isfinite(sims[int(i)])]

    # --- добор отношений --------------------------------------------------
    def add_relations(self, node, vec, engine_edges: list):
        arc = self.arc
        nid = node["id"]
        d = self.current_budget()
        e_budget = int(math.ceil(d / 2.0))
        # Перебор шире бюджета. Анти-хабовое правило отвергает насыщенные
        # цели, а нижний порог — далёкие по смыслу; при k = 2d атом упирался
        # в конец списка, набрав половину эмиссии (замер: 84 % отказов,
        # средняя степень 13.9 против нужных 22). Кандидат стоит один скаляр,
        # поэтому перебор расширен, а не бюджет ослаблен.
        k = max(self.oversample * d, 64)
        self.stat["бюджет_последний"] = d
        self.stat["_emit_want"] += e_budget

        # цели, уже взятые движком — точным списком, а не хвостом arc.edges
        own = {nid}
        for edge in engine_edges:
            if edge["from"] == nid:
                own.add(edge["to"])
            elif edge["to"] == nid:
                own.add(edge["from"])

        made = len(own) - 1                          # сколько целей уже есть
        returns = sum(1 for edge in engine_edges if edge["type"] == RETURN_TYPE)

        # канал 1 — головы чужих ветвей
        #
        # НИЖНИЙ ПОРОГ ОБЯЗАТЕЛЕН. Без него top-k по score отдаёт кандидатов
        # при любом сходстве, и атом, который движок счёл ни с чем не
        # связанным (sem < BRANCH_THRESH), получает ребро `reinforces` к
        # «наименее плохой» ветке. Это ложное утверждение о родстве: обход по
        # такому ребру уводит, а по выдаче этого не видно. Замер 21.09.2026
        # без порога: все 86 сирот «спаслись» именно так, молча.
        floor = self.core.BRANCH_THRESH
        for bid, sim in self._branch_candidates(vec, k):
            if made >= e_budget:
                break
            if bid == node.get("branch"):
                continue
            if sim < floor:
                self.stat["отказов_по_порогу"] += 1
                continue
            b = arc.branches.get(bid)
            if not b:
                continue
            tgt = b.get("head")
            if tgt is None or tgt in own:
                continue
            self.stat["кандидатов_просмотрено"] += 1
            if not self._accept(nid, tgt, d):
                self.stat["отказов_по_бюджету"] += 1
                continue
            arc.edges.append({"from": nid, "to": tgt, "type": REL_TYPE})
            self._bump(nid); self._bump(tgt)
            own.add(tgt); made += 1
            self.stat["рёбер_отношений"] += 1

        # канал 2 — глубокие узлы, только выше порога возврата
        if returns < self.max_returns and made < e_budget:
            self.stat["возвр_канал_вызван"] += 1
            thr = self.core.RETURN_THRESH
            deep = self._deep_candidates(vec, k)
            if not deep:
                self.stat["возвр_кэш_промах"] += 1
            self.stat["возвр_кандидатов"] += len(deep)
            for tgt, sim in deep:
                if made >= e_budget or returns >= self.max_returns:
                    break
                if sim <= thr:
                    continue
                self.stat["возвр_выше_порога"] += 1
                if tgt in own:
                    self.stat["возвр_уже_занято"] += 1
                    continue
                self.stat["кандидатов_просмотрено"] += 1
                if not self._accept(nid, tgt, d):
                    self.stat["отказов_по_бюджету"] += 1
                    continue
                arc.edges.append({"from": nid, "to": tgt, "type": RETURN_TYPE})
                self._bump(nid); self._bump(tgt)
                own.add(tgt); made += 1; returns += 1
                self.stat["возвратов_добавлено"] += 1
                # БЕЗ `_reactivate_branch`. Реактивация поднимает activation и
                # momentum ветви, а они входят в graft-score СЛЕДУЮЩИХ атомов,
                # то есть отношение начинает менять будущие решения о членстве.
                # Замер 21.09.2026 с реактивацией: ветка разошлась у 16 атомов
                # из 1200, `continues` 479 -> 477, ветвей 722 -> 724 —
                # инвариант сломан, и вместе с ним гарантия несхлопывания.
                #
                # ПРАВИЛО V5: отношения — это наблюдения, а не действия. Они
                # пишутся только в `edges` и не трогают ни состояние ветвей, ни
                # память пути, ни поля узлов. Менять состояние вправе одно
                # членство, и его выдаёт движок.

        self.stat["_emit_got"] += made
        w = self.stat["_emit_want"] or 1
        self.stat["эмиссия_заполнена_%"] = round(100.0 * self.stat["_emit_got"] / w, 1)

        # спасение сироты — отдельным типом
        if self.rescue and made == 0:
            cands = self._branch_candidates(vec, 1)
            tgt = None
            if cands:
                b = arc.branches.get(cands[0][0])
                tgt = b.get("head") if b else None
            if tgt is not None and tgt != nid:
                arc.edges.append({"from": nid, "to": tgt, "type": RESCUE_TYPE})
                self._bump(nid); self._bump(tgt)
                self.stat["спасено_сирот"] += 1

    def _accept(self, a, u, d: int) -> bool:
        """q = min(1, rem_a·rem_u/d²) плюс жёсткий потолок 2d."""
        if self.deg.get(u, 0) >= 2 * d or self.deg.get(a, 0) >= 2 * d:
            return False
        q = min(1.0, (self._rem(a, d) * self._rem(u, d)) / float(d * d))
        return self.rnd.random() < q

    def stats(self):
        return dict(self.stat)


# ---------------------------------------------------------------------------
def install(arc, core, fg, margin: float = 3.0, seed: int = 0,
            max_returns: int = MAX_RETURNS_PER_ATOM, rescue: bool = True,
            oversample: int = 8) -> BudgetGraft:
    """Надстроить бюджетный графт поверх уже установленного `graft_fast`.

    Порядок обязателен: сначала `graft_fast.install`, затем этот. V5 оборачивает
    ТО, что установлено, а не оригинальный метод класса, — и вызывает его
    первым, целиком, не вмешиваясь в выбор членства.
    """
    bg = BudgetGraft(arc, core, fg, margin=margin, seed=seed,
                     max_returns=max_returns, rescue=rescue, oversample=oversample)
    prev_add = arc._add_atom

    def _add_atom(atom):
        before = len(arc.edges)
        node = prev_add(atom)                      # членство, тип, путь — движок
        new_edges = list(arc.edges[before:])
        for edge in new_edges:
            bg._bump(edge["from"]); bg._bump(edge["to"])
        bg.stat["рёбер_движка"] += len(new_edges)
        # Вектор берём ТОТ ЖЕ объект, на котором graft_fast считал сканы:
        # иначе проверка `c[0] is vec` промахнётся и самый дорогой матвек
        # пересчитается второй раз на каждый атом.
        cache = fg._sem_cache
        vec = cache[0] if cache is not None else np.asarray(atom["vec"], dtype="float32")
        bg.add_relations(node, vec, new_edges)
        bg.stat["атомов"] += 1
        return node

    arc._add_atom = _add_atom
    return bg
