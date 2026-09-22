# -*- coding: utf-8 -*-
r"""PTG_V4/graft_fast.py — векторизация графта без единой правки ptg_core.

ЧТО ИМЕННО МЕДЛЕННО. `Archive._add_atom` на КАЖДЫЙ атом делает два полных
сканирования питоновским циклом:

  1. по всем веткам (`for bid, b in self.branches.items()`) — скалярный
     `np.dot` с центроидом тела ветки или, если тела ещё нет, с вектором её
     головы; из них берётся argmax graft-score;
  2. по всем узлам (`for nid in self.id_order`) — скалярный `np.dot` со
     всеми «глубокими» узлами, не являющимися головами; из них берётся
     ближайший (точка возврата).

К концу сборки v3 это 16 573 ветки и 27 476 узлов, то есть на последних атомах
каждый из двух сканов — десятки тысяч питоновских итераций. Замер по журналу
настоящего прогона (`logs/text_ptg_v3_build2.log`): фаза дерева 20.5 % против
79.5 % у эмбеддинга, и доля растёт с архивом — оба скана линейны по его
размеру, а атомов на файл столько же.

ПОЧЕМУ НЕ КОПИЯ МЕТОДА. `_add_atom` — 330 строк, где кроме этих двух циклов
лежит вся логика размещения (CASE A–E, density limiter, эпохи, рёбра). Копия
разошлась бы с движком при первой его правке, а проверять пришлось бы не
argmax, а весь метод.

ЧТО СДЕЛАНО ВМЕСТО. Оба argmax считаются заранее матвеком, а затем
оригинальному циклу подсовывается итерация ровно по одному победителю:

  * `self.branches` подменяется прокси, у которого `.items()` отдаёт одну
    пару, а `.values()`, `.get()`, `[]`, `len()` делегируются настоящему
    словарю — `head_ids` внутри метода считается по `.values()` и обязан
    остаться полным;
  * `self.id_order` подменяется прокси, у которого ИТЕРАЦИЯ отдаёт одного
    кандидата, а `len()`, индексация и `append()` делегируются — потому что
    `cur_index = len(self.id_order)` задаёт номер атома.

Вся скалярная арифметика внутри цикла остаётся ИСХОДНОЙ: `best_head_sim`,
`best_graft_score`, `best_old_sim` получаются тем же `float(np.dot(...))`.
Совпадение не приблизительное, а точное — при условии, что argmax выбран тот
же. За это отвечает сверка ниже.

ЗЕРКАЛО СОСТОЯНИЯ. Чтобы не вернуть ту же петлю с другой стороны, momentum /
activation / entropy ведутся массивами по тем же правилам, что в движке:
`_update_branch_states` гасит все ветки кроме выбранной (0.97 и 0.95),
`_decay_branches` каждые 25 атомов гасит простаивающие (0.8), `_reactivate_branch`
поднимает одну. Патчи вызывают оригинал (словари остаются источником истины для
сохранения) и повторяют то же на массивах — O(1) питоновских операций на атом
вместо O(веток).

ТОЧНОСТЬ ARGMAX. И скалярный `np.dot`, и матвек копят во float32, поэтому
значения расходятся на ~1e-5 — не на 1e-7. Оригинал берёт ПЕРВЫЙ строгий
максимум в порядке вставки. Поэтому всё, что лежит в пределах `EPS = 1e-4` от
максимума, пересчитывается скалярно, в исходном порядке, тем же правилом
«строго больше». Сколько раз это понадобилось, видно в `stats()`.

ПРОВЕРКА, А НЕ ВЕРА. `install(..., verify=N)` первые N атомов считает оба
argmax ещё и прямым перебором — буквальной транскрипцией циклов из ptg_core —
и сверяет, а заодно сверяет зеркало состояния со словарями. Проверка стоит
ровно столько, сколько стоил бы старый путь, поэтому N держат небольшим: это
ворота перед прогоном, а не режим прогона.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-4          # окно вокруг максимума, где порядок решает скалярный пересчёт
GROW = 4096         # на столько строк растут матрицы за раз
MOM_DECAY = 0.97    # те же константы, что в ptg_core._update_branch_states
ACT_DECAY = 0.95
IDLE_DECAY = 0.8    # ptg_core._decay_branches


# ---------------------------------------------------------------------------
# Прокси, сужающие итерацию, но не сам объект
# ---------------------------------------------------------------------------
class _OneBranch:
    """Словарь веток, у которого `.items()` отдаёт одну пару (или ни одной)."""

    __slots__ = ("_d", "_bid")

    def __init__(self, d, bid):
        self._d, self._bid = d, bid

    def items(self):
        if self._bid is None:
            return iter(())
        return iter(((self._bid, self._d[self._bid]),))

    def values(self):
        return self._d.values()

    def keys(self):
        return self._d.keys()

    def get(self, k, default=None):
        return self._d.get(k, default)

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v):
        self._d[k] = v

    def __contains__(self, k):
        return k in self._d

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)


class _OneOld:
    """id_order, у которого ИТЕРАЦИЯ отдаёт одного кандидата,
    а длина, индексация и append остаются настоящими."""

    __slots__ = ("_l", "_nid")

    def __init__(self, lst, nid):
        self._l, self._nid = lst, nid

    def __iter__(self):
        if self._nid is None:
            return iter(())
        return iter((self._nid,))

    def __len__(self):
        return len(self._l)

    def __getitem__(self, i):
        return self._l[i]

    def __contains__(self, k):
        return k in self._l

    def append(self, v):
        self._l.append(v)


# ---------------------------------------------------------------------------
class FastGraft:
    def __init__(self, arc, core, verify: int = 0, eps: float = EPS):
        self.arc, self.core = arc, core
        self.verify_left = int(verify)
        self.eps = float(eps)
        self.dim = None

        self.bid_list: list[str] = []
        self.bid_pos: dict[str, int] = {}
        self.C: np.ndarray | None = None
        self.mom = np.zeros(GROW, dtype="float64")
        self.act = np.zeros(GROW, dtype="float64")
        self.ent = np.zeros(GROW, dtype="float64")
        self.seen = np.zeros(GROW, dtype="int64")
        self.dirty: set[str] = set()

        self.nid_list: list[str] = []
        self.npos: dict[str, int] = {}
        self.M: np.ndarray | None = None
        self.nidx = np.zeros(GROW, dtype="int64")

        # живое множество голов — чтобы не пересобирать его на каждый атом
        self.head_of: dict[str, str] = {}
        self.head_set: set[str] = set()

        # длинный путь: позиции веток в C; -1 = у ветки ещё нет центроида,
        # такие перепроверяются при следующем вызове (их число только убывает)
        self.long_idx = np.zeros(GROW, dtype="int64")
        self.long_n = 0

        # сходство со всеми ветками, посчитанное для текущего атома в
        # best_branch: path_coherence нужен тот же вектор и та же матрица,
        # и пересчитывать его второй раз незачем
        self._sem_cache = None
        self._old_cache = None

        self.stat = {"атомов": 0, "сверено": 0, "расхождений_ветка": 0,
                     "расхождений_узел": 0, "расхождений_состояние": 0,
                     "пересчётов_ветка": 0, "пересчётов_узел": 0}

    # --- рост ---------------------------------------------------------------
    def _ensure_dim(self, vec):
        if self.dim is None:
            self.dim = int(np.asarray(vec).shape[0])
            self.C = np.zeros((GROW, self.dim), dtype="float32")
            self.M = np.zeros((GROW, self.dim), dtype="float32")

    @staticmethod
    def _grow2(mat, need):
        if need <= mat.shape[0]:
            return mat
        new = np.zeros((max(need, mat.shape[0] * 2), mat.shape[1]), dtype=mat.dtype)
        new[:mat.shape[0]] = mat
        return new

    @staticmethod
    def _grow1(arr, need):
        if need <= arr.shape[0]:
            return arr
        new = np.zeros(max(need, arr.shape[0] * 2), dtype=arr.dtype)
        new[:arr.shape[0]] = arr
        return new

    # --- строка ветки: ровно то, что берёт оригинал -------------------------
    def _branch_vec(self, bid):
        arc = self.arc
        b = arc.branches.get(bid)
        if b is None:
            return None
        body = arc.branch_bodies.get(bid)
        if body and len(body["atom_ids"]) > 0 and body["centroid"] is not None:
            cv = body["centroid"]
            cv = cv if isinstance(cv, np.ndarray) else np.array(cv, dtype="float32")
            if cv.shape == (self.dim,):
                return cv.astype("float32", copy=False)
        head = arc.nodes.get(b["head"])
        return None if head is None else np.asarray(head["vec"], dtype="float32")

    def add_branch(self, bid):
        """Новая ветка: строка в матрице + начальное состояние из словаря."""
        if bid in self.bid_pos:
            self.dirty.add(bid)
            return
        i = len(self.bid_list)
        self.bid_pos[bid] = i
        self.bid_list.append(bid)
        self.C = self._grow2(self.C, i + 1)
        for name in ("mom", "act", "ent", "seen"):
            setattr(self, name, self._grow1(getattr(self, name), i + 1))
        bs = self.arc._ensure_branch_state(bid)
        self.mom[i] = bs["momentum"]
        self.act[i] = bs["activation"]
        self.ent[i] = bs["entropy"]
        self.seen[i] = bs.get("last_seen", 0)
        self.dirty.add(bid)

    def _flush_dirty(self):
        for bid in self.dirty:
            i = self.bid_pos.get(bid)
            if i is None:
                continue
            v = self._branch_vec(bid)
            if v is not None:
                self.C[i] = v
        self.dirty.clear()

    def pull_state(self, bid):
        """Считать состояние одной ветки из словаря в массивы."""
        i = self.bid_pos.get(bid)
        if i is None:
            return
        bs = self.arc.branch_states.get(bid)
        if bs is None:
            return
        self.mom[i] = bs["momentum"]
        self.act[i] = bs["activation"]
        self.ent[i] = bs["entropy"]
        self.seen[i] = bs.get("last_seen", 0)

    # --- выбор ветки --------------------------------------------------------
    def best_branch(self, vec):
        core = self.core
        n = len(self.bid_list)
        if n == 0:
            return None
        self._flush_dirty()
        sem = self.C[:n] @ vec
        self._sem_cache = (vec, sem)
        # W_PATH*path_score и W_ROOT*root_sim одинаковы для всех веток —
        # на argmax не влияют и потому не считаются.
        score = (core.W_SEM * sem
                 + core.W_MOM * np.tanh(self.mom[:n] / 5.0)
                 + core.W_ACT * self.act[:n]
                 - core.W_ENT * self.ent[:n])
        i = int(np.argmax(score))
        near = np.flatnonzero(score >= score[i] - self.eps)
        if near.size > 1:
            self.stat["пересчётов_ветка"] += 1
            rest = score - core.W_SEM * sem          # часть, не зависящая от sem
            best_i, best_v = None, -np.inf
            for j in near:                            # порядок вставки, «строго больше»
                v = core.W_SEM * float(np.dot(vec, self.C[j])) + float(rest[j])
                if v > best_v:
                    best_v, best_i = v, int(j)
            i = best_i
        return self.bid_list[i]

    # --- выбор точки возврата ----------------------------------------------
    def best_old(self, vec, cur_index, head_ids):
        n = len(self.nid_list)
        if n == 0:
            return None
        limit = cur_index - self.core.DEEP_NODE_MIN_AGE
        ok = self.nidx[:n] < limit
        if not ok.any():
            return None
        sims = self.M[:n] @ vec
        sims = np.where(ok, sims, -np.inf)
        for hid in head_ids:
            j = self.npos.get(hid)
            if j is not None and j < n:
                sims[j] = -np.inf
        # Кэш для надстроек (PTG_V5): это самый дорогой скан на атом, и
        # считать его второй раз ради тех же кандидатов нельзя.
        self._old_cache = (vec, sims)
        i = int(np.argmax(sims))
        if not np.isfinite(sims[i]):
            return None
        near = np.flatnonzero(sims >= sims[i] - self.eps)
        if near.size > 1:
            self.stat["пересчётов_узел"] += 1
            best_i, best_v = None, -np.inf
            for j in near:
                v = float(np.dot(vec, self.M[j]))
                if v > best_v:
                    best_v, best_i = v, int(j)
            i = best_i
        return self.nid_list[i]

    # --- связность пути ------------------------------------------------------
    def _has_centroid(self, bid) -> bool:
        body = self.arc.branch_bodies.get(bid)
        if not body or body["centroid"] is None:
            return False
        cv = body["centroid"]
        cv = cv if isinstance(cv, np.ndarray) else np.asarray(cv)
        return cv.shape == (self.dim,)

    def path_coherence(self, vec):
        """То же, что ptg_core._path_coherence, но двумя матвеками.

        Короткий путь — последние CURRENT_PATH_WINDOW узлов, длинный — ветки
        lineage с кратностями (повтор ветки при возврате считается дважды,
        как и в оригинале). Отбор «у ветки есть годный центроид» повторён
        буквально; записи, у которых его ещё нет, помечаются -1 и
        перепроверяются позже, потому что центроид появляется и не исчезает.
        """
        arc, core = self.arc, self.core
        self._flush_dirty()

        short = arc.path_memory["short"][-core.CURRENT_PATH_WINDOW:]
        idx = [self.npos[p] for p in short if p in self.npos]
        short_coh = float(np.mean(self.M[idx] @ vec)) if idx else 0.5

        lp = arc.path_memory["long"]
        if len(lp) > self.long_n:
            self.long_idx = self._grow1(self.long_idx, len(lp))
            for t in range(self.long_n, len(lp)):
                bid = lp[t]
                self.long_idx[t] = (self.bid_pos.get(bid, -1)
                                    if self._has_centroid(bid) else -1)
            self.long_n = len(lp)
        pend = np.flatnonzero(self.long_idx[:self.long_n] < 0)
        for t in pend:                       # обычно пусто: центроид уже есть
            bid = lp[int(t)]
            if self._has_centroid(bid):
                self.long_idx[t] = self.bid_pos.get(bid, -1)
        li = self.long_idx[:self.long_n]
        li = li[li >= 0]
        if li.size:
            # НЕ self.C[li] @ vec: выборка строк копирует их — при 4000 ветках
            # это 41 МБ на атом, и векторизация выходит медленнее цикла.
            # Матвек считается по всей матрице (или берётся из кэша best_branch),
            # а индексируется уже одномерный результат.
            nb = len(self.bid_list)
            c = self._sem_cache
            sims = (c[1] if (c is not None and c[0] is vec and c[1].shape[0] == nb)
                    else self.C[:nb] @ vec)
            long_coh = float(np.mean(sims[li]))
        else:
            long_coh = 0.5

        return short_coh, long_coh, 0.7 * short_coh + 0.3 * long_coh

    # --- прямой перебор для сверки (транскрипция циклов ptg_core) -----------
    def brute(self, vec, cur_index, head_ids):
        arc, core = self.arc, self.core
        best_b, best_score = None, -1.0
        for bid, b in arc.branches.items():
            body = arc.branch_bodies.get(bid)
            if body and len(body["atom_ids"]) > 0 and body["centroid"] is not None:
                cv = body["centroid"]
                cv = cv if isinstance(cv, np.ndarray) else np.array(cv, dtype="float32")
                sem = (float(np.dot(vec, cv)) if cv.shape == vec.shape
                       else float(np.dot(vec, arc.nodes[b["head"]]["vec"])))
            else:
                sem = float(np.dot(vec, arc.nodes[b["head"]]["vec"]))
            bs = arc._ensure_branch_state(bid)
            score = (core.W_SEM * sem
                     + core.W_MOM * float(np.tanh(bs["momentum"] / 5.0))
                     + core.W_ACT * float(bs["activation"])
                     - core.W_ENT * bs["entropy"])
            if score > best_score:
                best_score, best_b = score, bid
        best_o, best_sim = None, -1.0
        for nid in arc.id_order:
            nd = arc.nodes[nid]
            if nd["index"] >= cur_index - core.DEEP_NODE_MIN_AGE or nid in head_ids:
                continue
            s = float(np.dot(vec, nd["vec"]))
            if s > best_sim:
                best_sim, best_o = s, nid
        return best_b, best_o

    def check_mirror(self):
        """Сверка зеркала со словарями — ловит расхождение правил затухания."""
        bad = 0
        for bid, i in self.bid_pos.items():
            bs = self.arc.branch_states.get(bid)
            if bs is None:
                continue
            if (abs(self.mom[i] - bs["momentum"]) > 1e-9
                    or abs(self.act[i] - bs["activation"]) > 1e-9
                    or abs(self.ent[i] - bs["entropy"]) > 1e-9):
                bad += 1
        return bad

    # --- учёт нового узла ---------------------------------------------------
    def note_node(self, node):
        i = len(self.nid_list)
        self.nid_list.append(node["id"])
        self.npos[node["id"]] = i
        self.M = self._grow2(self.M, i + 1)
        self.nidx = self._grow1(self.nidx, i + 1)
        self.M[i] = np.asarray(node["vec"], dtype="float32")
        self.nidx[i] = int(node["index"])
        bid = node.get("branch")
        if bid:
            if bid not in self.bid_pos:
                self.add_branch(bid)
            else:
                self.dirty.add(bid)
            # после атома он и есть голова своей ветки — и в новой ветке,
            # и при continues; прежняя голова головой быть перестаёт
            prev = self.head_of.get(bid)
            if prev is not None:
                self.head_set.discard(prev)
            self.head_of[bid] = node["id"]
            self.head_set.add(node["id"])

    def stats(self):
        return dict(self.stat)


# ---------------------------------------------------------------------------
def install(arc, core, verify: int = 0, eps: float = EPS) -> FastGraft:
    """Поставить быстрый графт на экземпляр ptg_core.Archive.

    arc    — Archive (можно уже загруженный с диска: зеркало поднимется по нему)
    core   — модуль ptg_core (веса и пороги берутся оттуда, не копируются)
    verify — сверять первые N атомов прямым перебором и зеркало со словарями
    """
    fg = FastGraft(arc, core, verify=verify, eps=eps)

    for nid in arc.id_order:
        nd = arc.nodes.get(nid)
        if nd is None or nd.get("vec") is None:
            continue
        fg._ensure_dim(nd["vec"])
        fg.note_node(nd)
    for bid, b in arc.branches.items():
        fg.add_branch(bid)
        h = b.get("head")
        if h:
            fg.head_of[bid] = h
            fg.head_set.add(h)

    orig_add = arc._add_atom
    orig_body = arc._update_branch_body
    orig_states = arc._update_branch_states
    orig_decay = arc._decay_branches
    orig_react = arc._reactivate_branch

    def _update_branch_body(branch_id, new_vec):
        r = orig_body(branch_id, new_vec)
        fg.dirty.add(branch_id)
        return r

    def _update_branch_states(chosen_bid, root_sim):
        orig_states(chosen_bid, root_sim)          # словари — источник истины
        n = len(fg.bid_list)
        if n:
            fg.mom[:n] *= MOM_DECAY
            fg.act[:n] *= ACT_DECAY
        if chosen_bid not in fg.bid_pos:
            fg.add_branch(chosen_bid)
        fg.pull_state(chosen_bid)                  # выбранная — точно из словаря

    def _decay_branches():
        orig_decay()
        n = len(fg.bid_list)
        if not n:
            return
        idle = len(arc.id_order) - fg.seen[:n]
        m = idle >= core.BRANCH_DECAY_IDLE_THRESH
        if m.any():
            fg.act[:n][m] *= IDLE_DECAY
            fg.mom[:n][m] *= IDLE_DECAY

    def _reactivate_branch(branch_id):
        orig_react(branch_id)
        if branch_id not in fg.bid_pos:
            fg.add_branch(branch_id)
        fg.pull_state(branch_id)

    def _add_atom(atom):
        vec = np.asarray(atom["vec"], dtype="float32")
        fg._ensure_dim(vec)
        cur_index = len(arc.id_order)

        head_ids = fg.head_set          # живое множество, не пересборка на атом
        best_b = fg.best_branch(vec)
        best_o = fg.best_old(vec, cur_index, head_ids)

        if fg.verify_left > 0:
            fg.verify_left -= 1
            fg.stat["сверено"] += 1
            bb, bo = fg.brute(vec, cur_index, head_ids)
            if bb != best_b:
                fg.stat["расхождений_ветка"] += 1
                print("[graft_fast] РАСХОЖДЕНИЕ ветка: быстрый %s, перебор %s" % (best_b, bb))
            if bo != best_o:
                fg.stat["расхождений_узел"] += 1
                print("[graft_fast] РАСХОЖДЕНИЕ узел: быстрый %s, перебор %s" % (best_o, bo))
            bad = fg.check_mirror()
            if bad:
                fg.stat["расхождений_состояние"] += bad
                print("[graft_fast] зеркало разошлось со словарями: %d веток" % bad)

        real_branches, real_order = arc.branches, arc.id_order
        arc.branches = _OneBranch(real_branches, best_b)
        arc.id_order = _OneOld(real_order, best_o)
        try:
            node = orig_add(atom)
        finally:
            arc.branches = real_branches
            arc.id_order = real_order

        fg.stat["атомов"] += 1
        fg.note_node(node)
        return node

    arc._path_coherence = fg.path_coherence
    arc._add_atom = _add_atom
    arc._update_branch_body = _update_branch_body
    arc._update_branch_states = _update_branch_states
    arc._decay_branches = _decay_branches
    arc._reactivate_branch = _reactivate_branch
    return fg
