# -*- coding: utf-8 -*-
r"""PTG_V4/metatree.py — метадрево над несколькими движками эмбеддинга.

ЗАЧЕМ. `meta_tree.json` внутри архива — это дерево ОДНОГО архива: порядок узлов,
соответствие вектор↔атом, модель, которой он построен. Смена модели по правилам
движка означает новый архив с нуля (ptg_core останавливает сборку: «архив уже
построен моделью X» — и останавливает правильно, косинусы несравнимы, а пороги
графтинга откалиброваны под конкретное их распределение). Из-за этого каждая
новая модель стоила полной пересборки, а старая работа выбрасывалась.

Метадрево здесь — слой НАД архивами: оно хранит то, что от движка не зависит, и
реестр того, что зависит. Тогда новая модель добавляется слоем, а не заменой.

РАЗДЕЛЕНИЕ, НА КОТОРОМ ВСЁ ДЕРЖИТСЯ

  от движка НЕ зависит          от движка зависит
  --------------------          -----------------
  текст атома и его тождество   вектор
  происхождение (файл, время,   индекс
    сессия, смещение)           назначение ветки
  файловый граф                 пороги графтинга
  утверждение ребра как факт    слой близости (какие пары стали рёбрами)
    «A уточняет B»              значение косинуса

ПРАВИЛА ВЫРАЩИВАНИЯ (обоснование каждого — в METATREE-ENGINES.md)

  П1. Тождество атома — по содержимому: sha1 нормализованного текста плюс путь
      источника. Идентификаторы узлов у разных движков разные и для стыковки
      непригодны.
  П2. Структурный владелец — ровно один движок. Ветки и типизированные рёбра
      канонической структуры берутся у него; остальные движки дают слои.
      Иначе структура оказывается откалибрована ни подо что.
  П3. Объединять можно слои БЛИЗОСТИ (continues, reinforces, refines, fixes,
      returns_to) и нельзя слои ПОРЯДКА (supersedes, contradicts): перенос
      свойств требует монотонности, а отношения порядка и отрицания не
      монотонны (PTG-SANDWICH-2026-09-20.md §4.4).
  П4. Объединение слоёв близости монотонно: рёбра только добавляются, значит
      средняя степень растёт и условие режима d >> ln n становится ДОСТИЖИМЕЕ.
  П5. Поиск по нескольким движкам — слияние РАНГОВ, не оценок. Косинусы из
      разных пространств несравнимы; RRF не требует никакой калибровки.
  П6. Общие координаты — только через измеренную привязку. Якорь = один и тот
      же текст, посчитанный обоими движками; преобразование ортогональное
      (Прокруст, сохраняет косинусную геометрию); качество меряется на
      ОТЛОЖЕННЫХ якорях и проходит ворота, иначе проекция не применяется.
  П7. Бюджет связей считается на числе УЗЛОВ объединения, а не файлов:
      d = ceil(3 * ln n).
  П8. Реестр движков обязателен: модель, dim, пороги, дата, хеш корпуса. Без
      него «какой моделью это построено» невосстановимо — ровно тот инцидент,
      ради которого в ptg_core появился ПАТЧ 18.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict, field

import numpy as np

PROXIMITY_TYPES = ("continues", "reinforces", "refines", "fixes", "returns_to")
ORDER_TYPES = ("supersedes", "contradicts")
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# П1 — тождество атома
# ---------------------------------------------------------------------------
def atom_key(text: str, source: str = "") -> str:
    """Ключ стыковки атомов между движками.

    Нормализация та же, что у манифеста инкремента: переводы строк и кратные
    пробелы не считаются правкой. Путь источника входит в ключ, потому что один
    и тот же абзац в двух разных файлах — это два разных атома с разным
    происхождением, и склеивать их нельзя.
    """
    norm = _WS.sub(" ", (text or "").strip())
    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
    return h if not source else hashlib.sha1((h + "\x00" + source).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# П8 — реестр движков
# ---------------------------------------------------------------------------
@dataclass
class EngineSpec:
    name: str                      # короткое имя слоя, оно же имя файла векторов
    model_id: str                  # ровно та строка, которой подписан архив
    dim: int
    store: str = ""                # путь к .ptg, если движок — владелец архива
    vectors: str = ""              # .npy с векторами в порядке ключей (см. П1)
    structural: bool = False       # П2: владелец веток и типизированных рёбер
    thresholds: dict = field(default_factory=dict)
    built_at: str = ""
    corpus_manifest: str = ""      # хеш манифеста корпуса, на котором построен

    def check(self) -> list[str]:
        bad = []
        if not self.model_id:
            bad.append("нет model_id — движок неопознаваем (П8)")
        if self.dim <= 0:
            bad.append("нет dim")
        if self.vectors and not os.path.exists(self.vectors):
            bad.append("нет файла векторов %s" % self.vectors)
        return bad


# ---------------------------------------------------------------------------
class MetaTree:
    """Нейтральный к движку слой: атомы, происхождение, слои рёбер, реестр."""

    def __init__(self):
        self.keys: list[str] = []                  # порядок = порядок строк в .npy
        self.pos: dict[str, int] = {}
        self.atoms: dict[str, dict] = {}           # key -> {text, file, timestamp, ...}
        self.engines: dict[str, EngineSpec] = {}
        self.vectors: dict[str, np.ndarray] = {}   # имя движка -> матрица (n x dim)
        self.proximity: dict[str, list] = {}       # имя движка -> [(k1, k2, sim)]
        self.order_layer: list[dict] = []          # П3: хранится с провенансом
        self.structural: str | None = None

    # --- наполнение -------------------------------------------------------
    def add_atom(self, text, source="", **meta) -> str:
        k = atom_key(text, source)
        if k not in self.pos:
            self.pos[k] = len(self.keys)
            self.keys.append(k)
            self.atoms[k] = {"text": text, "file": source, **meta}
        else:
            self.atoms[k].update(meta)
        return k

    def register(self, spec: EngineSpec, vectors: np.ndarray | None = None):
        bad = spec.check()
        if bad:
            raise ValueError("движок %s не годится: %s" % (spec.name, "; ".join(bad)))
        if spec.structural:
            if self.structural and self.structural != spec.name:
                raise ValueError("П2: структурный владелец уже есть — %s" % self.structural)
            self.structural = spec.name
        self.engines[spec.name] = spec
        if vectors is not None:
            if vectors.shape[0] != len(self.keys):
                raise ValueError("векторов %d при %d атомах" % (vectors.shape[0], len(self.keys)))
            if vectors.shape[1] != spec.dim:
                raise ValueError("dim %d против заявленного %d" % (vectors.shape[1], spec.dim))
            v = vectors.astype("float32", copy=True)
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            self.vectors[spec.name] = v

    def add_order_edge(self, k_from: str, k_to: str, etype: str, engine: str):
        """П3: supersedes/contradicts кладутся с указанием, КТО это утверждает."""
        if etype not in ORDER_TYPES:
            raise ValueError("%s — не отношение порядка" % etype)
        self.order_layer.append({"from": k_from, "to": k_to, "type": etype,
                                 "engine": engine})

    # --- П7 ---------------------------------------------------------------
    def budget(self, margin: float = 3.0) -> int:
        n = len(self.keys)
        return 1 if n <= 2 else max(2, int(math.ceil(margin * math.log(n))))

    # --- слой близости одного движка -------------------------------------
    def build_proximity(self, engine: str, budget: int | None = None,
                        per_node: int = 64, block: int = 1024):
        """Кандидаты — top-k по косинусу, отбор — правилом с бюджетом.

        Считается блоками: матрица сходства на 27 тысячах атомов целиком это
        3 ГБ, а блоками по 1024 строки — 100 МБ.
        """
        import sandwich_graft
        V = self.vectors[engine]
        n = V.shape[0]
        budget = budget or self.budget()
        k = min(per_node, n - 1)
        cand = []
        for s in range(0, n, block):
            e = min(n, s + block)
            S = V[s:e] @ V.T
            for r in range(e - s):
                S[r, s + r] = -np.inf
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            for r in range(e - s):
                i = s + r
                for j in idx[r]:
                    j = int(j)
                    if i == j:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    cand.append((self.keys[a], self.keys[b], float(S[r, j])))
        cand = sorted(set(cand), key=lambda t: -t[2])
        self.proximity[engine] = sandwich_graft.graft(cand, budget=budget)
        return self.proximity[engine]

    # --- П4 ---------------------------------------------------------------
    def fuse_proximity(self, engines: list[str] | None = None,
                       budget: int | None = None, per_node: int = 64,
                       rrf_k: int = 60, block: int = 1024) -> list:
        """ПРАВИЛЬНОЕ слияние слоёв: сливаются ПРЕДЛОЖЕНИЯ, а не принятые рёбра.

        Почему не union принятых. У каждого движка слой уже упёрт в бюджет d, и
        объединение даёт среднюю степень до 2d, а потолок до 4d — тот самый
        хабовый перекос, против которого бюджет и вводился. Выигрыш по покрытию
        оплачивался бы потерей контроля над степенью.

        Почему по рангам. Общий порядок пар нужен раскладке (`graft` ожидает
        убывание близости), а косинусы двух пространств несравнимы: 0.62 у
        одного и 0.62 у другого — разные события. Ранг пары внутри своего
        движка сравним, поэтому порядки сливаются по RRF, и по объединённому
        порядку проходит ОДНА раскладка с тем же бюджетом d.

        Выигрыш тогда берётся не из роста степени, а из независимости ошибок:
        пару, которую один движок не предложил, может предложить другой — при
        равном бюджете и равном потолке.
        """
        import sandwich_graft
        engines = engines or [e for e in self.engines if e in self.vectors]
        budget = budget or self.budget()
        rank: dict[tuple, dict[str, int]] = {}
        for name in engines:
            prop = self._propose(name, per_node=per_node, block=block)
            for r, (u, v, _s) in enumerate(prop):
                key = (u, v) if u < v else (v, u)
                rank.setdefault(key, {})[name] = r
        fused = []
        for key, rk in rank.items():
            score = sum(1.0 / (rrf_k + r + 1) for r in rk.values())
            fused.append((key[0], key[1], score))
        fused.sort(key=lambda t: -t[2])
        taken = sandwich_graft.graft(fused, budget=budget)
        self.proximity["__fused__"] = taken
        return taken

    def _propose(self, engine: str, per_node: int = 64, block: int = 1024) -> list:
        """Предложения одного движка: пары top-k, отсортированные по его косинусу."""
        V = self.vectors[engine]
        n = V.shape[0]
        k = min(per_node, n - 1)
        out = []
        for s in range(0, n, block):
            e = min(n, s + block)
            S = V[s:e] @ V.T
            for r in range(e - s):
                S[r, s + r] = -np.inf
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            for r in range(e - s):
                i = s + r
                for j in idx[r]:
                    j = int(j)
                    if i == j:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    out.append((self.keys[a], self.keys[b], float(S[r, j])))
        out = list({(u, v): (u, v, s) for u, v, s in out}.values())
        out.sort(key=lambda t: -t[2])
        return out

    def union_proximity(self, engines: list[str] | None = None) -> list:
        """Сырое объединение ПРИНЯТЫХ рёбер — для сравнения с fuse_proximity.

        Держится в модуле именно как то, чего делать не следует: степень
        выходит за бюджет. Полезно только как верхняя граница покрытия.
        """
        engines = engines or list(self.proximity)
        best: dict[tuple, list] = {}
        for name in engines:
            for u, v, s in self.proximity.get(name, ()):
                key = (u, v) if u < v else (v, u)
                prev = best.get(key)
                if prev is None:
                    best[key] = [s, {name}]
                else:
                    if s > prev[0]:
                        prev[0] = s          # сходство — максимум по движкам,
                    prev[1].add(name)        # источники — объединение, не замена
        return [(u, v, s, sorted(src)) for (u, v), (s, src) in best.items()]

    def layer_stats(self, edges) -> dict:
        deg = dict.fromkeys(self.keys, 0)
        simple = set()
        for e in edges:
            u, v = e[0], e[1]
            key = (u, v) if u < v else (v, u)
            if key in simple:
                continue
            simple.add(key)
            deg[u] += 1
            deg[v] += 1
        n = len(self.keys)
        c = 2.0 * len(simple) / max(1, n)
        iso = sum(1 for d in deg.values() if d == 0)
        ln_n = math.log(n) if n > 1 else 0.0
        return {"узлов": n, "рёбер": len(simple), "средняя_степень": round(c, 2),
                "ln_n": round(ln_n, 2), "в_режиме": bool(c >= 3 * ln_n),
                "изолированных": iso,
                "доля_изолированных_%": round(100.0 * iso / max(1, n), 2),
                "нулевая_модель_%": round(100.0 * math.exp(-c), 2),
                "максимум_степени": max(deg.values()) if deg else 0,
                "глубина_обхода": (1 if c <= 1 else
                                   max(1, int(math.ceil(ln_n / math.log(c)))))}

    # --- П5 ---------------------------------------------------------------
    def search(self, query_vecs: dict[str, np.ndarray], top_k: int = 15,
               rrf_k: int = 60) -> list[tuple[str, float, dict]]:
        """Слияние рангов по движкам (RRF): score = sum 1/(k + rank).

        query_vecs — вектор запроса ОТДЕЛЬНО для каждого движка, посчитанный
        его же моделью. Никакой калибровки между пространствами не требуется:
        складываются ранги, а не косинусы.
        """
        ranks: dict[str, dict[str, int]] = {}
        for name, qv in query_vecs.items():
            V = self.vectors.get(name)
            if V is None:
                continue
            q = np.asarray(qv, dtype="float32")
            q /= max(float(np.linalg.norm(q)), 1e-9)
            sims = V @ q
            order = np.argsort(-sims)
            ranks[name] = {self.keys[int(i)]: r for r, i in enumerate(order[:max(top_k * 10, 100)])}
        fused: dict[str, float] = {}
        where: dict[str, dict] = {}
        for name, rk in ranks.items():
            for key, r in rk.items():
                fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + r + 1)
                where.setdefault(key, {})[name] = r + 1
        out = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [(k, round(s, 6), where[k]) for k, s in out]

    # --- П6 ---------------------------------------------------------------
    def align(self, src: str, dst: str, train_frac: float = 0.7,
              gate_cos: float = 0.80, gate_corr: float = 0.90, seed: int = 11) -> dict:
        """Ортогональная привязка src -> dst по якорям с проверкой на отложенных.

        Возвращает отчёт и, если ворота пройдены, матрицу W. Не пройдены —
        матрицы нет: проекция без подтверждённого качества хуже её отсутствия,
        потому что молча портит выдачу.
        """
        A, B = self.vectors[dst], self.vectors[src]
        n = min(len(A), len(B))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        cut = int(train_frac * n)
        tr, te = perm[:cut], perm[cut:]
        U, _s, Vt = np.linalg.svd(B[tr].T @ A[tr], full_matrices=False)
        W = U @ Vt
        P = B[te] @ W
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        cos = float(np.einsum("ij,ij->i", P, A[te]).mean())
        Sa, Sb = A[te] @ A[te].T, P @ P.T
        iu = np.triu_indices(len(te), 1)
        corr = float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])
        ok = cos >= gate_cos and corr >= gate_corr
        rep = {"якорей_обучение": int(cut), "якорей_проверка": int(len(te)),
               "косинус_к_цели": round(cos, 4), "корреляция_сходств": round(corr, 4),
               "ворота": {"косинус": gate_cos, "корреляция": gate_corr},
               "пройдено": ok}
        return {"отчёт": rep, "W": W if ok else None}

    # --- хранение ---------------------------------------------------------
    def save(self, path: str):
        """Векторы — в .npy, а не в JSON: центроиды текстом это 498 МБ и 16 с
        против 67 МБ и 0.012 с записями фиксированной длины."""
        os.makedirs(path, exist_ok=True)
        for name, V in self.vectors.items():
            np.save(os.path.join(path, "vec_%s.npy" % name), V)
        head = {
            "версия": 1,
            "создано": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "структурный_движок": self.structural,
            "ключи": self.keys,
            "атомы": self.atoms,
            "движки": {k: asdict(v) for k, v in self.engines.items()},
            "слой_порядка": self.order_layer,
        }
        with io.open(os.path.join(path, "metatree.json"), "w", encoding="utf-8") as f:
            json.dump(head, f, ensure_ascii=False)
        for name, edges in self.proximity.items():
            with io.open(os.path.join(path, "prox_%s.json" % name), "w", encoding="utf-8") as f:
                json.dump([{"from": u, "to": v, "sim": round(s, 5)} for u, v, s in edges],
                          f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "MetaTree":
        mt = cls()
        with io.open(os.path.join(path, "metatree.json"), encoding="utf-8") as f:
            head = json.load(f)
        mt.keys = head["ключи"]
        mt.pos = {k: i for i, k in enumerate(mt.keys)}
        mt.atoms = head["атомы"]
        mt.structural = head.get("структурный_движок")
        mt.order_layer = head.get("слой_порядка", [])
        for name, d in head.get("движки", {}).items():
            mt.engines[name] = EngineSpec(**d)
            p = os.path.join(path, "vec_%s.npy" % name)
            if os.path.exists(p):
                mt.vectors[name] = np.load(p)
            pp = os.path.join(path, "prox_%s.json" % name)
            if os.path.exists(pp):
                with io.open(pp, encoding="utf-8") as f:
                    mt.proximity[name] = [(e["from"], e["to"], e["sim"]) for e in json.load(f)]
        return mt
