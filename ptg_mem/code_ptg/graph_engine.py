"""
graph_engine.py — мост к НАСТОЯЩЕМУ графовому движку Text PTG.

Это не пересказ архитектуры PTG и не новый формат — это прямой импорт
vendor_ptg/ptg_core.Archive (тот самый класс, что использует Text PTG) и
буквальный вызов его внутреннего метода _add_atom() для семантического
графтинга карточек Code PTG. Единственное, что не переиспользуется —
Archive.build()/scan_folder() и вся Q+A/docx-специфика: она относится к
чтению чат-логов, а не к самому графовому движку, и Code PTG эту часть
никогда бы не вызывал.

Два независимых пути записи в граф, оба пишут в ОДИН И ТОТ ЖЕ
Archive.nodes/edges/branches/branch_states/branch_bodies:

  1. add_concept_atom() — семантический графтинг через эмбеддинг +
     пороги (CONTINUE_THRESH/BRANCH_THRESH/...), ПОЛНОСТЬЮ идентичный
     механизм, что и в Text PTG. Используется для layer_2-интерпретации
     карточки (это "накопленное инженерное знание", тот же тип сущности,
     что чат-атом — короткий текст, который стоит поместить в тот же
     semantic memory graph, что и мысли из Text PTG).

  2. add_structural_edge() — прямое добавление ребра в archive.edges,
     В ОБХОД графтинга. Обоснование: calls/contains/implements — это
     детерминированные факты статического анализа, а не результат
     порогового сравнения эмбеддингов. Применять эмбеддинг-графтинг к
     детерминированному факту было бы категорической ошибкой — PTG сам
     никогда не решает "AST этой функции продолжает AST той" через
     косинусное сходство, и Code PTG не должен изобретать это для
     структурных фактов. Различаться должны только типы карточек и типы
     связей (буквальное требование ТЗ) — здесь это и есть та точка,
     где формат идентичен, а типы рёбер — новые (calls/contains/...).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(os.path.dirname(_THIS_DIR), "vendor_ptg")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

import ptg_core as _ptg_core_module  # noqa: E402
from ptg_core import Archive, Embedder  # noqa: E402

# ---------------------------------------------------------------------
# Единственный монки-патч на модуль: имя служебной директории.
# PTG_DIRNAME задаёт ИМЯ папки (".ptg"), а не формат её содержимого — формат
# (nodes.json/edges.json/branch_state.json/branch_bodies.json) остаётся
# тем же самым файлом того же самого Archive.save(). Переименование папки
# нужно только чтобы не перепутать архив Code PTG с архивом Text PTG на
# одном диске — это не "новый формат", это имя каталога.
# ---------------------------------------------------------------------
_ptg_core_module.PTG_DIRNAME = ".code_ptg"

# ---------------------------------------------------------------------
# Второй патч на модуль: векторы НЕ отображаются в память.
#
# `load_if_exists()` открывает `faiss.index.npy` через `np.load(mmap_mode="r")`,
# если файл больше MMAP_THRESHOLD_BYTES (50 МБ). Процесс держит файл
# отображённым весь сеанс, и следующее `save()` не может его перезаписать:
# Windows отвечает ERROR_USER_MAPPED_FILE, в Python это OSError Errno 22.
#
# Не гипотеза: на этом и упала сборка 21.09.2026, когда индекс кодового графа
# перевалил за 50 МБ (2048 измерений вместо 384) — второй проход по
# fork-tree/substrate загрузил архив, отобразил индекс и умер на сохранении.
# Ровно ту же ловушку в текстовом пути уже лечат `run_text_ptg_v3.py` и
# `mcp_launch.py`, и лечат так же — поднятием порога.
# ---------------------------------------------------------------------
_ptg_core_module.MMAP_THRESHOLD_BYTES = 1 << 62

# ---------------------------------------------------------------------
# Третий патч: центроиды ветвей — записями фиксированной длины, не текстом.
#
# `_save_unlocked()` кладёт центроид каждой ветки в `branch_bodies.json`
# списком чисел, да ещё с `indent=2`. Замер 21.09.2026 на графе из 8436 узлов:
# файл **403.6 МБ** из 526 МБ всего хранилища. Для распространения это
# приговор — никто не потянет полгигабайта индекса на 749 файлов кода.
#
# То же самое уже измерено на текстовом архиве (PTG_V4/index_pack.py):
# центроиды текстом — 498 МБ и 16 с на чтение, те же числа в .npy — 67 МБ и
# 0.012 с. Разница не в формате как вкусе: JSON хранит каждое число как
# ~20 символов вместо 4 байт и требует разбора, а .npy — это заголовок и
# сплошной блок памяти.
#
# Здесь то же самое сделано не отдельным пакетом рядом, а в самом хранилище:
# перед сохранением центроиды вынимаются из тел, пишутся одной матрицей в
# `branch_centroids.npy` (порядок — в `branch_centroid_ids.json`), а в JSON
# уходит пустой список. При загрузке они возвращаются на место.
#
# Обратная совместимость в обе стороны: нет .npy — центроиды читаются из
# JSON, как раньше; ветка без центроида (тело ещё не построено) в матрицу не
# попадает и остаётся `None`.
#
# ВНИМАНИЕ: заглушка — пустой СПИСОК, а не None. `np.array(None,
# dtype="float32")` не падает, а молча возвращает `array(nan)`, и такой
# центроид отравил бы все косинусы ветки, ничего не сообщив.
# ---------------------------------------------------------------------
_CENTROIDS_NPY = "branch_centroids.npy"
_CENTROID_IDS = "branch_centroid_ids.json"


def _centroid_dim(bodies) -> int:
    for b in bodies.values():
        c = b.get("centroid")
        if c is not None and getattr(c, "size", len(c) if hasattr(c, "__len__") else 0):
            return int(np.asarray(c).shape[0])
    return 0


def _save_unlocked_packed(self):
    bodies = getattr(self, "branch_bodies", None) or {}
    dim = _centroid_dim(bodies)
    if not dim:
        return _ORIG_SAVE_UNLOCKED(self)

    ids, rows, stash = [], [], {}
    for bid, body in bodies.items():
        c = body.get("centroid")
        if c is None:
            continue
        arr = np.asarray(c, dtype="float32")
        if arr.shape != (dim,):
            continue
        ids.append(bid)
        rows.append(arr)
        stash[bid] = c
        body["centroid"] = []          # см. предупреждение выше: НЕ None
    try:
        if rows:
            os.makedirs(self.ptg_dir, exist_ok=True)
            np.save(os.path.join(self.ptg_dir, _CENTROIDS_NPY),
                    np.stack(rows).astype("float32"))
            with open(os.path.join(self.ptg_dir, _CENTROID_IDS), "w", encoding="utf-8") as f:
                json.dump(ids, f, ensure_ascii=False)
        return _ORIG_SAVE_UNLOCKED(self)
    finally:
        for bid, c in stash.items():
            bodies[bid]["centroid"] = c


def _load_if_exists_packed(self):
    ok = _ORIG_LOAD_IF_EXISTS(self)
    if not ok:
        return ok
    npy = os.path.join(self.ptg_dir, _CENTROIDS_NPY)
    idsf = os.path.join(self.ptg_dir, _CENTROID_IDS)
    if not (os.path.exists(npy) and os.path.exists(idsf)):
        return ok                      # старое хранилище — центроиды уже в JSON
    mat = np.load(npy)
    with open(idsf, encoding="utf-8") as f:
        ids = json.load(f)
    for i, bid in enumerate(ids):
        body = self.branch_bodies.get(bid)
        if body is not None and i < mat.shape[0]:
            body["centroid"] = mat[i]
    # ветки без центроида в матрице: оригинал мог оставить пустой массив
    for body in self.branch_bodies.values():
        c = body.get("centroid")
        if c is not None and getattr(c, "size", 1) == 0:
            body["centroid"] = None
    return ok


_ORIG_SAVE_UNLOCKED = Archive._save_unlocked
_ORIG_LOAD_IF_EXISTS = Archive.load_if_exists
Archive._save_unlocked = _save_unlocked_packed
Archive.load_if_exists = _load_if_exists_packed

CODE_DOMAIN = "code"


class GraphEngine:
    """
    Тонкая обёртка вокруг настоящего Archive. Archive ничего не знает о
    том, что теперь в него графтятся код-карточки, а не чат-атомы — с его
    точки зрения это просто атомы с другим набором domain-полей.
    """

    def __init__(self, project_folder: str, output_dir: Optional[str] = None,
                 use_embeddings: bool = True, embedder=None):
        self.archive = Archive(folder=project_folder, output_dir=output_dir)
        self.use_embeddings = use_embeddings
        self.embedder = None
        self.embeddings_available = False
        if use_embeddings:
            # Порядок выбора: переданный извне -> внутрипроцессный (если задана
            # модель) -> LM Studio. Внутрипроцессный предпочтительнее: сервер
            # моделей на localhost — единая точка отказа, и для распространения
            # требовать его установки нельзя. См. local_embedder.py.
            if embedder is not None:
                self.embedder = embedder
            elif os.environ.get("CODE_PTG_MODEL"):
                from .local_embedder import LocalEmbedder
                self.embedder = LocalEmbedder()
            else:
                self.embedder = Embedder()
            try:
                self.embeddings_available = self.embedder.test_connection()
            except Exception:
                self.embeddings_available = False
        if not self.embeddings_available:
            where = getattr(self.embedder, "backend", "lm-studio")
            self.archive.log(
                f"Эмбеддер ({where}) недоступен или use_embeddings=False — "
                "семантический графтинг concept-атомов отключён. Структурные "
                "рёбра (contains/calls/implements/...) при этом строятся как "
                "обычно, т.к. они не требуют эмбеддингов вовсе "
                "(см. add_structural_edge)."
            )

    # ---------- путь 1: семантический графтинг (нужен эмбеддинг) ----------

    def add_concept_atom(self, atom_id: str, text: str, file_path: str,
                          timestamp: Optional[float] = None, extra: Optional[dict] = None) -> Optional[dict]:
        """
        Графтит текст (интерпретацию карточки) в общий semantic memory graph
        ЧЕРЕЗ РЕАЛЬНЫЙ Archive._add_atom — тот же код, что использует Text PTG.
        Возвращает построенный node dict или None, если эмбеддинги недоступны
        (в этом случае вызывающий код должен использовать add_isolated_node).
        """
        if not self.embeddings_available:
            return None
        import numpy as np
        raw_vec = self.embedder.embed([text])[0]
        norm = float(np.linalg.norm(raw_vec))
        vec = raw_vec / norm if norm > 0 else raw_vec
        atom = {
            "id": atom_id,
            "file": file_path,
            "order": len(self.archive.id_order),
            "text": text,
            "question": "", "answer": text,
            "confidence": "structural",
            "timestamp": timestamp or time.time(),
            "vec": vec,
        }
        if extra:
            atom.update(extra)
        node = self.archive._add_atom(atom)
        node["domain"] = CODE_DOMAIN
        return node

    # ---------- путь 1b: изолированный узел без эмбеддинга ----------

    def add_isolated_node(self, atom_id: str, text: str, file_path: str,
                           timestamp: Optional[float] = None) -> dict:
        """
        Fallback, когда эмбеддинги недоступны: узел получает собственную
        тривиальную ветку (root=head=сам себя), БЕЗ попытки сравнить с
        существующими ветками через несуществующий вектор. Это честная
        деградация, а не "графтинг с фальшивым вектором" — сравнивать
        по вектору, которого нет, было бы хуже, чем не сравнивать вовсе.
        """
        import numpy as np
        cur_index = len(self.archive.id_order)
        dim = self.embedder.dim if (self.embedder and self.embedder.dim) else 384
        # Нулевой вектор — НЕ "фальшивая семантика": cosine(0, что угодно) = 0,
        # то есть ниже BRANCH_THRESH автоматически — граф честно трактует такой
        # узел как несравнимый ни с чем, той же математикой, что и обычный
        # графтинг, без специальных исключений. Но формат хранения (см.
        # Archive.save()/load_if_exists(): единый (N, dim) np.array) требует
        # РЕАЛЬНОГО вектора фиксированной размерности у любого узла — None
        # ломает сериализацию (проверено: np.array([None,...]) схлопывается
        # в 1-D массив NaN, load_if_exists() падает на vecs.shape[1]).
        node = {
            "id": atom_id, "file": file_path, "order": cur_index,
            "question": "", "answer": text, "text": text,
            "confidence": "structural", "timestamp": timestamp or time.time(),
            "depth": 0, "parent": None, "index": cur_index,
            "status": "active", "vec": np.zeros(dim, dtype="float32"),
            "root_similarity": 0.0, "off_root": True,
            "reinforcement_count": 0, "reinforcement_links": [], "repetition_epochs": [],
            "placement_case": "NO_EMBEDDER", "lateral_expansion": False,
            "branch": atom_id, "semantic_epoch": 0,
            "path_coherence": 0.0, "short_path_coherence": 0.0, "long_path_coherence": 0.0,
            "domain": CODE_DOMAIN,
        }
        self.archive.nodes[atom_id] = node
        self.archive.id_order.append(atom_id)
        self.archive.branches[atom_id] = {
            "root": atom_id, "head": atom_id, "created_order": cur_index, "semantic_epoch": 0,
        }
        self.archive._ensure_branch_state(atom_id)
        self.archive._ensure_branch_body(atom_id)
        return node

    def graft_or_isolate(self, atom_id: str, text: str, file_path: str,
                          timestamp: Optional[float] = None) -> dict:
        node = self.add_concept_atom(atom_id, text, file_path, timestamp)
        if node is None:
            node = self.add_isolated_node(atom_id, text, file_path, timestamp)
        return node

    # ---------- путь 2: структурные рёбра (детерминированные факты) ----------

    def add_structural_edge(self, src_id: str, dst_id: str, edge_type: str) -> None:
        """
        Прямая запись в archive.edges, БЕЗ графтинга. edge_type — любой из
        Code PTG-специфичных типов: contains/calls/implements/tests/imports.
        Формат ребра {"from","to","type"} идентичен формату Text PTG рёбер —
        различается только словарь допустимых type.
        """
        self.archive.edges.append({"from": src_id, "to": dst_id, "type": edge_type})

    def mark_superseded(self, old_node_id: str) -> None:
        """Буквально то же самое присвоение, что PTG делает при ребре
        supersedes (ptg_core.py: self.nodes[target_id]["status"] = "superseded")."""
        if old_node_id in self.archive.nodes:
            self.archive.nodes[old_node_id]["status"] = "superseded"

    def outgoing(self, node_id: str) -> list[dict]:
        return [e for e in self.archive.edges if e["from"] == node_id]

    def incoming(self, node_id: str) -> list[dict]:
        return [e for e in self.archive.edges if e["to"] == node_id]

    def clear_outgoing(self, node_id: str) -> None:
        self.archive.edges = [e for e in self.archive.edges if e["from"] != node_id]

    # ---------- сохранение/загрузка — буквально Archive.save()/load_if_exists() ----------

    def save(self) -> None:
        self.archive.save()

    def load(self) -> bool:
        return self.archive.load_if_exists()
