"""
ptg_core.py — основная логика Personal Thought Graph (Личного графа мыслей).

ВЕРСИЯ ПОСЛЕ ПАТЧЕЙ 1-7.
Это не переписывание с нуля — патч поверх прежней архитектуры.
Хранилище по-прежнему: плоский JSON + векторный индекс в .ptg/.
Без сервера, без CLI, без базы данных, без NetworkX/PySide6
(сохранена существующая связка CTk + tkinter.ttk + словари/JSON —
смена GUI-фреймворка и графовой библиотеки была бы редизайном,
а не патчем).

Что изменилось относительно предыдущей версии:

ПАТЧ 1 — атом теперь = ОДИН вопрос + ОДИН ответ (Q+A), а не абзац.
ПАТЧ 2 — эмбеддинги только через LM Studio (http://localhost:1234),
          без sentence-transformers и без hash-фолбэка. Если LM Studio
          недоступен — построение архива блокируется.
ПАТЧ 3 — перед обработкой чат-логов сканируются README/ARCHITECTURE/
          INSTALL/docs/*.md и строится root_project.json — семантический
          корень, к которому крепится всё дерево.
ПАТЧ 4 — каждый файл становится семантическим объектом: центроид
          эмбеддингов его атомов, главные ветки, доминирующие концепты.
          Это file-level meta-tree для навигации между файлами.
ПАТЧ 5 — добавлены типы рёбер: contradicts, fixes, refines, supersedes
          (старые continues/branches/returns_to сохранены).
ПАТЧ 7 — инкрементальное прививание (append-only, mtime-проверки)
          сохранено без изменений; узлы при supersedes помечаются
          статусом, но никогда не удаляются и не перезаписываются.

Конвейер для одного атома (Q+A пара):
    1. эмбеддинг через LM Studio
    2. сравнение с "головой" каждой активной ветки
    3. сравнение со старыми (не "голова") узлами
    4. решение: continues / branches / returns_to
       + дополнительно: contradicts / fixes / refines / supersedes

Пороги (косинусное сходство, векторы нормализованы):
    > 0.78  -> continues
    < 0.60  -> branches (новая ветка)
    > 0.83 к старому узлу -> returns_to (дополнительно к основному)
"""

import os
import re
import json
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
import hashlib
import datetime as _dt
import numpy as np
import requests

from ptg_py_comments import extract_py_comments_text
from ptg_chat_exports import extract_chat_pairs, extract_chat_text_stream
from ptg_logging import setup_logging, get_logger

_log = get_logger("core")

try:
    import olefile  # для .doc (legacy OLE-формат) — см. _extract_doc_text
    _HAS_OLEFILE = True
except ImportError:
    _HAS_OLEFILE = False

# ---------------------------------------------------------------------------
# ПАТЧ 2 — настройки LM Studio
# ---------------------------------------------------------------------------
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
# Точное имя модели зависит от того, что загружено в LM Studio (см. вкладку
# Local Server). Если это имя не совпадёт со списком из GET /v1/models,
# Embedder автоматически возьмёт первую доступную модель из списка.
LM_STUDIO_EMBED_MODEL = "qwen3.5-embedding"
LM_STUDIO_TIMEOUT_CONNECT = 3
# ПАТЧ 18, фикс бага #1: раньше _probe_embed() использовал
# LM_STUDIO_TIMEOUT_CONNECT (3 сек) — рассчитан на лёгкий GET /v1/models,
# а не на реальный POST /v1/embeddings. Если GPU занят предыдущим большим
# батчем (например, эмбеддинг длинных текстов структуры проекта), пробный
# запрос может встать в очередь дольше 3 сек и словить ложный timeout —
# из-за чего код ошибочно считал рабочую модель нерабочей и переключался
# на другую (в реальном логе именно так — рабочая text-embedding-qwen3-
# embedding-4b была отброшена, LM Studio выгрузил её из VRAM и загрузил
# nomic-embed-text-v1.5 ВМЕСТО неё, посреди уже строящегося архива).
LM_STUDIO_TIMEOUT_PROBE = 30
LM_STUDIO_TIMEOUT_EMBED = 120
# ПАТЧ 20, фикс реального инцидента: батч из 17 текстов структуры проекта
# (до 6000 симв. каждый) реально обрабатывался LM Studio ~121 секунду на
# GTX 1660 Super 6ГБ (модель+KV-кэш+compute buffer требуют ~8.4ГБ — больше,
# чем есть VRAM, отсюда медленная обработка) — наш плоский таймаут 120с
# оборвал соединение буквально за секунду до реального завершения запроса
# на сервере. Timeout должен МАСШТАБИРОВАТЬСЯ с объёмом батча, а не быть
# одним числом на любой размер запроса — см. Embedder.embed()/_embed_timeout_for().
EMBED_TIMEOUT_BASE = LM_STUDIO_TIMEOUT_EMBED   # минимум — как раньше, для маленьких батчей
EMBED_TIMEOUT_PER_1K_CHARS = 5                  # доп. секунд на каждую 1000 симв. суммарного батча
EMBED_TIMEOUT_MAX = 1800                        # потолок 30 минут — не ждать буквально бесконечно

# ---------------------------------------------------------------------------
# Пороги решений по рёбрам
# ---------------------------------------------------------------------------
CONTINUE_THRESH = 0.78
BRANCH_THRESH = 0.60
RETURN_THRESH = 0.83
DEEP_NODE_MIN_AGE = 5

CONTRADICT_SIM_THRESH = 0.75
SUPERSEDE_SIM_THRESH = 0.70
REFINE_MIN_LEN_RATIO = 1.3

FIX_KEYWORDS = ("fix", "correct", "update", "wrong", "mistake", "actually", "revise", "oops", "bug",
                "исправ", "ошиб", "поправ", "на самом деле", "опечат", "баг", "неверн")

NEGATION_MARKERS = ("on the contrary", "actually no", "that's wrong", "incorrect", "i disagree", "not true",
                     "however", "but actually", "instead of that", "this is false",
                     "наоборот", "это неверно", "не так", "ошибаюсь", "однако", "не согласен",
                     "неправильно", "это не так", "вообще-то нет",
                     "на самом деле нет", "не подходит", "не верно", "нет, ", "не стоит")

SUPERSEDE_MARKERS = ("instead of", "replace with", "deprecated", "no longer use", "superseded by",
                      "use this instead", "replaced by",
                      "вместо", "замени", "устарел", "больше не используй", "заменено на", "взамен")

# ---------------------------------------------------------------------------
# ПАТЧ 1 — речевые метки для эвристического Q+A-парсера
# ---------------------------------------------------------------------------
_Q_LABEL_RE = re.compile(
    r'^[#>\*\s]{0,6}(you said|user|human|question|q|ты|вы сказали|пользователь|вопрос)\s*:',
    re.IGNORECASE,
)
_A_LABEL_RE = re.compile(
    r'^[#>\*\s]{0,6}(chatgpt said|claude said|assistant|ai|bot|chatgpt|gpt|claude|answer|a|'
    r'ассистент|клод|ответ)\s*:',
    re.IGNORECASE,
)

TEXT_EXTS = {
    # -- документы Word (ОБЯЗАТЕЛЬНО — извлекаются через отдельные
    # экстракторы _extract_docx_text/_extract_doc_text, НЕ через
    # _read_text_file: это составные бинарные форматы, не plain text) --
    ".docx", ".doc",
    # -- ПАТЧ 16/17: .py — ТОЛЬКО докстринги и #-комментарии, через
    # ptg_py_comments.extract_py_comments_text() (вынесено в отдельный
    # модуль). Раньше .py either
    # отсутствовал в TEXT_EXTS, либо (в самой первой версии) читался как
    # обычный текст и рубился sequential-фолбэком на бессмысленные пары
    # "вопрос/ответ" прямо из кода — это порождало мусорные векторы.
    # Теперь код НИКОГДА не попадает в атом — только человеческие мысли,
    # оставленные в комментариях/докстрингах.
    ".py",
    # -- простой текст / заметки / документация --
    ".txt", ".md", ".markdown", ".mdx", ".rst", ".org", ".tex", ".adoc", ".asciidoc",
    ".text", ".me", ".nfo",
    # -- логи и переписка --
    ".log", ".eml", ".chatlog",
    # -- структурированные данные --
    ".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".xml", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".env", ".properties",
    # -- веб/разметка (как текст, без рендеринга) --
    ".html", ".htm", ".xhtml", ".svg", ".css", ".scss", ".less",
    # -- субтитры / транскрипты --
    ".srt", ".vtt", ".ass", ".ssa", ".sub",
    # -- патчи/дифы/скрипты сборки --
    ".diff", ".patch", ".sql", ".gradle", ".cmake",
    # -- исходный код (кроме .py — по явному запросу) --
    ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".lua",
    ".pl", ".r", ".m", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".vue",
}
# ПАТЧ 11 — по запросу: "все возможные текстовики любых форматов, кроме .py"
# + ".doc и .docx обязательны" (ПАТЧ 12). .doc/.docx теперь поддержаны через
# отдельные экстракторы (_extract_docx_text/_extract_doc_text) — см. ниже.
# Явно НЕ включены (и не должны включаться без отдельного парсера, т.к. это
# бинарные/составные форматы, а _read_text_file читает файл как plain текст,
# что даст мусор или ошибку): .pdf, .rtf, .odt, .xls, .xlsx, .ppt, .pptx,
# .zip, .7z, изображения, аудио/видео. Для них нужен отдельный этап
# извлечения текста (см. skills: pdf/xlsx/pptx) ДО того, как файл попадёт
# в папку, которую сканирует PTG.
ROOT_DOC_PREFIXES = ("README", "ARCHITECTURE", "INSTALL")
MIN_BLOCK_CHARS = 10
EMBED_MAX_CHARS = 6000           # обрезка текста перед отправкой в LM Studio (не влияет на хранимый текст)

PTG_DIRNAME = ".ptg"
NODES_FILE = "nodes.json"
EDGES_FILE = "edges.json"
META_FILE = "meta_tree.json"
ROOT_PROJECT_FILE = "root_project.json"
FILE_EDGES_FILE = "file_edges.json"       # ПАТЧ 2 — файловый граф
INDEX_FILE = "faiss.index"

# ПАТЧ 1 — активный корневой приор
OFF_ROOT_THRESH = 0.35       # ниже — атом помечается off_root=True
FILE_GRAPH_THRESH = 0.72     # ПАТЧ 2 — минимальное сходство для рёбра файлового графа
LAYERED_TOP_FILES = 5        # ПАТЧ 3 — сколько файлов брать на первом шаге поиска

# Path-dependent graph growth
BRANCH_STATE_FILE        = "branch_state.json"
BRANCH_BODIES_FILE       = "branch_bodies.json"  # ПАТЧ 1 — тело ветки
CURRENT_PATH_WINDOW      = 20    # скользящее окно (short path)
BRANCH_DECAY_INTERVAL    = 25    # каждые N атомов — decay
BRANCH_DECAY_IDLE_THRESH = 50    # ветка не трогалась N+ атомов → decay
BRANCH_DORMANT_THRESH    = 0.10  # activation ниже → dormant
# Веса graft-score (ПАТЧ 5)
W_SEM   = 0.40
W_PATH  = 0.25
W_MOM   = 0.15
W_ROOT  = 0.10
W_ACT   = 0.10  # branch activation
W_ENT   = 0.20  # штраф за энтропию

# ---------------------------------------------------------------------------
# ПАТЧ 9 — Multi-axis placement score + cosine safety logic
# ---------------------------------------------------------------------------
# ВАЖНО: это НЕ замена graft-score (W_SEM..W_ENT выше). Graft-score отвечает
# на вопрос "к какой ветке этот атом гравитационно ближе всего" (выбор
# КАНДИДАТА). Placement-score отвечает на другой вопрос: "раз кандидат
# найден и cosine с его головой высок — можно ли считать это буквальным
# продолжением ветки (continues), или это то же самое по смыслу, но
# отдельное по времени/файлу/локальному кластеру событие (reinforces)".
# Без этого слоя высокий cosine автоматически превращался в continues —
# это и есть "semantic coagulation" из аудита.
PLACEMENT_W_SEM  = 0.45   # semantic_similarity
PLACEMENT_W_TEMP = 0.20   # temporal_affinity
PLACEMENT_W_FILE = 0.15   # file_origin_affinity
PLACEMENT_W_META = 0.10   # metadata_affinity
PLACEMENT_W_PATH = 0.10   # path_history_affinity (переиспользует уже
                           # существующий path_score из _path_coherence)

# ПАТЧ 10 — расширение placement-score двумя осями: entropy_affinity
# (насколько ветка-кандидат уже "спокойна" семантически) и density_penalty
# (насколько ветка уже перегружена похожими атомами). Обе оси УЖЕ
# использовались раньше как отдельные gate'ы (_local_semantic_density,
# bs['entropy']) — здесь они дополнительно попадают в единый score как
# диагностика (решение по-прежнему принимается явными CASE-условиями,
# см. комментарий в _add_atom про "один скаляр — источник багов").
PLACEMENT_W_ENT  = 0.05   # entropy_affinity
PLACEMENT_W_DENS = 0.05   # density_penalty (1 - нормированная плотность)
# Веса Ws/Wt/Wf/Wm сохраняют старые значения из ПАТЧ 9 (0.45/0.20/0.15/0.10);
# We/Wd добавлены СВЕРХУ по запросу задания, сумма весов намеренно НЕ
# нормируется к 1.0 — placement_score используется только как диагностика,
# не как единственный判断 критерий (см. CASE-логику ниже).


TEMPORAL_HALF_LIFE_DAYS = 30.0      # affinity падает вдвое каждые N дней
REINFORCE_TEMPORAL_DAYS = 21.0      # порог "низкой временной дистанции" для CASE A
FILE_GRAPH_LINEAGE_MIN  = FILE_GRAPH_THRESH  # порог "той же файловой линии" через file_edges

# Semantic density limiter (ПАТЧ 9, часть VIII) — принудительное
# латеральное расширение вместо бесконечного углубления одной ветки
LOCAL_DENSITY_WINDOW      = 8     # сколько последних атомов ветки смотрим
LOCAL_DENSITY_HIGH_COSINE = 0.85  # что считаем "высоким" cosine к центроиду
LOCAL_DENSITY_LIMIT       = 5     # >= этого числа высоких cosine подряд → латеральное расширение

# ПАТЧ 10 — фикс подтверждённого бага аудита #9 (mmap safety). mmap имеет
# смысл только для действительно больших векторных файлов; для типичных
# архивов Алексея (десятки-сотни МБ логов, но вектор-файл обычно << 50 МБ)
# safer default — полная загрузка в RAM (без открытого файлового хэндла,
# который иначе живёт всё время работы MCP-сервера и может столкнуться с
# перезаписью того же .npy на диске из GUI-процесса при следующем build()).
MMAP_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 МБ

# Semantic epoch — "монотонный" счётчик повторных возвращений к одной
# линии мышления через CASE B/C (см. _place_against_candidate). Хранится
# на самой ветке (self.branches[bid]['semantic_epoch']), 0 для веток без
# предшественника-реинфорсмента.


def _has_marker(text, markers):
    low = text.lower()
    return any(m in low for m in markers)


# ---------------------------------------------------------------------------
# ПАТЧ 9 — чистые функции для multi-axis placement score
# ---------------------------------------------------------------------------
def _temporal_affinity(ts_a: float, ts_b: float,
                        half_life_days: float = TEMPORAL_HALF_LIFE_DAYS) -> float:
    """Экспоненциальное затухание сходства по времени: 1.0 при нулевой
    дистанции, 0.5 через half_life_days, дальше — меньше. ts_* — unix time
    (секунды). Никогда не возвращает 0 ровно (asymptotic), что осознанно:
    даже очень старая перекличка идей сохраняет ненулевой сигнал."""
    days = abs(float(ts_a) - float(ts_b)) / 86400.0
    if half_life_days <= 0:
        return 1.0 if days == 0 else 0.0
    return float(0.5 ** (days / half_life_days))


def _metadata_affinity(meta_a: dict, meta_b: dict) -> float:
    """Взвешенное совпадение метаданных: одна и та же папка — сильный
    сигнал (0.6), одна и та же сессия импорта — дополнительный (0.4).
    Оба поля обязательны на узле (см. _add_atom): folder_path, session_id."""
    score = 0.0
    if meta_a.get("folder_path") and meta_a.get("folder_path") == meta_b.get("folder_path"):
        score += 0.6
    if meta_a.get("session_id") and meta_a.get("session_id") == meta_b.get("session_id"):
        score += 0.4
    return score


# ---------------------------------------------------------------------------
# ПАТЧ 2 — Embedder: только LM Studio, без фолбэков
# ---------------------------------------------------------------------------
class Embedder:
    """Эмбеддинги ИСКЛЮЧИТЕЛЬНО через локальный сервер LM Studio.
    Никакого sentence-transformers, никакого hash-фолбэка — если LM Studio
    недоступен, вызывающий код обязан остановить построение архива."""

    def __init__(self, base_url=LM_STUDIO_BASE_URL, model=LM_STUDIO_EMBED_MODEL):
        self.base_url = base_url
        self.model = model
        self.dim = None
        self.backend = "lm-studio"

    def test_connection(self, preferred_model: str = None):
        """GET /v1/models, затем ПРОБНЫЙ вызов /v1/embeddings для кандидатов,
        пока не найдётся реально рабочая embedding-модель.

        ПАТЧ 13 — раньше выбор модели был основан на имени: если настроенная
        LM_STUDIO_EMBED_MODEL не находилась в списке, код либо падал, либо
        (после прошлого фикса) молча брал первую попавшуюся модель — и это
        могло оказаться VL/чат-моделью, не поддерживающей /v1/embeddings
        (см. случай с qwen3-vl-embedding, давший 400 Bad Request). Имя
        модели НИЧЕГО не гарантирует — единственный надёжный способ узнать,
        работает ли она как embedder, это реально её спросить.

        ПАТЧ 18 — preferred_model: если архив уже был построен КОНКРЕТНОЙ
        моделью (см. build(): self.embedder_model_used из meta), эта модель
        пробуется ПЕРВОЙ, раньше даже настроенной по умолчанию
        LM_STUDIO_EMBED_MODEL — продолжать архив другой моделью означало бы
        смешать несравнимые векторные пространства (см. Archive.build():
        жёсткая проверка после test_connection(), которая не даст этому
        случиться молча, даже если сюда закралась ошибка).

        Теперь: пробуем сначала preferred_model (если задан и есть в
        списке), потом настроенную по умолчанию (если есть), затем
        остальные по порядку — каждую одним маленьким пробным запросом
        embed(["ping"]). Первая, что ответит валидным вектором, становится
        self.model."""
        self.fallback_used = False
        self.probe_error = None
        try:
            r = requests.get(f"{self.base_url}/models", timeout=LM_STUDIO_TIMEOUT_CONNECT)
            if r.status_code != 200:
                return False
            data = r.json()
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        except Exception as ex:
            self.probe_error = str(ex)
            return False
        if not ids:
            self.probe_error = "LM Studio не сообщил ни одной загруженной модели"
            return False

        original_model = self.model
        priority = [m for m in (preferred_model, self.model) if m and m in ids]
        candidates = priority + [m for m in ids if m not in priority]
        _log.debug(f"test_connection: кандидаты в порядке пробы: {candidates}")

        last_error = None
        for candidate in candidates:
            try:
                probe_vecs = self._probe_embed(candidate)
            except Exception as ex:
                # ПАТЧ 18: таймаут/сетевая ошибка — это НЕ "модель не умеет
                # embeddings", а "не удалось спросить прямо сейчас". Раньше
                # это тоже просто переходило к следующему кандидату — при
                # коротком таймауте (см. фикс #1 выше) именно так рабочая
                # модель ошибочно бракуется и код уезжает на другую.
                # Само по себе поведение (пробовать следующего кандидата)
                # оставлено прежним — теперь неверно был только таймаут;
                # с LM_STUDIO_TIMEOUT_PROBE=30 сек это должно происходить
                # только при реальной недоступности модели.
                last_error = f"{candidate}: {ex}"
                _log.warning(f"test_connection: проба «{candidate}» провалилась: {ex}")
                continue
            if probe_vecs is not None:
                self.model = candidate
                self.fallback_used = (candidate != original_model)
                self.dim = probe_vecs.shape[1] if probe_vecs.size else self.dim
                _log.info(f"test_connection: выбрана модель «{candidate}» (dim={self.dim}, "
                          f"fallback_used={self.fallback_used})")
                return True
            _log.warning(f"test_connection: «{candidate}» вернула HTTP-ошибку (см. embed())")

        # Ни одна модель не прошла пробный вызов — оставляем self.model как
        # был, но честно сообщаем причину последней ошибки вызывающему коду.
        self.probe_error = last_error or "ни одна модель не ответила валидным эмбеддингом"
        _log.error(f"test_connection: ни одна модель не прошла пробу. {self.probe_error}")
        return False

    def _probe_embed(self, model_id: str):
        """Единственный надёжный тест "эта модель реально умеет embeddings":
        маленький настоящий вызов /v1/embeddings. Возвращает np.ndarray
        векторов при успехе, None при HTTP-ошибке. Исключения (сеть,
        таймаут) пробрасываются вызывающему коду (test_connection) —
        там они конвертируются в last_error и проверяется следующий кандидат."""
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": model_id, "input": ["ping"]},
            timeout=LM_STUDIO_TIMEOUT_PROBE,
        )
        if resp.status_code >= 400:
            return None
        data = resp.json()
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        vecs = np.array([it["embedding"] for it in items], dtype="float32")
        if vecs.size == 0:
            return None
        return vecs

    def _embed_timeout_for(self, texts) -> float:
        """ПАТЧ 20 — таймаут embed()-запроса, масштабированный по объёму
        батча, а не плоское число. Реальный инцидент: батч из 17 текстов
        структуры проекта (~100 000 симв. суммарно) честно обрабатывался
        LM Studio ~121 секунду на слабом GPU (VRAM впритык/с overflow) —
        плоский LM_STUDIO_TIMEOUT_EMBED=120 обрывал соединение на последней
        секунде перед реальным завершением. Формула: базовый таймаут (как
        раньше) + EMBED_TIMEOUT_PER_1K_CHARS секунд на каждую 1000 символов
        суммарного батча, с потолком EMBED_TIMEOUT_MAX (не ждать буквально
        бесконечно при настоящем зависании)."""
        total_chars = sum(len(t) for t in texts)
        scaled = EMBED_TIMEOUT_BASE + (total_chars / 1000.0) * EMBED_TIMEOUT_PER_1K_CHARS
        return min(EMBED_TIMEOUT_MAX, max(EMBED_TIMEOUT_BASE, scaled))

    def embed(self, texts):
        texts = [t[:EMBED_MAX_CHARS] for t in texts]
        t0 = time.time()
        timeout = self._embed_timeout_for(texts)
        _log.debug(f"embed(): модель={self.model}, батч={len(texts)} текст(ов), "
                  f"макс.длина={max((len(t) for t in texts), default=0)} симв., "
                  f"таймаут={timeout:.0f}с")
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            timeout=timeout,
        )
        elapsed = time.time() - t0
        if resp.status_code >= 400:
            # Фикс: raise_for_status() даёт только "400 Client Error: ..." без
            # тела ответа — а именно в теле LM Studio обычно пишет РЕАЛЬНУЮ
            # причину (например, "model does not support embeddings" для
            # VL/чат-моделей, ошибочно выбранных как embedding-модель).
            body = resp.text[:500]
            _log.error(f"embed(): HTTP {resp.status_code} от модели «{self.model}» "
                      f"за {elapsed:.1f}с: {body}")
            raise RuntimeError(
                f"LM Studio вернул {resp.status_code} для модели '{self.model}': {body}"
            )
        data = resp.json()
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        vecs = np.array([it["embedding"] for it in items], dtype="float32")
        if self.dim is None and vecs.size:
            self.dim = vecs.shape[1]
        _log.debug(f"embed(): успех за {elapsed:.1f}с, получено {len(vecs)} вектор(ов)")
        return vecs


# ---------------------------------------------------------------------------
# Сканирование файлов
# ---------------------------------------------------------------------------
def scan_folder(folder, include_py: bool = True):
    """ПАТЧ 17 — include_py: выключатель обработки .py (через GUI-чекбокс
    или Archive(..., extract_py_comments=False)). Если выключен — .py
    файлы просто не попадают в список для сканирования (не тратится
    время на ast/tokenize вообще). Ранее добавленные .py-атомы, если
    переключатель выключили ПОСЛЕ того, как они уже попали в архив,
    остаются (append-only — старое не удаляется при смене настройки)."""
    files = []
    for root, dirs, fnames in os.walk(folder):
        dirs[:] = [d for d in dirs if d != PTG_DIRNAME and not d.startswith(".")]
        for fn in fnames:
            ext = os.path.splitext(fn)[1].lower()
            if ext == ".py" and not include_py:
                continue
            if ext in TEXT_EXTS:
                files.append(os.path.join(root, fn))
    files.sort()
    return files


# ---------------------------------------------------------------------------
# ПАТЧ 1 — парсер: атом = вопрос + ответ, а не абзац
# ---------------------------------------------------------------------------
def _split_into_blocks(content):
    """Базовая единица перед склейкой в Q+A — по-прежнему абзац, но это
    промежуточный блок, а не финальный атом."""
    paragraphs = re.split(r"\n\s*\n", content)
    blocks, buf = [], ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(p) < MIN_BLOCK_CHARS:
            buf = (buf + "\n\n" + p).strip()
            continue
        if buf:
            blocks.append(buf)
            buf = ""
        blocks.append(p)
    if buf:
        blocks.append(buf)
    return blocks


def _label_blocks(blocks):
    labels = []
    for b in blocks:
        if _Q_LABEL_RE.match(b):
            labels.append("q")
        elif _A_LABEL_RE.match(b):
            labels.append("a")
        else:
            labels.append(None)
    return labels


def _group_by_labels(blocks, labels):
    """Группировка по найденным речевым меткам: подряд идущие 'q'-блоки —
    вопрос, последующие 'a'-блоки — ответ."""
    pairs = []
    cur_q, cur_a, mode = [], [], None
    for b, lab in zip(blocks, labels):
        if lab == "q":
            if cur_q and cur_a:
                pairs.append((cur_q, cur_a))
                cur_q, cur_a = [], []
            cur_q.append(b)
            mode = "q"
        elif lab == "a":
            cur_a.append(b)
            mode = "a"
        else:
            # без метки — приклеиваем к текущей роли (внутренний перенос строки реплики)
            if mode == "a":
                cur_a.append(b)
            else:
                cur_q.append(b)
                mode = "q"
    if cur_q or cur_a:
        pairs.append((cur_q, cur_a))
    return pairs


def _pair_sequential(blocks):
    """Фолбэк низкой уверенности: просто склеиваем блоки последовательно
    парами (вопрос, ответ)."""
    pairs = []
    i = 0
    while i < len(blocks):
        q = [blocks[i]]
        a = [blocks[i + 1]] if i + 1 < len(blocks) else []
        pairs.append((q, a))
        i += 2
    return pairs


_DOCX_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _extract_docx_text(path):
    """ПАТЧ 12 — .docx обязателен. .docx — это ZIP-архив с XML внутри
    (word/document.xml), поэтому извлечение работает через stdlib
    (zipfile + xml.etree.ElementTree), без новых зависимостей.
    Берём текст всех <w:t> узлов внутри каждого <w:p> (абзац), включая
    текст внутри таблиц (ячейки — это тоже <w:p> внутри документа) —
    таблицы теряют структуру (не превращаются в CSV-подобный текст), но
    сам текст не теряется, что достаточно для смыслового атома PTG.
    Возвращает None при ЛЮБОЙ ошибке (битый .docx, защищённый паролем,
    не-Word zip и т.п.) — вызывающий код (parse_file_to_atoms) в этом
    случае просто пропускает файл, как и раньше делал для нечитаемых файлов."""
    try:
        with zipfile.ZipFile(path) as z:
            with z.open("word/document.xml") as f:
                tree = ET.parse(f)
    except Exception as ex:
        _log.warning(f".docx не удалось разобрать ({path}): {ex}")
        return None

    paragraphs = []
    for p in tree.getroot().iter(_DOCX_WORD_NS + "p"):
        text = "".join(node.text or "" for node in p.iter(_DOCX_WORD_NS + "t"))
        paragraphs.append(text)
    return "\n".join(paragraphs)


def _extract_doc_text(path):
    """ПАТЧ 12 — .doc обязателен. Старый (до 2007) бинарный формат Word —
    OLE Compound File с потоками WordDocument/0Table/1Table. Полный разбор
    формата ([MS-DOC]) сложен; здесь — практичный двухуровневый подход:

    1) Основной путь: через olefile (чистый Python, pip-пакет) читаем
       поток WordDocument, находим FIB (File Information Block), по флагу
       fWhichTblStream (бит 0x0200 в fibbase.flags1, offset 0x0A) выбираем
       "0Table" или "1Table", по fcClx/lcbClx из FIB (offset 0x01A2/0x01A6)
       достаём Clx → PlcPcd (таблицу "кусков" текста) и склеиваем текст
       кусок за куском, учитывая, что каждый кусок либо 8-бит (cp1252),
       либо 16-бит (UTF-16LE) в зависимости от бита компрессии в PCD.fc.

    2) Fallback (если olefile не установлен ИЛИ разбор Clx не удался —
       нестандартные/повреждённые файлы): грубое сканирование потока
       WordDocument на предмет длинных последовательностей "печатных"
       UTF-16LE символов. Качество ниже (могут остаться служебные
       фрагменты), но текст обычно читаем — лучше, чем полный отказ,
       раз .doc помечен как обязательный формат.

    Возвращает None только если И основной путь, И fallback не дали текста
    (например, файл не является OLE-файлом вообще)."""
    if _HAS_OLEFILE:
        text = _extract_doc_text_piecetable(path)
        if text:
            return text
    return _extract_doc_text_heuristic(path)


def _extract_doc_text_piecetable(path):
    """Основной путь для .doc — см. docstring _extract_doc_text(). Только
    ввод-вывод (OLE-контейнер); сам разбор — в _parse_doc_piecetable()."""
    try:
        ole = olefile.OleFileIO(path)
    except Exception as ex:
        _log.debug(f".doc: не удалось открыть как OLE-файл ({path}): {ex} — пробую эвристику")
        return None
    try:
        if not ole.exists("WordDocument"):
            _log.debug(f".doc: нет потока WordDocument ({path}) — пробую эвристику")
            return None
        with ole.openstream("WordDocument") as f:
            word_doc = f.read()
        if len(word_doc) < 0x200:
            _log.debug(f".doc: поток WordDocument подозрительно мал ({path}) — пробую эвристику")
            return None

        flags1 = int.from_bytes(word_doc[0x0A:0x0C], "little")
        table_name = "1Table" if (flags1 & 0x0200) else "0Table"
        if not ole.exists(table_name):
            _log.debug(f".doc: нет потока {table_name} ({path}) — пробую эвристику")
            return None
        with ole.openstream(table_name) as f:
            table_stream = f.read()

        return _parse_doc_piecetable(word_doc, table_stream)
    except Exception as ex:
        _log.debug(f".doc: piece-table разбор не удался ({path}): {ex} — пробую эвристику")
        return None
    finally:
        try:
            ole.close()
        except Exception:
            pass


def _parse_doc_piecetable(word_doc: bytes, table_stream: bytes):
    """Чистая логика разбора FIB → Clx → PlcPcd → текст, без зависимости от
    olefile/файловой системы — можно юнит-тестировать на синтетических
    байтовых буферах, не собирая настоящий OLE-контейнер."""
    fc_clx = int.from_bytes(word_doc[0x01A2:0x01A6], "little")
    lcb_clx = int.from_bytes(word_doc[0x01A6:0x01AA], "little")
    if lcb_clx <= 0 or fc_clx + lcb_clx > len(table_stream):
        return None
    clx = table_stream[fc_clx:fc_clx + lcb_clx]

    # Clx = последовательность Prc-блоков (начинаются с 0x01, содержат
    # grpprl — форматирование, нам не нужно) и ОДИН Pcdt-блок (начинается
    # с 0x02) — вот он нам и нужен: 4-байтная длина + PlcPcd.
    i = 0
    plc_pcd = None
    while i < len(clx):
        marker = clx[i]
        if marker == 0x02:
            lcb = int.from_bytes(clx[i + 1:i + 5], "little")
            plc_pcd = clx[i + 5:i + 5 + lcb]
            break
        elif marker == 0x01:
            i += 1
            cb_grpprl = int.from_bytes(clx[i:i + 2], "little")
            i += 2 + cb_grpprl
        else:
            break
    if plc_pcd is None:
        return None

    # PlcPcd: (n+1) CPs по 4 байта, затем n PCD по 8 байт.
    n = (len(plc_pcd) - 4) // 12
    if n <= 0:
        return None
    cps = [int.from_bytes(plc_pcd[k * 4:k * 4 + 4], "little") for k in range(n + 1)]
    pcd_start = (n + 1) * 4
    pieces = []
    for k in range(n):
        pcd = plc_pcd[pcd_start + k * 8: pcd_start + k * 8 + 8]
        if len(pcd) < 8:
            continue
        fc_raw = int.from_bytes(pcd[2:6], "little")
        is_ansi = bool(fc_raw & 0x40000000)   # бит компрессии: 1 = cp1252 (8-бит), 0 = UTF-16LE
        fc = (fc_raw & ~0x40000000)
        cp_start, cp_end = cps[k], cps[k + 1]
        n_chars = cp_end - cp_start
        if n_chars <= 0:
            continue
        if is_ansi:
            fc = fc // 2
            raw = word_doc[fc: fc + n_chars]
            piece_text = raw.decode("cp1252", errors="ignore")
        else:
            raw = word_doc[fc: fc + n_chars * 2]
            piece_text = raw.decode("utf-16-le", errors="ignore")
        pieces.append(piece_text)

    text = "".join(pieces)
    # В .doc абзацы разделены символом \r (0x0D), не \n
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x07", "\t")
    # Служебные управляющие символы (кроме \n\t) — убираем как шум
    text = "".join(ch for ch in text if ch in "\n\t" or ch >= " ")
    return text.strip() or None


def _extract_doc_text_heuristic(path):
    """Fallback для .doc без olefile или при неудаче основного пути:
    грубый скан бинарных данных всего файла на предмет длинных прогонов
    "печатных" UTF-16LE символов (ASCII 0x20-0x7E либо кириллица
    0x0400-0x04FF, оба диапазона распознаются напрямую по 16-битному коду
    символа — старший байт НЕ обязан быть нулевым, кириллица как раз имеет
    старший байт 0x04). Качество хуже
    piece-table метода (могут остаться разрывы/мусор на границах), но не
    требует olefile и почти никогда не даёт полный отказ."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return None

    chars = []
    i = 0
    run = []
    def _flush():
        if len(run) >= 4:  # отбрасываем случайные короткие шумовые обрывки
            chars.append("".join(run))
        run.clear()

    while i + 1 < len(data):
        lo, hi = data[i], data[i + 1]
        code = lo | (hi << 8)
        is_printable = (0x20 <= code <= 0x7E) or (0x0400 <= code <= 0x04FF) or code in (0x09, 0x0A, 0x0D)
        if is_printable:
            run.append(chr(code))
            i += 2
        else:
            _flush()
            i += 1
    _flush()

    text = "\n".join(chars)
    return text.strip() or None


def _read_text_file(path):
    """ПАТЧ 11 — честное чтение текста в разных кодировках. Раньше файл
    читался ЖЁСТКО как utf-8 с errors='ignore': если исходный файл был в
    другой кодировке (например, старые русские .txt/.log в cp1251/
    windows-1251), не-UTF8 байты молча выбрасывались, а не декодировались —
    текст оказывался обрезан/испорчен без единого предупреждения.
    Теперь пробуем по порядку: utf-8-sig (снимает BOM, если есть) → utf-8
    (строго) → cp1251 (частый случай для русского текста в старых
    экспортах) → cp1252 (частый случай для англ./зап.-европ. Windows-текста)
    → latin-1 как последний рубеж (latin-1 не бросает исключений вообще,
    так что до него дело почти никогда не доходит, но файл в любом случае
    будет прочитан, а не пропущен)."""
    for enc in ("utf-8-sig", "utf-8", "cp1251", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as ex:
            _log.warning(f"не удалось открыть файл ({path}): {ex}")
            return None
    _log.debug(f"файл не декодировался ни в одной обычной кодировке ({path}) — latin-1 fallback")
    try:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()
    except Exception as ex:
        _log.warning(f"latin-1 fallback тоже не сработал ({path}): {ex}")
        return None


# ПАТЧ 23 — .json/.jsonl/.ndjson: структурированные экспорты чатов с ИИ
# (ChatGPT, Claude, Grok, Gemini/Google Takeout, Meta AI). Разбор форматов
# живёт в отдельном модуле ptg_chat_exports.py (по тому же принципу, что и
# ptg_py_comments.py) — см. его docstring за обоснованием разделения на
# "точные" парсеры (ChatGPT/Claude) и "generic" (Grok/Meta) vs. журнал
# активности (Google Takeout).
_JSON_CHAT_EXTS = {".json", ".jsonl", ".ndjson"}


def _load_json_for_chat_detection(path, ext):
    """Попытаться разобрать .json/.jsonl/.ndjson как структурированный JSON
    ДО того, как (при неудаче) откатиться на чтение файла как plain text.
    .jsonl/.ndjson — по объекту на строку; отдельные невалидные строки
    (например, хвостовая пустая строка) пропускаются, но если НИ ОДНА
    строка не распарсилась, возвращается None — файл честно не похож на
    построчный JSON, вызывающий код откатится на текстовое чтение."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        return None
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp1251", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        return None

    if ext == ".json":
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return records or None


def parse_file_to_atoms(path, session_id=None):
    """Разобрать сырой экспорт чата на атомы вида {question, answer}.

    ВАЖНО (Патч 1): изолированные абзацы НИКОГДА не считаются атомом сами
    по себе. Каждый атом — это пара вопрос+ответ, определённая либо по
    речевым меткам (User:/Assistant: и т.п.), либо, при низкой уверенности,
    последовательной склейкой блоков по два.

    ПАТЧ 23: для .json/.jsonl/.ndjson сначала пробуем разобрать файл как
    СТРУКТУРИРОВАННЫЙ экспорт чата (ChatGPT/Claude/Grok/Gemini/Meta AI —
    см. ptg_chat_exports.py). Если формат распознан как список готовых пар
    question/answer — атомы строятся напрямую из них (в обход
    _split_into_blocks/_label_blocks), с собственной меткой времени каждой
    пары, если формат её предоставил. Если распознан только как текстовый
    поток без строгой структуры (Google Takeout) — этот текст идёт в ТОТ ЖЕ
    текстовый пайплайн ниже, что и обычный .txt. Если файл не похож ни на
    один известный формат чата (обычный .json-конфиг, package.json и т.п.)
    — поведение полностью прежнее: чтение как plain text."""
    ext = os.path.splitext(path)[1].lower()
    chat_pairs = None
    if ext == ".docx":
        content = _extract_docx_text(path)
    elif ext == ".doc":
        content = _extract_doc_text(path)
    elif ext == ".py":
        content = extract_py_comments_text(path)
    elif ext in _JSON_CHAT_EXTS:
        data = _load_json_for_chat_detection(path, ext)
        content = None
        if data is not None:
            chat_pairs = extract_chat_pairs(data, source_label=path)
            if chat_pairs is None:
                content = extract_chat_text_stream(data, source_label=path)
        if chat_pairs is None and content is None:
            content = _read_text_file(path)
    else:
        content = _read_text_file(path)

    mtime = os.path.getmtime(path)
    session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
    folder_path = os.path.dirname(path)
    filename = os.path.basename(path)

    if chat_pairs is not None:
        # ПАТЧ 23 — структурированный чат-экспорт: у каждой пары уже может
        # быть СОБСТВЕННАЯ метка времени (не файловый mtime) — используем
        # её, когда формат её предоставил (честнее общего mtime всего файла,
        # см. §5 PTG_EMBEDDER.md); иначе — общий fallback на mtime, как и
        # раньше для текстовых экспортов.
        atoms = []
        for i, pair in enumerate(chat_pairs):
            question = pair.get("question", "")
            answer = pair.get("answer", "")
            if not question and not answer:
                continue
            combined = (question + "\n\n" + answer).strip()
            if len(combined) < MIN_BLOCK_CHARS:
                continue
            ts = pair.get("timestamp")
            atom_time = float(ts) if isinstance(ts, (int, float)) else mtime
            atoms.append({
                "id": str(uuid.uuid4()), "file": path, "order": i,
                "question": question, "answer": answer, "text": combined,
                "confidence": "structured_export", "timestamp": atom_time,
                "source_path": path,
                "filename": filename,
                "folder_path": folder_path,
                "created_at": atom_time,
                "modified_at": atom_time,
                "session_id": session_id,
            })
        return atoms

    if content is None:
        return []

    blocks = _split_into_blocks(content)
    if not blocks:
        return []

    labels = _label_blocks(blocks)
    labeled_ratio = sum(1 for l in labels if l) / len(labels)
    has_both_roles = "q" in labels and "a" in labels

    if labeled_ratio >= 0.3 and has_both_roles:
        pairs = _group_by_labels(blocks, labels)
        confidence = "label"
    else:
        pairs = _pair_sequential(blocks)
        confidence = "sequential"

    atoms = []
    for i, (qparts, aparts) in enumerate(pairs):
        question = "\n\n".join(qparts).strip()
        answer = "\n\n".join(aparts).strip()
        if not question and not answer:
            continue
        combined = (question + "\n\n" + answer).strip()
        if len(combined) < MIN_BLOCK_CHARS:
            continue
        atoms.append({
            "id": str(uuid.uuid4()), "file": path, "order": i,
            "question": question, "answer": answer, "text": combined,
            "confidence": confidence, "timestamp": mtime,
            # ПАТЧ 9 — часть VI: временной/метаданный слой.
            # created_at/modified_at честно равны mtime всего файла-источника:
            # у отдельного Q+A внутри чат-экспорта нет собственной ОС-метки
            # времени, и выдумывать её было бы хуже, чем явно задокументировать
            # это ограничение. import_timestamp (момент попадания в граф,
            # не в этот атом-словарь) проставляется позже в _add_atom().
            "source_path": path,
            "filename": filename,
            "folder_path": folder_path,
            "created_at": mtime,
            "modified_at": mtime,
            "session_id": session_id,
        })
    return atoms


# ---------------------------------------------------------------------------
# ПАТЧ 3 — корневой семантический скелет проекта
# ---------------------------------------------------------------------------
STRUCTURE_FILE_MARKER = "__project_structure__"   # сентинел-путь для синтетических атомов структуры
STRUCTURE_MAX_ENTRIES_PER_ATOM = 200               # порог дробления по топ-папкам


def _build_project_structure_texts(folder):
    """ПАТЧ 16, часть II — структурный слепок проекта (иерархия папок и
    файлов с метаданными: размер, дата изменения, тип) как ТЕКСТ, готовый
    стать атомом(ами) графа. Это НЕ то же самое, что file_manifest()
    (тот — API-вызов для агента по требованию) — цель здесь другая: сама
    структура должна быть НАЙДЕНА обычным семантическим/branch-first
    поиском наравне с любой другой мыслью в архиве, а не только через
    отдельный инструмент.

    Возвращает список (label, text). Если файлов немного — один общий
    срез; если много — дробится по топ-уровневым папкам, чтобы не
    получить один неповоротливый атом на весь проект."""
    entries_by_top = {}
    total_entries = 0
    for root, dirs, fnames in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d != PTG_DIRNAME and not d.startswith("."))
        rel_root = os.path.relpath(root, folder)
        top = "." if rel_root == "." else rel_root.split(os.sep)[0]
        for fn in sorted(fnames):
            path = os.path.join(root, fn)
            try:
                st = os.stat(path)
            except OSError:
                continue
            rel_path = os.path.relpath(path, folder)
            mtime_str = _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
            ext = os.path.splitext(fn)[1].lower()
            kind = "текст" if ext in TEXT_EXTS else "прочее"
            entries_by_top.setdefault(top, []).append(
                f"{rel_path}  [{kind}, {st.st_size / 1024:.1f}KB, изм. {mtime_str}]"
            )
            total_entries += 1

    project_name = os.path.basename(os.path.normpath(folder)) or "Проект"
    out = []
    if total_entries == 0:
        return out
    if total_entries <= STRUCTURE_MAX_ENTRIES_PER_ATOM:
        lines = []
        for top in sorted(entries_by_top):
            lines.extend(entries_by_top[top])
        text = f"Структура проекта «{project_name}» ({total_entries} файлов):\n" + "\n".join(lines)
        out.append(("root", text))
    else:
        for top, lines in sorted(entries_by_top.items()):
            text = (f"Структура проекта «{project_name}», раздел «{top}» "
                     f"({len(lines)} файлов):\n" + "\n".join(lines))
            out.append((top, text))
    return out


def _build_root_project(folder):
    found = []
    for root, dirs, fnames in os.walk(folder):
        dirs[:] = [d for d in dirs if d != PTG_DIRNAME and not d.startswith(".")]
        in_docs = os.path.basename(root).lower() == "docs"
        for fn in fnames:
            base_upper = fn.upper()
            is_md = fn.lower().endswith(".md")
            is_named = any(base_upper.startswith(p) for p in ROOT_DOC_PREFIXES)
            if is_md or is_named or in_docs:
                path = os.path.join(root, fn)
                content = _read_text_file(path)
                if content is not None:
                    found.append((path, content))

    headings = []
    concepts = set()
    project_name = os.path.basename(os.path.normpath(folder)) or "Проект"

    for path, text in found:
        for line in text.splitlines():
            m = re.match(r"^(#{1,2})\s+(.+)", line.strip())
            if m:
                headings.append(m.group(2).strip())
        concepts.update(re.findall(r"`([a-zA-Z0-9_./-]{3,40})`", text))
        concepts.update(re.findall(r"\b([A-Z][a-zA-Z0-9]{2,}(?:[A-Z][a-zA-Z0-9]*)+)\b", text))
        if os.path.basename(path).upper().startswith("README"):
            m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
            if m:
                project_name = m.group(1).strip()

    domains = sorted(
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d)) and d != PTG_DIRNAME and not d.startswith(".")
    )

    return {
        "project_name": project_name,
        "domains": domains,
        "root_modules": sorted(set(headings))[:50],
        "known_concepts": sorted(concepts)[:100],
    }


def _root_project_to_text(rp):
    """Сериализовать root_project в единый текст для эмбеддинга."""
    parts = []
    if rp.get("project_name"):
        parts.append(f"Project: {rp['project_name']}")
    if rp.get("domains"):
        parts.append("Domains: " + ", ".join(rp["domains"]))
    if rp.get("root_modules"):
        parts.append("Modules: " + ", ".join(rp["root_modules"][:30]))
    if rp.get("known_concepts"):
        parts.append("Concepts: " + ", ".join(rp["known_concepts"][:50]))
    return "\n".join(parts) or "Unknown project"


# ---------------------------------------------------------------------------
# Archive: дерево/DAG + хранение
# ---------------------------------------------------------------------------
class Archive:
    def __init__(self, folder, progress_cb=None, output_dir=None, extract_py_comments=True):
        """folder — папка с исходными логами/документами (сканируется).
        output_dir — ПАТЧ 14: отдельная папка для векторной базы (.ptg/).
        Если не указана — как и раньше, .ptg/ лежит внутри folder (полная
        обратная совместимость со старыми архивами). Если указана — вся
        векторная база (nodes.json, edges.json, faiss.index*, snapshots/ и
        т.д.) пишется туда, а folder используется ТОЛЬКО для чтения
        исходных файлов — это позволяет, например, держать логи на одном
        диске, а векторную базу на другом (SSD), или не засорять папку с
        документами служебными файлами.
        extract_py_comments — ПАТЧ 17: выключатель обработки .py (только
        докстринги/#-комментарии, см. ptg_py_comments.py). По умолчанию
        включён (True) для обратной совместимости с ПАТЧ 16. GUI даёт
        чекбокс, управляющий этим параметром при вызове build()."""
        self.folder = folder
        self.extract_py_comments = extract_py_comments
        # ПАТЧ 18 — какой моделью/размерностью архив УЖЕ был построен
        # (заполняется из meta_tree.json в load_if_exists(), если архив
        # существует). Используется build() как обязательный якорь: нельзя
        # молча продолжить архив другой embedding-моделью — это разные,
        # несравнимые векторные пространства.
        self.embedder_model_used = None
        self.embedder_dim_used = None
        self.ptg_dir = os.path.join(output_dir, PTG_DIRNAME) if output_dir else os.path.join(folder, PTG_DIRNAME)
        self.output_dir = output_dir  # None = .ptg рядом с folder (старое поведение)
        self.output_root = output_dir or folder  # куда класть snapshots/ и прочие производные артефакты
        self.progress_cb = progress_cb or (lambda msg: None)

        # ПАТЧ 19 — подробное файловое логирование, отдельно от progress_cb.
        # Пишет в <ptg_dir>/ptg.log независимо от того, кто создал Archive
        # (GUI, MCP-сервер) — это тот самый лог, которого не хватило при
        # разборе инцидента с молчаливой сменой embedding-модели.
        setup_logging(self.ptg_dir)
        _log.info(f"Archive инициализирован: folder={folder!r}, output_dir={output_dir!r}, "
                  f"extract_py_comments={extract_py_comments}")

        self.nodes = {}
        self.edges = []
        self.branches = {}
        self.id_order = []
        self.processed_files = {}
        self.last_structure_hash = None   # ПАТЧ 16 — хэш последнего структурного слепка

        # ПАТЧ 3 / ПАТЧ 4
        self.root_project = {}
        self.files = {}           # путь -> {file_id, centroid, main_branches, dominant_concepts, connected_files}

        # ПАТЧ 1 — активный корневой приор (вычисляется один раз за build)
        self.root_embedding = None  # type: np.ndarray | None

        # ПАТЧ 2 — файловый граф
        self.file_edges = []      # [{source_file, target_file, similarity, relation}]

        # Path-dependent growth
        # branch_id -> {weight, momentum, entropy, activation, last_seen, atom_count, dormant}
        self.branch_states: dict = {}
        # ПАТЧ 1 — тело ветки: branch_id -> {atom_ids, centroid, variance}
        self.branch_bodies: dict = {}
        # ПАТЧ 2 — иерархическая память пути
        self.path_memory: dict = {"short": [], "long": []}  # short=атомы, long=переходы между ветками
        self._prev_branch_id: str | None = None  # для отслеживания смены ветки
        # счётчик decay
        self._atoms_since_decay: int = 0

        self.embedder = Embedder()

    def log(self, msg):
        self.progress_cb(msg)
        _log.info(msg)

    # -- ПАТЧ 10, фикс бага #5 (race conditions между GUI и MCP-процессом) --
    def _lock_path(self):
        return os.path.join(self.ptg_dir, ".lock")

    def _acquire_lock(self, stale_seconds: int = 120):
        """Advisory (НЕ ОС-уровня flock/fcntl — платформозависимы и не
        работают одинаково в Windows/Linux) файловая блокировка между
        процессами, работающими с одной папкой архива: main.py (GUI,
        читает+пишет) и ptg_mcp_server.py (обычно только читает, но
        сохраняет снапшоты в ту же папку). Не гарантирует атомарность на
        100%, но устраняет типичный сценарий "оба процесса пишут
        одновременно в те же файлы" — по крайней мере честно предупреждает
        об этом в лог вместо тихого повреждения данных."""
        os.makedirs(self.ptg_dir, exist_ok=True)
        lock_path = self._lock_path()
        if os.path.exists(lock_path):
            try:
                with open(lock_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                age = time.time() - info.get("ts", 0)
                if age < stale_seconds and info.get("pid") != os.getpid():
                    self.log(
                        f"ВНИМАНИЕ: архив помечен как занятый другим процессом "
                        f"(pid={info.get('pid')}, {age:.0f} сек назад). Запись "
                        f"продолжается, но возможна гонка — не запускайте build() "
                        f"из GUI, пока MCP-сервер сохраняет снапшот, и наоборот."
                    )
            except Exception:
                pass  # повреждённый/пустой lock-файл — не блокирует работу
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "ts": time.time()}, f)

    def _release_lock(self):
        try:
            os.remove(self._lock_path())
        except OSError:
            pass

    # -- сохранение / загрузка -------------------------------------------
    def load_if_exists(self):
        if not os.path.isdir(self.ptg_dir):
            return False
        try:
            with open(os.path.join(self.ptg_dir, NODES_FILE), "r", encoding="utf-8") as f:
                raw_nodes = json.load(f)
            with open(os.path.join(self.ptg_dir, EDGES_FILE), "r", encoding="utf-8") as f:
                self.edges = json.load(f)
            with open(os.path.join(self.ptg_dir, META_FILE), "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.branches = meta.get("branches", {})
            self.processed_files = meta.get("processed_files", {})
            self.last_structure_hash = meta.get("last_structure_hash")
            # ПАТЧ 18 — запомнить, какой моделью архив был построен
            self.embedder_model_used = meta.get("embedder_model")
            self.embedder_dim_used = meta.get("embedder_dim")
            self.files = meta.get("files", {})

            root_path = os.path.join(self.ptg_dir, ROOT_PROJECT_FILE)
            if os.path.exists(root_path):
                with open(root_path, "r", encoding="utf-8") as f:
                    self.root_project = json.load(f)

            # ПАТЧ 2 — загрузить файловый граф
            fe_path = os.path.join(self.ptg_dir, FILE_EDGES_FILE)
            if os.path.exists(fe_path):
                with open(fe_path, "r", encoding="utf-8") as f:
                    self.file_edges = json.load(f)

            # ПАТЧ 1 — загрузить сохранённый root_embedding
            re_path = os.path.join(self.ptg_dir, "root_embedding.npy")
            if os.path.exists(re_path):
                self.root_embedding = np.load(re_path)

            # Path-dependent growth: branch_states, branch_bodies, path_memory
            bs_path = os.path.join(self.ptg_dir, BRANCH_STATE_FILE)
            if os.path.exists(bs_path):
                with open(bs_path, "r", encoding="utf-8") as f:
                    self.branch_states = json.load(f)
            bb_path = os.path.join(self.ptg_dir, BRANCH_BODIES_FILE)
            if os.path.exists(bb_path):
                with open(bb_path, "r", encoding="utf-8") as f:
                    raw_bb = json.load(f)
                # centroid хранится как list, восстанавливаем в np.array при загрузке
                for bid, body in raw_bb.items():
                    body["centroid"] = np.array(body["centroid"], dtype="float32")
                    self.branch_bodies[bid] = body
            # иерархическая память пути
            pm = meta.get("path_memory")
            if pm:
                self.path_memory = pm
            else:
                # обратная совместимость со старым current_path
                self.path_memory = {"short": meta.get("current_path", []), "long": []}
            self._atoms_since_decay = meta.get("atoms_since_decay", 0)

            vec_path_npy = os.path.join(self.ptg_dir, INDEX_FILE + ".npy")
            # ПАТЧ 9/10, часть VII — lazy-loading только выше MMAP_THRESHOLD_BYTES.
            # Это ЧАСТИЧНАЯ мера (метаданные/nodes.json/edges.json всё ещё
            # грузятся целиком — их постраничная загрузка потребовала бы
            # смены формата хранения с плоского JSON, что запрещено
            # ограничениями патча). Для полного streaming ingestion (часть
            # VII целиком) нужен отдельный, гораздо более крупный патч —
            # см. аудит, пункт "честно не решено".
            #
            # ПАТЧ 10 фикс бага #9: раньше mmap применялся ВСЕГДА, включая
            # маленькие архивы — файловый хэндл держался открытым весь срок
            # жизни процесса (например, ptg_mcp_server.py) без необходимости,
            # и рисковал столкнуться с перезаписью .npy другим процессом
            # (GUI, следующий build()). Теперь: mmap только для файлов
            # действительно большого размера, где выигрыш по памяти реален;
            # иначе — обычная полная загрузка (безопаснее, файл сразу
            # закрывается). При ЛЮБОЙ ошибке mmap — честный fallback на
            # полную загрузку, а не падение процесса.
            vecs = None
            if os.path.exists(vec_path_npy):
                file_size = os.path.getsize(vec_path_npy)
                try:
                    if file_size >= MMAP_THRESHOLD_BYTES:
                        vecs = np.load(vec_path_npy, mmap_mode="r")
                    else:
                        vecs = np.load(vec_path_npy)
                except Exception as ex:
                    self.log(f"mmap/загрузка векторов не удалась ({ex}), пробую полную загрузку заново...")
                    vecs = np.load(vec_path_npy)

            self.id_order = meta.get("id_order", list(raw_nodes.keys()))
            dim = meta.get("embedder_dim") or (vecs.shape[1] if vecs is not None and len(vecs) else 384)
            for i, nid in enumerate(self.id_order):
                n = dict(raw_nodes[nid])
                n.setdefault("status", "active")
                n["vec"] = vecs[i] if vecs is not None and i < len(vecs) else np.zeros(dim, dtype="float32")
                self.nodes[nid] = n
            return True
        except Exception:
            self.nodes, self.edges, self.branches = {}, [], {}
            self.id_order, self.processed_files = [], {}
            self.files, self.root_project = {}, {}
            return False

    def save(self):
        os.makedirs(self.ptg_dir, exist_ok=True)
        self._acquire_lock()
        try:
            self._save_unlocked()
        finally:
            self._release_lock()

    def _save_unlocked(self):
        nodes_out = {}
        vecs = []
        for nid in self.id_order:
            n = self.nodes[nid]
            vecs.append(n["vec"])
            nodes_out[nid] = {k: v for k, v in n.items() if k != "vec"}

        with open(os.path.join(self.ptg_dir, NODES_FILE), "w", encoding="utf-8") as f:
            json.dump(nodes_out, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.ptg_dir, EDGES_FILE), "w", encoding="utf-8") as f:
            json.dump(self.edges, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.ptg_dir, ROOT_PROJECT_FILE), "w", encoding="utf-8") as f:
            json.dump(self.root_project, f, ensure_ascii=False, indent=2)

        # ПАТЧ 2 — файловый граф
        with open(os.path.join(self.ptg_dir, FILE_EDGES_FILE), "w", encoding="utf-8") as f:
            json.dump(self.file_edges, f, ensure_ascii=False, indent=2)

        # ПАТЧ 1 — root_embedding
        if self.root_embedding is not None:
            np.save(os.path.join(self.ptg_dir, "root_embedding.npy"), self.root_embedding)

        # Path-dependent growth: branch_state.json
        with open(os.path.join(self.ptg_dir, BRANCH_STATE_FILE), "w", encoding="utf-8") as f:
            json.dump(self.branch_states, f, ensure_ascii=False, indent=2)

        # ПАТЧ 1: branch_bodies.json (centroid сериализуем как list)
        # ПАТЧ 10: + temporal_span/dominant_files/dominant_paths/
        # dominant_metadata/semantic_density/reinforcement_count — без
        # этого новые поля сигнатуры ветки терялись бы при каждом save().
        bb_out = {}
        for bid, body in self.branch_bodies.items():
            bb_out[bid] = {
                "atom_ids":  body["atom_ids"],
                "centroid":  body["centroid"].tolist() if hasattr(body["centroid"], "tolist") else body["centroid"],
                "variance":  body["variance"],
                "temporal_span": body.get("temporal_span", {"start": None, "end": None}),
                "dominant_files": body.get("dominant_files", {}),
                "dominant_paths": body.get("dominant_paths", {}),
                "dominant_metadata": body.get("dominant_metadata", {}),
                "semantic_density": body.get("semantic_density", 0.0),
                "reinforcement_count": body.get("reinforcement_count", 0),
            }
        with open(os.path.join(self.ptg_dir, BRANCH_BODIES_FILE), "w", encoding="utf-8") as f:
            json.dump(bb_out, f, ensure_ascii=False, indent=2)

        meta = {
            "branches": self.branches,
            "processed_files": self.processed_files,
            "last_structure_hash": self.last_structure_hash,
            "id_order": self.id_order,
            "files": self.files,                          # ПАТЧ 4 — file-level meta-tree
            "embedder_backend": self.embedder.backend,
            "embedder_model": self.embedder.model,
            "embedder_dim": self.embedder.dim,
            "built_at": time.time(),
            "total_nodes": len(self.nodes),
            "total_branches": len(self.branches),
            # Path-dependent growth
            "path_memory": self.path_memory,
            "atoms_since_decay": self._atoms_since_decay,
        }
        with open(os.path.join(self.ptg_dir, META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        dim = self.embedder.dim or 384
        vec_arr = np.array(vecs, dtype="float32") if vecs else np.zeros((0, dim), dtype="float32")
        self._save_index(vec_arr)

    def _save_index(self, vec_arr):
        try:
            import faiss
            dim = vec_arr.shape[1] if vec_arr.size else (self.embedder.dim or 384)
            index = faiss.IndexFlatIP(dim)
            if vec_arr.size:
                index.add(vec_arr)
            faiss.write_index(index, os.path.join(self.ptg_dir, INDEX_FILE))
        except Exception:
            pass
        np.save(os.path.join(self.ptg_dir, INDEX_FILE + ".npy"), vec_arr)

    # -- построение --------------------------------------------------------
    def build(self):
        """ПАТЧ 19 — тонкая обёртка над _build_impl(): логирует старт/финиш
        и полный traceback при ЛЮБОМ исключении (не только уже явно
        обработанных вроде ConnectionError) в постоянный файловый лог,
        прежде чем пробросить исключение дальше вызывающему коду (GUI/MCP)
        как и раньше. Сама логика сборки не переписывалась и не
        переотступалась — она целиком в _build_impl()."""
        _log.info(f"build(): старт (folder={self.folder!r})")
        t0 = time.time()
        try:
            result = self._build_impl()
            _log.info(f"build(): успешно завершён за {time.time() - t0:.1f}с, "
                      f"{len(self.nodes)} узлов, {len(self.branches)} веток")
            return result
        except Exception:
            _log.exception(f"build(): ОШИБКА после {time.time() - t0:.1f}с")
            raise

    def _build_impl(self):
        self.log("Сканирование файлов...")
        had_existing = self.load_if_exists()
        if had_existing:
            self.log(f"Найден существующий архив с {len(self.nodes)} узлами — продолжаем его (append-only).")

        # ПАТЧ 2 — без LM Studio построение не выполняется, фолбэков нет
        self.log("Проверка соединения с LM Studio...")
        if not self.embedder.test_connection(preferred_model=self.embedder_model_used):
            reason = getattr(self.embedder, "probe_error", None)
            msg = ("LM Studio недоступен на http://localhost:1234, либо ни одна из "
                   "загруженных моделей не отвечает на /v1/embeddings. Запустите LM Studio, "
                   "включите Local Server и загрузите ЛЮБУЮ модель эмбеддингов, затем повторите.")
            if reason:
                msg += f" Причина последней попытки: {reason}"
            self.log(f"ОШИБКА: {msg}")
            raise ConnectionError(msg)

        # ПАТЧ 18 — ГЛАВНЫЙ ФИКС: жёсткая проверка согласованности векторного
        # пространства. Если архив уже содержит атомы, построенные КОНКРЕТНОЙ
        # моделью (self.embedder_model_used), а test_connection() (например,
        # из-за того, что предпочитаемая модель сейчас недоступна) выбрал
        # ДРУГУЮ модель — продолжать НЕЛЬЗЯ: cosine-сравнения между векторами
        # из разных embedding-пространств бессмысленны и молча испортят
        # placement/поиск для всего архива. Раньше (до этого патча) такой
        # проверки не было вообще — реальный инцидент: LM Studio на секунду
        # не ответил рабочей моделью (короткий таймаут, см. фикс #1 выше),
        # код тихо переключился на другую модель ПОСРЕДИ уже строящегося
        # архива, и это прошло бы полностью незамеченным.
        if (self.nodes and self.embedder_model_used
                and self.embedder.model != self.embedder_model_used):
            msg = (
                f"ОСТАНОВЛЕНО: архив уже построен моделью «{self.embedder_model_used}» "
                f"(размерность {self.embedder_dim_used}), а сейчас LM Studio предоставил "
                f"«{self.embedder.model}» (размерность {self.embedder.dim}). Это разные "
                f"векторные пространства — продолжать сборку означало бы молча испортить "
                f"согласованность всего архива (косинусные сравнения между старыми и новыми "
                f"атомами станут бессмысленными). Загрузите в LM Studio именно "
                f"«{self.embedder_model_used}» и повторите, либо создайте новый архив "
                f"(другая папка/output_dir) для новой модели."
            )
            self.log(f"ОШИБКА: {msg}")
            raise ConnectionError(msg)

        self.log(f"LM Studio подключён (модель: {self.embedder.model}).")
        if getattr(self.embedder, "fallback_used", False):
            # ПАТЧ 13: это уже не угадывание по имени — модель прошла
            # реальный пробный вызов /v1/embeddings, так что предупреждение
            # чисто информационное (какая именно модель отличается от
            # изначально настроенной), а не тревога о возможном сбое.
            self.log(
                f"ℹ Настроенная по умолчанию модель '{LM_STUDIO_EMBED_MODEL}' не найдена "
                f"или не прошла проверку — используется '{self.embedder.model}' (она "
                f"успешно ответила на тестовый запрос эмбеддинга)."
            )

        # ПАТЧ 3 — корневой скелет строится при каждом запуске (дёшево, всегда актуален)
        self.log("Построение корневого скелета проекта (root_project.json)...")
        self.root_project = _build_root_project(self.folder)

        # ПАТЧ 1 — вычислить root_embedding как семантический приор
        self.log("Вычисление root_embedding (семантический корень проекта)...")
        root_text = _root_project_to_text(self.root_project)
        rv = self.embedder.embed([root_text])[0]
        n_rv = float(np.linalg.norm(rv))
        self.root_embedding = rv / n_rv if n_rv > 0 else rv
        self.log(f"  root_embedding dim={self.root_embedding.shape[0]}, проект='{self.root_project.get('project_name','?')}'")

        files = scan_folder(self.folder, include_py=self.extract_py_comments)
        new_files = [f for f in files if self.processed_files.get(f) != os.path.getmtime(f)]

        # ПАТЧ 16, часть II — структурный слепок проекта как атомы графа.
        # Делается ДО раннего return "новых файлов нет": структура (папки/
        # файлы/метаданные) может измениться независимо от содержимого
        # TEXT_EXTS-файлов (например, добавили .pdf или новую пустую папку).
        # Хэш сравнивается с прошлым build()-проходом — если структура не
        # менялась, атом не пересоздаётся (не тратим вызов LM Studio зря).
        self.log("Проверка структурного слепка проекта...")
        structure_texts = _build_project_structure_texts(self.folder)
        structure_combined = "\n".join(t for _, t in structure_texts)
        structure_hash = hashlib.sha1(structure_combined.encode("utf-8")).hexdigest()
        if structure_texts and structure_hash != getattr(self, "last_structure_hash", None):
            self.log(f"  Структура изменилась — создаю {len(structure_texts)} атом(ов) слепка...")
            # ПАТЧ 21 — ГЛАВНЫЙ ФИКС реального инцидента: раньше все N текстов
            # структуры эмбеддились ОДНИМ HTTP-запросом без обработки ошибок —
            # при любом сбое (таймаут, смена модели) исключение пробрасывалось
            # наверх и обрывало ВЕСЬ build(), НЕ ДОХОДЯ до разбора настоящих
            # файлов проекта (сотен документов) — структурный слепок,
            # задуманный как приятное дополнение, на практике становился
            # блокирующим препятствием перед основной ценностью build().
            # Теперь: по одному разделу за раз, каждый в своём try/except —
            # частичный сбой одного раздела не теряет остальные и, что
            # важнее всего, НИКОГДА не мешает дойти до реальных файлов ниже.
            build_session_id_struct = f"session-{uuid.uuid4().hex[:12]}"
            structure_hash_partial_ok = True
            for i, (label, text) in enumerate(structure_texts, 1):
                try:
                    self.log(f"  Эмбеддинг структуры: раздел {i}/{len(structure_texts)} «{label}»...")
                    v = self.embedder.embed([text])[0]
                except Exception as ex:
                    self.log(f"  ⚠ Раздел структуры «{label}» не удалось эмбеддить ({ex}) — "
                             f"пропускаю, продолжаю с остальными/файлами.")
                    _log.warning(f"структурный атом «{label}» пропущен: {ex}")
                    structure_hash_partial_ok = False
                    continue
                now_ts = time.time()
                n = float(np.linalg.norm(v))
                struct_file = f"{STRUCTURE_FILE_MARKER}/{label}"
                struct_atom = {
                    "id": str(uuid.uuid4()), "file": struct_file, "order": 0,
                    "question": f"Структура проекта: {label}", "answer": text,
                    "text": text, "confidence": "structure", "timestamp": now_ts,
                    "vec": v / n if n > 0 else v,
                    "source_path": struct_file, "filename": f"structure_{label}",
                    "folder_path": STRUCTURE_FILE_MARKER, "created_at": now_ts,
                    "modified_at": now_ts, "session_id": build_session_id_struct,
                }
                self._add_atom(struct_atom)
            # Хэш фиксируем как "обработанный" только если ВСЕ разделы прошли
            # успешно — иначе следующий build() честно попробует снова
            # недостающие/все разделы, а не решит, что структура "уже учтена".
            if structure_hash_partial_ok:
                self.last_structure_hash = structure_hash
            else:
                self.log("  ⚠ Слепок структуры собран частично — при следующей сборке будет предпринята повторная попытка.")
        else:
            self.log("  Структура не изменилась — новых атомов слепка не требуется.")

        if not new_files:
            self.log("Новых или изменённых файлов не найдено.")
            self.save()
            return self.summary()

        self.log(f"Разбор {len(new_files)} файл(ов): атом = вопрос+ответ...")
        build_session_id = f"session-{uuid.uuid4().hex[:12]}"  # ПАТЧ 9 — один на весь build()-проход
        FILES_PER_CHECKPOINT = 20  # ПАТЧ 21 — периодическое сохранение (см. ниже)
        for i, path in enumerate(new_files, 1):
            atoms = parse_file_to_atoms(path, session_id=build_session_id)
            if not atoms:
                self.log(f"  {os.path.basename(path)}: пригодных Q+A атомов не найдено.")
                self.processed_files[path] = os.path.getmtime(path)
                continue

            # ПАТЧ 21 — ГЛАВНЫЙ ФИКС реального инцидента: раньше сбой embed()
            # НА ОДНОМ файле (таймаут, смена модели, сетевая ошибка) обрывал
            # исключением ВЕСЬ build() — при сотнях файлов в проекте это
            # означало потерю обработки ВСЕХ оставшихся файлов из-за одного
            # неудачного вызова. Теперь: сбой на файле логируется, файл НЕ
            # помечается обработанным (self.processed_files не обновляется —
            # значит на следующей сборке будет предпринята повторная попытка,
            # ничего не потеряно навсегда), и цикл идёт дальше со следующим
            # файлом — так же, как структурные атомы (см. фикс выше).
            try:
                self.log(f"  Эмбеддинг {len(atoms)} атом(ов) из {os.path.basename(path)} через LM Studio...")
                vecs = self.embedder.embed([a["text"] for a in atoms])
            except Exception as ex:
                self.log(f"  ⚠ {os.path.basename(path)}: эмбеддинг не удался ({ex}) — "
                         f"пропускаю, повторная попытка на следующей сборке.")
                _log.warning(f"файл {path} пропущен из-за сбоя embed(): {ex}")
                continue
            for a, v in zip(atoms, vecs):
                n = float(np.linalg.norm(v))
                a["vec"] = v / n if n > 0 else v

            self.log(f"  Построение дерева для {os.path.basename(path)}...")
            file_vecs, file_branches = [], set()
            for atom in atoms:
                node = self._add_atom(atom)
                file_vecs.append(node["vec"])
                file_branches.add(node["branch"])

            # ПАТЧ 4 — файл как семантический объект
            dim = self.embedder.dim or (len(file_vecs[0]) if file_vecs else 384)
            centroid = np.mean(np.array(file_vecs), axis=0) if file_vecs else np.zeros(dim, dtype="float32")
            combined_text = " ".join(a["text"] for a in atoms).lower()
            dominant = [c for c in self.root_project.get("known_concepts", []) if c.lower() in combined_text][:20]
            self.files[path] = {
                "file_id": path,
                "centroid": centroid.tolist(),
                "main_branches": sorted(file_branches),
                "dominant_concepts": dominant,
            }

            self.processed_files[path] = os.path.getmtime(path)

            # ПАТЧ 21 — периодическое сохранение: если что-то прервёт процесс
            # ПОЗЖЕ (закрытие приложения, обрыв питания, не пойманное здесь
            # исключение) — уже обработанные файлы не потеряются полностью.
            if i % FILES_PER_CHECKPOINT == 0:
                self.log(f"  Промежуточное сохранение ({i}/{len(new_files)} файлов обработано)...")
                self._build_file_graph()
                self.save()

        # ПАТЧ 2 — построить файловый граф после обработки всех файлов
        self.log("Построение файлового графа (попарное сравнение центроидов)...")
        self._build_file_graph()

        self.log("Сохранение...")
        self.save()
        self.log("Готово.")
        return self.summary()


    # -- Path-dependent growth: вспомогательные методы (ПАТЧ 1, 5, 6, 7) ----

    def _ensure_branch_state(self, branch_id: str) -> dict:
        """Вернуть (и при необходимости создать) branch_state для ветки."""
        if branch_id not in self.branch_states:
            self.branch_states[branch_id] = {
                "branch_id": branch_id,
                "weight":     0.0,
                "momentum":   0.0,
                "entropy":    0.5,   # начальная неопределённость
                "activation": 1.0,
                "last_seen":  len(self.id_order),
                "atom_count": 0,
                "dormant":    False,
            }
        return self.branch_states[branch_id]

    def _update_branch_states(self, chosen_bid: str, root_sim: float) -> None:
        """Обновить branch_states после графтинга.
        Выбранная ветка: weight/momentum/activation растут.
        Остальные: momentum и activation затухают.
        ПАТЧ 4 — True Entropy: берём variance из branch_body.
        ПАТЧ 10 — фикс подтверждённого бага аудита #3: параметр node_vec
        был объявлен, но нигде не использовался внутри метода (мёртвый
        параметр). Убран из сигнатуры; вызов ниже обновлён соответственно.
        Поведение метода не изменилось — только устранён мёртвый код."""
        cur_index = len(self.id_order)

        for bid, bs in self.branch_states.items():
            if bid == chosen_bid:
                bs["weight"]     += root_sim
                bs["momentum"]   += 1.0
                bs["activation"]  = 1.0
                bs["last_seen"]   = cur_index
                bs["atom_count"] += 1
                # ПАТЧ 4: entropy = normalized variance из branch_body
                body = self.branch_bodies.get(bid)
                if body and body["atom_ids"]:
                    # variance уже нормирована в [0,1] через _update_branch_body
                    bs["entropy"] = float(np.clip(body["variance"], 0.0, 1.0))
            else:
                bs["momentum"]   *= 0.97
                bs["activation"] *= 0.95

    def _decay_branches(self) -> None:
        """ПАТЧ 7: запускается каждые BRANCH_DECAY_INTERVAL атомов.
        Ветки, не видевшие новых атомов BRANCH_DECAY_IDLE_THRESH+ шагов, затухают.
        Никакие ветки не удаляются."""
        cur_index = len(self.id_order)
        for bid, bs in self.branch_states.items():
            idle = cur_index - bs.get("last_seen", 0)
            if idle >= BRANCH_DECAY_IDLE_THRESH:
                bs["activation"] *= 0.8
                bs["momentum"]   *= 0.8
            if bs["activation"] < BRANCH_DORMANT_THRESH:
                bs["dormant"] = True

    def _reactivate_branch(self, branch_id: str) -> None:
        """ПАТЧ 6: вызывается при returns_to — реактивирует дремлющую ветку."""
        bs = self._ensure_branch_state(branch_id)
        bs["activation"] = min(1.0, bs["activation"] + 0.4)
        bs["momentum"]   = min(5.0, bs["momentum"]   + 0.5)
        bs["dormant"]    = bs["activation"] < BRANCH_DORMANT_THRESH

    def _path_coherence(self, vec) -> tuple[float, float, float]:
        """ПАТЧ 3: двойная path coherence.
        short: среднее сходство с short path (последние 20 атомов).
        long:  среднее сходство с центроидами веток из long path.
        Возвращает (short_coh, long_coh, path_score)."""
        # Short coherence
        short_sims = []
        for pid in self.path_memory["short"][-CURRENT_PATH_WINDOW:]:
            pn = self.nodes.get(pid)
            if pn is not None:
                short_sims.append(float(np.dot(vec, pn["vec"])))
        short_coh = float(np.mean(short_sims)) if short_sims else 0.5

        # Long coherence — сравниваем с центроидами веток из lineage
        long_sims = []
        for bid in self.path_memory["long"]:
            body = self.branch_bodies.get(bid)
            if body is None:
                continue
            cv = body["centroid"]
            if cv is not None:
                try:
                    cv_arr = cv if isinstance(cv, np.ndarray) else np.array(cv, dtype="float32")
                    if cv_arr.shape == vec.shape:
                        long_sims.append(float(np.dot(vec, cv_arr)))
                except Exception:
                    pass
        long_coh = float(np.mean(long_sims)) if long_sims else 0.5

        path_score = 0.7 * short_coh + 0.3 * long_coh
        return short_coh, long_coh, path_score


    def _file_origin_affinity(self, file_a: str, file_b: str) -> float:
        """ПАТЧ 9 — ось Wf placement-score. 1.0 если тот же файл (одна
        физическая история), иначе — сходство из уже существующего
        file_edges (та же "файловая линия" по центроидам), иначе 0.0
        (разные, никак не связанные корни архива — часть III, CASE C)."""
        if file_a == file_b:
            return 1.0
        for fe in self.file_edges:
            if {fe["source_file"], fe["target_file"]} == {file_a, file_b}:
                return float(fe["similarity"])
        return 0.0

    def _local_semantic_density(self, branch_id: str) -> int:
        """ПАТЧ 9, часть VIII — сколько из последних LOCAL_DENSITY_WINDOW
        атомов тела ветки имеют cosine к ТЕКУЩЕМУ центроиду ветки выше
        LOCAL_DENSITY_HIGH_COSINE. Высокое значение = ветка схлопывается
        в плотный семантический сгусток ("semantic black hole") — сигнал
        форсировать латеральное расширение вместо дальнейшего углубления."""
        body = self.branch_bodies.get(branch_id)
        if not body or body.get("centroid") is None:
            return 0
        cv = body["centroid"]
        cv = cv if isinstance(cv, np.ndarray) else np.array(cv, dtype="float32")
        recent = body["atom_ids"][-LOCAL_DENSITY_WINDOW:]
        high = 0
        for nid in recent:
            n = self.nodes.get(nid)
            if n is None:
                continue
            if float(np.dot(n["vec"], cv)) >= LOCAL_DENSITY_HIGH_COSINE:
                high += 1
        return high

    def _ensure_branch_body(self, branch_id: str) -> dict:
        """ПАТЧ 1: создать запись тела ветки при первом обращении.
        centroid=None до добавления первого атома — инициализируется в _update_branch_body.
        ПАТЧ 10 — добавлены обязательные поля "ветка как активное состояние"
        (temporal_span/dominant_files/dominant_paths/dominant_metadata/
        semantic_density/reinforcement_count), заполняются в _update_branch_signature()."""
        if branch_id not in self.branch_bodies:
            self.branch_bodies[branch_id] = {
                "atom_ids": [],
                "centroid":  None,  # будет инициализирован при первом атоме
                "variance":  0.0,
                "temporal_span": {"start": None, "end": None},
                "dominant_files": {},      # {source_path: count}
                "dominant_paths": {},      # {folder_path: count}
                "dominant_metadata": {},   # {session_id: count}
                "semantic_density": 0.0,   # см. _local_semantic_density, нормировано [0,1]
                "reinforcement_count": 0,  # сумма reinforces-событий, нацеленных на атомы этой ветки
            }
        return self.branch_bodies[branch_id]

    def _update_branch_signature(self, branch_id: str, node: dict) -> None:
        """ПАТЧ 10, часть III/VIII — "ветка как активное состояние". Вызывается
        из _add_atom() сразу после _update_branch_body() для КАЖДОГО атома
        (не пост-обработкой). Обновляет temporal_span, dominant_files/
        dominant_paths/dominant_metadata (частотные счётчики — файловая
        топология активно участвует в описании ветки, а не только в
        placement-score) и semantic_density (для density_penalty оси)."""
        body = self._ensure_branch_body(branch_id)
        ts = node.get("timestamp")
        span = body["temporal_span"]
        if ts is not None:
            span["start"] = ts if span["start"] is None else min(span["start"], ts)
            span["end"]   = ts if span["end"]   is None else max(span["end"],   ts)
        fpath = node.get("source_path") or node.get("file")
        if fpath:
            body["dominant_files"][fpath] = body["dominant_files"].get(fpath, 0) + 1
        fdir = node.get("folder_path")
        if fdir:
            body["dominant_paths"][fdir] = body["dominant_paths"].get(fdir, 0) + 1
        sid = node.get("session_id")
        if sid:
            body["dominant_metadata"][sid] = body["dominant_metadata"].get(sid, 0) + 1
        body["semantic_density"] = round(
            self._local_semantic_density(branch_id) / float(LOCAL_DENSITY_WINDOW), 4
        )

    def _update_branch_body(self, branch_id: str, new_vec: np.ndarray) -> float:
        """ПАТЧ 1: инкрементально обновить центроид и variance ветки.

        Формула центроида:
            new_centroid = (old_centroid * n + new_vec) / (n + 1)

        Variance (mean cosine deviation от центроида):
            обновляется как rolling mean отклонений."""
        body = self._ensure_branch_body(branch_id)
        n = len(body["atom_ids"])

        old_c = body["centroid"]
        if old_c is None:
            # первый атом — инициализируем центроид его вектором
            old_c = np.zeros_like(new_vec)
        elif isinstance(old_c, list):
            old_c = np.array(old_c, dtype="float32")

        # Инкрементальный центроид
        new_c = (old_c * n + new_vec) / (n + 1)
        # нормализуем центроид чтобы dot-произведения оставались в [-1,1]
        nc_norm = float(np.linalg.norm(new_c))
        if nc_norm > 0:
            new_c = new_c / nc_norm

        # Cosine deviation нового атома от обновлённого центроида
        dev = 1.0 - float(np.dot(new_vec, new_c))

        # Rolling variance: среднее отклонение по всем атомам (приближение)
        old_var = body["variance"]
        new_var = (old_var * n + dev) / (n + 1)
        # нормируем в [0,1]: max отклонение = 2.0 (противоположные векторы)
        new_var_norm = float(np.clip(new_var / 2.0, 0.0, 1.0))

        body["centroid"]  = new_c
        body["variance"]  = new_var_norm
        return float(np.dot(new_vec, new_c))  # возвращаем sim к новому центроиду

    def _add_atom(self, atom):
        vec = atom["vec"]
        node_id = atom["id"]
        cur_index = len(self.id_order)

        # Корневой приор
        root_sim = 0.0
        if self.root_embedding is not None:
            root_sim = float(np.dot(vec, self.root_embedding))
        off_root = root_sim < OFF_ROOT_THRESH

        # ПАТЧ 3: dual path coherence (short + long)
        short_coh, long_coh, path_score = self._path_coherence(vec)

        # ПАТЧ 1+5: ранжировать ветки по graft-score через branch CENTROID, не head
        # score = 0.40*sem + 0.25*path + 0.15*mom + 0.10*root + 0.10*act - 0.20*entropy
        head_ids = {b["head"] for b in self.branches.values()}
        best_head_branch, best_head_sim, best_graft_score = None, -1.0, -1.0
        for bid, b in self.branches.items():
            # ПАТЧ 1: семантическое сходство с ЦЕНТРОИДОМ тела ветки (не с head)
            body = self.branch_bodies.get(bid)
            if body and len(body["atom_ids"]) > 0 and body["centroid"] is not None:
                cv = body["centroid"]
                cv_arr = cv if isinstance(cv, np.ndarray) else np.array(cv, dtype="float32")
                if cv_arr.shape == vec.shape:
                    sem_sim = float(np.dot(vec, cv_arr))
                else:
                    sem_sim = float(np.dot(vec, self.nodes[b["head"]]["vec"]))
            else:
                # тело ещё не построено — откат к head
                hv = self.nodes[b["head"]]["vec"]
                sem_sim = float(np.dot(vec, hv))
            bs = self._ensure_branch_state(bid)
            mom_n = float(np.tanh(bs["momentum"] / 5.0))
            act_n = float(bs["activation"])  # уже в [0,1]
            score = (W_SEM  * sem_sim
                   + W_PATH * path_score
                   + W_MOM  * mom_n
                   + W_ROOT * root_sim
                   + W_ACT  * act_n
                   - W_ENT  * bs["entropy"])
            if score > best_graft_score:
                best_graft_score = score
                best_head_branch = bid
                best_head_sim    = sem_sim   # используем centroid-sim для порогов

        # 2. сравнение со старыми узлами, не являющимися "головами"
        best_old_id, best_old_sim = None, -1.0
        for nid in self.id_order:
            n = self.nodes[nid]
            if n["index"] >= cur_index - DEEP_NODE_MIN_AGE or nid in head_ids:
                continue
            sim = float(np.dot(vec, n["vec"]))
            if sim > best_old_sim:
                best_old_sim, best_old_id = sim, nid

        # 3. ПАТЧ 9 — cosine safety logic (часть IV) + multi-axis placement score
        #    (часть III). Раньше решение continues/branches зависело ТОЛЬКО от
        #    sem_sim (best_head_sim). Это и есть источник semantic coagulation:
        #    высокий cosine автоматически трактовался как "буквальное
        #    продолжение той же ветки", даже если атом отстоит на месяцы,
        #    из другого файла, из другой сессии импорта. Ниже — явные кейсы
        #    A/B/C/D из аудита; "merge" (=continues в этой модели) разрешён
        #    ТОЛЬКО в кейсе D (тот же файл + тот же локальный кластер).
        placement_case = None
        placement_info = None
        candidate_id = None

        if best_head_branch is None or best_head_sim < BRANCH_THRESH:
            # низкое сходство — новая, семантически несвязанная ветка.
            # Это не CASE-логика (не о чем спорить: связи с существующим нет).
            branch_id, parent = None, None
        else:
            candidate_id = self.branches[best_head_branch]["head"]
            candidate = self.nodes[candidate_id]
            cand_ts = candidate.get("timestamp", atom["timestamp"])
            cand_bs = self.branch_states.get(best_head_branch, {})

            # ПАТЧ 10 — явные оси placement-score (ни одна не implicit):
            temporal_aff  = _temporal_affinity(atom["timestamp"], cand_ts)               # Wt
            file_aff      = self._file_origin_affinity(atom["file"], candidate.get("file"))  # Wf
            meta_aff      = _metadata_affinity(atom, candidate)                          # Wm
            # path_history_affinity = path_score, уже посчитан выше через _path_coherence  # Wp
            entropy_aff   = 1.0 - float(np.clip(cand_bs.get("entropy", 0.0), 0.0, 1.0))  # We:
            #   чем "спокойнее" (менее энтропийна) ветка-кандидат, тем выше
            #   безопасность деформировать/расширять именно её.
            density_now   = self._local_semantic_density(best_head_branch)               # Wd:
            density_penalty = 1.0 - min(1.0, density_now / float(LOCAL_DENSITY_WINDOW))
            #   density_penalty — это "запас плотности" (1 - заполненность
            #   окна высоко-косинусными соседями), а не сырой штраф: так он
            #   складывается с остальными affinity-осями по одной и той же
            #   логике "выше = безопаснее сливать".

            placement_score = (PLACEMENT_W_SEM  * best_head_sim
                              + PLACEMENT_W_TEMP * temporal_aff
                              + PLACEMENT_W_FILE * file_aff
                              + PLACEMENT_W_META * meta_aff
                              + PLACEMENT_W_PATH * path_score
                              + PLACEMENT_W_ENT  * entropy_aff
                              + PLACEMENT_W_DENS * density_penalty)
            same_file = atom["file"] == candidate.get("file")
            in_local_cluster = candidate_id in self.path_memory["short"]
            time_days = abs(atom["timestamp"] - cand_ts) / 86400.0

            if best_head_sim <= CONTINUE_THRESH:
                # ПАТЧ 9 чинит старый баг: раньше зона [BRANCH_THRESH,
                # CONTINUE_THRESH] молча вела себя как continues (см. аудит,
                # находка №1). Теперь это никогда не буквальное продолжение.
                allow_continue = False
                placement_case = "MEDIUM_SIBLING"
            elif same_file and in_local_cluster:
                allow_continue = True
                placement_case = "D_LOCAL_MERGE"
            elif same_file and time_days <= REINFORCE_TEMPORAL_DAYS and file_aff >= FILE_GRAPH_LINEAGE_MIN:
                allow_continue = False
                placement_case = "A_REINFORCEMENT"
            elif time_days > REINFORCE_TEMPORAL_DAYS:
                allow_continue = False
                placement_case = "B_TEMPORAL_SIBLING"
            elif not same_file and file_aff < FILE_GRAPH_LINEAGE_MIN:
                allow_continue = False
                placement_case = "C_ROOT_SIBLING"
            else:
                allow_continue = False
                placement_case = "A_REINFORCEMENT"

            # Semantic density limiter (часть IX задания / VIII аудита):
            # даже легитимный CASE D блокируется, если ветка уже перегружена
            # высоко-косинусными атомами подряд — принудительное латеральное
            # расширение вместо бесконечного углубления одной ветки.
            # ПАТЧ 10: используем тот же density_now, что уже посчитан для
            # оси Wd выше — не пересчитываем дважды.
            if allow_continue and density_now >= LOCAL_DENSITY_LIMIT:
                allow_continue = False
                placement_case = "E_DENSITY_LIMIT_LATERAL"

            placement_info = {
                "placement_score": round(placement_score, 4),
                "semantic_similarity": round(best_head_sim, 4),
                "temporal_affinity": round(temporal_aff, 4),
                "file_origin_affinity": round(file_aff, 4),
                "metadata_affinity": round(meta_aff, 4),
                "path_history_affinity": round(path_score, 4),
                "entropy_affinity": round(entropy_aff, 4),
                "density_penalty": round(density_penalty, 4),
                "case": placement_case,
                "candidate_id": candidate_id,
            }

            if allow_continue:
                branch_id, parent = best_head_branch, candidate_id
            else:
                # Кейсы A/B/C/MEDIUM/E: узел не наследует ветку кандидата —
                # это либо новая sibling-ветка (реинфорсит candidate, но не
                # деформирует его линию), либо (E) латеральный сплит той же
                # линии. И то, и другое реализуется одинаково на уровне
                # branches/edges: новая ветка + причинная связь к candidate.
                branch_id, parent = None, candidate_id

        depth = 0 if parent is None else self.nodes[parent]["depth"] + 1
        node = {
            "id": node_id, "file": atom["file"], "order": atom["order"],
            "question": atom.get("question", ""), "answer": atom.get("answer", ""),
            "text": atom["text"], "confidence": atom.get("confidence", "sequential"),
            "timestamp": atom["timestamp"], "depth": depth, "parent": parent, "index": cur_index,
            "status": "active", "vec": vec,
            # ПАТЧ 1 — семантическое выравнивание с корнем проекта
            "root_similarity": round(root_sim, 4),
            "off_root": off_root,
            # ПАТЧ 9, часть VI — временной/метаданный слой на самом узле
            "source_path": atom.get("source_path", atom["file"]),
            "filename": atom.get("filename"),
            "folder_path": atom.get("folder_path"),
            "created_at": atom.get("created_at", atom["timestamp"]),
            "modified_at": atom.get("modified_at", atom["timestamp"]),
            "import_timestamp": time.time(),
            "session_id": atom.get("session_id"),
            # ПАТЧ 9, часть V — модель избыточности (reinforcement)
            "reinforcement_count": 0,
            "reinforcement_links": [],
            "repetition_epochs": [],
            "placement_case": placement_case,
            "lateral_expansion": placement_case == "E_DENSITY_LIMIT_LATERAL",
        }

        if branch_id is None:
            branch_id = node_id
            new_epoch = 0
            if candidate_id is not None:
                cand_branch = self.nodes[candidate_id].get("branch")
                cand_epoch = self.branches.get(cand_branch, {}).get("semantic_epoch", 0)
                # E: латеральный сплит той же линии — эпоха не растёт.
                # A/B/C/MEDIUM: реинфорсмент прежней идеи — новая эпоха.
                new_epoch = cand_epoch if placement_case == "E_DENSITY_LIMIT_LATERAL" else cand_epoch + 1
            self.branches[branch_id] = {
                "root": node_id, "head": node_id, "created_order": cur_index,
                "semantic_epoch": new_epoch,
            }
            node["semantic_epoch"] = new_epoch
            if parent is not None:
                # ПАТЧ 9, часть V — reinforces вместо молчаливого continues,
                # КРОМЕ density-limiter (E): там это буквально та же линия,
                # просто структурно расщеплённая — сохраняем причинность
                # через continues, а не reinforces.
                edge_type = "continues" if placement_case == "E_DENSITY_LIMIT_LATERAL" else "reinforces"
                self.edges.append({"from": node_id, "to": parent, "type": edge_type})
                if edge_type == "reinforces":
                    cand_node = self.nodes[parent]
                    cand_node["reinforcement_count"] = cand_node.get("reinforcement_count", 0) + 1
                    cand_node.setdefault("reinforcement_links", []).append(node_id)
                    # ПАТЧ 10, часть VI — repetition_epochs: список эпох, в
                    # которых эта идея была повторена (не перезаписывается,
                    # только растёт — структурная персистентность повторов).
                    cand_node.setdefault("repetition_epochs", []).append(new_epoch)
                    # ПАТЧ 10, часть III — reinforcement_count на самой ветке
                    # кандидата (не только на её головном узле): чтобы branch-
                    # first поиск мог ранжировать ветки по "часто ли к этой
                    # линии возвращались", не заглядывая в каждый узел.
                    cand_branch_body = self.branch_bodies.get(cand_node.get("branch"))
                    if cand_branch_body is not None:
                        cand_branch_body["reinforcement_count"] = cand_branch_body.get("reinforcement_count", 0) + 1
        else:
            self.branches[branch_id]["head"] = node_id
            node["semantic_epoch"] = self.branches[branch_id].get("semantic_epoch", 0)
            self.edges.append({"from": parent, "to": node_id, "type": "continues"})

        if placement_info is not None:
            node["placement_info"] = placement_info
            _log.debug(
                f"placement: атом {node_id[:8]} file={atom['file']} "
                f"case={placement_case} branch={branch_id[:8] if branch_id else '?'} "
                f"score={placement_info['placement_score']} sem={placement_info['semantic_similarity']} "
                f"temp={placement_info['temporal_affinity']} file_aff={placement_info['file_origin_affinity']}"
            )
        else:
            _log.debug(f"placement: атом {node_id[:8]} file={atom['file']} "
                      f"case=NEW_UNRELATED branch={branch_id[:8] if branch_id else '?'} (нет кандидата)")

        # ПАТЧ 3: записать обе компоненты path coherence
        node["path_coherence"]       = round(path_score,  4)
        node["short_path_coherence"] = round(short_coh,   4)
        node["long_path_coherence"]  = round(long_coh,    4)
        node["branch"] = branch_id
        self.nodes[node_id] = node
        self.id_order.append(node_id)

        # ПАТЧ 2: обновить иерархическую память пути
        self.path_memory["short"].append(node_id)
        if len(self.path_memory["short"]) > CURRENT_PATH_WINDOW:
            self.path_memory["short"].pop(0)
        # long path: добавляем branch_id только при СМЕНЕ ветки
        if branch_id != self._prev_branch_id:
            self.path_memory["long"].append(branch_id)
            self._prev_branch_id = branch_id

        # ПАТЧ 1: обновить тело ветки (centroid + variance)
        self._ensure_branch_body(branch_id)
        self.branch_bodies[branch_id]["atom_ids"].append(node_id)
        self._update_branch_body(branch_id, vec)
        # ПАТЧ 10: обновить сигнатуру ветки (temporal_span/dominant_*/density)
        self._update_branch_signature(branch_id, node)

        # обновить branch_states (true entropy берётся из branch_body)
        self._ensure_branch_state(branch_id)
        self._update_branch_states(branch_id, root_sim)

        # ПАТЧ 7: branch decay каждые N атомов
        self._atoms_since_decay += 1
        if self._atoms_since_decay >= BRANCH_DECAY_INTERVAL:
            self._decay_branches()
            self._atoms_since_decay = 0

        # returns_to: реактивация + ПАТЧ 6: восстановление в long path lineage
        if best_old_id is not None and best_old_sim > RETURN_THRESH:
            self.edges.append({"from": node_id, "to": best_old_id, "type": "returns_to"})
            return_target_branch = self.nodes[best_old_id].get("branch")
            if return_target_branch and return_target_branch != branch_id:
                self._reactivate_branch(return_target_branch)
                # ПАТЧ 6: реинтродуцировать ветку в lineage long path
                self.path_memory["long"].append(return_target_branch)

        # Расширенные семантические рёбра (contradicts/fixes/refines/supersedes).
        # Старые типы рёбер не убираются, эти добавляются поверх.
        #
        # ПАТЧ 3 — root-aware steering:
        # root_sim >= 0.55 означает, что атом выровнен с концептами проекта.
        # Это смещает пороги в сторону более сильных утверждений об отношениях:
        #   A) deepens root concepts  → bias к refines (порог снижается)
        #   B) opposes root concepts  → bias к contradicts (порог снижается)
        #   C) repairs root-aligned   → bias к fixes (порог снижается)
        #   D) replaces root-aligned  → bias к supersedes (порог снижается)
        # Если root_sim < 0.55 — атом периферийный, пороги остаются стандартными.
        ROOT_STEERING_THRESH = 0.55
        root_aligned = root_sim >= ROOT_STEERING_THRESH
        # bias-коэффициент: root_aligned → 0.90 (мягче), иначе 1.0 (стандарт)
        _bias = 0.90 if root_aligned else 1.0

        for target_id, sim in ((parent, best_head_sim), (best_old_id, best_old_sim)):
            if target_id is None or target_id not in self.nodes:
                continue
            target_text = self.nodes[target_id]["text"]
            target_root_sim = self.nodes[target_id].get("root_similarity", 0.0)
            target_root_aligned = target_root_sim >= ROOT_STEERING_THRESH

            # B) contradicts:
            # root_aligned атом с маркерами отрицания к root_aligned цели —
            # это противоречие внутри области проекта: снижаем порог (_bias)
            contradict_thresh = CONTRADICT_SIM_THRESH * _bias if (root_aligned and target_root_aligned) else CONTRADICT_SIM_THRESH
            if sim > contradict_thresh and _has_marker(node["text"], NEGATION_MARKERS):
                self.edges.append({"from": node_id, "to": target_id, "type": "contradicts"})

            # C) fixes:
            # root_aligned атом, чинящий root_aligned цель — значимее:
            # добавляем ребро fixes даже без маркера, если sim высок
            has_fix_marker = _has_marker(node["text"], FIX_KEYWORDS)
            fix_by_sim = root_aligned and target_root_aligned and sim > CONTINUE_THRESH
            if has_fix_marker or fix_by_sim:
                self.edges.append({"from": node_id, "to": target_id, "type": "fixes"})

            # A) refines:
            # root_aligned атом, углубляющий root_aligned цель — snижаем порог
            refine_sim_thresh = CONTINUE_THRESH * _bias
            if sim > refine_sim_thresh and len(node["text"]) > len(target_text) * REFINE_MIN_LEN_RATIO:
                self.edges.append({"from": node_id, "to": target_id, "type": "refines"})

            # D) supersedes:
            # root_aligned атом с маркерами замены к root_aligned цели —
            # семантически значимее: снижаем порог
            supersede_thresh = SUPERSEDE_SIM_THRESH * _bias if (root_aligned and target_root_aligned) else SUPERSEDE_SIM_THRESH
            if sim > supersede_thresh and _has_marker(node["text"], SUPERSEDE_MARKERS):
                self.edges.append({"from": node_id, "to": target_id, "type": "supersedes"})
                self.nodes[target_id]["status"] = "superseded"

        return node

    # -- ПАТЧ 2: файловый граф (попарное сравнение центроидов) --------
    def _build_file_graph(self):
        """После обработки всех файлов: сравнить центроиды попарно.
        Результат — file_edges.json + поле connected_files в каждом file-объекте.
        Ни одного атомного ребра здесь не используется — только центроиды."""
        paths = [p for p, fd in self.files.items() if fd.get("centroid")]
        if len(paths) < 2:
            return

        centroids = np.array(
            [self.files[p]["centroid"] for p in paths], dtype="float32"
        )
        # нормализация (на случай если центроид не нормирован)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        centroids = centroids / norms

        # сбросить старые рёбра файлового графа (пересчитываем при каждом build)
        self.file_edges = []
        connected: dict[str, list[str]] = {p: [] for p in paths}

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                sim = float(np.dot(centroids[i], centroids[j]))
                if sim >= FILE_GRAPH_THRESH:
                    self.file_edges.append({
                        "source_file": paths[i],
                        "target_file": paths[j],
                        "similarity": round(sim, 4),
                        "relation": "related",
                    })
                    connected[paths[i]].append(paths[j])
                    connected[paths[j]].append(paths[i])

        for p in paths:
            self.files[p]["connected_files"] = connected[p]

        self.log(f"  Файловый граф: {len(self.file_edges)} рёбер между {len(paths)} файлами.")

    def summary(self):
        by_type = {}
        for e in self.edges:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "total_branches": len(self.branches),
            "total_files": len(self.files),
            "edges_by_type": by_type,
            "embedder_backend": self.embedder.backend,
            "embedder_model": self.embedder.model,
        }

    # -- вспомогательные методы для GUI -----------------------------------
    def search(self, query, top_k=15):
        """ПАТЧ 3 — иерархический поиск:
          Шаг 1: эмбеддинг запроса
          Шаг 2: сравнить с центроидами файлов → TOP-N файлов
          Шаг 3: только внутри этих файлов искать атомы
          Шаг 4: ранжировать и вернуть top_k
        Глобальный перебор всех атомов запрещён."""
        if not self.nodes:
            return []

        # Шаг 1
        qv = self.embedder.embed([query])[0]
        n = float(np.linalg.norm(qv))
        qv = qv / n if n > 0 else qv

        # Шаг 2 — ранжирование файлов по центроиду
        file_scores = []
        for fpath, fdata in self.files.items():
            c = fdata.get("centroid")
            if not c:
                continue
            cv = np.array(c, dtype="float32")
            cn = float(np.linalg.norm(cv))
            cv = cv / cn if cn > 0 else cv
            file_scores.append((float(np.dot(qv, cv)), fpath))
        file_scores.sort(key=lambda x: -x[0])
        primary_top = [fpath for _, fpath in file_scores[:LAYERED_TOP_FILES]]
        top_files: set[str] = set(primary_top)

        # ПАТЧ 2 — расширить кандидатный набор через файловый граф:
        # для каждого top-файла добавить его connected_files (ranked по sim ребра)
        LAYERED_MAX_FILES = 12
        if len(top_files) < LAYERED_MAX_FILES:
            # собрать все рёбра файлового графа в словарь: файл -> [(sim, сосед)]
            _graph: dict[str, list[tuple[float, str]]] = {}
            for fe in self.file_edges:
                s, t, sim_fe = fe["source_file"], fe["target_file"], fe["similarity"]
                _graph.setdefault(s, []).append((sim_fe, t))
                _graph.setdefault(t, []).append((sim_fe, s))
            # обходим primary_top в порядке убывания релевантности
            for pf in primary_top:
                for edge_sim, neighbor in sorted(_graph.get(pf, []), reverse=True):
                    if neighbor not in top_files:
                        top_files.add(neighbor)
                    if len(top_files) >= LAYERED_MAX_FILES:
                        break
                if len(top_files) >= LAYERED_MAX_FILES:
                    break

        # Если файловый граф пуст (архив без file-объектов — старый формат),
        # откатываемся к поиску по всем файлам присутствующих в архиве
        if not top_files:
            top_files = {self.nodes[nid]["file"] for nid in self.id_order}

        # Шаг 3+4 — атомы только из топ-файлов
        scored = []
        for nid in self.id_order:
            node = self.nodes[nid]
            if node["file"] not in top_files:
                continue
            sim = float(np.dot(qv, node["vec"]))
            scored.append((sim, nid))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    # =====================================================================
    # ПАТЧ 10 — Branch-first retrieval
    # =====================================================================
    # search() выше — file-first (L1 файлы → L2 файловый граф → L3 атомы).
    # Он НЕ удаляется и НЕ заменяется (обратная совместимость: GUI и
    # существующие MCP-вызовы могут на него полагаться). search_branch_first()
    # — параллельный маршрут: query → root branch scoring → top branches →
    # branch traversal → local atom harvest → context assembly. Это то, что
    # явно требуется заданием как ОСНОВНОЙ путь для агента (см. ptg_mcp_server.py:
    # ptg_search теперь вызывает именно этот метод).
    #
    # Почему это не то же самое, что file-first: branch — это единица
    # ТРАЕКТОРИИ мышления (может пересекать несколько файлов через
    # continues/reinforces), а file — единица ФИЗИЧЕСКОГО источника. Поиск
    # "с какой веткой я сейчас работаю" и "в каком файле это лежит" —
    # разные вопросы; branch-first отвечает на первый.
    def search_branch_first(self, query: str, top_branches: int = 6,
                             top_k: int = 15) -> list:
        """Часть X задания. Возвращает [(score, node_id), ...] — тот же
        формат, что и search(), для обратной совместимости вызывающего кода.

        Шаги:
          1. embed query
          2. root branch scoring — cosine(query, branch centroid) с лёгким
             temporal-bias (часть V: время влияет и на traversal, не только
             на insertion) — недавно активные ветки получают небольшой бонус,
             т.к. они вероятнее продолжают текущий контекст работы.
          3. top_branches веток по итоговому score
          4. branch traversal — внутри каждой ветки берём её atom_ids
             (реальная топология ветки, не глобальный список атомов)
          5. local atom harvest — ранжируем атомы внутри выбранных веток по
             cosine к query, с ограничением атомов-на-файл (часть VIII:
             файловая топология активна в traversal — не даём одному файлу
             монополизировать harvest внутри ветки)
          6. возвращаем top_k атомов, отсортированных по итоговому score
        """
        if not self.nodes or not self.branch_bodies:
            return []

        qv = self.embedder.embed([query])[0]
        qn = float(np.linalg.norm(qv))
        qv = qv / qn if qn > 0 else qv
        now = time.time()

        # 2-3. root branch scoring (+ лёгкий temporal-bias на traversal)
        branch_scores = []
        for bid, body in self.branch_bodies.items():
            cv = body.get("centroid")
            if cv is None:
                continue
            cv = cv if isinstance(cv, np.ndarray) else np.array(cv, dtype="float32")
            sem = float(np.dot(qv, cv))
            head_id = self.branches.get(bid, {}).get("head")
            head_ts = self.nodes.get(head_id, {}).get("timestamp", now) if head_id else now
            recency = _temporal_affinity(head_ts, now, half_life_days=90.0)
            # 85% семантика / 15% recency — семантика остаётся ведущей осью,
            # recency — тай-брейкер между близкими по смыслу ветками
            branch_scores.append((0.85 * sem + 0.15 * recency, bid))
        branch_scores.sort(key=lambda x: -x[0])
        chosen_branches = [bid for _, bid in branch_scores[:top_branches]]

        # 4-5. branch traversal + local atom harvest (файловая топология
        # ограничивает вклад одного файла внутри ветки — часть VIII)
        MAX_ATOMS_PER_FILE_IN_BRANCH = 4
        scored = []
        for bid in chosen_branches:
            body = self.branch_bodies[bid]
            per_file_count: dict[str, int] = {}
            for nid in body["atom_ids"]:
                node = self.nodes.get(nid)
                if node is None:
                    continue
                fpath = node.get("source_path", node.get("file"))
                if per_file_count.get(fpath, 0) >= MAX_ATOMS_PER_FILE_IN_BRANCH:
                    continue
                sim = float(np.dot(qv, node["vec"]))
                scored.append((sim, nid))
                per_file_count[fpath] = per_file_count.get(fpath, 0) + 1

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def preview(self, node_id):
        """Полный срез одного узла для GUI (PreviewPanel). ВОССТАНОВЛЕН
        (был утерян при более раннем патче — тело метода стало мёртвым
        кодом после return внутри search_branch_first(), а сам preview()
        перестал существовать как атрибут класса; main.py при этом продолжал
        его вызывать, что привело бы к AttributeError при первом клике по
        атому в дереве/поиске). Заголовок и is_unresolved — единственное
        отличие от исходной версии (см. ПАТЧ 8: _is_unresolved_head).
        Дополнение — тип ребра "reinforces" в relations и поле
        "reinforced_by" (кто реинфорсил этот узел) — эти данные уже
        существуют в графе с ПАТЧ 9/10, но раньше не были видны в preview."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        branch = self.branches.get(node["branch"], {})
        root_id = branch.get("root", node_id)
        root = self.nodes.get(root_id, node)

        children = [self.nodes[nid] for nid in self.id_order if self.nodes[nid].get("parent") == node_id]
        returns_out = [e for e in self.edges if e["type"] == "returns_to" and e["from"] == node_id]
        returns_in = [e for e in self.edges if e["type"] == "returns_to" and e["to"] == node_id]

        relations = {"contradicts": [], "fixes": [], "refines": [], "supersedes": [], "reinforces": []}
        for e in self.edges:
            if e["from"] == node_id and e["type"] in relations:
                relations[e["type"]].append({"to": e["to"], "text": self.nodes[e["to"]]["text"][:150]})

        is_unresolved = self._is_unresolved_head(node_id)

        # branch_state + branch_body для GUI
        branch_state_data = self.branch_states.get(node.get("branch"))
        body = self.branch_bodies.get(node.get("branch"))
        branch_body_data = None
        if body:
            branch_body_data = {
                "atom_count": len(body["atom_ids"]),
                "centroid_dim": len(body["centroid"]) if hasattr(body["centroid"], "__len__") else 0,
                "variance":  round(body["variance"], 4),
                "entropy":   round(float(np.clip(body["variance"], 0.0, 1.0)), 4),
            }

        # ПАТЧ 10 (фикс при восстановлении preview) — узлы, реинфорсившие
        # ЭТОТ узел (обратное направление к relations["reinforces"] выше,
        # которое показывает, КОГО реинфорсит сам node_id). Данные уже
        # хранятся на узле с ПАТЧ 9 (reinforcement_links), просто не были
        # прежде видны в preview.
        reinforced_by = [
            {"from": rid, "text": self.nodes[rid]["text"][:150]}
            for rid in node.get("reinforcement_links", []) if rid in self.nodes
        ]

        return {
            "node": node,
            "status": node.get("status", "active"),
            "origin": {"file": root["file"], "text": root["text"][:200], "id": root_id},
            "continuation": [{"file": c["file"], "text": c["text"][:150], "id": c["id"]} for c in children],
            "returns": {
                "out": [{"to": e["to"], "text": self.nodes[e["to"]]["text"][:150]} for e in returns_out],
                "in": [{"from": e["from"], "text": self.nodes[e["from"]]["text"][:150]} for e in returns_in],
            },
            "relations": relations,
            "reinforced_by": reinforced_by,
            "is_unresolved": is_unresolved,
            "branch_state": branch_state_data,
            "branch_body":  branch_body_data,
        }


    # =====================================================================
    # ПАТЧ 8 — Query-слой для MCP Toolset
    # =====================================================================
    # Ничего из существующей модели не меняется: ни формат nodes/edges,
    # ни branches/branch_states/branch_bodies/path_memory/files/file_edges,
    # ни пороги, ни build()/save()/_add_atom(). Это ЧИСТО read-only
    # проекция уже существующего графа наружу.
    #
    # Зачем это нужно:
    #   Большая модель (клиент MCP) не должна получать весь архив — она
    #   должна САМА решать, куда идти дальше. Для этого ей нужны дешёвые,
    #   компактные "точки входа" в граф: список веток вместо всех атомов,
    #   развёрнутая линия ветки вместо всего архива, отношения одного узла
    #   вместо полного edges[], и т.д. Раньше этого слоя не было — были
    #   только build()/search()/preview(), которых недостаточно для
    #   автономной навигации: preview() даёт срез ОДНОГО узла, search()
    #   не различает contradicts/fixes/supersedes, а полный nodes.json
    #   отдавать нельзя (это и есть проблема "пустой контекст vs
    #   переполнение токенов", которую должен решить MCP-слой).
    #
    # Что ломается без этого патча:
    #   MCP-серверу пришлось бы либо лезть напрямую в приватные поля
    #   Archive (self.nodes, self.edges, ...) из другого модуля, дублируя
    #   логику (например, разрешение supersedes-цепочки или is_unresolved
    #   уже частично реализованы внутри preview() и их пришлось бы
    #   копировать), либо отдавать модели сырые json-файлы целиком, что
    #   прямо противоречит цели проекта.
    # =====================================================================

    def _is_unresolved_head(self, node_id: str) -> bool:
        """Правда только для головы ветки без продолжений/возвратов.
        Вынесено из preview() (ПАТЧ 8) — используется также в
        find_unresolved(). Поведение preview() не изменилось."""
        node = self.nodes.get(node_id)
        if node is None:
            return False
        branch = self.branches.get(node.get("branch"), {})
        if branch.get("head") != node_id:
            return False
        has_continuation = any(
            self.nodes[nid].get("parent") == node_id for nid in self.id_order
        )
        has_return = any(
            e["type"] == "returns_to" and (e["from"] == node_id or e["to"] == node_id)
            for e in self.edges
        )
        return not has_continuation and not has_return

    # -- 1. Ориентация: чем модель открывает исследование графа ----------
    def root_overview(self, top_branches: int = 12) -> dict:
        """Компактная точка входа: кто мы (root_project), сколько всего
        узлов/веток/файлов, и top-N веток по весу (branch_state.weight) —
        линии мышления, накопившие больше всего "гравитации". Не отдаёт
        ни одного полного текста атома — только project-level ориентиры."""
        branches = self.list_branches(include_dormant=True, limit=top_branches)
        return {
            "project": self.root_project,
            "total_nodes": len(self.nodes),
            "total_branches": len(self.branches),
            "total_files": len(self.files),
            "top_branches": branches,
        }

    # -- 2. Список веток с их состоянием ----------------------------------
    def list_branches(self, include_dormant: bool = True,
                       min_root_sim: float = None,
                       file: str = None,
                       limit: int = 200) -> list:
        """Список веток вместо всех атомов — модель выбирает, куда
        углубляться, прочитав ~200 байт на ветку, а не весь её текст."""
        out = []
        for bid, b in self.branches.items():
            bs = self.branch_states.get(bid, {})
            if not include_dormant and bs.get("dormant"):
                continue
            body = self.branch_bodies.get(bid, {})
            root_node = self.nodes.get(b["root"])
            head_node = self.nodes.get(b["head"])
            if file is not None and (not root_node or root_node.get("file") != file):
                continue
            root_sim = root_node.get("root_similarity", 0.0) if root_node else 0.0
            if min_root_sim is not None and root_sim < min_root_sim:
                continue
            out.append({
                "branch_id": bid,
                "atom_count": len(body.get("atom_ids", [])),
                "variance": round(body.get("variance", 0.0), 4),
                "weight": round(bs.get("weight", 0.0), 4),
                "activation": round(bs.get("activation", 0.0), 4),
                "dormant": bool(bs.get("dormant", False)),
                "root_similarity": round(root_sim, 4),
                "root_preview": (root_node["text"][:120] if root_node else ""),
                "head_preview": (head_node["text"][:120] if head_node else ""),
                "file": root_node["file"] if root_node else None,
                "has_unresolved_head": self._is_unresolved_head(b["head"]),
            })
        out.sort(key=lambda x: (-x["weight"], -x["atom_count"]))
        return out[:limit]

    # -- 3. Развёрнутая линия одной ветки ---------------------------------
    def branch_lineage(self, branch_id: str, max_atoms: int = 40,
                        text_chars: int = 320):
        """Атомы ветки в порядке continues (root -> head), с инлайн-
        аннотациями relations (supersedes/fixes/contradicts/refines),
        чтобы модели не приходилось делать отдельный вызов на каждый атом.
        Источник порядка — уже существующий branch_bodies[bid]['atom_ids']
        (порядок добавления атомов в тело ветки), новых полей не вводится."""
        body = self.branch_bodies.get(branch_id)
        if body is None:
            return None
        all_ids = body["atom_ids"]
        atom_ids = all_ids[-max_atoms:] if max_atoms else all_ids
        by_from = {}
        for e in self.edges:
            if e["from"] in atom_ids or e["to"] in atom_ids:
                by_from.setdefault(e["from"], []).append(e)

        atoms = []
        for nid in atom_ids:
            n = self.nodes.get(nid)
            if n is None:
                continue
            outgoing = [e for e in by_from.get(nid, []) if e["from"] == nid
                        and e["type"] in ("supersedes", "fixes", "contradicts", "refines")]
            atoms.append({
                "id": nid,
                "index": n["index"],
                "status": n.get("status", "active"),
                "question": n.get("question", "")[:text_chars],
                "answer": n.get("answer", "")[:text_chars],
                "root_similarity": n.get("root_similarity", 0.0),
                "relations": [{"type": e["type"], "to": e["to"]} for e in outgoing],
            })
        return {
            "branch_id": branch_id,
            "atom_count": len(all_ids),
            "truncated": len(all_ids) > len(atom_ids),
            "variance": round(body["variance"], 4),
            "atoms": atoms,
        }

    # -- 4. Отношения одного узла (с расширением по глубине) --------------
    def node_relations(self, node_id: str,
                        types=("contradicts", "fixes", "refines",
                               "supersedes", "returns_to"),
                        depth: int = 1, text_chars: int = 200) -> dict:
        """BFS по типизированным рёбрам от node_id на depth шагов.
        depth=1 — это то, что раньше было "жёстко зашито" в preview();
        depth>1 нужен для сбора цепочек вида: A fixes B, B supersedes C —
        модель хочет видеть всю цепочку, а не только прямых соседей."""
        seen = {node_id}
        frontier = [node_id]
        collected = []
        for _ in range(max(1, depth)):
            nxt = []
            for e in self.edges:
                if e["type"] not in types:
                    continue
                if e["from"] in frontier and e["to"] not in seen:
                    tgt = self.nodes.get(e["to"])
                    collected.append({
                        "from": e["from"], "to": e["to"], "type": e["type"],
                        "to_text": (tgt["text"][:text_chars] if tgt else ""),
                        "to_status": (tgt.get("status", "active") if tgt else "?"),
                    })
                    seen.add(e["to"]); nxt.append(e["to"])
                if e["to"] in frontier and e["from"] not in seen:
                    src = self.nodes.get(e["from"])
                    collected.append({
                        "from": e["from"], "to": e["to"], "type": e["type"],
                        "from_text": (src["text"][:text_chars] if src else ""),
                        "from_status": (src.get("status", "active") if src else "?"),
                    })
                    seen.add(e["from"]); nxt.append(e["from"])
            if not nxt:
                break
            frontier = nxt
        return {"node_id": node_id, "depth": depth, "relations": collected}

    # -- 5. Незакрытые линии мышления (bulk) -------------------------------
    def find_unresolved(self, limit: int = 50, min_root_sim: float = 0.0,
                         only_active_branches: bool = True) -> list:
        """Все головы веток без продолжения и без returns_to — оборванные
        линии рассуждения. Именно это модель должна проверять перед тем,
        как отвечать "мы это уже решили": решённое имеет continuation/
        fix/supersedes, нерешённое — нет."""
        out = []
        for bid, b in self.branches.items():
            head_id = b["head"]
            if not self._is_unresolved_head(head_id):
                continue
            bs = self.branch_states.get(bid, {})
            if only_active_branches and bs.get("dormant"):
                continue
            n = self.nodes.get(head_id)
            if n is None or n.get("root_similarity", 0.0) < min_root_sim:
                continue
            out.append({
                "branch_id": bid, "node_id": head_id,
                "text": n["text"][:250],
                "root_similarity": n.get("root_similarity", 0.0),
                "file": n.get("file"),
                "index": n.get("index"),
            })
        out.sort(key=lambda x: -x["root_similarity"])
        return out[:limit]

    # -- 6. Разрешение цепочки supersedes ----------------------------------
    def resolve_supersession(self, node_id: str, max_hops: int = 20) -> dict:
        """Дан любой узел — найти актуальную "последнюю правду" по цепочке
        supersedes. edges: {from: новый, to: старый, type: supersedes}.
        Идём от node_id ВПЕРЁД: ищем рёбра supersedes, где to == текущий,
        берём from (более новый узел), повторяем, пока не упрёмся в узел,
        которого никто не заменил. Ничего не удаляется и не помечается —
        только читаем существующие edges/status."""
        chain = [node_id]
        current = node_id
        for _ in range(max_hops):
            newer = next((e["from"] for e in self.edges
                          if e["type"] == "supersedes" and e["to"] == current), None)
            if newer is None or newer in chain:
                break
            chain.append(newer)
            current = newer
        final = self.nodes.get(current)
        return {
            "requested": node_id,
            "is_superseded": len(chain) > 1,
            "chain": chain,
            "current_node_id": current,
            "current_text": final["text"][:400] if final else None,
            "current_status": final.get("status", "active") if final else None,
        }

    # -- 7. Типизированные подмножества рёбер (contradicts/fixes/...) -----
    # -- 9. Файловый манифест: "что где лежит и что за что отвечает" -----
    def file_manifest(self, sort_by: str = "path") -> list:
        """ПАТЧ 15 — прямой ответ на "я вернулся к проекту и не помню, что
        где лежит и что за что отвечает". Один вызов вместо реконструкции
        файловой картины по кусочкам через ptg_list_branches.

        Для каждого файла:
          - folder/filename — где физически лежит
          - dominant_concepts — уже посчитанные ключевые термины файла
            (пересечение с known_concepts из root_project)
          - atom_count — сколько атомов (мыслей) в нём распознано
          - last_modified — mtime на момент последней обработки
          - avg_root_similarity — насколько файл в среднем "по теме"
            проекта (низкое значение = периферийный/несвязанный контент)
          - connected_files — с какими файлами семантически пересекается
            (уже посчитано в file_edges при построении файлового графа)
          - is_isolated — True, если у файла НЕТ связанных файлов И его
            avg_root_similarity ниже медианы по архиву — типичный признак
            забытого черновика/устаревшей версии, оторванной от остального
            проекта. Это НЕ говорит "удали" — только "обрати внимание"."""
        if not self.files:
            return []

        # avg_root_similarity по файлу — считаем один раз проходом по nodes
        sums, counts = {}, {}
        for n in self.nodes.values():
            f = n.get("file")
            sums[f] = sums.get(f, 0.0) + n.get("root_similarity", 0.0)
            counts[f] = counts.get(f, 0) + 1
        avg_root_sim = {f: (sums[f] / counts[f]) for f in sums if counts[f]}
        median_sim = sorted(avg_root_sim.values())[len(avg_root_sim) // 2] if avg_root_sim else 0.0

        out = []
        for path, finfo in self.files.items():
            connected = finfo.get("connected_files", [])
            a_sim = avg_root_sim.get(path, 0.0)
            out.append({
                "path": path,
                "folder": os.path.dirname(path),
                "filename": os.path.basename(path),
                "atom_count": counts.get(path, 0),
                "last_modified": self.processed_files.get(path),
                "dominant_concepts": finfo.get("dominant_concepts", []),
                "connected_files": connected,
                "avg_root_similarity": round(a_sim, 4),
                "is_isolated": (len(connected) == 0 and a_sim < median_sim),
            })

        if sort_by == "last_modified":
            out.sort(key=lambda x: -(x["last_modified"] or 0))
        elif sort_by == "folder":
            out.sort(key=lambda x: (x["folder"], x["filename"]))
        else:
            out.sort(key=lambda x: x["path"])
        return out

    # -- 10. ПОЛНЫЙ контекст проекта — "2 клика, без ручного поиска" ------
    def export_full_context(self, max_atom_chars: int = 4000,
                             include_dormant_branches: bool = True) -> dict:
        """ПАТЧ 16, часть III — главный запрошенный функционал: собрать
        ВЕСЬ проект в один структурированный текст, автоматически, без
        участия пользователя (никакого поиска/выбора seed-узлов — в
        отличие от assemble_context_snapshot(), который собирает контекст
        ПОД конкретный запрос).

        Организация текста: root_project → структура проекта →
        file_manifest (с пометкой изолированных файлов) → незакрытые линии
        мышления → противоречия → ПОЛНЫЙ контент по каждому файлу (folder
        → file → атомы по порядку), с разрешением supersedes: устаревшая
        (явно замещённая) версия не дублируется полным текстом, а
        компактно помечается указателем на актуальную — это не потеря
        нюанса, а устранение шума устаревших черновиков, история которых
        всё равно видна через resolve_supersession при необходимости.
        Reinforcement НЕ схлопывается — повторяющиеся идеи остаются
        видны как отдельные записи с пометкой "повторено N раз(а)",
        именно так требует anti-coagulation модель PTG.

        Цель: результат можно вставить в НОВЫЙ чат с любым ИИ-клиентом и
        продолжить работу без ручного пересказа архива."""
        L = []
        proj = self.root_project.get("project_name", "?")
        L.append(f"# ПОЛНЫЙ КОНТЕКСТ ПРОЕКТА: {proj}")
        domains = self.root_project.get("domains", [])
        if domains:
            L.append(f"Разделы (папки верхнего уровня): {', '.join(domains)}")
        modules = self.root_project.get("root_modules", [])
        if modules:
            L.append(f"Ключевые заголовки/модули: {', '.join(modules[:40])}")
        concepts = self.root_project.get("known_concepts", [])
        if concepts:
            L.append(f"Ключевые концепты: {', '.join(concepts[:60])}")

        # -- Файловая структура (манифест) --------------------------------
        manifest = self.file_manifest(sort_by="folder")
        L.append(f"\n## Файловая структура ({len(manifest)} файлов)")
        for f in manifest:
            if f["path"].startswith(STRUCTURE_FILE_MARKER):
                continue  # синтетические структурные атомы сюда не дублируем
            when = (_dt.datetime.fromtimestamp(f["last_modified"]).strftime("%Y-%m-%d")
                    if f["last_modified"] else "?")
            flag = "  ⚠ ИЗОЛИРОВАН (не связан ни с чем — возможно забытый черновик)" if f["is_isolated"] else ""
            connected = f", связан с: {', '.join(f['connected_files'])}" if f["connected_files"] else ""
            L.append(f"- {f['path']}  ({f['atom_count']} ат., изм. {when}){connected}{flag}")

        # -- Незакрытые линии мышления — важно не потерять -----------------
        unresolved = self.find_unresolved(limit=10_000, min_root_sim=0.0,
                                           only_active_branches=not include_dormant_branches)
        L.append(f"\n## Незакрытые линии мышления ({len(unresolved)})")
        for u in unresolved:
            L.append(f"- [{u['file']}] {u['text'][:200]}")

        # -- Противоречия — важно не потерять ------------------------------
        contradictions = self.edges_by_type("contradicts", limit=10_000)
        if contradictions:
            L.append(f"\n## Противоречия ({len(contradictions)})")
            for c in contradictions:
                L.append(f"- «{c['from_text'][:150]}» ПРОТИВОРЕЧИТ «{c['to_text'][:150]}»")

        # -- Полный контент по файлам: folder -> file -> атомы по порядку --
        L.append("\n## Полное содержимое по файлам")
        by_folder: dict = {}
        for path in self.files:
            if path.startswith(STRUCTURE_FILE_MARKER):
                continue
            by_folder.setdefault(os.path.dirname(path), []).append(path)

        included_ids: set = set()
        for folder in sorted(by_folder):
            L.append(f"\n### 📁 {folder or '.'}")
            for path in sorted(by_folder[folder]):
                L.append(f"\n#### 📄 {os.path.basename(path)}")
                atoms_in_file = sorted(
                    (n for n in self.nodes.values() if n.get("file") == path),
                    key=lambda n: n["index"],
                )
                for n in atoms_in_file:
                    if n["id"] in included_ids:
                        continue
                    resolved = self.resolve_supersession(n["id"])
                    if resolved["is_superseded"]:
                        # Компактный указатель вместо полного дублирования
                        # устаревшего текста — история всё равно доступна
                        # через resolve_supersession(n["id"]) при необходимости.
                        L.append(f"[УСТАРЕЛО → см. {resolved['current_node_id'][:8]}] {n['text'][:150]}")
                        included_ids.add(n["id"])
                        continue
                    included_ids.add(n["id"])
                    snippet = n["text"][:max_atom_chars]
                    reinforced = n.get("reinforcement_count", 0)
                    tag = f"  [повторено/усилено {reinforced} раз(а)]" if reinforced else ""
                    L.append(snippet + tag)

        full_text = "\n".join(L)
        return {
            "project_name": proj,
            "full_text": full_text,
            "char_count": len(full_text),
            "approx_tokens": len(full_text) // 4,
            "file_count": len(manifest),
            "atom_count": len(included_ids),
            "unresolved_count": len(unresolved),
            "contradiction_count": len(contradictions),
        }

    # -- 11. Типизированные подмножества рёбер (contradicts/fixes/...) ---
    def edges_by_type(self, edge_type: str, node_id: str = None,
                       branch_id: str = None, file: str = None,
                       limit: int = 100, text_chars: int = 200) -> list:
        """Единая точка доступа к contradicts/fixes/refines/supersedes/
        returns_to/continues с опциональным скоупом (по узлу/ветке/файлу).
        Раньше эти рёбра были видны только внутри preview() одного узла —
        глобально проверить "какие вообще есть противоречия в проекте"
        было нельзя без ручного обхода self.edges."""
        out = []
        for e in self.edges:
            if e["type"] != edge_type:
                continue
            if node_id is not None and node_id not in (e["from"], e["to"]):
                continue
            src, tgt = self.nodes.get(e["from"]), self.nodes.get(e["to"])
            if branch_id is not None:
                if not ((src and src.get("branch") == branch_id) or
                        (tgt and tgt.get("branch") == branch_id)):
                    continue
            if file is not None:
                if not ((src and src.get("file") == file) or
                        (tgt and tgt.get("file") == file)):
                    continue
            out.append({
                "from": e["from"], "to": e["to"], "type": e["type"],
                "from_text": src["text"][:text_chars] if src else "",
                "to_text": tgt["text"][:text_chars] if tgt else "",
                "from_root_sim": src.get("root_similarity", 0.0) if src else 0.0,
                "to_root_sim": tgt.get("root_similarity", 0.0) if tgt else 0.0,
            })
            if len(out) >= limit:
                break
        return out

    # -- 8. Сборка CONTEXT SNAPSHOT ----------------------------------------
    def assemble_context_snapshot(self, seed_node_ids: list,
                                   char_budget: int = 6000,
                                   include_lineage: bool = True,
                                   include_relations: bool = True) -> dict:
        """Финальный шаг автономного исследования: из набора seed-узлов
        (найденных моделью через search/list_branches/branch_lineage)
        собрать компактный, дедуплицированный, хронологически
        упорядоченный контекст — без дублей, с явной пометкой статуса
        (superseded/contradicts/fixes).

        Приоритет включения при нехватке бюджета (greedy, по убыванию):
          1. сами seed-атомы (полный текст)
          2. их supersedes-разрешение (актуальная версия, если seed устарел)
          3. прямые relations (contradicts/fixes/refines) — один хоп
          4. окружающая lineage их веток (короткие превью)
        Каждый включённый атом добавляется РОВНО ОДИН РАЗ по id
        (дедупликация через used_ids) — даже если он достижим несколькими
        путями (lineage + relation), в снапшот он попадёт один раз."""
        used_ids = set()
        blocks = []       # (global_index, text) — для хронологической сортировки
        prov_raw = {}     # node_id -> {tag, chars, semantic_score}
        budget_left = char_budget
        seed_ts_ref = None
        for sid in seed_node_ids:
            sn = self.nodes.get(sid)
            if sn is not None:
                seed_ts_ref = sn.get("timestamp")
                break

        def _emit(node_id, tag, chars, semantic_score=None):
            nonlocal budget_left
            if node_id in used_ids:
                return True
            n = self.nodes.get(node_id)
            if n is None:
                return True
            snippet = n["text"][:chars]
            piece = f"[{tag} | {node_id[:8]} | status={n.get('status','active')}]\n{snippet}\n"
            if len(piece) > budget_left:
                return False
            blocks.append((n["index"], piece))
            used_ids.add(node_id)
            prov_raw[node_id] = {"tag": tag, "chars": len(snippet), "semantic_score": semantic_score}
            budget_left -= len(piece)
            return True

        overflow = False
        # 1. seeds — полный текст, с приоритетным разрешением supersedes
        for sid in seed_node_ids:
            resolved = self.resolve_supersession(sid)
            target = resolved["current_node_id"]
            tag = "SEED" if target == sid else f"SEED_SUPERSEDED_BY_{target[:8]}"
            if not _emit(sid, tag, 500, semantic_score=1.0):
                overflow = True
            if target != sid and not _emit(target, "CURRENT_VERSION", 500, semantic_score=1.0):
                overflow = True

        # 2. прямые relations (1 хоп) для каждого seed
        if include_relations and not overflow:
            for sid in list(used_ids):
                rel = self.node_relations(sid, depth=1, text_chars=250)
                for r in rel["relations"]:
                    other = r["to"] if r["from"] == sid else r["from"]
                    if not _emit(other, r["type"].upper(), 250):
                        overflow = True
                        break
                if overflow:
                    break

        # 3. окружающая lineage веток seed-узлов (короткие превью)
        if include_lineage and not overflow:
            branch_ids = {self.nodes[nid]["branch"] for nid in list(used_ids) if nid in self.nodes}
            for bid in branch_ids:
                lin = self.branch_lineage(bid, max_atoms=15, text_chars=150)
                if not lin:
                    continue
                for a in lin["atoms"]:
                    if not _emit(a["id"], "LINEAGE", 150):
                        overflow = True
                        break
                if overflow:
                    break

        blocks.sort(key=lambda t: t[0])
        snapshot_text = "\n---\n".join(p for _, p in blocks)

        # ПАТЧ 9, часть X — Context Provenance Map: по одной записи на
        # каждый включённый чанк, в итоговом порядке (position).
        provenance = []
        for pos, (gidx, _piece) in enumerate(blocks):
            nid = next(n for n, info in prov_raw.items()
                       if self.nodes[n]["index"] == gidx)
            n = self.nodes[nid]
            info = prov_raw[nid]
            temporal_score = (_temporal_affinity(n.get("timestamp", 0), seed_ts_ref)
                               if seed_ts_ref is not None else None)
            sem_score = info["semantic_score"] if info["semantic_score"] is not None \
                else n.get("root_similarity", 0.0)
            final_weight = round(
                0.6 * sem_score + 0.4 * (temporal_score if temporal_score is not None else 0.5), 4
            )
            provenance.append({
                "position": pos,
                "node_id": nid,
                "branch_id": n.get("branch"),
                "tag": info["tag"],
                "source_path": n.get("source_path", n.get("file")),
                "filename": n.get("filename"),
                "folder_path": n.get("folder_path"),
                "created_at": n.get("created_at"),
                "modified_at": n.get("modified_at"),
                "tree_depth": n.get("depth", 0),
                "semantic_score": round(float(sem_score), 4),
                "temporal_score": round(temporal_score, 4) if temporal_score is not None else None,
                "placement_score": n.get("placement_info", {}).get("placement_score"),
                "final_weight": final_weight,
                "token_count": max(1, info["chars"] // 4),  # грубая оценка (chars/4)
            })

        # ПАТЧ 9, часть IX — context_entropy: энтропия Шеннона по
        # распределению final_weight включённых чанков. Низкая энтропия =
        # снапшот однобокий (почти весь вес на 1-2 чанках), высокая =
        # информация размазана по многим источникам поровну. Диагностика,
        # не решение — помогает увидеть "чёрную дыру" одного узла в снапшоте.
        weights = np.array([p["final_weight"] for p in provenance], dtype="float64")
        weights = np.clip(weights, 1e-9, None)
        probs = weights / weights.sum() if weights.sum() > 0 else weights
        context_entropy = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0

        token_total = sum(p["token_count"] for p in provenance)

        return {
            "seed_node_ids": seed_node_ids,
            "included_node_ids": sorted(used_ids),
            "char_budget": char_budget,
            "char_used": char_budget - budget_left,
            "truncated_by_budget": overflow,
            "snapshot_text": snapshot_text,
            "provenance": provenance,
            "token_total": token_total,
            "context_entropy": round(context_entropy, 4),
        }
