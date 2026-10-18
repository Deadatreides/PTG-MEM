"""Данные ядра памяти memcore: атомы файла и план синхронизации, движки поиска и план поиска, находки и бюджет
подсказки, связи между атомами, правила отбора файлов, вычищенный текст.
"""
from dataclasses import dataclass

# Виды связи между атомами (схема Graphiti: связь действует с valid_at до invalid_at; старое не удаляется).
# supersedes — src заменяет dst; остальные — уточняет, противоречит, чинит, продолжает, возвращается к, подкрепляет.
RELATION_KINDS: tuple[str, ...] = ("supersedes", "contradicts", "refines", "fixes", "continues", "returns_to", "reinforces")

# Форматы ключей поставщиков, которых нет в detect-secrets (замер bench/secrets_exact.py): Anthropic и OpenAI sk-,
# Google AIza, Hugging Face hf_, GitHub github_pat_, Composio uak_/ak_/ck_; закрытый ключ PEM — блоком целиком.
VENDOR_KEY_RE: str = (
    r"(?<![A-Za-z0-9_])(?:sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|uak_[A-Za-z0-9_-]{16,}|ak_[A-Za-z0-9_-]{20,}|ck_[A-Za-z0-9_-]{16,})"
    r"|(?s:-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----)"
)

# Чем заменяется найденный секрет.
REDACTION_MARK: str = "<secret>"


@dataclass(frozen=True)
class Atom:
    """Свежий атом перечитанного файла: id, файл, номер места в файле, отпечаток текста (считает оболочка)."""
    id: str
    file: str
    position: int
    digest: str


@dataclass(frozen=True)
class Stored:
    """Атом из хранилища; live=False — снят раньше."""
    id: str
    file: str
    position: int
    digest: str
    live: bool


@dataclass(frozen=True)
class SyncPlan:
    """Что сделать с хранилищем после перечитки: add — id свежих атомов к записи; keep — id хранимых, что остаются;
    retract — id хранимых к снятию; revises — пары (новый, прежний): новый атом встал на место снятого."""
    add: tuple[str, ...]
    keep: tuple[str, ...]
    retract: tuple[str, ...]
    revises: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Engine:
    """Модель эмбеддинга и её столбец в хранилище: загружена ли модель, сколько строк в столбце, к каким столбцам
    есть адаптер перевода запроса (Drift-Adapter)."""
    model: str
    loaded: bool
    rows: int
    adapters: tuple[str, ...]


@dataclass(frozen=True)
class SearchPlan:
    """Как искать: mode — vector, translated или lexical; column — столбец векторов; encoder — модель, которой
    кодируется запрос. У lexical column и encoder пустые."""
    mode: str
    column: str
    encoder: str


@dataclass(frozen=True)
class Hit:
    """Находка поиска: id атома, файл, текст, сходство, id преемника (пусто — не заменён)."""
    id: str
    file: str
    text: str
    score: float
    successor: str


@dataclass(frozen=True)
class Budget:
    """Бюджет подсказки: символов, записей, порог сходства, записей из одного файла."""
    chars: int
    items: int
    min_score: float
    per_file: int


@dataclass(frozen=True)
class Item:
    """Запись подсказки: id атома, текст, пометка (пусто или «заменено: <id>»)."""
    id: str
    text: str
    note: str


@dataclass(frozen=True)
class Edge:
    """Связь src → dst вида kind: действует с valid_at; invalid_at — когда закрыта (0.0 — действует); source — кто
    заявил (agent, model, user)."""
    src: str
    dst: str
    kind: str
    valid_at: float
    invalid_at: float
    source: str


@dataclass(frozen=True)
class Claim:
    """Заявка на связь src → dst вида kind в момент at от source."""
    src: str
    dst: str
    kind: str
    at: float
    source: str


@dataclass(frozen=True)
class Rules:
    """Правила отбора файлов: шаблоны включения и исключения в синтаксисе .gitignore, предел размера в байтах."""
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    max_bytes: int


@dataclass(frozen=True)
class Admission:
    """Решение по файлу: ok и причина отказа (пусто, если принят)."""
    ok: bool
    reason: str


@dataclass(frozen=True)
class Scrubbed:
    """Текст после вычистки и число заменённых кусков."""
    text: str
    found: int
