"""Сущности ядра PTG-MEM. Связь — memcore.types.Edge; пространство, путь запроса, находка, узел и ребро кода —
ptgfull.types.

node: leaf
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """Источник памяти: путь и вид — 'file' (файл папки), 'claude' (сеанс Claude Code), 'codex' (сеанс Codex)."""

    path: str
    kind: str


@dataclass(frozen=True)
class Draft:
    """Атом источника до записи: место, текст, вопрос и ответ (у реплик и пар чата), время (реплики или правки файла),
    сеанс ('' — не сеанс), откуда ('file', 'claude', 'codex', 'agent', 'user')."""

    position: int
    text: str
    question: str
    answer: str
    at: float
    session: str
    source: str


@dataclass(frozen=True)
class Entry:
    """Атом в базе без вектора: id, источник, место, отпечаток текста, текст, вопрос, ответ, время, состояние ('active'
    или 'retracted'), когда снят (0.0 — не снят), сеанс, откуда."""

    id: str
    file: str
    position: int
    digest: str
    text: str
    question: str
    answer: str
    at: float
    status: str
    retracted_at: float
    session: str
    source: str


@dataclass(frozen=True)
class Batch:
    """План приёма одного источника: новые атомы (векторов ещё нет), id к снятию, пары (новый, старый) «на месте
    снятого», сколько атомов осталось как было, сколько секретов вычищено."""

    file: str
    fresh: tuple[Entry, ...]
    retract: tuple[str, ...]
    revises: tuple[tuple[str, str], ...]
    kept: int
    secrets: int


@dataclass(frozen=True)
class Report:
    """Итог приёма или снятия источника."""

    file: str
    added: int
    retracted: int
    kept: int


@dataclass(frozen=True)
class Link:
    """Связь у карточки: направление ('out' — от карточки), вид, другой конец, кто заявил, состояние другого конца,
    его дата, начало его текста. Шов с кодом — вид 'edits', 'reads' или 'mentions', другой конец — 'file:<путь>' или
    полное имя символа, кто заявил — 'metatree', состояние — 'code'."""

    dir: str
    kind: str
    id: str
    source: str
    status: str
    date: str
    text: str


@dataclass(frozen=True)
class Card:
    """Атом для выдачи: id, оценка (None — без поиска), состояние ('active', 'superseded', 'retracted'), дата, файл,
    файл от корня проекта, вопрос, текст (обрезан), длина полного текста, связи, текущая версия ('' — сам атом),
    сеанс."""

    id: str
    score: float | None
    status: str
    date: str
    file: str
    rel_file: str
    question: str
    text: str
    chars: int
    links: tuple[Link, ...]
    current: str
    session: str


@dataclass(frozen=True)
class Recall:
    """Подсказка к запросу: блок <ptg-memory> ('' — нечего сказать), показанные карточки, лучшая оценка."""

    context: str
    items: tuple[Card, ...]
    top: float


@dataclass(frozen=True)
class Side:
    """Конец решения или сосед по файлу: id, начало текста, файл от корня, состояние, дата."""

    id: str
    text: str
    file: str
    status: str
    date: str


@dataclass(frozen=True)
class Decision:
    """Решение: вид связи, дата и время записи, кто заявил, закрыта ли связь, новое и старое."""

    kind: str
    date: str
    at: float
    source: str
    closed: bool
    new: Side
    old: Side


@dataclass(frozen=True)
class Node:
    """Атом целиком: карточка (полный текст), ответ, соседи по файлу (до четырёх с каждой стороны, сам атом внутри)."""

    card: Card
    answer: str
    lineage: tuple[Side, ...]


@dataclass(frozen=True)
class GNode:
    """Узел эго-сети: id, подпись (60 знаков), состояние ('code' — узел кода или метадерева), дата, файл, центр ли."""

    id: str
    label: str
    status: str
    date: str
    file: str
    center: bool


@dataclass(frozen=True)
class GEdge:
    """Ребро эго-сети: от, к, вид."""

    src: str
    dst: str
    kind: str


@dataclass(frozen=True)
class Graph:
    """Эго-сеть атома для окна."""

    nodes: tuple[GNode, ...]
    edges: tuple[GEdge, ...]


@dataclass(frozen=True)
class Plan:
    """Что устарело: источники к приёму (новые или изменённые), пути к снятию (исчезли)."""

    todo: tuple[Source, ...]
    gone: tuple[str, ...]


@dataclass(frozen=True)
class Stats:
    """Состояние базы: атомов, живых (не снятых), снятых, заменённых, связей, обработанных файлов, символов кода, рёбер
    метадерева, пространства."""

    atoms: int
    live: int
    retracted: int
    superseded: int
    relations: int
    files: int
    code: int
    edges: int
    spaces: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    """Настройки приёма проекта: корень, папка базы (её файлы не принимаются), брать ли папку, сеансы Claude Code,
    сеансы Codex, исключённые папки сверх правил, расширения ('()' — правила по умолчанию), язык меток .py."""

    root: str
    store: str
    folder: bool
    claude: bool
    codex: bool
    exclude_dirs: tuple[str, ...]
    include_exts: tuple[str, ...]
    lang: str
