"""Сущности полного PTG: пространства векторов, находки, граф кода, метадерево, подсказка.

node: leaf
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Space:
    """Пространство эмбеддингов — векторы одной модели.

    name — имя (латиница в нижнем регистре, цифры, «_»); provider — имя в реестре эмбеддеров LanceDB
    (sentence-transformers, openai, ollama, huggingface …) или 'hash' — проверочный; model — модель у поставщика;
    dim — размерность.
    """

    name: str
    provider: str
    model: str
    dim: int


@dataclass(frozen=True)
class Hit:
    """Находка: id атома или узла графа, оценка (больше — лучше), откуда — пространство, 'lexical' или 'graph'."""

    id: str
    score: float
    via: str


@dataclass(frozen=True)
class Route:
    """Путь запроса в пространство space: 'direct' — запрос кодирует модель самого space (source == space);
    'adapter' — модель source, затем адаптер source → space."""

    space: str
    how: str
    source: str


@dataclass(frozen=True)
class CodeNode:
    """Символ графа кода: qn — полное имя без префикса проекта (memcore.filtering.admit), name — короткое имя,
    label — Function, Method или Class, file — путь от корня репозитория через «/», start и end — строки."""

    qn: str
    name: str
    label: str
    file: str
    start: int
    end: int


@dataclass(frozen=True)
class Edge:
    """Ребро графа: кода (CALLS, USAGE) или метадерева (contains, next, edits, reads, mentions)."""

    src: str
    dst: str
    kind: str


@dataclass(frozen=True)
class Atom:
    """Атом текста: id — отпечаток, file — источник, session — id сеанса переписки ('' — не переписка),
    order — место в источнике."""

    id: str
    file: str
    text: str
    session: str
    order: int


@dataclass(frozen=True)
class Piece:
    """Кусок подсказки: kind — 'atom' или 'code', ref — id атома или полное имя символа, text, score."""

    kind: str
    ref: str
    text: str
    score: float


@dataclass(frozen=True)
class Counts:
    """Итог приёма: атомов в базе, векторов во всех доступных пространствах, символов кода, рёбер метадерева."""

    atoms: int
    vectors: int
    code: int
    edges: int
