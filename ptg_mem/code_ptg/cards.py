"""
Модель семантической карточки инженерного объекта.

Отличие от предыдущей версии MVP: карточка больше не сериализуется в
собственную SQLite-строку — она сериализуется В ТОЧНО ТОТ ЖЕ node-формат,
что использует настоящий Archive из vendor_ptg/ptg_core.py (плоский dict,
поля id/branch/status/vec/timestamp/... общие для обеих систем, плюс
domain-специфичные поля поверх). Card остаётся удобной типизированной
обёрткой для работы В КОДЕ Code PTG, но единица хранения — node dict.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Confidence(str, Enum):
    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class Evidence:
    type: str
    ref: str


@dataclass
class Provenance:
    """Универсальная обёртка происхождения для ЛЮБОГО поля карточки."""

    source: str
    method: str
    algorithm_version: str
    confidence: Confidence
    evidence: list[Evidence] = field(default_factory=list)
    recomputable: bool = True
    computed_at: float = field(default_factory=time.time)

    @staticmethod
    def objective(method: str, ref: str, algorithm_version: str = "1.0.0",
                  source: str = "python_ast") -> "Provenance":
        """source указывается явно: факт из tree-sitter и факт из ast Python
        имеют разную силу (по имени против разрешённого), и в провенансе это
        должно быть видно, а не подразумеваться."""
        return Provenance(
            source=source, method=method, algorithm_version=algorithm_version,
            confidence=Confidence.VERIFIED, evidence=[Evidence(type="ast_span", ref=ref)],
            recomputable=True,
        )

    @staticmethod
    def unknown(method: str) -> "Provenance":
        return Provenance(
            source="none", method=method, algorithm_version="1.0.0",
            confidence=Confidence.NONE, evidence=[], recomputable=True,
        )


@dataclass
class Field:
    value: Any
    provenance: Provenance

    def __post_init__(self):
        if not self.provenance.evidence and self.provenance.confidence != Confidence.NONE:
            raise ValueError(
                f"Нарушение инварианта provenance: confidence != NONE без evidence "
                f"(метод {self.provenance.method!r})"
            )
        if self.provenance.confidence == Confidence.NONE and self.value not in (None, "Unknown"):
            raise ValueError("Нарушение инварианта: confidence=NONE требует value=Unknown")

    @staticmethod
    def unknown(method: str) -> "Field":
        return Field(value="Unknown", provenance=Provenance.unknown(method))


@dataclass
class Identity:
    card_id: str
    kind: str
    subkind: str
    qualified_name: str
    file_path: str
    span: tuple[int, int]
    content_hash: str
    signature_hash: str


@dataclass
class Card:
    identity: Identity
    layer_1: dict[str, Any]
    layer_1_provenance: Provenance
    layer_2: dict[str, Field]

    # ---------- сериализация в единый PTG node-формат ----------

    def interpretation_text(self) -> str:
        """
        Текст, который эмбеддится для семантического графтинга (аналог
        поля "text" у Text PTG атома). Строится ТОЛЬКО из известных полей —
        если layer_2 пуст (Unknown), текст сводится к объективным фактам,
        и графтинг работает по чистой структуре, а не по выдуманному смыслу.
        """
        parts = [f"{self.identity.kind}:{self.identity.subkind} {self.identity.qualified_name}"]
        doc = self.layer_1.get("docstring")
        if doc:
            parts.append(doc)
        for name in ("purpose", "architectural_role", "side_effects_summary"):
            f = self.layer_2.get(name)
            if f and f.value != "Unknown":
                parts.append(f"{name}: {f.value}")
        return "\n".join(parts)

    def to_node_payload(self) -> dict:
        """Code-специфичные поля, добавляемые ПОВЕРХ универсальных полей
        PTG-узла (id/branch/vec/status/... — их проставляет graph_engine)."""
        return {
            "domain": "code",
            "kind": self.identity.kind,
            "subkind": self.identity.subkind,
            "qualified_name": self.identity.qualified_name,
            "file_path": self.identity.file_path,
            "span": list(self.identity.span),
            "content_hash": self.identity.content_hash,
            "signature_hash": self.identity.signature_hash,
            "layer_1": self.layer_1,
            "layer_1_provenance": asdict(self.layer_1_provenance),
            "layer_2": {k: asdict(v) for k, v in self.layer_2.items()},
        }


def content_hash_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def signature_hash_of(signature_repr: str) -> str:
    return hashlib.sha256(signature_repr.encode("utf-8")).hexdigest()[:16]


def make_card_id(file_path: str, qualified_name: str, version_salt: str = "") -> str:
    """
    card_id теперь версионируется (version_salt = content_hash момента
    создания) — потому что PTG append-only: каждая версия карточки это
    НОВЫЙ узел, связанный с предыдущим через supersedes/continues, а не
    перезапись старого. qualified_name используется отдельно для поиска
    "текущей активной" версии (см. archive.py: get_active_card_by_qname).
    """
    raw = f"{file_path}::{qualified_name}::{version_salt}::{time.time_ns()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
