"""Данные оболочки memshell: черновик атома документа и запись подсказки; постоянные тексты блока памяти.
"""
from dataclasses import dataclass

# Блок памяти в подсказке модели: так его видят хуки Claude Code и Codex (формат PTG-MEM 0.2).
OPEN: str = "<ptg-memory>"
CLOSE: str = "</ptg-memory>"
RECALL_HEAD: str = ("Auto-recalled from this project's memory (PTG-MEM). Older notes may be outdated — check the date and the "
                    "markers before relying on them.")
BRIEF_HEAD: str = ("PTG-MEM is active for this project: {live} memory atoms. Relevant memory is attached to each prompt "
                   "automatically. Tools: ptg_search, ptg_node, ptg_decisions, ptg_remember (store a decision the moment it is made).")
RECENT_HEAD: str = "Recent notes and decisions (last two weeks):"

# Начала сообщений обвязки (не человека) в переписке Claude Code — как в ptg_mem/transcripts.py.
AUTOMATED: tuple[str, ...] = ("<local-command", "<command-", "Caveat:", "<task-notification", "<system-reminder",
                              "[SYSTEM NOTIFICATION", "<bash-", "<user-memory")
# Метка хода в переписке, сохранённой документом: «Пользователь: [2026-09-22 14:57] …».
TURN_PREFIX: str = r"^(?:Пользователь|Claude|User|Assistant)\s*:\s*\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]\s*"

# Связи, которые показывает лента решений (memcore.types.RELATION_KINDS): правка абзаца записывается как supersedes
# с source="edit".
DECISION_KINDS: tuple[str, ...] = ("supersedes", "contradicts")


@dataclass(frozen=True)
class Draft:
    """Атом документа до записи: файл, место в файле, отпечаток текста, текст."""
    file: str
    position: int
    digest: str
    text: str


@dataclass(frozen=True)
class Entry:
    """Запись для показа: дата (ГГГГ-ММ-ДД), файл от корня проекта, сходство, текст, строки пометок."""
    date: str
    file: str
    score: float
    text: str
    notes: tuple[str, ...]
