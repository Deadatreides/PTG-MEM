"""Атомы документа: абзацы текста; абзац длиннее предела режет Chonkie (RecursiveChunker, правила по умолчанию — без
сети). Замер bench/chunk_locality.py: при правке одного абзаца меняется ровно один атом, а упаковка Chonkie целиком
сдвигала границы (в среднем 6 сменённых атомов, до 25).

node: leaf
"""
import hashlib
import re

import deal
from chonkie import RecursiveChunker

from memshell.types import Draft


@deal.pre(lambda text, limit: limit >= 64)
@deal.ensure(lambda text, limit, result: all(0 < len(p) <= limit and p == p.strip() for p in result))
@deal.ensure(lambda text, limit, result: "".join("".join(result).split()) == "".join(text.split()))
@deal.has()
def split_text(text: str, limit: int) -> list[str]:
    """Абзацы текста. Переводы строк \\r\\n и \\r — как \\n; абзацы разделяет пустая строка (строка из одних пробелов
    тоже пустая); каждый абзац — без пробелов по краям, пустые пропускаются. Абзац длиннее limit — куски
    RecursiveChunker(tokenizer="character", chunk_size=limit).chunk(абзац), каждый без пробелов по краям, пустые
    пропускаются.

    D: 4

    >>> split_text("Первый абзац.\\n\\nВторой абзац.\\n", 1200)
    ['Первый абзац.', 'Второй абзац.']
    >>> split_text("Строка\\r\\nеё продолжение\\r\\n  \\r\\nНовый абзац", 1200)
    ['Строка\\nеё продолжение', 'Новый абзац']
    >>> split_text(" \\n\\n\\t\\n", 1200)
    []
    >>> split_text("Первое предложение длинного абзаца. Второе предложение того же абзаца. Третье, последнее.", 64)
    ['Первое предложение длинного абзаца.', 'Второе предложение того же абзаца. Третье, последнее.']
    """
    normalized = re.sub(r"\r\n?", "\n", text)
    blocks = re.split(r"\n(?:[^\S\n]*\n)+", normalized)
    result: list[str] = []
    for block in blocks:
        paragraph = block.strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            result.append(paragraph)
            continue
        chunker = RecursiveChunker(tokenizer="character", chunk_size=limit)
        for chunk in chunker.chunk(paragraph):
            piece = chunk.text.strip()
            if piece:
                result.append(piece)
    return result


@deal.ensure(lambda text, result: len(result) == 16 and all(c in "0123456789abcdef" for c in result))
@deal.has()
def digest(text: str) -> str:
    """Отпечаток атома: hashlib.blake2b(digest_size=8).hexdigest() — 16 знаков hex — от текста со схлопнутыми
    пробелами (" ".join(text.split())) в UTF-8 с errors="replace". Правка одних пробелов — тот же атом.

    D: 1

    >>> digest("Первый абзац.")
    '8403a71b67ab969e'
    >>> digest("  Первый\\n абзац. ")
    '8403a71b67ab969e'
    >>> digest("")
    'e4a6a0577479b2b4'
    >>> digest("Второй абзац.")
    'd948e1fb94c8d49a'
    """
    collapsed = " ".join(text.split())
    return hashlib.blake2b(collapsed.encode("utf-8", errors="replace"), digest_size=8).hexdigest()


@deal.pre(lambda file, text, limit: limit >= 64)
@deal.ensure(lambda file, text, limit, result: [d.position for d in result] == list(range(len(result))))
@deal.ensure(lambda file, text, limit, result: all(d.file == file and d.digest == digest(d.text) for d in result))
@deal.ensure(lambda file, text, limit, result: [d.text for d in result] == split_text(text, limit))
@deal.has()
def atomize(file: str, text: str, limit: int) -> list[Draft]:
    """Черновики атомов файла: абзацы split_text(text, limit) по порядку, место — номер с нуля, отпечаток — digest.

    D: 1

    >>> atomize("plan.md", "Первый абзац.\\n\\nВторой абзац.\\n", 1200)
    [Draft(file='plan.md', position=0, digest='8403a71b67ab969e', text='Первый абзац.'), Draft(file='plan.md', position=1, digest='d948e1fb94c8d49a', text='Второй абзац.')]
    >>> atomize("empty.md", " \\n\\n\\t\\n", 1200)
    []
    """
    return [
        Draft(file=file, position=position, digest=digest(paragraph), text=paragraph)
        for position, paragraph in enumerate(split_text(text, limit))
    ]
