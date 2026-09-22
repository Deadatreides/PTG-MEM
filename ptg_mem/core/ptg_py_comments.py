"""
ptg_py_comments.py — извлечение ТОЛЬКО докстрингов и #-комментариев из .py.

Вынесено в отдельный модуль (по запросу пользователя): изолирует
Python-специфичную логику от общего текстового пайплайна ptg_core.py и
позволяет независимо включать/выключать обработку .py — см. main.py
(чекбокс «Обрабатывать .py-комментарии») и Archive(..., extract_py_comments=...).

НИКОГДА не возвращает код — только человеческую мысль, оставленную в
комментариях/докстрингах. Раньше (до этого решения) .py либо не входил в
обработку вовсе, либо (в самой первой версии проекта) читался как обычный
текст и рубился sequential-фолбэком прямо по коду — функция могла
разорваться посередине, комментарий — склеиться с несвязанным кодом
снизу, и всё это эмбеддилось как реальные атомы графа. Здесь этого не
происходит: код физически не может попасть в возвращаемый текст.
"""

import ast
import tokenize
import io

from ptg_logging import get_logger

_log = get_logger("py_comments")


def extract_py_comments_text(path: str):
    """Извлечь только докстринги и #-комментарии из .py-файла.

    Два независимых источника:
    1. Docstring'и модуля/классов/функций — через ast.parse() +
       ast.get_docstring(). Требует синтаксически валидного файла.
    2. #-комментарии — через tokenize.generate_tokens(). Работает даже
       если ast.parse() упал (например, файл с частичным/незавершённым
       синтаксисом) — токенайзер терпимее к некоторым отклонениям, но
       тоже может не справиться с совсем битым файлом.

    Оба источника пытаются сработать независимо: если ast упал, но
    tokenize отработал — вернутся хотя бы комментарии, и наоборот.
    Комментарии, собранные ДО падения tokenize (например, из-за незакрытой
    скобки дальше по файлу), не отбрасываются — используется то, что уже
    успели собрать.

    Возвращает None, только если файл вообще не прочитать как текст, или
    если ни докстрингов, ни комментариев не нашлось."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        return None

    source = None
    for enc in ("utf-8-sig", "utf-8", "cp1251", "cp1252"):
        try:
            source = raw.decode(enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if source is None:
        source = raw.decode("latin-1", errors="ignore")

    pieces = []

    # 1. Docstring'и через ast
    try:
        tree = ast.parse(source)
        mod_doc = ast.get_docstring(tree)
        if mod_doc:
            pieces.append(f"[docstring модуля]\n{mod_doc.strip()}")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                if doc:
                    kind = "класс" if isinstance(node, ast.ClassDef) else "функция"
                    pieces.append(f"[docstring {kind} {node.name}]\n{doc.strip()}")
    except (SyntaxError, ValueError, RecursionError) as ex:
        _log.debug(f"ast.parse не удался для {path} ({ex}) — докстринги пропущены, "
                  f"пробую tokenize для комментариев")

    # 2. #-комментарии через tokenize
    comments = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                text = tok.string.lstrip("#").strip()
                if text and not text.startswith("!"):  # пропускаем shebang-подобные строки
                    comments.append(text)
    except (tokenize.TokenError, IndentationError, SyntaxError, UnicodeDecodeError) as ex:
        _log.debug(f"tokenize упал для {path} ({ex}) — использую {len(comments)} "
                  f"уже собранных комментариев до этого места")
    if comments:
        pieces.append("[комментарии в коде]\n" + "\n".join(comments))

    if not pieces:
        return None
    return "\n\n".join(pieces)
