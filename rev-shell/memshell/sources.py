"""Текст файла-источника в Markdown: MarkItDown (Microsoft) читает .md/.txt как есть, HTML и таблицы переводит в
Markdown, .docx/.pdf/.xlsx/.pptx — в их текст (замер ревизии: .pdf 17 из 25, .xlsx 25 из 25, кириллица .docx — как
у своего разбора).

node: leaf
effects: io
"""
import deal
from markitdown import MarkItDown, UnsupportedFormatException


@deal.raises(FileNotFoundError)
@deal.ensure(lambda path, result: "\r" not in result)
@deal.has("read")
def convert(path: str) -> str:
    """Markdown-текст файла path: MarkItDown(enable_plugins=False).convert(path).text_content, переводы строк \\r\\n
    и \\r — как \\n. Формат, который MarkItDown не берёт (UnsupportedFormatException, например двоичный файл), —
    пустая строка. Нет файла — FileNotFoundError.

    D: 2

    >>> convert("tests/fixtures/note.md")
    '# План\\n\\nПервый абзац.\\n\\nВторой абзац.\\n'
    >>> convert("tests/fixtures/page.html")
    '# Заголовок\\n\\nАбзац **жирный**.\\n\\n* раз\\n* два'
    >>> convert("tests/fixtures/blob.bin")
    ''
    """
    with open(path, "rb"):
        pass
    try:
        text = MarkItDown(enable_plugins=False).convert(path).text_content
    except UnsupportedFormatException:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")
