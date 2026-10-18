"""Дата и путь для показа.

node: leaf
"""
import os
import time

import deal


@deal.ensure(lambda ts, result: len(result) == 10 or result == '?')
@deal.has()
def day(ts: float) -> str:
    """Дата для показа: time.strftime('%Y-%m-%d', time.localtime(ts)); если localtime или strftime не принимают ts
    (OverflowError, OSError, ValueError) или в дате не 10 знаков — '?'.

    D: 2

    >>> day(1760000000.0)
    '2025-10-09'
    >>> day(1e30)
    '?'
    """
    try:
        text = time.strftime('%Y-%m-%d', time.localtime(ts))
    except (OverflowError, OSError, ValueError):
        return '?'
    return text if len(text) == 10 else '?'


@deal.ensure(lambda path, root, result: (result == path or not result.startswith('..')) and (path != '' or result == ''))
@deal.has()
def rel(path: str, root: str) -> str:
    """Путь для показа: os.path.relpath(path, root), если path не пуст, не начинается с 'memory://' и лежит внутри root
    (relpath не равен '..' и не начинается с '..' + os.sep); иначе, а также когда relpath не считается (ValueError или
    OSError — например, другой диск), — path как есть.

    D: 4

    >>> rel('memory://notes', '.')
    'memory://notes'
    >>> rel('n.md', '.')
    'n.md'
    >>> rel('', '.')
    ''
    >>> rel('../чужой/n.md', '.')
    '../чужой/n.md'
    """
    if path == '' or path.startswith('memory://'):
        return path
    try:
        rel_path = os.path.relpath(path, root)
    except (ValueError, OSError):
        return path
    if rel_path == '..' or rel_path.startswith('..' + os.sep):
        return path
    return rel_path
