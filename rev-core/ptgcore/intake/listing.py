"""Какие источники у проекта есть: файлы папки, сеансы Claude Code и Codex; время правки файла; какие пути пропали.

node: leaf
effects: io
"""
import os

import deal
from ptg_mem import filters, transcripts

from ptgcore.types import Source


@deal.ensure(lambda root, store, exclude_dirs, include_exts, result: len(set(result)) == len(result))
@deal.has('io')
def folder_files(root: str, store: str, exclude_dirs: tuple[str, ...], include_exts: tuple[str, ...]) -> tuple[str, ...]:
    """ФТ-002. Файлы папки root, которые стоит принять, по порядку обхода: filters.FileFilter(root, exclude_dirs,
    include_exts или None при (), store).walk(). Нет папки — ().

    D: 2

    >>> folder_files('нет-такой-папки', 'нет-базы', (), ())
    ()
    >>> folder_files('нет-такой-папки', 'нет-базы', ('node_modules',), ('.md',))
    ()
    """
    if not os.path.isdir(root):
        return ()
    walked = filters.FileFilter(root, exclude_dirs, include_exts or None, store).walk()
    return tuple(dict.fromkeys(walked))


@deal.ensure(lambda root, result: list(result) == sorted(result) and len(set(result)) == len(result)
             and all(p.endswith('.jsonl') for p in result))
@deal.has('io')
def claude_files(root: str) -> tuple[str, ...]:
    """ФТ-001. Сеансы Claude Code проекта root: transcripts.claude_transcripts(root) по порядку; нет сеансов — ().

    D: 1

    >>> claude_files('no-such-project-root')
    ()
    >>> claude_files('no/such/project/root')
    ()
    """
    return tuple(sorted({p for p in transcripts.claude_transcripts(root) if p.endswith('.jsonl')}))


@deal.ensure(lambda root, result: list(result) == sorted(result) and len(set(result)) == len(result)
             and all(p.endswith('.jsonl') for p in result))
@deal.has('io')
def codex_files(root: str) -> tuple[str, ...]:
    """ФТ-001. Сеансы Codex проекта root: transcripts.codex_transcripts(root) по порядку; нет сеансов — ().

    D: 1

    >>> codex_files('no-such-project-root')
    ()
    >>> codex_files('no/such/project/root')
    ()
    """
    return tuple(sorted({p for p in transcripts.codex_transcripts(root) if p.endswith('.jsonl')}))


@deal.ensure(lambda path, result: result >= 0.0)
@deal.has('io')
def file_time(path: str) -> float:
    """Время правки файла (os.path.getmtime); нет файла, путь не читается или время отрицательное — 0.0.

    D: 2

    >>> file_time('нет-такого.md')
    0.0
    >>> file_time('')
    0.0
    """
    try:
        moment = os.path.getmtime(path)
    except (OSError, ValueError):
        return 0.0
    return moment if moment > 0.0 else 0.0


@deal.ensure(lambda paths, result: set(result) <= set(paths) and len(set(result)) == len(result))
@deal.ensure(lambda paths, result: list(result) == [p for p in dict.fromkeys(paths) if p in result])
@deal.has('io')
def gone(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Пути, которых больше нет на диске (os.path.exists), по порядку, без повторов.

    D: 2

    >>> gone(('нет-такого.md', 'tests/fixtures/session.jsonl', 'нет-такого.md'))
    ('нет-такого.md',)
    >>> gone(())
    ()
    """
    return tuple(p for p in dict.fromkeys(paths) if not os.path.exists(p))


@deal.ensure(lambda entries, result: len({s.path for s, _ in result}) == len(result) and all(t > 0.0 for _, t in result))
@deal.ensure(lambda entries, result: all(item in entries for item in result))
@deal.has()
def stamped(entries: tuple[tuple[Source, float], ...]) -> tuple[tuple[Source, float], ...]:
    """ФТ-001. Источники с известным временем правки (больше нуля): исчезнувшие при чтении времени (0.0) пропускаются,
    путь, уже встретившийся раньше, — тоже; порядок как был.

    D: 3

    >>> stamped(())
    ()
    >>> stamped(((Source('a.md', 'file'), 5.0), (Source('b.md', 'file'), 0.0), (Source('a.md', 'claude'), 7.0), (Source('c.jsonl', 'claude'), 2.0)))
    ((Source(path='a.md', kind='file'), 5.0), (Source(path='c.jsonl', kind='claude'), 2.0))
    """
    seen: set[str] = set()
    kept: list[tuple[Source, float]] = []
    for source, moment in entries:
        if source.path in seen:
            continue
        seen.add(source.path)
        if moment > 0.0:
            kept.append((source, moment))
    return tuple(kept)
