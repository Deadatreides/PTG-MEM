"""Состояние атома по связям: текущая версия (memcore.current_version), состояние для показа и сколько атомов заменено.

node: leaf
"""
import math

import deal
import memcore
from memcore.types import Edge

from ptgcore.types import Entry


@deal.ensure(lambda atom, edges, result: result != atom)
@deal.has()
def successor(atom: str, edges: tuple[Edge, ...]) -> str:
    """Текущая версия атома: memcore.current_version(atom, список связей с конечными valid_at и invalid_at — прочие не
    учитываются); версия совпала с atom — ''. atom = '' (это не id атома) — ' ', чтобы result != atom.

    D: 3

    >>> successor('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),))
    'a2'
    >>> successor('a2', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),))
    ''
    >>> successor('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 9.0, 'edit'),))
    ''
    >>> successor('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'supersedes', 7.0, 0.0, 'agent')))
    'a3'
    """
    if atom == '':
        return ' '
    live = [e for e in edges if math.isfinite(e.valid_at) and math.isfinite(e.invalid_at)]
    version = memcore.current_version(atom, live)
    return '' if version == atom else version


@deal.ensure(lambda entry, edges, result: result in ('active', 'superseded', 'retracted'))
@deal.has()
def status(entry: Entry, edges: tuple[Edge, ...]) -> str:
    """ФТ-006. 'retracted', если entry.status == 'retracted'; иначе 'superseded', если successor(entry.id, edges) не пуст;
    иначе 'active'.

    D: 3

    >>> status(Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1.0, 'active', 0.0, '', 'file'), (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),))
    'superseded'
    >>> status(Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1.0, 'retracted', 5.0, '', 'file'), ())
    'retracted'
    >>> status(Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1.0, 'retracted', 5.0, '', 'file'), (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),))
    'retracted'
    >>> status(Entry('a2', 'n.md', 1, 'd2', 'новое', '', 'новое', 1.0, 'active', 0.0, '', 'file'), (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),))
    'active'
    """
    if entry.status == 'retracted':
        return 'retracted'
    if successor(entry.id, edges) != '':
        return 'superseded'
    return 'active'


@deal.ensure(lambda edges, ids, result: 0 <= result <= len(set(ids)))
@deal.has()
def superseded(edges: tuple[Edge, ...], ids: tuple[str, ...]) -> int:
    """Сколько разных атомов из ids заменены: successor(i, edges) не '' (повторы в ids считаются один раз).

    D: 2

    >>> superseded((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ('a1', 'a2'))
    1
    >>> superseded((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ('a1', 'a1'))
    1
    >>> superseded((), ('a1',))
    0
    """
    return len({atom for atom in ids if successor(atom, edges) != ''})
