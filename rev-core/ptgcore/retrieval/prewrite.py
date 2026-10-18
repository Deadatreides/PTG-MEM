"""Подготовка записи: можно ли занять имя пространства и в какую форму хранилища ptgfull переходят атомы.

node: leaf
"""
import deal
from ptgfull.types import Atom, Space

from ptgcore.types import Entry


@deal.ensure(lambda known, space, result: (result == '') == all(s.name != space.name or s == space for s in known))
@deal.has()
def clash(known: tuple[Space, ...], space: Space) -> str:
    """ФТ-015. Можно ли записать пространство в базу, где уже есть known: пусто — можно (имени нет или там то же самое
    пространство); иначе описание обоих для первого занятого имени — «<имя>: в базе <поставщик>:<модель>@<размерность>,
    пришло <поставщик>:<модель>@<размерность>».

    D: 3

    >>> clash((Space('h', 'hash', 'words', 4),), Space('h', 'hash', 'words', 4))
    ''
    >>> clash((Space('h', 'hash', 'words', 4),), Space('h', 'openai', 'm', 8))
    'h: в базе hash:words@4, пришло openai:m@8'
    >>> clash((Space('h', 'hash', 'words', 4),), Space('g', 'hash', 'words', 4))
    ''
    >>> clash((), Space('h', 'hash', 'words', 4))
    ''
    """
    for s in known:
        if s.name == space.name and s != space:
            return f"{s.name}: в базе {s.provider}:{s.model}@{s.dim}, пришло {space.provider}:{space.model}@{space.dim}"
    return ''


@deal.ensure(lambda entries, result: [a.id for a in result] == [e.id for e in entries]
             and all(a.file == e.file and a.text == e.text and a.session == e.session and a.order == e.position
                     for a, e in zip(result, entries, strict=False)))
@deal.has()
def atoms(entries: tuple[Entry, ...]) -> tuple[Atom, ...]:
    """Атомы ptgfull для записи текста: Atom(id, file, text, session, position) по каждому атому entries, по порядку.

    D: 1

    >>> atoms((Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'),))
    (Atom(id='a1', file='n.md', text='LanceDB выбран.', session='', order=0),)
    >>> atoms((Entry('a2', 's/x.jsonl', 3, 'd2', 'реплика', 'вопрос', 'реплика', 5.0, 'active', 0.0, 'sess1', 'claude'),))
    (Atom(id='a2', file='s/x.jsonl', text='реплика', session='sess1', order=3),)
    >>> atoms(())
    ()
    """
    return tuple(Atom(id=e.id, file=e.file, text=e.text, session=e.session, order=e.position) for e in entries)
