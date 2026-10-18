"""Карточки находок: какие находки идут в выдачу и как собрать карточку из готовых частей.

node: leaf
"""
import deal
from ptgfull.types import Hit

from ptgcore.types import Card, Entry, Link


@deal.ensure(lambda entry, exclude_ids, with_retracted, result:
             result == (entry.id not in exclude_ids and (with_retracted or entry.status != 'retracted')))
@deal.has()
def visible(entry: Entry, exclude_ids: tuple[str, ...], with_retracted: bool) -> bool:
    """ФТ-007. Можно ли показывать атом: его id не в exclude_ids и он не снят (status != 'retracted'), если не
    with_retracted.

    D: 2

    >>> visible(Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file'), (), False)
    True
    >>> visible(Entry('a4', 'n.md', 3, 'd4', 'четыре', '', 'четыре', 4.0, 'retracted', 9.0, '', 'file'), (), False)
    False
    >>> visible(Entry('a4', 'n.md', 3, 'd4', 'четыре', '', 'четыре', 4.0, 'retracted', 9.0, '', 'file'), (), True)
    True
    >>> visible(Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file'), ('a1',), True)
    False
    """
    return entry.id not in exclude_ids and (with_retracted or entry.status != 'retracted')


@deal.pre(lambda hits, found, exclude_ids, with_retracted, k: k >= 0)
@deal.ensure(lambda hits, found, exclude_ids, with_retracted, k, result: len(result) <= k
             and all(h in hits and e in found and h.id == e.id and e.id not in exclude_ids
                     and (with_retracted or e.status != 'retracted') for h, e in result)
             and len({e.digest for _, e in result}) == len(result))
@deal.has()
def shortlist(hits: tuple[Hit, ...], found: tuple[Entry, ...], exclude_ids: tuple[str, ...], with_retracted: bool,
              k: int) -> tuple[tuple[Hit, Entry], ...]:
    """ФТ-007. Находки по порядку hits, которые идут в выдачу, парами (находка, атом): атом с таким id есть среди found,
    visible(атом, exclude_ids, with_retracted), отпечаток (digest) ещё не встречался среди взятых; взято k — стоп.

    D: 6

    >>> shortlist((Hit('a1', 0.0325, 'lexical'),), (Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'),), (), False, 5)
    ((Hit(id='a1', score=0.0325, via='lexical'), Entry(id='a1', file='n.md', position=0, digest='d1', text='LanceDB выбран.', question='', answer='LanceDB выбран.', at=1760000000.0, status='active', retracted_at=0.0, session='', source='file')),)
    >>> shortlist((Hit('a1', 0.03, 'lexical'),), (Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'retracted', 5.0, '', 'file'),), (), False, 5)
    ()
    >>> shortlist((Hit('a2', 0.03, 'lexical'), Hit('a1', 0.02, 'h')), (Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'два', '', 'два', 2.0, 'active', 0.0, '', 'file')), (), False, 1)
    ((Hit(id='a2', score=0.03, via='lexical'), Entry(id='a2', file='n.md', position=1, digest='d2', text='два', question='', answer='два', at=2.0, status='active', retracted_at=0.0, session='', source='file')),)
    >>> shortlist((Hit('a1', 0.03, 'lexical'), Hit('a3', 0.02, 'lexical')), (Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file'), Entry('a3', 'n.md', 2, 'd1', 'раз', '', 'раз', 3.0, 'active', 0.0, '', 'file')), (), False, 5)
    ((Hit(id='a1', score=0.03, via='lexical'), Entry(id='a1', file='n.md', position=0, digest='d1', text='раз', question='', answer='раз', at=1.0, status='active', retracted_at=0.0, session='', source='file')),)
    >>> shortlist((Hit('zz', 0.04, 'lexical'), Hit('a4', 0.03, 'lexical'), Hit('a1', 0.02, 'lexical'), Hit('a2', 0.01, 'lexical')), (Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'два', '', 'два', 2.0, 'active', 0.0, '', 'file'), Entry('a4', 'n.md', 3, 'd4', 'четыре', '', 'четыре', 4.0, 'retracted', 9.0, '', 'file')), ('a1',), False, 5)
    ((Hit(id='a2', score=0.01, via='lexical'), Entry(id='a2', file='n.md', position=1, digest='d2', text='два', question='', answer='два', at=2.0, status='active', retracted_at=0.0, session='', source='file')),)
    """
    taken: list[tuple[Hit, Entry]] = []
    digests: set[str] = set()
    for hit in hits:
        if len(taken) >= k:
            break
        entry = next((e for e in found if e.id == hit.id), None)
        if entry is None or not visible(entry, exclude_ids, with_retracted) or entry.digest in digests:
            continue
        digests.add(entry.digest)
        taken.append((hit, entry))
    return tuple(taken)


@deal.pre(lambda entry, score, status, date, rel_file, links, current, chars: chars >= 1 and current != entry.id)
@deal.ensure(lambda entry, score, status, date, rel_file, links, current, chars, result:
             result.id == entry.id and result.chars == len(entry.text) and len(result.text) <= chars
             and len(result.question) <= chars and result.current != entry.id and (result.score is None) == (score is None))
@deal.has()
def assemble(entry: Entry, score: float | None, status: str, date: str, rel_file: str, links: tuple[Link, ...], current: str,
             chars: int) -> Card:
    """Карточка из готовых частей: id, файл и сеанс атома; оценка round(score, 4) (None — None); status, date, rel_file,
    links и current как даны; вопрос и текст — первые chars знаков; chars карточки — длина полного текста.

    D: 2

    >>> assemble(Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'), None, 'active', '2025-10-09', 'n.md', (), '', 700)
    Card(id='a1', score=None, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session='')
    >>> assemble(Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'), 0.0325, 'active', '2025-10-09', 'n.md', (), '', 700)
    Card(id='a1', score=0.0325, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session='')
    >>> assemble(Entry('a5', 'docs/n.md', 0, 'd5', 'LanceDB выбран.', 'Какая база?', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, 'sess', 'claude'), 0.03252, 'superseded', '2025-10-09', 'n.md', (), 'a6', 5)
    Card(id='a5', score=0.0325, status='superseded', date='2025-10-09', file='docs/n.md', rel_file='n.md', question='Какая', text='Lance', chars=15, links=(), current='a6', session='sess')
    >>> assemble(Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'), 0.9, 'superseded', '2025-10-09', 'n.md', (Link('in', 'supersedes', 'a2', 'edit', 'active', '2025-10-09', 'новое'),), 'a2', 700)
    Card(id='a1', score=0.9, status='superseded', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(Link(dir='in', kind='supersedes', id='a2', source='edit', status='active', date='2025-10-09', text='новое'),), current='a2', session='')
    """
    return Card(
        id=entry.id,
        score=None if score is None else round(score, 4),
        status=status,
        date=date,
        file=entry.file,
        rel_file=rel_file,
        question=entry.question[:chars],
        text=entry.text[:chars],
        chars=len(entry.text),
        links=links,
        current=current,
        session=entry.session,
    )
