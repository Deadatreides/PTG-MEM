"""Поиск: пространства базы и пути запроса, запись текста и векторов в хранилище ptgfull, слияние мест RRF, выбор
карточек.

Готовое: ptgfull.store (atoms с полнотекстовым индексом, vec_<пространство>, spaces, adapters), ptgfull.spaces
(plan_routes, fuse), ptgfull.adapter.adapt. Векторы запроса даёт демон: пары (пространство модели, вектор) — по
путям query_routes; пространство с путём 'adapter' получает вектор источника через адаптер.

node: composite
effects: io
"""
import deal
import ptgfull.adapter
import ptgfull.spaces
import ptgfull.store
from memcore.types import Edge
from ptgfull.types import Hit, Route, Space

from ptgcore import relations
from ptgcore.retrieval import cards, prewrite, queries
from ptgcore.types import Card, Entry

RRF_K = 60


@deal.pre(lambda home, space: home != "" and space.dim >= 1)
@deal.ensure(lambda home, space, result: space in result)
@deal.raises(ValueError)
@deal.has("io")
def register(home: str, space: Space) -> tuple[Space, ...]:
    """ФТ-015. Зарегистрировать пространство (ptgfull.store.write_space); то же имя с другим поставщиком, моделью или
    размерностью — ValueError с обоими описаниями; итог — пространства базы.

    D: 2

    >>> register('ret_register', Space('h', 'hash', 'words', 4))
    (Space(name='h', provider='hash', model='words', dim=4),)
    """
    problem = prewrite.clash(ptgfull.store.read_spaces(home), space)
    if problem:
        raise ValueError(problem)
    ptgfull.store.write_space(home, space)
    return ptgfull.store.read_spaces(home)


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: len(set(result)) == len(result))
@deal.has("io")
def spaces(home: str) -> tuple[str, ...]:
    """Имена пространств базы.

    D: 1

    >>> spaces('нет-базы')
    ()
    """
    return tuple(s.name for s in ptgfull.store.read_spaces(home))


@deal.pre(lambda home, ready: home != "")
@deal.ensure(lambda home, ready, result: all(r.source in ready for r in result))
@deal.has("io")
def routes(home: str, ready: tuple[str, ...]) -> tuple[Route, ...]:
    """Пути запроса: ptgfull.spaces.plan_routes(пространства базы, ready, адаптеры базы); ready — пространства, чьи
    модели демон может запустить.

    D: 1

    >>> routes('нет-базы', ('h',))
    ()
    """
    return ptgfull.spaces.plan_routes(ptgfull.store.read_spaces(home), ready, ptgfull.store.adapters(home))


@deal.pre(lambda home, entries, space, vectors: home != "" and len(vectors) in (0, len(entries))
          and (not vectors or space != ""))
@deal.ensure(lambda home, entries, space, vectors, result: result == len(entries))
@deal.has("io")
def index(home: str, entries: tuple[Entry, ...], space: str, vectors: tuple[tuple[float, ...], ...]) -> int:
    """Текст атомов — в ptgfull.store.write_atoms (ptgfull Atom(id, file, text, session, position)), векторы — в
    ptgfull.store.write_vectors(space) по тем же id; vectors = () — эмбеддера нет, только текст; итог — сколько атомов.

    D: 2

    >>> index('ret_index', (Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'),), 'h', ((1.0, 0.0, 0.0, 0.0),))
    1
    >>> index('нет-базы-индекса', (), '', ())
    0
    """
    ptgfull.store.write_atoms(home, prewrite.atoms(entries))
    if vectors:
        ptgfull.store.write_vectors(home, space, tuple(e.id for e in entries), vectors)
    return len(entries)


@deal.pre(lambda home, query, vectors, n: home != "" and n >= 1 and len({s for s, _ in vectors}) == len(vectors))
@deal.ensure(lambda home, query, vectors, n, result: len(result) <= n and len({h.id for h in result}) == len(result))
@deal.has("io")
def ranked(home: str, query: str, vectors: tuple[tuple[str, tuple[float, ...]], ...], n: int) -> tuple[Hit, ...]:
    """ФТ-007. Места по запросу: полнотекст (ptgfull.store.search_text) и каждое пространство с путём из routes(home,
    имена из vectors) — 'direct' — свой вектор, 'adapter' — ptgfull.adapter.adapt(вектор источника, адаптер); поиск
    ptgfull.store.search_vectors; слияние ptgfull.spaces.fuse(списки, RRF_K, n).

    D: 3

    >>> ranked('нет-базы', 'кэш', (), 10)
    ()
    """
    lists = [ptgfull.store.search_text(home, query, n)]
    for route in routes(home, tuple(name for name, _ in vectors)):
        vector = queries.source_vector(route, vectors)
        if route.how == 'adapter':
            vector = queries.carry(vector, ptgfull.store.read_adapter(home, route.source, route.space))
        lists.append(ptgfull.store.search_vectors(home, route.space, vector, n))
    return ptgfull.spaces.fuse(tuple(lists), RRF_K, n)


@deal.pre(lambda home, query, vectors, n: home != "" and n >= 0)
@deal.ensure(lambda home, query, vectors, n, result: len(result[0]) <= n
             and all(a[1] >= b[1] for a, b in zip(result[1], result[1][1:], strict=False)))
@deal.has("io")
def evidence(home: str, query: str, vectors: tuple[tuple[str, tuple[float, ...]], ...],
             n: int) -> tuple[tuple[Hit, ...], tuple[tuple[str, float], ...], tuple[str, ...]]:
    """ФТ-007, ФТ-008. Места по запросу и их опора за один проход поиска: (слияние — как ranked(home, query, vectors,
    n); косинусы — пары (id, наибольшая оценка id в списках пространств), по убыванию оценки, при равенстве — по id;
    слова — id списка полнотекста по порядку).

    D: 2

    >>> evidence('нет-базы', 'кэш', (), 10)
    ((), (), ())
    """
    words = ptgfull.store.search_text(home, query, n)
    lists = [words]
    best: dict[str, float] = {}
    for route in routes(home, tuple(name for name, _ in vectors)):
        vector = queries.source_vector(route, vectors)
        if route.how == 'adapter':
            vector = queries.carry(vector, ptgfull.store.read_adapter(home, route.source, route.space))
        found = ptgfull.store.search_vectors(home, route.space, vector, n)
        lists.append(found)
        for hit in found:
            best[hit.id] = max(best.get(hit.id, hit.score), hit.score)
    near = tuple(sorted(best.items(), key=lambda pair: (-pair[1], pair[0])))
    return ptgfull.spaces.fuse(tuple(lists), RRF_K, n), near, tuple(hit.id for hit in words)


@deal.pre(lambda hits, found, edges, others, k, exclude_ids, with_retracted, root: k >= 0)
@deal.ensure(lambda hits, found, edges, others, k, exclude_ids, with_retracted, root, result:
             len(result) <= k and not {c.id for c in result} & set(exclude_ids)
             and (with_retracted or all(c.status != "retracted" for c in result)))
@deal.has()
def pick(hits: tuple[Hit, ...], found: tuple[Entry, ...], edges: tuple[Edge, ...], others: tuple[Entry, ...], k: int,
         exclude_ids: tuple[str, ...], with_retracted: bool, root: str) -> tuple[Card, ...]:
    """Карточки находок по порядку hits: только атомы из found, без exclude_ids, без снятых (если не with_retracted),
    один на отпечаток; карточка — card(атом, оценка находки, связи, others, root, 700), где связи —
    relations.scope(edges, id отобранных атомов), один раз на вызов; не больше k.

    D: 5

    >>> pick((Hit('a1', 0.0325, 'lexical'),), (Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'),), (), (), 5, (), False, '.')
    (Card(id='a1', score=0.0325, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session=''),)
    >>> pick((Hit('a1', 0.03, 'lexical'),), (Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'retracted', 5.0, '', 'file'),), (), (), 5, (), False, '.')
    ()
    """
    chosen = cards.shortlist(hits, found, exclude_ids, with_retracted, k)
    scoped = relations.scope(edges, tuple(entry.id for _, entry in chosen))
    return tuple(card(entry, hit.score, scoped, others, root, 700) for hit, entry in chosen)


@deal.pre(lambda entry, score, edges, others, root, chars: chars >= 1)
@deal.ensure(lambda entry, score, edges, others, root, chars, result: result.id == entry.id and result.chars == len(entry.text)
             and len(result.text) <= chars and result.current != entry.id)
@deal.has()
def card(entry: Entry, score: float | None, edges: tuple[Edge, ...], others: tuple[Entry, ...], root: str, chars: int) -> Card:
    """Карточка атома: оценка round(score, 4) (None — None), состояние (relations.status), дата (relations.day),
    файл, rel_file (relations.rel), вопрос и текст — первые chars знаков, длина полного текста, связи
    (relations.links(id, edges, others)), текущая версия (relations.successor), сеанс.

    D: 1

    >>> card(Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'), None, (), (), '.', 700)
    Card(id='a1', score=None, status='active', date='2025-10-09', file='n.md', rel_file='n.md', question='', text='LanceDB выбран.', chars=15, links=(), current='', session='')
    """
    return cards.assemble(
        entry,
        score,
        relations.status(entry, edges),
        relations.day(entry.at),
        relations.rel(entry.file, root),
        relations.links(entry.id, edges, others),
        relations.successor(entry.id, edges),
        chars,
    )
