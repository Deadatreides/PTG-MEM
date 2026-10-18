"""Связи атомов как в Graphiti: memcore.types.Edge(src, dst, kind, valid_at, invalid_at, source), действует, пока
invalid_at == 0.0; новая заявка закрывает прежнюю связь пары (memcore.record_relation), ничего не стирается.
Таблица relations в LanceDB папки home. Отсюда же — состояние атома, связи у карточки, лента решений, соседи по
файлу, эго-сеть, дата и путь от корня для показа.

Вид связи: supersedes (src заменяет dst), contradicts, fixes, refines, continues, returns_to, reinforces. Порядок
показа — этот же (TYPE_ORDER).

node: composite
effects: io
"""
import math
import time

import deal
import lancedb
import memcore
import memshell
import pyarrow
from memcore.types import Edge

from ptgcore.relations import chain, claim, feed, hood, show, state, table
from ptgcore.types import Decision, Entry, GEdge, GNode, Graph, Link, Side

TYPE_ORDER = ('supersedes', 'contradicts', 'fixes', 'refines', 'continues', 'returns_to', 'reinforces')


@deal.pre(lambda home: home != '')
@deal.ensure(lambda home, result: all(a.valid_at <= b.valid_at for a, b in zip(result, result[1:], strict=False)))
@deal.has('io')
def claims(home: str) -> tuple[Edge, ...]:
    """Все связи по времени записи.

    D: 1

    >>> claims('нет-базы')
    ()
    """
    return table.read_edges(home)


@deal.pre(lambda home, src, dst, kind, at, source: home != '' and src != dst and kind in TYPE_ORDER and at > 0.0 and source != '')
@deal.ensure(lambda home, src, dst, kind, at, source, result: all({e.src, e.dst} == {src, dst} for e in result)
             and sum(e.invalid_at == 0.0 for e in result) == 1)
@deal.has('io')
def record(home: str, src: str, dst: str, kind: str, at: float, source: str) -> tuple[Edge, ...]:
    """Записать заявку memcore.record_relation(claims, Claim(src, dst, kind, at, source)); итог — все связи пары
    (действующая — одна).

    D: 1

    >>> record('rel_record', 'a2', 'a1', 'supersedes', 5.0, 'edit')
    (Edge(src='a2', dst='a1', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='edit'),)
    """
    edges = claim.apply(table.read_edges(home), src, dst, kind, at, source)
    table.write_edges(home, edges)
    return claim.of_pair(edges, src, dst)


@deal.pre(lambda home, pairs, kind, at, source: home != '' and kind in TYPE_ORDER and at > 0.0 and source != ''
          and all(a != b for a, b in pairs))
@deal.ensure(lambda home, pairs, kind, at, source, result: result == len(pairs))
@deal.has('io')
def record_pairs(home: str, pairs: tuple[tuple[str, str], ...], kind: str, at: float, source: str) -> int:
    """Записать заявки (src, dst) одного вида разом; итог — сколько заявок.

    D: 2

    >>> record_pairs('нет-базы-пар', (), 'supersedes', 5.0, 'edit')
    0
    """
    if not pairs:
        return 0
    table.write_edges(home, claim.apply_all(table.read_edges(home), pairs, kind, at, source))
    return len(pairs)


@deal.pre(lambda home, claims, source: home != '' and source != ''
          and all(s != d and k in TYPE_ORDER and math.isfinite(at) and at > 0.0 for s, d, k, at in claims))
@deal.ensure(lambda home, claims, source, result: result == len(claims))
@deal.has('io')
def record_many(home: str, claims: tuple[tuple[str, str, str, float], ...], source: str) -> int:
    """Записать заявки (src, dst, вид, время) разных видов и времени разом, одной записью таблицы (перенос базы: 13.10
    запись пачками по record_pairs перечитывала таблицу на каждую пачку — 38 659 связей за 54 минуты): заявки
    сгруппированы по паре (src, dst), внутри пары — memcore.record_relation по порядку времени (при равенстве — по
    порядку claims) поверх прежних связей этой пары; связи других пар не меняются; итог — сколько заявок.

    D: 3

    >>> record_many('rel_many', (('a2', 'a1', 'supersedes', 5.0), ('a3', 'a1', 'contradicts', 6.0)), 'v6')
    2
    >>> record_many('нет-базы-связей', (), 'v6')
    0
    """
    if not claims:
        return 0
    edges = table.read_edges(home)
    for _, (src, dst, kind, at) in sorted(enumerate(claims), key=lambda p: (p[1][3], p[0])):
        edges = claim.apply(edges, src, dst, kind, at, source)
    table.write_edges(home, edges)
    return len(claims)


@deal.ensure(lambda edges, ids, result: len(set(result)) == len(result))
@deal.has()
def ends(edges: tuple[Edge, ...], ids: tuple[str, ...]) -> tuple[str, ...]:
    """Другие концы связей, касающихся ids (ids = () — все концы всех связей), по первому появлению; ids и уже
    найденные концы — множествами (14.10: поиск в кортеже — 2,35 с на ленте решений).

    D: 3

    >>> ends((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'contradicts', 7.0, 0.0, 'agent')), ('a2',))
    ('a1', 'a3')
    >>> ends((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ())
    ('a2', 'a1')
    """
    return hood.ends(edges, ids)


@deal.ensure(lambda atom, edges, result: result != atom)
@deal.has()
def successor(atom: str, edges: tuple[Edge, ...]) -> str:
    """Текущая версия атома (memcore.current_version по действующим связям); '' — атом и есть текущий.

    D: 1

    >>> successor('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),))
    'a2'
    >>> successor('a2', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),))
    ''
    """
    return state.successor(atom, edges)


@deal.ensure(lambda edges, ids, result: set(result) <= set(edges)
             and all(e in result for e in edges if e.src in ids or e.dst in ids))
@deal.has()
def scope(edges: tuple[Edge, ...], ids: tuple[str, ...]) -> tuple[Edge, ...]:
    """Связи, от которых зависят карточки атомов ids, за один проход по edges (индекс по концам строится один раз):
    все связи, касающиеся ids, и цепочки замен (supersedes, от заменённого к заменившему, циклы — один раз) от ids и
    от других концов этих связей — successor, status и links атомов ids и состояния их соседей по scope те же, что по
    edges; порядок — как в edges. 14.10: карточка на каждую из 120 находок проходила по всем 16 702 связям — recall
    3,2 с.

    D: 3

    >>> scope((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'supersedes', 6.0, 0.0, 'edit'), Edge('b2', 'b1', 'supersedes', 7.0, 0.0, 'edit')), ('a1',))
    (Edge(src='a2', dst='a1', kind='supersedes', valid_at=5.0, invalid_at=0.0, source='edit'), Edge(src='a3', dst='a2', kind='supersedes', valid_at=6.0, invalid_at=0.0, source='edit'))
    >>> scope((Edge('a2', 'a1', 'contradicts', 5.0, 0.0, 'agent'), Edge('a3', 'a2', 'supersedes', 6.0, 0.0, 'edit')), ('a1',))
    (Edge(src='a2', dst='a1', kind='contradicts', valid_at=5.0, invalid_at=0.0, source='agent'), Edge(src='a3', dst='a2', kind='supersedes', valid_at=6.0, invalid_at=0.0, source='edit'))
    >>> scope((), ('a1',))
    ()
    """
    if not ids:
        return ()
    return chain.pick(edges, chain.touching(edges, ids) + chain.walk(edges, ids + hood.ends(edges, ids)))


@deal.ensure(lambda entry, edges, result: result in ('active', 'superseded', 'retracted'))
@deal.has()
def status(entry: Entry, edges: tuple[Edge, ...]) -> str:
    """ФТ-006. 'retracted', если атом снят; иначе 'superseded', если есть текущая версия другая; иначе 'active'.

    D: 2

    >>> status(Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1.0, 'active', 0.0, '', 'file'), (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),))
    'superseded'
    >>> status(Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1.0, 'retracted', 5.0, '', 'file'), ())
    'retracted'
    """
    return state.status(entry, edges)


@deal.ensure(lambda ts, result: len(result) == 10 or result == '?')
@deal.has()
def day(ts: float) -> str:
    """Дата для показа: time.strftime('%Y-%m-%d', time.localtime(ts)); не время — '?'.

    D: 1

    >>> day(1760000000.0)
    '2025-10-09'
    """
    return show.day(ts)


@deal.has()
def rel(path: str, root: str) -> str:
    """Путь для показа: от root, если внутри него (os.path.relpath без «..»); 'memory://…', пустой и чужой — как есть.

    D: 3

    >>> rel('memory://notes', '.')
    'memory://notes'
    >>> rel('n.md', '.')
    'n.md'
    """
    return show.rel(path, root)


@deal.ensure(lambda atom, edges, others, result: all(x.dir in ('in', 'out') and x.kind in TYPE_ORDER for x in result)
             and [TYPE_ORDER.index(x.kind) for x in result] == sorted(TYPE_ORDER.index(x.kind) for x in result))
@deal.has()
def links(atom: str, edges: tuple[Edge, ...], others: tuple[Entry, ...]) -> tuple[Link, ...]:
    """Действующие связи атома с известными концами (others): направление, вид, другой конец, кто заявил, его
    состояние (status), дата (day), первые 240 знаков текста; по TYPE_ORDER, внутри — по времени записи.

    D: 4

    >>> links('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), (Entry('a2', 'n.md', 0, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file'),))
    (Link(dir='in', kind='supersedes', id='a2', source='edit', status='active', date='2025-10-09', text='новое'),)
    >>> links('a1', (), ())
    ()
    """
    return feed.links(atom, edges, others)


@deal.ensure(lambda entry, siblings, edges, root, result: len(result) <= 9 and (not result or entry.id in {s.id for s in result}))
@deal.has()
def lineage(entry: Entry, siblings: tuple[Entry, ...], edges: tuple[Edge, ...], root: str) -> tuple[Side, ...]:
    """Соседи по файлу: до четырёх атомов до и до четырёх после entry в siblings (порядок siblings), сам атом внутри;
    Side(id, первые 160 знаков текста, rel(file, root), status, day(at)). entry нет в siblings — ().

    D: 2

    >>> lineage(Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1760000000.0, 'active', 0.0, '', 'file'), (Entry('a1', 'n.md', 0, 'd1', 'раз', '', 'раз', 1760000000.0, 'active', 0.0, '', 'file'),), (), '.')
    (Side(id='a1', text='раз', file='n.md', status='active', date='2025-10-09'),)
    """
    return feed.lineage(entry, siblings, edges, root)


@deal.pre(lambda edges, entries, root, limit: limit >= 0)
@deal.ensure(lambda edges, entries, root, limit, result: len(result) <= limit
             and all(a.at >= b.at for a, b in zip(result, result[1:], strict=False)))
@deal.has()
def decisions(edges: tuple[Edge, ...], entries: tuple[Entry, ...], root: str, limit: int) -> tuple[Decision, ...]:
    """ФТ-010. Лента решений: memshell.decision_pairs(edges, limit), только пары с известными концами; Decision(вид,
    day(valid_at), valid_at, кто заявил, закрыта ли (invalid_at != 0.0), новое — src, старое — dst; Side с первыми 280
    знаками текста). Состояния сторон — по scope(edges, концы выбранных пар), а не по всем связям.

    D: 3

    >>> decisions((Edge('a2', 'a1', 'supersedes', 1760003600.0, 0.0, 'edit'),), (Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1760000000.0, 'retracted', 1760003600.0, '', 'file'), Entry('a2', 'n.md', 0, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file')), '.', 50)
    (Decision(kind='supersedes', date='2025-10-09', at=1760003600.0, source='edit', closed=False, new=Side(id='a2', text='новое', file='n.md', status='active', date='2025-10-09'), old=Side(id='a1', text='старое', file='n.md', status='retracted', date='2025-10-09')),)
    >>> decisions((), (), '.', 50)
    ()
    """
    chosen = feed.picked(edges, entries, limit)
    return feed.decide(chosen, scope(edges, ends(chosen, ())), entries, root)


@deal.pre(lambda atom, edges, depth, limit: depth >= 1 and limit >= 1)
@deal.ensure(lambda atom, edges, depth, limit, result: len(result) <= limit and result[:1] == (atom,))
@deal.has()
def reach(atom: str, edges: tuple[Edge, ...], depth: int, limit: int) -> tuple[str, ...]:
    """ФТ-011. Атомы эго-сети: сам атом, затем соседи по связям (любое направление, и закрытые) слоями до глубины
    depth, не больше limit; в слое — по порядку edges.

    D: 4

    >>> reach('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'contradicts', 7.0, 0.0, 'agent')), 1, 60)
    ('a1', 'a2')
    >>> reach('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'contradicts', 7.0, 0.0, 'agent')), 2, 60)
    ('a1', 'a2', 'a3')
    """
    seen: tuple[str, ...] = (atom,)
    frontier: tuple[str, ...] = (atom,)
    for _ in range(depth):
        frontier = hood.layer(edges, frontier, seen, limit)
        if not frontier:
            break
        seen = seen + frontier
    return seen


@deal.ensure(lambda atom, edges, entries, root, result: {n.id for n in result.nodes} <= {e.id for e in entries}
             and all(e.src in {n.id for n in result.nodes} and e.dst in {n.id for n in result.nodes} for e in result.edges))
@deal.has()
def ego(atom: str, edges: tuple[Edge, ...], entries: tuple[Entry, ...], root: str) -> Graph:
    """Эго-сеть по известным атомам: узлы — entries по порядку (подпись — первые 60 знаков вопроса или текста в одну
    строку, status, day, rel, center — atom); рёбра — связи между ними (GEdge(src, dst, kind)), по порядку edges.

    D: 3

    >>> ego('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), (Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1760000000.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file')), '.')
    Graph(nodes=(GNode(id='a1', label='старое', status='superseded', date='2025-10-09', file='n.md', center=True), GNode(id='a2', label='новое', status='active', date='2025-10-09', file='n.md', center=False)), edges=(GEdge(src='a2', dst='a1', kind='supersedes'),))
    """
    return hood.ego(atom, edges, entries, root)


@deal.pre(lambda graph, extra, limit: limit >= 1)
@deal.ensure(lambda graph, extra, limit, result: len(result.nodes) <= max(limit, len(graph.nodes))
             and set(graph.edges) <= set(result.edges))
@deal.has()
def merge(graph: Graph, extra: tuple[GEdge, ...], limit: int) -> Graph:
    """Добавить к эго-сети рёбра метадерева: ребро берётся, если оба конца уже узлы или новых узлов не больше limit
    всего; новый узел — GNode(id, id, 'code', '', '', False); без повторов рёбер.

    D: 4

    >>> merge(Graph((GNode('a1', 'старое', 'active', '2025-10-09', 'n.md', True),), ()), (GEdge('a1', 'file:PTG-MEM/x.py', 'edits'),), 60)
    Graph(nodes=(GNode(id='a1', label='старое', status='active', date='2025-10-09', file='n.md', center=True), GNode(id='file:PTG-MEM/x.py', label='file:PTG-MEM/x.py', status='code', date='', file='', center=False)), edges=(GEdge(src='a1', dst='file:PTG-MEM/x.py', kind='edits'),))
    """
    return hood.merge(graph, extra, limit)


@deal.ensure(lambda edges, ids, result: 0 <= result <= len(set(ids)))
@deal.has()
def superseded(edges: tuple[Edge, ...], ids: tuple[str, ...]) -> int:
    """Сколько атомов ids заменены (successor не ''), за один проход по edges: индекс действующих supersedes по
    заменённому строится один раз, цепочка каждого атома — по индексу (14.10: successor по всем 16 702 связям для
    каждого из 24 184 атомов — 340 с внутри save).

    D: 1

    >>> superseded((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ('a1', 'a2'))
    1
    """
    return chain.replaced(edges, ids)
