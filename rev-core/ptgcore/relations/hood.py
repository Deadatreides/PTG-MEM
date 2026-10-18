"""Окрестность атома в графе связей: другие концы связей, слой эго-сети, сама эго-сеть и её слияние с рёбрами метадерева.

node: leaf
"""
import deal
from memcore.types import Edge

from ptgcore.relations import show, state
from ptgcore.types import Entry, GEdge, GNode, Graph


@deal.ensure(lambda edges, ids, result: len(set(result)) == len(result))
@deal.has()
def ends(edges: tuple[Edge, ...], ids: tuple[str, ...]) -> tuple[str, ...]:
    """Другие концы связей, касающихся ids, по первому появлению, без повторов; закрытые связи тоже. ids = (): по порядку
    edges — src, затем dst каждой связи. Иначе по порядку edges: если src в ids — dst, затем, если dst в ids — src. ids и
    уже найденные концы — множествами (14.10: поиск в кортеже — 2,35 с на ленте решений).

    D: 5

    >>> ends((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'contradicts', 7.0, 0.0, 'agent')), ('a2',))
    ('a1', 'a3')
    >>> ends((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), ())
    ('a2', 'a1')
    >>> ends((), ())
    ()
    >>> ends((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a2', 'a1', 'refines', 6.0, 0.0, 'agent')), ('a1',))
    ('a2',)
    """
    wanted = set(ids)
    found: list[str] = []
    seen: set[str] = set()

    def take(atom: str) -> None:
        if atom not in seen:
            seen.add(atom)
            found.append(atom)

    for e in edges:
        if not wanted:
            take(e.src)
            take(e.dst)
            continue
        if e.src in wanted:
            take(e.dst)
        if e.dst in wanted:
            take(e.src)
    return tuple(found)


@deal.pre(lambda edges, frontier, seen, limit: limit >= 0)
@deal.ensure(lambda edges, frontier, seen, limit, result: len(set(result)) == len(result) and not set(result) & set(seen)
             and len(seen) + len(result) <= max(limit, len(seen)))
@deal.has()
def layer(edges: tuple[Edge, ...], frontier: tuple[str, ...], seen: tuple[str, ...], limit: int) -> tuple[str, ...]:
    """Следующий слой эго-сети: по порядку edges, для каждой связи (любого вида, и закрытой), у которой src или dst в
    frontier, — сначала src, затем dst, если этого атома нет ни в seen, ни среди уже взятых, и len(seen) + число взятых <
    limit; итог — взятые атомы по порядку.

    D: 7

    >>> layer((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'contradicts', 7.0, 0.0, 'agent')), ('a1',), ('a1',), 60)
    ('a2',)
    >>> layer((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'contradicts', 7.0, 0.0, 'agent')), ('a2',), ('a1', 'a2'), 60)
    ('a3',)
    >>> layer((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'contradicts', 7.0, 0.0, 'agent')), ('a1',), ('a1',), 1)
    ()
    >>> layer((Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'), Edge('a3', 'a2', 'contradicts', 7.0, 0.0, 'agent')), ('a2',), ('a2',), 2)
    ('a1',)
    >>> layer((), ('a1',), ('a1',), 60)
    ()
    """
    front = set(frontier)
    blocked = set(seen)
    taken: list[str] = []
    for e in edges:
        if e.src not in front and e.dst not in front:
            continue
        for atom in (e.src, e.dst):
            if atom not in blocked and len(seen) + len(taken) < limit:
                blocked.add(atom)
                taken.append(atom)
    return tuple(taken)


@deal.ensure(lambda atom, edges, entries, root, result: {n.id for n in result.nodes} <= {e.id for e in entries}
             and all(e.src in {n.id for n in result.nodes} and e.dst in {n.id for n in result.nodes} for e in result.edges)
             and len(set(result.edges)) == len(result.edges))
@deal.has()
def ego(atom: str, edges: tuple[Edge, ...], entries: tuple[Entry, ...], root: str) -> Graph:
    """Эго-сеть по известным атомам: узлы — entries по порядку, GNode(id, подпись, state.status(атом, edges),
    show.day(at), show.rel(file, root), id == atom); подпись — первые 60 знаков строки ' '.join((question or text).split()).
    Рёбра — все связи edges (и закрытые), у которых и src, и dst — id узлов, по порядку edges, GEdge(src, dst, kind); связь с
    тем же (src, dst, kind) берётся один раз.

    D: 6

    >>> ego('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 0.0, 'edit'),), (Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1760000000.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file')), '.')
    Graph(nodes=(GNode(id='a1', label='старое', status='superseded', date='2025-10-09', file='n.md', center=True), GNode(id='a2', label='новое', status='active', date='2025-10-09', file='n.md', center=False)), edges=(GEdge(src='a2', dst='a1', kind='supersedes'),))
    >>> ego('a1', (), (), '.')
    Graph(nodes=(), edges=())
    >>> ego('a1', (Edge('a2', 'a1', 'supersedes', 5.0, 6.0, 'edit'), Edge('a2', 'a1', 'supersedes', 7.0, 0.0, 'edit'), Edge('a9', 'a1', 'refines', 8.0, 0.0, 'agent')), (Entry('a1', 'n.md', 0, 'd1', 'старое', '', 'старое', 1760000000.0, 'active', 0.0, '', 'file'), Entry('a2', 'n.md', 1, 'd2', 'новое', '', 'новое', 1760003600.0, 'active', 0.0, '', 'file')), '.')
    Graph(nodes=(GNode(id='a1', label='старое', status='superseded', date='2025-10-09', file='n.md', center=True), GNode(id='a2', label='новое', status='active', date='2025-10-09', file='n.md', center=False)), edges=(GEdge(src='a2', dst='a1', kind='supersedes'),))
    >>> ego('a9', (), (Entry('a1', 'n.md', 0, 'd1', 'Храним в LanceDB.', 'Какой  базой   храним?', 'Храним в LanceDB.', 1760000000.0, 'active', 0.0, '', 'file'),), '.')
    Graph(nodes=(GNode(id='a1', label='Какой базой храним?', status='active', date='2025-10-09', file='n.md', center=False),), edges=())
    """
    nodes: list[GNode] = []
    for entry in entries:
        label = ' '.join((entry.question or entry.text).split())[:60]
        nodes.append(
            GNode(
                entry.id,
                label,
                state.status(entry, edges),
                show.day(entry.at),
                show.rel(entry.file, root),
                entry.id == atom,
            )
        )
    ids = {n.id for n in nodes}
    pairs: list[GEdge] = []
    for e in edges:
        if e.src in ids and e.dst in ids:
            g = GEdge(e.src, e.dst, e.kind)
            if g not in pairs:
                pairs.append(g)
    return Graph(tuple(nodes), tuple(pairs))


@deal.pre(lambda graph, extra, limit: limit >= 1)
@deal.ensure(lambda graph, extra, limit, result: len(result.nodes) <= max(limit, len(graph.nodes))
             and set(graph.edges) <= set(result.edges))
@deal.has()
def merge(graph: Graph, extra: tuple[GEdge, ...], limit: int) -> Graph:
    """Добавить к эго-сети рёбра метадерева: по порядку extra ребро пропускается, если оно уже есть среди рёбер результата;
    иначе берётся, если оба его конца уже узлы или после добавления новых концов (src, затем dst; без повторов) узлов
    становится не больше limit. Новый узел — GNode(id, id, 'code', '', '', False). Узлы и рёбра graph остаются как были,
    новые — в конец.

    D: 6

    >>> merge(Graph((GNode('a1', 'старое', 'active', '2025-10-09', 'n.md', True),), ()), (GEdge('a1', 'file:PTG-MEM/x.py', 'edits'),), 60)
    Graph(nodes=(GNode(id='a1', label='старое', status='active', date='2025-10-09', file='n.md', center=True), GNode(id='file:PTG-MEM/x.py', label='file:PTG-MEM/x.py', status='code', date='', file='', center=False)), edges=(GEdge(src='a1', dst='file:PTG-MEM/x.py', kind='edits'),))
    >>> merge(Graph((GNode('a1', 'старое', 'active', '2025-10-09', 'n.md', True), GNode('a2', 'новое', 'active', '2025-10-09', 'n.md', False)), ()), (GEdge('a1', 'a2', 'next'), GEdge('a1', 'a2', 'next')), 1)
    Graph(nodes=(GNode(id='a1', label='старое', status='active', date='2025-10-09', file='n.md', center=True), GNode(id='a2', label='новое', status='active', date='2025-10-09', file='n.md', center=False)), edges=(GEdge(src='a1', dst='a2', kind='next'),))
    >>> merge(Graph((GNode('a1', 'старое', 'active', '2025-10-09', 'n.md', True),), ()), (GEdge('a1', 'file:PTG-MEM/x.py', 'edits'),), 1)
    Graph(nodes=(GNode(id='a1', label='старое', status='active', date='2025-10-09', file='n.md', center=True),), edges=())
    >>> merge(Graph((), ()), (GEdge('s', 't', 'contains'),), 2)
    Graph(nodes=(GNode(id='s', label='s', status='code', date='', file='', center=False), GNode(id='t', label='t', status='code', date='', file='', center=False)), edges=(GEdge(src='s', dst='t', kind='contains'),))
    """
    nodes = list(graph.nodes)
    ids = {n.id for n in nodes}
    edges = list(graph.edges)
    for g in extra:
        if g in edges:
            continue
        new = [a for a in dict.fromkeys((g.src, g.dst)) if a not in ids]
        if new and len(nodes) + len(new) > limit:
            continue
        for a in new:
            ids.add(a)
            nodes.append(GNode(a, a, 'code', '', '', False))
        edges.append(g)
    return Graph(tuple(nodes), tuple(edges))
