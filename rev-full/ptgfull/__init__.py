"""Полный PTG: текст, код, метадерево с графом и любые эмбеддинги — склейка готовых частей.

Текст — memshell (MarkItDown, атомы-абзацы) и memcore (вычистка секретов); код — codebase-memory-mcp; хранилище и
модели — LanceDB; перенос между моделями — Drift-Adapter; обход — личный PageRank networkx.

node: composite
effects: io
"""
import re

import deal
import memcore
import memshell

from ptgfull import adapter, code, metatree, models, spaces, store, walk
from ptgfull.types import Atom, Counts, Edge, Hit, Piece, Space

LIMIT = 1200
RRF_K = 60
ALPHA = 0.85


@deal.pre(lambda home, files, specs, repo, binary: home != "" and all(re.fullmatch(spaces.SPEC, s) for s in specs))
@deal.ensure(lambda home, files, specs, repo, binary, result: min(result.atoms, result.vectors, result.code, result.edges) >= 0)
@deal.raises(FileNotFoundError, RuntimeError, ValueError)
@deal.has("io", "network")
def index(home: str, files: tuple[str, ...], specs: tuple[str, ...], repo: str, binary: str) -> Counts:
    """P1, P2, P3, P5. Принять файлы и код в базу home: новые пространства specs, атомы файлов (convert → scrub →
    atomize), векторы атомов во всех пространствах, где модель доступна; при repo — граф кода; затем рёбра
    метадерева заново: дерево кода, источники текста, швы и рёбра кода.

    D: 6

    >>> index('home_index', ('tests/fixtures/notes.md',), ('h=hash:words@16',), '', '')
    Counts(atoms=2, vectors=2, code=0, edges=3)
    """
    for spec in specs:
        store.write_space(home, spaces.parse_space(spec))
    drafts = [d for path in files for d in memshell.atomize(path, memcore.scrub(memshell.convert(path)).text, LIMIT)]
    fresh = tuple(Atom(d.digest, d.file, d.text, "", d.position) for d in drafts)
    total = store.write_atoms(home, fresh)
    vectors = 0
    for space in store.read_spaces(home):
        if models.can_embed(space):
            done = store.write_vectors(home, space.name, tuple(a.id for a in fresh), models.embed(space, tuple(a.text for a in fresh)))
            vectors += done
    if repo:
        nodes, links = code.code_graph(binary, repo)
        store.write_code(home, nodes, links)
    nodes, links = store.read_code(home)
    atoms = store.read_atoms(home, ())
    edges = metatree.tree_edges(nodes) + metatree.source_edges(atoms) + metatree.stitch(atoms, nodes) + links
    store.write_edges(home, edges)
    return Counts(total, vectors, len(nodes), len(edges))


@deal.pre(lambda home, spec, atoms, vectors: home != "" and re.fullmatch(spaces.SPEC, spec) is not None
          and len(atoms) == len(vectors) and all(len(v) == int(spec.rsplit("@", 1)[1]) for v in vectors))
@deal.ensure(lambda home, spec, atoms, vectors, result: result >= len({a.id for a in atoms}))
@deal.has("io")
def attach(home: str, spec: str, atoms: tuple[Atom, ...], vectors: tuple[tuple[float, ...], ...]) -> int:
    """P1. Подключить готовую базу без пересчёта: пространство spec, атомы и их векторы как есть; итог — векторов
    пространства в базе.

    D: 1

    >>> attach('home_attach', 'old=openai:text-embedding-3-small@2', (Atom('a1', 'n.md', 'текст', '', 0),), ((1.0, 0.0),))
    1
    """
    space = spaces.parse_space(spec)
    store.write_space(home, space)
    store.write_atoms(home, atoms)
    return store.write_vectors(home, space.name, tuple(a.id for a in atoms), vectors)


@deal.pre(lambda home, source, target: home != "" and source != target)
@deal.ensure(lambda home, source, target, result: result >= 0)
@deal.has("io")
def bridge(home: str, source: str, target: str) -> int:
    """P1. Адаптер source → target по атомам, у которых есть векторы обоих пространств; итог — число пар; пар нет —
    0, адаптер не пишется.

    D: 2

    >>> bridge('нет-базы', 'h', 'old')
    0
    """
    theirs = dict(store.read_vectors(home, target))
    pairs = [(vec, theirs[i]) for i, vec in store.read_vectors(home, source) if i in theirs]
    if not pairs:
        return 0
    store.write_adapter(home, source, target, adapter.fit(tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)))
    return len(pairs)


@deal.pre(lambda home, query, budget, limit: home != "" and budget >= 0 and limit >= 1)
@deal.ensure(lambda home, query, budget, limit, result: len(result) <= budget)
@deal.raises(ValueError)
@deal.has("io", "network")
def ask(home: str, query: str, budget: int, limit: int) -> str:
    """P4. Подсказка на запрос: поиск по словам и по всем пространствам, куда есть путь, слияние RRF, соседи по
    метадереву; блок <ptg-context> не длиннее budget. Базы нет — ''.

    D: 5

    >>> ask('нет-базы', 'кэш', 500, 5)
    ''
    """
    known = store.read_spaces(home)
    ready = tuple(s.name for s in known if models.can_embed(s))
    by_name = {s.name: s for s in known}
    rankings = [store.search_text(home, query, limit)]
    for route in spaces.plan_routes(known, ready, store.adapters(home)):
        vector = models.embed(by_name[route.source], (query,))[0]
        if route.how == "adapter":
            vector = adapter.adapt(vector, store.read_adapter(home, route.source, route.space))
        rankings.append(store.search_vectors(home, route.space, vector, limit))
    hits = spaces.fuse(tuple(rankings), RRF_K, limit)
    near = walk.walk(store.read_edges(home), hits, limit, ALPHA)
    order = tuple(h.id for h in hits) + near
    return walk.assemble(walk.pieces(order, store.read_atoms(home, order), store.read_code(home)[0]), budget)

