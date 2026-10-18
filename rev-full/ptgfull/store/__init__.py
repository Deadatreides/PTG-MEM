"""Хранилище полного PTG — LanceDB в папке home (lancedb.connect(home)).

Таблицы:
* spaces (name, provider, model, dim);
* atoms (id, file, text, session, order) с полнотекстовым индексом по text;
* vec_<пространство> (id, vector — список float32 длины dim);
* code_nodes (qn, name, label, file, start, end), code_edges (src, dst, kind);
* edges (src, dst, kind);
* adapters (source, target, matrix — список строк матрицы).

Запись spaces, atoms, vec_* и adapters — upsert по ключу (merge_insert): name, id, id, (source, target). code_* и
edges заменяются целиком. Нет папки или таблицы — чтение и поиск отдают пустое. Числа векторов и матриц при
чтении — round(float(x), 6) + 0.0.

node: composite
effects: io
"""
import math

import deal
import lancedb

from ptgfull.store import answer, clause, lookup, tables, unique
from ptgfull.types import Atom, CodeNode, Edge, Hit, Space

SPACES = "spaces"
ATOMS = "atoms"
CODE_NODES = "code_nodes"
CODE_EDGES = "code_edges"
EDGES = "edges"
ADAPTERS = "adapters"


@deal.pre(lambda home, space: home != "" and space.dim >= 1)
@deal.ensure(lambda home, space, result: result >= 1)
@deal.has("io")
def write_space(home: str, space: Space) -> int:
    """Записать пространство (upsert по name); итог — пространств в базе.

    D: 2

    >>> write_space('st_space', Space('h', 'hash', 'words', 4))
    1
    """
    row = {"name": space.name, "provider": space.provider, "model": space.model, "dim": space.dim}
    return tables.put_rows(home, SPACES, ("name",), (row,))


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: len({s.name for s in result}) == len(result))
@deal.has("io")
def read_spaces(home: str) -> tuple[Space, ...]:
    """Пространства базы по имени.

    D: 2

    >>> read_spaces('нет-базы')
    ()
    """
    found = {str(r["name"]): Space(str(r["name"]), str(r["provider"]), str(r["model"]), int(r["dim"]))
             for r in tables.table_rows(home, SPACES, (), "")}
    return tuple(found[name] for name in sorted(found))


@deal.pre(lambda home, atoms: home != "")
@deal.ensure(lambda home, atoms, result: result >= len({a.id for a in atoms}))
@deal.has("io")
def write_atoms(home: str, atoms: tuple[Atom, ...]) -> int:
    """Записать атомы (upsert по id) и обновить полнотекстовый индекс; итог — атомов в базе. Пустой atoms ничего не
    создаёт.

    D: 3

    >>> write_atoms('st_atoms', (Atom('a1', 'n.md', 'LanceDB выбран.', '', 0), Atom('a2', 'n.md', 'кэш', '', 1)))
    2
    >>> write_atoms('нет-базы-атомов', ())
    0
    """
    rows = tuple({"id": a.id, "file": a.file, "text": a.text, "session": a.session, "order": a.order}
                 for a in unique.distinct_atoms(atoms))
    total = tables.put_rows(home, ATOMS, ("id",), rows)
    if rows:
        lookup.index_words(home, ATOMS, "text")
    return total


@deal.pre(lambda home, ids: home != "")
@deal.ensure(lambda home, ids, result: len({a.id for a in result}) == len(result)
             and (not ids or [a.id for a in result] == [i for i in dict.fromkeys(ids) if i in {a.id for a in result}]))
@deal.has("io")
def read_atoms(home: str, ids: tuple[str, ...]) -> tuple[Atom, ...]:
    """Атомы по ids в их порядке, без повторов, отсутствующие пропускаются; ids = () — все атомы по (file, order).

    D: 3

    >>> read_atoms('нет-базы', ('a1',))
    ()
    """
    rows = tables.table_rows(home, ATOMS, (), clause.where_in("id", ids))
    found = tuple(Atom(str(r["id"]), str(r["file"]), str(r["text"]), str(r["session"]), int(r["order"])) for r in rows)
    return answer.arrange_atoms(found, ids)


@deal.pre(lambda home, space, ids, vectors: home != "" and space != "" and len(ids) == len(vectors)
          and len({len(v) for v in vectors}) <= 1 and all(math.isfinite(x) for v in vectors for x in v))
@deal.ensure(lambda home, space, ids, vectors, result: result >= len(set(ids)))
@deal.has("io")
def write_vectors(home: str, space: str, ids: tuple[str, ...], vectors: tuple[tuple[float, ...], ...]) -> int:
    """Записать векторы пространства в vec_<space> (upsert по id); итог — векторов в таблице. Пустой ids ничего не
    создаёт.

    D: 2

    >>> write_vectors('st_vec', 'h', ('a1', 'a2'), ((1.0, 0.0), (0.0, 1.0)))
    2
    >>> write_vectors('нет-базы-векторов', 'h', (), ())
    0
    """
    pairs = unique.distinct_vectors(tuple(zip(ids, vectors, strict=True)))
    rows = tuple({"id": i, "vector": list(v)} for i, v in pairs)
    return tables.put_rows(home, "vec_" + space, ("id",), rows)


@deal.pre(lambda home, space: home != "" and space != "")
@deal.ensure(lambda home, space, result: len({i for i, _ in result}) == len(result))
@deal.has("io")
def read_vectors(home: str, space: str) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Все пары (id, вектор) пространства по id.

    D: 2

    >>> read_vectors('нет-базы', 'h')
    ()
    """
    found = {str(r["id"]): tuple(round(float(x), 6) + 0.0 for x in r["vector"])
             for r in tables.table_rows(home, "vec_" + space, (), "")}
    return tuple((i, found[i]) for i in sorted(found))


@deal.pre(lambda home, space, vector, k: home != "" and space != "" and k >= 0 and all(math.isfinite(x) for x in vector))
@deal.ensure(lambda home, space, vector, k, result: len(result) <= k and all(h.via == space for h in result)
             and all(a.score >= b.score for a, b in zip(result, result[1:], strict=False)))
@deal.has("io")
def search_vectors(home: str, space: str, vector: tuple[float, ...], k: int) -> tuple[Hit, ...]:
    """Ближайшие по косинусу в vec_<space>: оценка round(1 − расстояние, 6), via = space; нулевой вектор или нет
    таблицы — ().

    D: 3

    >>> search_vectors('нет-базы', 'h', (1.0, 0.0), 3)
    ()
    """
    if k == 0 or not any(vector):
        return ()
    return answer.nearest_hits(lookup.find_nearest(home, "vec_" + space, vector, k), space, k)


@deal.pre(lambda home, query, k: home != "" and k >= 0)
@deal.ensure(lambda home, query, k, result: len(result) <= k and all(h.via == "lexical" for h in result)
             and all(a.score >= b.score for a, b in zip(result, result[1:], strict=False)))
@deal.has("io")
def search_text(home: str, query: str, k: int) -> tuple[Hit, ...]:
    """Полнотекстовый поиск LanceDB по atoms.text: оценка round(_score, 6), via = 'lexical'; пустой запрос, нет
    слов или нет таблицы — ().

    D: 3

    >>> search_text('нет-базы', 'кэш', 3)
    ()
    """
    if k == 0 or not clause.has_words(query):
        return ()
    return answer.word_hits(lookup.find_words(home, ATOMS, query, k), k)


@deal.pre(lambda home, nodes, edges: home != "")
@deal.ensure(lambda home, nodes, edges, result: result == len(nodes))
@deal.has("io")
def write_code(home: str, nodes: tuple[CodeNode, ...], edges: tuple[Edge, ...]) -> int:
    """Заменить граф кода целиком; итог — символов.

    D: 2

    >>> write_code('st_code', (CodeNode('m.f', 'f', 'Function', 'm.py', 1, 2),), (Edge('m.f', 'm.f', 'CALLS'),))
    1
    """
    tables.replace_rows(home, CODE_EDGES, tuple({"src": e.src, "dst": e.dst, "kind": e.kind} for e in edges))
    return tables.replace_rows(home, CODE_NODES, tuple(
        {"qn": n.qn, "name": n.name, "label": n.label, "file": n.file, "start": n.start, "end": n.end} for n in nodes))


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: len({n.qn for n in result[0]}) == len(result[0]))
@deal.has("io")
def read_code(home: str) -> tuple[tuple[CodeNode, ...], tuple[Edge, ...]]:
    """Граф кода: символы по qn и рёбра в порядке записи.

    D: 2

    >>> read_code('нет-базы')
    ((), ())
    """
    nodes = tuple(CodeNode(str(r["qn"]), str(r["name"]), str(r["label"]), str(r["file"]), int(r["start"]), int(r["end"]))
                  for r in tables.table_rows(home, CODE_NODES, (), ""))
    links = tuple(Edge(str(r["src"]), str(r["dst"]), str(r["kind"])) for r in tables.table_rows(home, CODE_EDGES, (), ""))
    return unique.distinct_nodes(nodes), links


@deal.pre(lambda home, edges: home != "")
@deal.ensure(lambda home, edges, result: result == len(edges))
@deal.has("io")
def write_edges(home: str, edges: tuple[Edge, ...]) -> int:
    """Заменить рёбра метадерева целиком; итог — рёбер.

    D: 2

    >>> write_edges('st_edges', (Edge('src:n.md', 'a1', 'contains'),))
    1
    """
    return tables.replace_rows(home, EDGES, tuple({"src": e.src, "dst": e.dst, "kind": e.kind} for e in edges))


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: isinstance(result, tuple))
@deal.has("io")
def read_edges(home: str) -> tuple[Edge, ...]:
    """Рёбра метадерева в порядке записи.

    D: 2

    >>> read_edges('нет-базы')
    ()
    """
    return tuple(Edge(str(r["src"]), str(r["dst"]), str(r["kind"])) for r in tables.table_rows(home, EDGES, (), ""))


@deal.pre(lambda home, source, target, matrix: home != "" and source != "" and target != ""
          and len({len(r) for r in matrix}) <= 1 and all(math.isfinite(x) for r in matrix for x in r))
@deal.ensure(lambda home, source, target, matrix, result: result >= 1)
@deal.has("io")
def write_adapter(home: str, source: str, target: str, matrix: tuple[tuple[float, ...], ...]) -> int:
    """Записать адаптер source → target (upsert по паре); итог — адаптеров в базе.

    D: 2

    >>> write_adapter('st_adapter', 'h', 'old', ((0.0, 1.0), (-1.0, 0.0)))
    1
    """
    row = {"source": source, "target": target, "matrix": [list(r) for r in matrix]}
    return tables.put_rows(home, ADAPTERS, ("source", "target"), (row,))


@deal.pre(lambda home, source, target: home != "")
@deal.ensure(lambda home, source, target, result: len({len(r) for r in result}) <= 1)
@deal.has("io")
def read_adapter(home: str, source: str, target: str) -> tuple[tuple[float, ...], ...]:
    """Матрица адаптера source → target; нет — ().

    D: 2

    >>> read_adapter('нет-базы', 'h', 'old')
    ()
    """
    for r in tables.table_rows(home, ADAPTERS, (), ""):
        if r["source"] == source and r["target"] == target:
            return tuple(tuple(round(float(x), 6) + 0.0 for x in row) for row in r["matrix"])
    return ()


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: len(set(result)) == len(result))
@deal.has("io")
def adapters(home: str) -> tuple[tuple[str, str], ...]:
    """Пары (источник, цель) адаптеров базы по порядку пар.

    D: 2

    >>> adapters('нет-базы')
    ()
    """
    rows = tables.table_rows(home, ADAPTERS, ("source", "target"), "")
    return tuple(sorted({(str(r["source"]), str(r["target"])) for r in rows}))
