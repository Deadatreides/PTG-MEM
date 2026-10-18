"""Перенос живой базы PTG-MEM формата 6 без пересчёта векторов (ФТ-014).

Формат 6 — папка data/:
* meta.json: {"format": 6, "id_order": [...], "processed_files": {путь: время правки}, "embedder_model", "embedder_dim"};
* nodes.json: {id: узел}, у узла file, order, question, answer, text, timestamp, status ('active' / 'retracted'),
  retracted_at, session_id, source — числа бывают строками, отсутствие — 'None' или нет ключа;
* vectors.npy — векторы в порядке id_order;
* edges.json — [{"from", "to", "type"}], тип — вид связи (см. relations.TYPE_ORDER).

Куда: атомы — ledger.write; текст и векторы — retrieval.index кусками по chunk; связи — только виды ленты решений
(V6_KINDS), одной записью relations.record_many (valid_at — время атома from, источник 'v6'); обработанные файлы — одной
записью ledger.mark_all. Прочие связи формата 6 — догадки по порогу косинуса (13.10 в живой базе: 417 442 reinforces,
28 002 returns_to, 14 496 refines, 9 469 continues из 486 297), их не переносим: соседство даёт метадерево.

node: composite
effects: io
"""
import json
import os

import deal
import memshell
import numpy
from memcore.types import Edge
from ptgfull.types import Space

from ptgcore import ledger, relations, retrieval
from ptgcore.migration import atoms, disk, layout, links, scalars
from ptgcore.types import Entry, Report


# виды связей формата 6, которые переносятся, и их вид здесь: revises в формате 6 писала правка файла — это supersedes
V6_KINDS = {'supersedes': 'supersedes', 'revises': 'supersedes', 'contradicts': 'contradicts', 'fixes': 'fixes'}


@deal.pre(lambda nid, file, position, text, question, answer, at, status, retracted_at, session, source:
          nid != "" and position >= 0 and at >= 0.0 and retracted_at >= 0.0)
@deal.ensure(lambda nid, file, position, text, question, answer, at, status, retracted_at, session, source, result:
             result.id == nid and result.status in ("active", "retracted")
             and (result.status == "retracted") == (status == "retracted")
             and result.session != "None" and result.source != "")
@deal.has()
def v6_entry(nid: str, file: str, position: int, text: str, question: str, answer: str, at: float, status: str,
             retracted_at: float, session: str, source: str) -> Entry:
    """Атом формата 6: id как был, отпечаток memshell.digest(text); состояние 'retracted', если status == 'retracted'
    (retracted_at — данное, а если 0.0 — at), иначе 'active' и retracted_at 0.0; сеанс 'None' — ''; source '' — 'v6'.

    D: 3

    >>> v6_entry('n1', 'docs/a.md', 0, 'Векторы храним в SQLite.', '', 'Векторы храним в SQLite.', 1760000000.0, 'retracted', 0.0, 'None', '')
    Entry(id='n1', file='docs/a.md', position=0, digest='0470878cd8043cf6', text='Векторы храним в SQLite.', question='', answer='Векторы храним в SQLite.', at=1760000000.0, status='retracted', retracted_at=1760000000.0, session='', source='v6')
    >>> v6_entry('n2', 'docs/a.md', 0, 'Векторы храним в LanceDB.', '', 'Векторы храним в LanceDB.', 1760003600.0, 'active', 0.0, '', 'file')
    Entry(id='n2', file='docs/a.md', position=0, digest='1d91240b07a4dac5', text='Векторы храним в LanceDB.', question='', answer='Векторы храним в LanceDB.', at=1760003600.0, status='active', retracted_at=0.0, session='', source='file')
    """
    return atoms.build(nid, file, position, text, question, answer, at, status, retracted_at, session, source)


@deal.pre(lambda src, dst, kind, at: src != dst and at > 0.0)
@deal.ensure(lambda src, dst, kind, at, result: result.invalid_at == 0.0 and result.source == "v6")
@deal.has()
def v6_edge(src: str, dst: str, kind: str, at: float) -> Edge:
    """Связь формата 6: Edge(src, dst, kind, at, 0.0, 'v6').

    D: 1

    >>> v6_edge('n2', 'n1', 'supersedes', 1760003600.0)
    Edge(src='n2', dst='n1', kind='supersedes', valid_at=1760003600.0, invalid_at=0.0, source='v6')
    """
    return links.build(src, dst, kind, at)


@deal.pre(lambda home, store_dir, space, chunk: home != "" and chunk >= 1 and space.dim >= 1)
@deal.ensure(lambda home, store_dir, space, chunk, result: result.file == store_dir and result.added >= result.retracted >= 0)
@deal.raises(FileNotFoundError, ValueError)
@deal.has("io")
def import_v6(home: str, store_dir: str, space: Space, chunk: int) -> Report:
    """Перенести базу формата 6 из store_dir: retrieval.register(space); атомы (v6_entry) — ledger.write и
    retrieval.index(space, векторы) кусками по chunk; связи с известными концами только видов из V6_KINDS (вид —
    по V6_KINDS, valid_at — время атома from, v6_edge) — одной записью relations.record_many(…, 'v6'); processed_files —
    одной записью ledger.mark_all. Нет data/meta.json — FileNotFoundError(store_dir); format не 6 или размерность
    векторов не space.dim — ValueError. Итог: added — атомов, retracted — из них снятых.

    D: 6

    >>> import_v6('mig_v6', 'tests/fixtures/store_v6', Space('v6', 'gguf', 'Qwen3-Embedding-4B-Q4_K_M', 4), 1000)
    Report(file='tests/fixtures/store_v6', added=2, retracted=1, kept=0)
    >>> import_v6('mig_none', 'нет-такой-базы', Space('v6', 'gguf', 'Qwen3-Embedding-4B-Q4_K_M', 4), 1000)
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такой-базы
    """
    ids, processed = layout.parse_meta(disk.read_json(store_dir, 'meta.json'))
    layout.fit(disk.vector_shape(store_dir), len(ids), space.dim)
    nodes = disk.read_json(store_dir, 'nodes.json')
    rows = atoms.rows(ids, nodes)
    entries = tuple(v6_entry(*fields) for fields in atoms.args(ids, nodes))
    retrieval.register(home, space)
    for start in range(0, len(entries), chunk):
        part = entries[start:start + chunk]
        ledger.write(home, part)
        retrieval.index(home, part, space.name, disk.read_vectors(store_dir, rows[start:start + chunk]))
    raw = disk.read_json(store_dir, 'edges.json') if disk.has_file(store_dir, 'edges.json') else []
    edges = tuple(v6_edge(s, d, V6_KINDS[k], t) for s, d, k, t in links.known(raw, entries, tuple(V6_KINDS)))
    relations.record_many(home, tuple((e.src, e.dst, e.kind, e.valid_at) for e in edges), 'v6')
    ledger.mark_all(home, processed)
    return atoms.tally(store_dir, entries)
