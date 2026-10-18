"""Атомы формата 6: из nodes.json в аргументы v6_entry, сам атом Entry и итог переноса. Решения чистые.

node: leaf
"""
import deal
import memshell

from ptgcore.migration import scalars
from ptgcore.types import Entry, Report

Fields = tuple[str, str, int, str, str, str, float, str, float, str, str]  # аргументы v6_entry по порядку


@deal.ensure(lambda ids, nodes, result: list(result) == sorted(set(result))
             and all(0 <= r < len(ids) and ids[r] != '' for r in result)
             and len({ids[r] for r in result}) == len(result))
@deal.has()
def rows(ids: tuple[str, ...], nodes: object) -> tuple[int, ...]:
    """Номера строк vectors.npy у атомов, чей id есть и в id_order (ids), и в nodes.json (nodes — разобранный JSON; не
    объект — ()): по порядку id_order, пустой id и повтор пропускаются.

    D: 4

    >>> rows(('n1', 'n2'), {'n1': {'id': 'n1', 'file': 'docs/a.md', 'order': '0', 'question': '', 'answer': 'Векторы храним в SQLite.', 'text': 'Векторы храним в SQLite.', 'timestamp': '1760000000.0', 'status': 'retracted', 'session_id': '', 'source': 'file'}, 'n2': {'id': 'n2', 'file': 'docs/a.md', 'order': '0', 'question': '', 'answer': 'Векторы храним в LanceDB.', 'text': 'Векторы храним в LanceDB.', 'timestamp': '1760003600.0', 'status': 'active', 'session_id': '', 'source': 'file'}})
    (0, 1)
    >>> rows(('n1', 'x', 'n2', 'n1'), {'n1': {}, 'n2': {}})
    (0, 2)
    >>> rows((), 'мусор')
    ()
    """
    if not isinstance(nodes, dict):
        return ()
    out: list[int] = []
    seen: set[str] = set()
    for r, nid in enumerate(ids):
        if nid != '' and nid in nodes and nid not in seen:
            seen.add(nid)
            out.append(r)
    return tuple(out)


@deal.ensure(lambda ids, nodes, result: len(result) <= len(ids)
             and all(a[0] in ids and a[0] != '' and a[2] >= 0 and a[6] >= 0.0 and a[8] >= 0.0 for a in result))
@deal.has()
def args(ids: tuple[str, ...], nodes: object) -> tuple[Fields, ...]:
    """Аргументы v6_entry (Fields: nid, file, position, text, question, answer, at, status, retracted_at, session,
    source) для атомов rows(ids, nodes), по порядку (столько же и в том же порядке, что rows): nid — id; из узла:
    file, question, answer, text, status, session_id, source — текстом (scalars.text); order — position
    (целое), timestamp — at и retracted_at — числами (scalars.number). Узел не объект — все его поля
    пусты.

    D: 3

    >>> args(('n1', 'n2'), {'n1': {'id': 'n1', 'file': 'docs/a.md', 'order': '0', 'question': '', 'answer': 'Векторы храним в SQLite.', 'text': 'Векторы храним в SQLite.', 'timestamp': '1760000000.0', 'status': 'retracted', 'session_id': '', 'source': 'file'}, 'n2': {'id': 'n2', 'file': 'docs/a.md', 'order': '0', 'question': '', 'answer': 'Векторы храним в LanceDB.', 'text': 'Векторы храним в LanceDB.', 'timestamp': '1760003600.0', 'status': 'active', 'session_id': '', 'source': 'file'}})
    (('n1', 'docs/a.md', 0, 'Векторы храним в SQLite.', '', 'Векторы храним в SQLite.', 1760000000.0, 'retracted', 0.0, '', 'file'), ('n2', 'docs/a.md', 0, 'Векторы храним в LanceDB.', '', 'Векторы храним в LanceDB.', 1760003600.0, 'active', 0.0, '', 'file'))
    >>> args(('n3',), {'n3': {'file': 'None', 'text': 'x', 'order': 'None', 'timestamp': None, 'status': 'active'}})
    (('n3', '', 0, 'x', '', '', 0.0, 'active', 0.0, '', ''),)
    >>> args(('n1',), 'мусор')
    ()
    """
    if not isinstance(nodes, dict):
        return ()
    out: list[Fields] = []
    for r in rows(ids, nodes):
        nid = ids[r]
        node = nodes[nid]
        if not isinstance(node, dict):
            node = {}
        out.append((
            nid,
            scalars.text(node.get('file')),
            int(scalars.number(node.get('order'))),
            scalars.text(node.get('text')),
            scalars.text(node.get('question')),
            scalars.text(node.get('answer')),
            scalars.number(node.get('timestamp')),
            scalars.text(node.get('status')),
            scalars.number(node.get('retracted_at')),
            scalars.text(node.get('session_id')),
            scalars.text(node.get('source')),
        ))
    return tuple(out)


@deal.pre(lambda nid, file, position, text, question, answer, at, status, retracted_at, session, source:
          nid != '' and position >= 0 and at >= 0.0 and retracted_at >= 0.0)
@deal.ensure(lambda nid, file, position, text, question, answer, at, status, retracted_at, session, source, result:
             result.id == nid and result.status in ('active', 'retracted')
             and (result.status == 'retracted') == (status == 'retracted')
             and result.session != 'None' and result.source != '')
@deal.has()
def build(nid: str, file: str, position: int, text: str, question: str, answer: str, at: float, status: str,
          retracted_at: float, session: str, source: str) -> Entry:
    """Атом формата 6: id как был, отпечаток memshell.digest(text); состояние 'retracted', если status == 'retracted'
    (retracted_at — данное, а если 0.0 — at), иначе 'active' и retracted_at 0.0; сеанс 'None' — ''; source '' — 'v6'.

    D: 4

    >>> build('n1', 'docs/a.md', 0, 'Векторы храним в SQLite.', '', 'Векторы храним в SQLite.', 1760000000.0, 'retracted', 0.0, 'None', '')
    Entry(id='n1', file='docs/a.md', position=0, digest='0470878cd8043cf6', text='Векторы храним в SQLite.', question='', answer='Векторы храним в SQLite.', at=1760000000.0, status='retracted', retracted_at=1760000000.0, session='', source='v6')
    >>> build('n2', 'docs/a.md', 0, 'Векторы храним в LanceDB.', '', 'Векторы храним в LanceDB.', 1760003600.0, 'active', 0.0, '', 'file')
    Entry(id='n2', file='docs/a.md', position=0, digest='1d91240b07a4dac5', text='Векторы храним в LanceDB.', question='', answer='Векторы храним в LanceDB.', at=1760003600.0, status='active', retracted_at=0.0, session='', source='file')
    >>> build('n1', 'docs/a.md', 0, 'Векторы храним в SQLite.', '', 'Векторы храним в SQLite.', 1760000000.0, 'retracted', 0.0, '', 'file')
    Entry(id='n1', file='docs/a.md', position=0, digest='0470878cd8043cf6', text='Векторы храним в SQLite.', question='', answer='Векторы храним в SQLite.', at=1760000000.0, status='retracted', retracted_at=1760000000.0, session='', source='file')
    >>> build('n3', 'b.md', 2, 'Векторы храним в SQLite.', 'Чем?', 'SQLite.', 5.0, 'retracted', 9.0, 'abc', 'claude')
    Entry(id='n3', file='b.md', position=2, digest='0470878cd8043cf6', text='Векторы храним в SQLite.', question='Чем?', answer='SQLite.', at=5.0, status='retracted', retracted_at=9.0, session='abc', source='claude')
    >>> build('n4', 'n.md', 1, 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 5.0, 'None', '')
    Entry(id='n4', file='n.md', position=1, digest='0b00d70ea64ce933', text='LanceDB выбран.', question='', answer='LanceDB выбран.', at=1760000000.0, status='active', retracted_at=0.0, session='', source='v6')
    """
    digest = memshell.digest(text)
    if status == 'retracted':
        state = 'retracted'
        gone = retracted_at if retracted_at != 0.0 else at
    else:
        state = 'active'
        gone = 0.0
    return Entry(
        id=nid,
        file=file,
        position=position,
        digest=digest,
        text=text,
        question=question,
        answer=answer,
        at=at,
        status=state,
        retracted_at=gone,
        session='' if session == 'None' else session,
        source='v6' if source == '' else source,
    )


@deal.ensure(lambda store_dir, entries, result: result.file == store_dir and result.added == len(entries)
             and result.added >= result.retracted >= 0 and result.kept == 0)
@deal.has()
def tally(store_dir: str, entries: tuple[Entry, ...]) -> Report:
    """Итог переноса: Report(store_dir, сколько атомов, из них снятых (status == 'retracted'), 0).

    D: 1

    >>> tally('tests/fixtures/store_v6', (Entry('n1', 'docs/a.md', 0, '0470878cd8043cf6', 'Векторы храним в SQLite.', '', 'Векторы храним в SQLite.', 1760000000.0, 'retracted', 1760000000.0, '', 'file'), Entry('n2', 'docs/a.md', 0, '1d91240b07a4dac5', 'Векторы храним в LanceDB.', '', 'Векторы храним в LanceDB.', 1760003600.0, 'active', 0.0, '', 'file')))
    Report(file='tests/fixtures/store_v6', added=2, retracted=1, kept=0)
    >>> tally('нет-такой-базы', ())
    Report(file='нет-такой-базы', added=0, retracted=0, kept=0)
    """
    return Report(
        file=store_dir,
        added=len(entries),
        retracted=sum(1 for e in entries if e.status == 'retracted'),
        kept=0,
    )
