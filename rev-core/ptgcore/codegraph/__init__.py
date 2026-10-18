"""Кодовый PTG и метадерево: граф кода выбранных репозиториев (ptgfull.code.code_graph — codebase-memory-mcp целиком),
рёбра метадерева по живым атомам (ptgfull.metatree: tree_edges, source_edges, stitch), швы атомов с кодом для
подсказки, соседи атома по метадереву для эго-сети. Хранилище — ptgfull.store (code_nodes, code_edges, edges).

Узлы метадерева: атом — id; символ — полное имя; файл кода — 'file:<репозиторий>/<путь>'; папка — 'dir:<путь>';
источник текста — 'src:<файл>'.

node: composite
effects: io
"""
import os
import re

import deal
import ptgfull.code
import ptgfull.metatree
import ptgfull.store
from ptgfull.types import Atom, CodeNode, Edge

from ptgcore.codegraph.marks import mark_text
from ptgcore.codegraph.rebuild import assemble, to_atoms
from ptgcore.codegraph.scan import collect
from ptgcore.codegraph.seams import seam_lines, touching
from ptgcore.types import Entry, GEdge

TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit", "Read")
SEAMS = ("edits", "reads", "mentions")


@deal.ensure(lambda text, result: result.count("\n") == text.count("\n"))
@deal.has()
def tool_marks(text: str) -> str:
    """ФТ-013. Строки инструментов сеанса — в разметку метадерева: каждая строка вида «[Имя] путь» или «Tools: [Имя]
    путь» с Имя из TOOLS становится «[инструмент Имя: путь]» (приставка «Tools: » сохраняется); прочее — как было.

    D: 1

    >>> tool_marks('Ответ.\\n\\nTools: [Edit] H:/proj/memcore/filtering.py\\n[Read] H:/proj/spec.md\\n[Bash] ls')
    'Ответ.\\n\\nTools: [инструмент Edit: H:/proj/memcore/filtering.py]\\n[инструмент Read: H:/proj/spec.md]\\n[Bash] ls'
    >>> tool_marks('просто текст')
    'просто текст'
    """
    return mark_text(text)


@deal.pre(lambda home, repos, binary: home != "" and len(set(repos)) == len(repos))
@deal.ensure(lambda home, repos, binary, result: result >= 0)
@deal.raises(FileNotFoundError, RuntimeError)
@deal.has("io")
def index(home: str, repos: tuple[str, ...], binary: str) -> int:
    """ФТ-012. Граф кода репозиториев: ptgfull.code.code_graph(binary, repo) по каждому; путь файла символа — с именем
    папки репозитория впереди (os.path.basename(os.path.normpath(repo)) + '/' + file); всё вместе —
    ptgfull.store.write_code; итог — символов. repos = () — граф кода не трогается, итог — сколько символов в базе.

    D: 3

    >>> index('нет-базы', (), 'нет-такой-программы.exe')
    0
    >>> index('cg_index', ('.',), 'нет-такой-программы.exe')
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такой-программы.exe
    """
    if not repos:
        return len(ptgfull.store.read_code(home)[0])
    nodes, edges = collect(binary, repos)
    return int(ptgfull.store.write_code(home, nodes, edges))


@deal.pre(lambda home, entries: home != "")
@deal.ensure(lambda home, entries, result: result >= 0)
@deal.has("io")
def relink(home: str, entries: tuple[Entry, ...]) -> int:
    """ФТ-013. Рёбра метадерева заново по живым атомам entries: ptgfull Atom(id, file, tool_marks(text), session,
    position); tree_edges(символы базы) + source_edges(атомы) + stitch(атомы, символы) + рёбра кода базы →
    ptgfull.store.write_edges; итог — рёбер.

    D: 2

    >>> relink('cg_relink', (Entry('a1', 'n.md', 0, 'd1', 'LanceDB выбран.', '', 'LanceDB выбран.', 1760000000.0, 'active', 0.0, '', 'file'),))
    1
    >>> relink('нет-базы-рёбер', ())
    0
    """
    nodes, links = ptgfull.store.read_code(home)
    return int(ptgfull.store.write_edges(home, assemble(to_atoms(entries), nodes, links)))


@deal.pre(lambda home, ids: home != "")
@deal.ensure(lambda home, ids, result: {i for i, _ in result} <= set(ids))
@deal.has("io")
def code_links(home: str, ids: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Швы атомов ids с кодом для подсказки, по порядку ids и рёбер: (id, '<вид> <цель>'), цель — 'file:<путь>' или
    для символа '<label> <qn> — <file>:<start>-<end>'.

    D: 3

    >>> code_links('нет-базы', ('a1',))
    ()
    """
    nodes, _ = ptgfull.store.read_code(home)
    return seam_lines(ids, ptgfull.store.read_edges(home), nodes)


@deal.pre(lambda home, atom, limit: home != "" and limit >= 0)
@deal.ensure(lambda home, atom, limit, result: len(result) <= limit and all(atom in (e.src, e.dst) for e in result))
@deal.has("io")
def around(home: str, atom: str, limit: int) -> tuple[GEdge, ...]:
    """Рёбра метадерева, касающиеся атома (швы с кодом, соседи next, источник contains), по порядку записи, не больше
    limit.

    D: 2

    >>> around('нет-базы', 'a1', 10)
    ()
    """
    return touching(ptgfull.store.read_edges(home), atom, limit)


@deal.pre(lambda home: home != "")
@deal.ensure(lambda home, result: min(result) >= 0)
@deal.has("io")
def counts(home: str) -> tuple[int, int]:
    """Символов кода и рёбер метадерева в базе.

    D: 1

    >>> counts('нет-базы')
    (0, 0)
    """
    nodes, _ = ptgfull.store.read_code(home)
    return (len(nodes), len(ptgfull.store.read_edges(home)))
