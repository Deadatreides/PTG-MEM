"""Кодовый PTG: граф кода из codebase-memory-mcp (программа целиком, запуск `cli`) и разбор его ответов.

Ответ `cli --quiet query_graph` — текст. Пример:

    rows_refs: 1  (cols: id prefix)
      0 P.memcore.
    rows_ref_rule: @N+suffix=prefix+suffix
    rows: 2  (cols: a.qualified_name a.file_path a.start_line)
      @0+filtering.admit memcore/filtering.py "50"
      builtins.len <python-builtins> "1"
    returned: 2

Строка таблицы — после «rows: …» и с отступом в два пробела; значения разделены пробелами, кавычки — как в
shlex.split; `@N+хвост` = префикс N из блока rows_refs + хвост. Ответ `index_repository` — JSON с полем "project".
Запрос узлов: `MATCH (n:<вид>) RETURN n.qualified_name, n.name, n.file_path, n.start_line, n.end_line SKIP <s> LIMIT 200`,
рёбер: `MATCH (a)-[r:<вид>]->(b) RETURN a.qualified_name, b.qualified_name SKIP <s> LIMIT 200`; страница — 200 строк,
следующая — пока строк ровно 200.

node: composite
effects: io
"""
import deal

from ptgfull.code.process import command, execute
from ptgfull.code.queries import edge_query, index_args, node_query, project_name, query_args
from ptgfull.code.refs import expand, ref_map
from ptgfull.code.symbols import edge_of, inner_edges, node_of
from ptgfull.code.table import block, cells
from ptgfull.types import CodeNode, Edge

SYMBOLS = ("Function", "Method", "Class")
LINKS = ("CALLS", "USAGE")
PAGE = 200


@deal.ensure(lambda text, result: all(len(r) > 0 for r in result))
@deal.has()
def parse_rows(text: str) -> tuple[tuple[str, ...], ...]:
    """ФТ-006. Строки таблицы ответа codebase-memory-mcp с раскрытыми ссылками `@N+хвост`; нет таблицы — ().

    D: 6

    >>> parse_rows('rows_refs: 1  (cols: id prefix)\\n  0 P.memcore.\\nrows: 2  (cols: a b)\\n  @0+filtering.admit "50"\\n  builtins.len "1"\\nreturned: 2\\n')
    (('P.memcore.filtering.admit', '50'), ('builtins.len', '1'))
    >>> parse_rows('rows: 1  (cols: a b)\\n  "x y" z\\nreturned: 1\\n')
    (('x y', 'z'),)
    >>> parse_rows('total: 0\\nhint: "Query returned no results."\\n')
    ()
    """
    refs = ref_map(tuple(cells(line) for line in block(text, "rows_refs:")))
    table = (expand(cells(line), refs) for line in block(text, "rows:"))
    return tuple(row for row in table if row)


@deal.pre(lambda rows, label, prefix: label in SYMBOLS and prefix.endswith("."))
@deal.ensure(lambda rows, label, prefix, result: len(result) <= len(rows)
             and all(n.label == label and not n.file.startswith("<") for n in result))
@deal.has()
def code_nodes(rows: tuple[tuple[str, ...], ...], label: str, prefix: str) -> tuple[CodeNode, ...]:
    """ФТ-006. Символы из строк (полное имя, имя, файл, начало, конец): только с полным именем на prefix и файлом
    не в «<…>», префикс снят; строки другой длины или с нечисловыми строками кода пропускаются.

    D: 4

    >>> code_nodes((('P.memcore.filtering.admit', 'admit', 'memcore/filtering.py', '50', '74'), ('builtins.len', 'len', '<python-builtins>', '1', '1')), 'Function', 'P.')
    (CodeNode(qn='memcore.filtering.admit', name='admit', label='Function', file='memcore/filtering.py', start=50, end=74),)
    >>> code_nodes((('P.a.B', 'B', 'a.py', 'x', '9'),), 'Class', 'P.')
    ()
    """
    return tuple(n for row in rows for n in node_of(row, label, prefix))


@deal.pre(lambda rows, kind, prefix: kind in LINKS and prefix.endswith("."))
@deal.ensure(lambda rows, kind, prefix, result: len(result) <= len(rows) and all(e.kind == kind for e in result))
@deal.has()
def code_edges(rows: tuple[tuple[str, ...], ...], kind: str, prefix: str) -> tuple[Edge, ...]:
    """ФТ-006. Рёбра из строк (от, к): только где оба полных имени на prefix, префикс снят; строки другой длины
    пропускаются.

    D: 3

    >>> code_edges((('P.a.f', 'P.a.g'), ('P.a.f', 'builtins.len')), 'CALLS', 'P.')
    (Edge(src='a.f', dst='a.g', kind='CALLS'),)
    >>> code_edges((('P.a.f',),), 'USAGE', 'P.')
    ()
    """
    return tuple(e for row in rows for e in edge_of(row, kind, prefix))


@deal.ensure(lambda binary, tool, args, result: isinstance(result, str))
@deal.raises(FileNotFoundError, RuntimeError)
@deal.has("io")
def run_cbm(binary: str, tool: str, args: str) -> str:
    """Запуск `<binary> cli --quiet <tool> <args>`: вывод в UTF-8. Нет программы — FileNotFoundError(binary);
    код выхода не 0 — RuntimeError с текстом ошибки.

    D: 2

    >>> run_cbm('нет-такой-программы.exe', 'list_projects', '{}')
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такой-программы.exe
    """
    code, out, err = execute(command(binary, tool, args))
    if code != 0:
        raise RuntimeError(err.strip() or out.strip() or f"код выхода {code}")
    return out


@deal.ensure(lambda binary, repo, result: {e.src for e in result[1]} | {e.dst for e in result[1]} <= {n.qn for n in result[0]})
@deal.raises(FileNotFoundError, RuntimeError)
@deal.has("io")
def code_graph(binary: str, repo: str) -> tuple[tuple[CodeNode, ...], tuple[Edge, ...]]:
    """ФТ-006. Граф кода репозитория: index_repository → имя проекта; символы видов SYMBOLS и рёбра видов LINKS
    постранично; рёбра — только между найденными символами.

    D: 6

    >>> code_graph('нет-такой-программы.exe', '.')
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такой-программы.exe
    """
    out = run_cbm(binary, "index_repository", index_args(repo))
    project = project_name(out)
    if not project:
        raise RuntimeError(f"index_repository не вернул имя проекта: {out.strip()}")
    prefix = project + "."
    nodes: list[CodeNode] = []
    for label in SYMBOLS:
        skip = 0
        while True:
            rows = parse_rows(run_cbm(binary, "query_graph", query_args(project, node_query(label, skip, PAGE))))
            nodes.extend(code_nodes(rows, label, prefix))
            if len(rows) != PAGE:
                break
            skip += PAGE
    edges: list[Edge] = []
    for kind in LINKS:
        skip = 0
        while True:
            rows = parse_rows(run_cbm(binary, "query_graph", query_args(project, edge_query(kind, skip, PAGE))))
            edges.extend(code_edges(rows, kind, prefix))
            if len(rows) != PAGE:
                break
            skip += PAGE
    return tuple(nodes), inner_edges(tuple(edges), tuple(nodes))
