"""Запросы к codebase-memory-mcp: аргументы вызова (JSON), тексты запросов графа и имя проекта из ответа.

node: leaf
"""
import json

import deal


@deal.ensure(lambda repo, result: json.loads(result) == {'repo_path': repo})
@deal.has()
def index_args(repo: str) -> str:
    """ФТ-006. Аргументы index_repository: json.dumps({"repo_path": repo}, ensure_ascii=False).

    D: 1

    >>> index_args('.')
    '{"repo_path": "."}'
    >>> index_args('H:/работа/app')
    '{"repo_path": "H:/работа/app"}'
    """
    return json.dumps({"repo_path": repo}, ensure_ascii=False)


@deal.ensure(lambda project, query, result: json.loads(result) == {'project': project, 'query': query})
@deal.has()
def query_args(project: str, query: str) -> str:
    """ФТ-006. Аргументы query_graph: json.dumps({"project": project, "query": query}, ensure_ascii=False).

    D: 1

    >>> query_args('P', 'MATCH (n) RETURN n')
    '{"project": "P", "query": "MATCH (n) RETURN n"}'
    >>> query_args('H-работа-app', 'MATCH (n:Class) RETURN n.name')
    '{"project": "H-работа-app", "query": "MATCH (n:Class) RETURN n.name"}'
    """
    return json.dumps({"project": project, "query": query}, ensure_ascii=False)


@deal.pre(lambda label, skip, limit: skip >= 0 and limit >= 1)
@deal.ensure(lambda label, skip, limit, result: result == f'MATCH (n:{label}) RETURN n.qualified_name, n.name, n.file_path, n.start_line, n.end_line SKIP {skip} LIMIT {limit}')
@deal.has()
def node_query(label: str, skip: int, limit: int) -> str:
    """ФТ-006. Запрос страницы узлов вида label: `MATCH (n:<label>) RETURN n.qualified_name, n.name, n.file_path,
    n.start_line, n.end_line SKIP <skip> LIMIT <limit>` (одной строкой, через один пробел).

    D: 1

    >>> node_query('Function', 0, 200)
    'MATCH (n:Function) RETURN n.qualified_name, n.name, n.file_path, n.start_line, n.end_line SKIP 0 LIMIT 200'
    >>> node_query('Class', 400, 200)
    'MATCH (n:Class) RETURN n.qualified_name, n.name, n.file_path, n.start_line, n.end_line SKIP 400 LIMIT 200'
    """
    return (
        f"MATCH (n:{label}) RETURN n.qualified_name, n.name, n.file_path, "
        f"n.start_line, n.end_line SKIP {skip} LIMIT {limit}"
    )


@deal.pre(lambda kind, skip, limit: skip >= 0 and limit >= 1)
@deal.ensure(lambda kind, skip, limit, result: result == f'MATCH (a)-[r:{kind}]->(b) RETURN a.qualified_name, b.qualified_name SKIP {skip} LIMIT {limit}')
@deal.has()
def edge_query(kind: str, skip: int, limit: int) -> str:
    """ФТ-006. Запрос страницы рёбер вида kind: `MATCH (a)-[r:<kind>]->(b) RETURN a.qualified_name, b.qualified_name
    SKIP <skip> LIMIT <limit>` (одной строкой, через один пробел).

    D: 1

    >>> edge_query('CALLS', 0, 200)
    'MATCH (a)-[r:CALLS]->(b) RETURN a.qualified_name, b.qualified_name SKIP 0 LIMIT 200'
    >>> edge_query('USAGE', 200, 200)
    'MATCH (a)-[r:USAGE]->(b) RETURN a.qualified_name, b.qualified_name SKIP 200 LIMIT 200'
    """
    return (
        f"MATCH (a)-[r:{kind}]->(b) RETURN a.qualified_name, b.qualified_name "
        f"SKIP {skip} LIMIT {limit}"
    )


@deal.ensure(lambda text, result: len(result) <= len(text) and (result == '' or '{' in text))
@deal.has()
def project_name(text: str) -> str:
    """ФТ-006. Имя проекта из ответа index_repository: поле "project" JSON-объекта — текста от первой «{» до
    последней «}». Нет объекта, он не разбирается, поля нет или оно не строка — ''.

    D: 5

    >>> project_name('{"project": "H-work-app", "status": "indexed"}')
    'H-work-app'
    >>> project_name('индексирую\\n{\\n  "project": "P",\\n  "nodes": 3\\n}\\n')
    'P'
    >>> project_name('{"status": "error"}')
    ''
    >>> project_name('нет JSON')
    ''
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        return ""
    try:
        obj = json.loads(text[start : end + 1])
    except (ValueError, RecursionError):
        return ""
    if not isinstance(obj, dict):
        return ""
    value = obj.get("project")
    if not isinstance(value, str):
        return ""
    return value
