"""
Layer 1 extractor: Source -> Python ast -> объективные факты.

MVP-эквивалент этапов "Tree-sitter -> AST -> CFG -> Static Facts" из
Knowledge Pipeline (addendum v1.1 §4). Здесь это один Python-модуль вместо
отдельных версионируемых стадий DAG — для одного языка и MVP-масштаба
разделение на отдельные переиспользуемые/заменяемые стадии не даёт
практической выгоды, но контракт (чистая функция source_text -> список
карточек) специально сохранён так, чтобы этот модуль было можно позже
разбить на настоящие стадии DAG без изменения вызывающего кода.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .cards import Card, Identity, Field, Provenance, content_hash_of, signature_hash_of, make_card_id


def _span(node: ast.AST) -> tuple[int, int]:
    end = getattr(node, "end_lineno", node.lineno)
    return (node.lineno, end)


def _node_text(source_lines: list[str], node: ast.AST) -> str:
    start, end = _span(node)
    return "\n".join(source_lines[start - 1:end])


def _signature_repr(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in node.args.args]
    decorators = [ast.unparse(d) for d in node.decorator_list]
    return f"{node.name}({','.join(args)})|deco={decorators}"


def _cyclomatic_complexity(node: ast.AST) -> int:
    """Наивная цикломатическая сложность: 1 + число ветвящихся конструкций."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                               ast.BoolOp, ast.ExceptHandler)):
            complexity += 1
    return complexity


def _calls_out(node: ast.AST) -> list[str]:
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            f = child.func
            if isinstance(f, ast.Name):
                names.append(f.id)
            elif isinstance(f, ast.Attribute):
                names.append(f.attr)
    return sorted(set(names))


def _exceptions_raised(node: ast.AST) -> list[str]:
    out = []
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            exc = child.exc
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                out.append(exc.func.id)
            elif isinstance(exc, ast.Name):
                out.append(exc.id)
    return sorted(set(out))


def _mutates_globals(node: ast.AST) -> list[str]:
    out = []
    for child in ast.walk(node):
        if isinstance(child, ast.Global):
            out.extend(child.names)
    return sorted(set(out))


def _docstring(node: ast.AST) -> str | None:
    return ast.get_docstring(node)


def _stub_layer_2() -> dict[str, Field]:
    """
    Все поля layer_2 по умолчанию Unknown — LLM-интерпретация в MVP не
    подключена (можно передать interpreter в pipeline.py, см. README).
    Это демонстрирует инвариант "нет evidence -> Unknown", а не имитирует его.
    """
    fields = ["purpose", "architectural_role", "concepts", "algorithms",
              "patterns", "invariants_claimed", "side_effects_summary"]
    return {name: Field.unknown(method=f"stub:{name}") for name in fields}


@dataclass
class ExtractionResult:
    cards: list[Card]
    containment_edges: list[tuple[str, str]]  # (parent_card_id, child_card_id)
    reference_edges: list[tuple[str, str, str]]  # (src_card_id, dst_name, edge_type) - dst_name резолвится позже


def extract_module(file_path: str, source_text: str) -> ExtractionResult:
    tree = ast.parse(source_text, filename=file_path)
    source_lines = source_text.splitlines()

    cards: list[Card] = []
    containment: list[tuple[str, str]] = []
    references: list[tuple[str, str, str]] = []

    module_qname = file_path
    module_id = make_card_id(file_path, module_qname, version_salt=content_hash_of(source_text))
    module_children: list[str] = []

    def handle_function(node, owner_qname: str, owner_kind_hint: str) -> str:
        qname = f"{owner_qname}.{node.name}" if owner_qname else node.name
        text = _node_text(source_lines, node)
        sig = _signature_repr(node)
        card_id = make_card_id(file_path, qname, version_salt=content_hash_of(text))

        layer_1 = {
            "name": node.name,
            "params": [a.arg for a in node.args.args],
            "decorators": [ast.unparse(d) for d in node.decorator_list],
            "docstring": _docstring(node),
            "cyclomatic_complexity": _cyclomatic_complexity(node),
            "calls_out": _calls_out(node),
            "exceptions_raised": _exceptions_raised(node),
            "mutates_globals": _mutates_globals(node),
            "loc": _span(node)[1] - _span(node)[0] + 1,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }
        subkind = "method" if owner_kind_hint == "type" else "function"
        card = Card(
            identity=Identity(
                card_id=card_id,
                kind="callable",
                subkind=subkind,
                qualified_name=qname,
                file_path=file_path,
                span=_span(node),
                content_hash=content_hash_of(text),
                signature_hash=signature_hash_of(sig),
            ),
            layer_1=layer_1,
            layer_1_provenance=Provenance.objective(
                method="python_ast.FunctionDef", ref=f"{file_path}:{_span(node)}"
            ),
            layer_2=_stub_layer_2(),
        )
        cards.append(card)
        for callee in layer_1["calls_out"]:
            references.append((card_id, callee, "calls"))
        return card_id

    def handle_class(node: ast.ClassDef, owner_qname: str) -> str:
        qname = f"{owner_qname}.{node.name}" if owner_qname else node.name
        text = _node_text(source_lines, node)
        card_id = make_card_id(file_path, qname, version_salt=content_hash_of(text))
        bases = [ast.unparse(b) for b in node.bases]
        sig = f"class {node.name}({bases})"

        method_ids = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                mid = handle_function(item, qname, owner_kind_hint="type")
                method_ids.append(mid)
                containment.append((card_id, mid))

        layer_1 = {
            "name": node.name,
            "bases": bases,
            "docstring": _docstring(node),
            "methods": method_ids,
            "loc": _span(node)[1] - _span(node)[0] + 1,
        }
        card = Card(
            identity=Identity(
                card_id=card_id,
                kind="type",
                subkind="class",
                qualified_name=qname,
                file_path=file_path,
                span=_span(node),
                content_hash=content_hash_of(text),
                signature_hash=signature_hash_of(sig),
            ),
            layer_1=layer_1,
            layer_1_provenance=Provenance.objective(
                method="python_ast.ClassDef", ref=f"{file_path}:{_span(node)}"
            ),
            layer_2=_stub_layer_2(),
        )
        cards.append(card)
        for base in bases:
            references.append((card_id, base, "implements"))
        return card_id

    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            else:
                mod = node.module or ""
                imports.extend(f"{mod}.{a.name}" for a in node.names)
        elif isinstance(node, ast.ClassDef):
            cid = handle_class(node, owner_qname="")
            containment.append((module_id, cid))
            module_children.append(cid)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cid = handle_function(node, owner_qname="", owner_kind_hint="module")
            containment.append((module_id, cid))
            module_children.append(cid)

    module_card = Card(
        identity=Identity(
            card_id=module_id,
            kind="container",
            subkind="module",
            qualified_name=module_qname,
            file_path=file_path,
            span=(1, len(source_lines) or 1),
            content_hash=content_hash_of(source_text),
            signature_hash=signature_hash_of(",".join(sorted(imports))),
        ),
        layer_1={
            "imports": sorted(set(imports)),
            "children": module_children,
            "loc": len(source_lines),
        },
        layer_1_provenance=Provenance.objective(method="python_ast.Module", ref=file_path),
        layer_2=_stub_layer_2(),
    )
    cards.append(module_card)
    for imp in imports:
        references.append((module_id, imp, "imports"))

    return ExtractionResult(cards=cards, containment_edges=containment, reference_edges=references)
