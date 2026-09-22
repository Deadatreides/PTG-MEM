# -*- coding: utf-8 -*-
r"""Layer 1 extractor для C/C++: Source -> tree-sitter -> объективные факты.

Контракт тот же, что у `extractor.py`: чистая функция
`(file_path, source_text) -> ExtractionResult`, те же `Card`, та же
вложенность, те же типы рёбер. Отличается только источник фактов, и это
записано в провенансе (`source="tree_sitter_cpp"`), а не подразумевается.

ПОЧЕМУ TREE-SITTER, А НЕ LIBCLANG. libclang разрешает перегрузки, шаблоны и
макросы по-настоящему, и `compile_commands.json` у форка есть (сборка на
CMake+ninja). Но он требует от ПОЛЬЗОВАТЕЛЯ установленного LLVM подходящей
версии, а это несовместимо с распространением: индексатор, для пробы
которого надо ставить toolchain, никто не поставит. tree-sitter приезжает
колесом с pip, работает офлайн и без сборки.

ЧЕСТНАЯ ГРАНИЦА. Из этого следует, что рёбра `calls` здесь — **по имени**, а
не по разрешённой перегрузке: `foo(int)` и `foo(double)` неразличимы, вызовы
через указатель и через макрос не видны, шаблоны не инстанцируются. Это
сказано прямо, потому что ложное ребро в графе хуже отсутствующего: обход по
нему уводит, и по выдаче этого не видно.

КОММЕНТАРИЙ КАК ДОКСТРИНГ. В Python смысловой слой берётся из докстринга. В
C++ его роль играет блок комментариев НАД объявлением, и здесь он
извлекается именно так. Это не украшение: замер 21.09.2026 показал, что
осмысленность векторов держится на тексте карточки — где докстринг есть,
соседи верные, где нет, они вырождаются в похожесть имён.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cards import (Card, Identity, Provenance, content_hash_of,
                    signature_hash_of, make_card_id)
from .extractor import ExtractionResult, _stub_layer_2

CPP_EXTS = (".c", ".cc", ".cpp", ".cxx", ".c++", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh")

_DECL_WRAPPERS = {"pointer_declarator", "reference_declarator",
                  "parenthesized_declarator", "array_declarator",
                  "init_declarator"}
_NAME_TYPES = {"identifier", "field_identifier", "qualified_identifier",
               "destructor_name", "operator_name", "type_identifier"}

_parser = None


def _get_parser():
    global _parser
    if _parser is None:
        import tree_sitter_cpp
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_cpp.language()))
    return _parser


def available() -> bool:
    try:
        _get_parser()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
def _txt(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _span(node) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _fn_declarator(node):
    """Спуск к function_declarator сквозь обёртки указателя/ссылки/массива."""
    d = node.child_by_field_name("declarator")
    seen = 0
    while d is not None and d.type != "function_declarator" and seen < 8:
        if d.type not in _DECL_WRAPPERS:
            return None
        d = d.child_by_field_name("declarator")
        seen += 1
    return d if (d is not None and d.type == "function_declarator") else None


def _decl_name(fdecl, src: bytes) -> str | None:
    d = fdecl.child_by_field_name("declarator")
    while d is not None and d.type in _DECL_WRAPPERS:
        d = d.child_by_field_name("declarator")
    if d is None or d.type not in _NAME_TYPES:
        return None
    return _txt(d, src)


def _params(fdecl, src: bytes) -> list[str]:
    pl = fdecl.child_by_field_name("parameters")
    if pl is None:
        return []
    out = []
    for ch in pl.named_children:
        if ch.type in ("parameter_declaration", "optional_parameter_declaration",
                       "variadic_parameter_declaration"):
            out.append(" ".join(_txt(ch, src).split()))
    return out


def _leading_comment(node, src: bytes) -> str | None:
    """Блок комментариев прямо над объявлением — роль докстринга в C++."""
    parts = []
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        # разрыв в одну пустую строку ещё считается тем же блоком, больше — нет
        if parts and node.start_point[0] - prev.end_point[0] > 2:
            break
        parts.append(_txt(prev, src))
        node = prev
        prev = prev.prev_sibling
    if not parts:
        return None
    text = "\n".join(reversed(parts))
    cleaned = []
    for line in text.splitlines():
        s = line.strip()
        for pre in ("///", "//!", "//", "/**", "/*!", "/*", "*/", "*"):
            if s.startswith(pre):
                s = s[len(pre):].strip()
                break
        if s.endswith("*/"):
            s = s[:-2].strip()
        # строки-разделители («// -------», «==========») смысла не несут, но
        # попадают в эмбеддимый текст и сближают несвязанные карточки
        if s and len(s) >= 3 and not s.strip("-=*_~#+ "):
            continue
        cleaned.append(s)
    out = "\n".join(l for l in cleaned if l).strip()
    return out or None


def _calls_out(node, src: bytes, limit: int = 64) -> list[str]:
    """Имена вызываемых функций внутри тела. По имени — см. границу в шапке."""
    names, stack = [], [node]
    while stack and len(names) < limit:
        n = stack.pop()
        if n.type == "call_expression":
            f = n.child_by_field_name("function")
            if f is not None:
                if f.type == "field_expression":
                    fld = f.child_by_field_name("field")
                    nm = _txt(fld, src) if fld is not None else None
                elif f.type in _NAME_TYPES:
                    nm = _txt(f, src)
                else:
                    nm = None
                if nm and nm not in names:
                    names.append(nm)
        stack.extend(n.named_children)
    return sorted(names)


# ---------------------------------------------------------------------------
def extract_module_cpp(file_path: str, source_text: str) -> ExtractionResult:
    parser = _get_parser()
    src = source_text.encode("utf-8", "replace")
    tree = parser.parse(src)
    root = tree.root_node

    cards: list[Card] = []
    containment: list[tuple[str, str]] = []
    references: list[tuple[str, str, str]] = []

    module_qname = file_path
    module_id = make_card_id(file_path, module_qname,
                             version_salt=content_hash_of(source_text))
    module_children: list[str] = []
    includes: list[str] = []

    def prov(method: str, node) -> Provenance:
        return Provenance.objective(method=method, ref=f"{file_path}:{_span(node)}",
                                    source="tree_sitter_cpp")

    def add_function(node, owner_qname: str, owner_kind: str | None) -> str | None:
        fdecl = _fn_declarator(node)
        if fdecl is None:
            return None
        name = _decl_name(fdecl, src)
        if not name:
            return None
        qname = f"{owner_qname}::{name}" if owner_qname else name
        text = _txt(node, src)
        card_id = make_card_id(file_path, qname, version_salt=content_hash_of(text))
        params = _params(fdecl, src)
        body = node.child_by_field_name("body")
        calls = _calls_out(body, src) if body is not None else []
        subkind = "method" if owner_kind == "type" else (
            "function" if body is not None else "declaration")
        layer_1 = {
            "name": name,
            "params": params,
            "docstring": _leading_comment(node, src),
            "calls_out": calls,
            "loc": _span(node)[1] - _span(node)[0] + 1,
            "has_body": body is not None,
        }
        cards.append(Card(
            identity=Identity(
                card_id=card_id, kind="callable", subkind=subkind,
                qualified_name=qname, file_path=file_path, span=_span(node),
                content_hash=content_hash_of(text),
                signature_hash=signature_hash_of(f"{name}({', '.join(params)})"),
            ),
            layer_1=layer_1,
            layer_1_provenance=prov("tree_sitter_cpp.function_definition", node),
            layer_2=_stub_layer_2(),
        ))
        for callee in calls:
            references.append((card_id, callee, "calls"))
        return card_id

    def add_type(node, owner_qname: str) -> str | None:
        nm = node.child_by_field_name("name")
        if nm is None:
            return None
        name = _txt(nm, src)
        qname = f"{owner_qname}::{name}" if owner_qname else name
        text = _txt(node, src)
        card_id = make_card_id(file_path, qname, version_salt=content_hash_of(text))
        bases = []
        for ch in node.named_children:
            if ch.type == "base_class_clause":
                bases = [" ".join(_txt(b, src).split()) for b in ch.named_children]
        members: list[str] = []
        body = node.child_by_field_name("body")
        if body is not None:
            for ch in body.named_children:
                mid = None
                if ch.type == "function_definition":
                    mid = add_function(ch, qname, "type")
                elif ch.type in ("field_declaration", "declaration"):
                    if _fn_declarator(ch) is not None:
                        mid = add_function(ch, qname, "type")
                if mid:
                    members.append(mid)
                    containment.append((card_id, mid))
        cards.append(Card(
            identity=Identity(
                card_id=card_id, kind="type",
                subkind="struct" if node.type == "struct_specifier" else "class",
                qualified_name=qname, file_path=file_path, span=_span(node),
                content_hash=content_hash_of(text),
                signature_hash=signature_hash_of(f"{node.type} {name}({bases})"),
            ),
            layer_1={"name": name, "bases": bases,
                     "docstring": _leading_comment(node, src),
                     "methods": members,
                     "loc": _span(node)[1] - _span(node)[0] + 1},
            layer_1_provenance=prov(f"tree_sitter_cpp.{node.type}", node),
            layer_2=_stub_layer_2(),
        ))
        for b in bases:
            references.append((card_id, b.split()[-1], "implements"))
        return card_id

    def walk(node, owner_qname: str, top: bool):
        for ch in node.named_children:
            t = ch.type
            if t == "preproc_include":
                p = ch.child_by_field_name("path")
                if p is not None:
                    includes.append(_txt(p, src).strip('"<>'))
            elif t == "namespace_definition":
                nm = ch.child_by_field_name("name")
                inner = f"{owner_qname}::{_txt(nm, src)}" if nm is not None and owner_qname \
                    else (_txt(nm, src) if nm is not None else owner_qname)
                body = ch.child_by_field_name("body")
                if body is not None:
                    walk(body, inner, top)
            elif t in ("class_specifier", "struct_specifier"):
                cid = add_type(ch, owner_qname)
                if cid and top:
                    module_children.append(cid)
                    containment.append((module_id, cid))
            elif t == "function_definition":
                cid = add_function(ch, owner_qname, None)
                if cid and top:
                    module_children.append(cid)
                    containment.append((module_id, cid))
            elif t in ("declaration", "field_declaration"):
                if _fn_declarator(ch) is not None:
                    cid = add_function(ch, owner_qname, None)
                    if cid and top:
                        module_children.append(cid)
                        containment.append((module_id, cid))
            elif t in ("linkage_specification", "preproc_ifdef", "preproc_if",
                       "declaration_list", "translation_unit"):
                walk(ch, owner_qname, top)

    walk(root, "", True)

    cards.append(Card(
        identity=Identity(
            card_id=module_id, kind="container", subkind="module",
            qualified_name=module_qname, file_path=file_path,
            span=(1, max(1, source_text.count("\n") + 1)),
            content_hash=content_hash_of(source_text),
            signature_hash=signature_hash_of(f"module {file_path}"),
        ),
        layer_1={"includes": sorted(set(includes)), "children": module_children,
                 "docstring": _leading_comment(root.named_children[0], src)
                 if root.named_child_count else None,
                 "loc": source_text.count("\n") + 1},
        layer_1_provenance=prov("tree_sitter_cpp.translation_unit", root),
        layer_2=_stub_layer_2(),
    ))
    for inc in sorted(set(includes)):
        references.append((module_id, inc, "imports"))

    return ExtractionResult(cards=cards, containment_edges=containment,
                            reference_edges=references)
