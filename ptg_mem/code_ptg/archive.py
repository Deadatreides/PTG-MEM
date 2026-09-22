"""
archive.py — оркестрация Code PTG: Source -> Card -> node в РЕАЛЬНОМ
Archive-графе (через graph_engine.GraphEngine).

Инкрементальность реализована НЕ через выдуманный DecayState (Clean/Dirty/
DirtyDerived — этого нет в реальном PTG), а через настоящие механизмы,
которые уже есть в ptg_core.py:
  - "supersedes" + node.status="superseded"  — контракт карточки изменился
    (signature_hash отличается) — старая версия НЕ удаляется, помечается.
  - "continues"                                — тело карточки изменилось,
    контракт — нет (это ровно то же решение continues/branches, что PTG
    принимает для чат-атомов, только здесь причина continues — явная
    (тот же qualified_name), а не эмбеддинг-порог).
  - "removed_from_source" (доп. поле, добавлено по аналогии с тем, как
    сами патчи PTG добавляли новые поля узла — placement_case,
    lateral_expansion и т.д. — схема узла в PTG аддитивна по построению)
    — карточка исчезла из исходника; узел не удаляется (append-only).
"""

from __future__ import annotations

import os
import time
from typing import Optional

from .extractor import extract_module
from .extractor_cpp import CPP_EXTS
from .graph_engine import GraphEngine, CODE_DOMAIN

STRUCTURAL_EDGE_TYPES = {"contains", "calls", "implements", "tests", "imports"}


class CodeArchive:
    def __init__(self, project_folder: str, store_dir: Optional[str] = None,
                 use_embeddings: bool = True, embedder=None):
        self.project_folder = project_folder
        self.engine = GraphEngine(project_folder, output_dir=store_dir,
                                  use_embeddings=use_embeddings, embedder=embedder)
        self._unresolved: list[tuple[str, str, str]] = []  # (src, dst_name, edge_type)

    # ---------- индексация ----------

    def _embed_text(self, card) -> str:
        """Текст, который пойдёт в эмбеддер.

        У модуля `qualified_name` — абсолютный путь, и его постоянный префикс
        (корень проекта) сидит в тексте КАЖДОГО модуля. В векторе это общая
        составляющая, которая тянет все модульные карточки друг к другу:
        замер 20.09.2026 на 600 карточках дал медианный косинус 0.508 при
        максимуме 0.712, и в верхушке выдачи стояли модули, совпавшие по
        словам из пути, а не по смыслу. Убираем корень — остаётся то, что
        различает.
        """
        text = card.interpretation_text()
        root = self.project_folder
        if root:
            text = text.replace(root + os.sep, "").replace(root, "")
        return text

    def _extract(self, file_path: str, source: str):
        """Выбор экстрактора по расширению. Контракт у обоих один:
        (путь, текст) -> ExtractionResult."""
        if file_path.lower().endswith(CPP_EXTS):
            from .extractor_cpp import extract_module_cpp
            return extract_module_cpp(file_path, source)
        return extract_module(file_path, source)

    def index_file(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        result = self._extract(file_path, source)

        for card in result.cards:
            node = self.engine.graft_or_isolate(
                atom_id=card.identity.card_id,
                text=self._embed_text(card),
                file_path=card.identity.file_path,
            )
            node.update(card.to_node_payload())

        for parent, child in result.containment_edges:
            self.engine.add_structural_edge(parent, child, "contains")
        for src, dst_name, edge_type in result.reference_edges:
            self._unresolved.append((src, dst_name, edge_type))

    #: Каталоги, которые нельзя индексировать по умолчанию.
    #: Это не гигиена, а условие работоспособности: в типичном репозитории
    #: чужого кода (зависимости, сборка, кэши) на порядок больше, чем своего,
    #: и без отсева граф раздувается чужим. А цена обхода — ln n: лишние узлы
    #: поднимают и требуемый бюджет связей, и число прыжков до ответа.
    SKIP_DIRS = frozenset({
        ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
        ".tox", ".venv", "venv", "env", "site-packages", "node_modules",
        "dist", "build", ".eggs", ".idea", ".vscode",
    })

    def index_directory(self, root: str, skip_dirs: Optional[set] = None) -> dict:
        """Обход дерева. Сбой на одном файле НЕ обрывает проход.

        Раньше единственный неразбираемый `.py` ронял всю индексацию
        исключением из `ast.parse`. В чужом репозитории это не редкость, а
        норма: обрывки, шаблоны с подстановками, код на втором питоне, файл
        с текстом вместо кода. Индексатор, который в таком репозитории не
        даёт НИЧЕГО, бесполезен — поэтому файл с ошибкой пропускается,
        запоминается с причиной и называется в отчёте.

        Тот же приём стоит в ptg_core (ПАТЧ 21) и ровно по той же причине.
        """
        skip = set(self.SKIP_DIRS) | set(skip_dirs or ())
        ok, failed = 0, []
        for dirpath, dirnames, filenames in os.walk(root):
            # Каталог сборки CMake опознаём по содержимому, а не по имени:
            # их зовут build, build-fast, build-cuda, cmake-build-debug — списком
            # имён не угадать. Внутри лежит сгенерированный C++ (пробники
            # компилятора, moc, protobuf), который в граф кода проекта не
            # входит и только раздувает n.
            if "CMakeCache.txt" in filenames:
                dirnames[:] = []
                continue
            # правка на месте — os.walk после этого в отсечённые не заходит
            dirnames[:] = [d for d in dirnames if d not in skip and d != "CMakeFiles"]
            for fn in filenames:
                if not fn.lower().endswith((".py",) + CPP_EXTS):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    self.index_file(path)
                    ok += 1
                except (SyntaxError, UnicodeDecodeError, ValueError, OSError) as ex:
                    failed.append((path, f"{type(ex).__name__}: {ex}"))
        self._resolve_refs()
        if failed:
            self.engine.archive.log(
                "пропущено файлов: %d из %d (разбор не удался)" % (len(failed), ok + len(failed)))
        return {"проиндексировано": ok, "пропущено": len(failed),
                "причины": failed[:20]}

    def _resolve_refs(self):
        still_unresolved = []
        nodes = self.engine.archive.nodes
        for src, dst_name, edge_type in self._unresolved:
            src_file = (nodes.get(src) or {}).get("file_path")
            match = self._find_active_by_name(dst_name, src_file)
            if match:
                self.engine.add_structural_edge(src, match, edge_type)
            else:
                still_unresolved.append((src, dst_name, edge_type))
        self._unresolved = still_unresolved

    @staticmethod
    def _lang_of(path: str) -> str:
        return "cpp" if str(path).lower().endswith(CPP_EXTS) else "py"

    def _find_active_by_name(self, name: str, src_file: Optional[str] = None) -> Optional[str]:
        """Разрешение имени вызова в карточку.

        ДВА ПРАВИЛА, И ОБА ОПЛАЧЕНЫ ЛОЖНЫМИ РЁБРАМИ.

        1. НЕ ЧЕРЕЗ ЯЗЫКОВУЮ ГРАНИЦУ. Разделитель вложенности у Python «.»,
           у C++ «::». Когда суффикс `::` добавили, чтобы заработали вызовы
           внутри C++, питоновский `list.append()` начал разрешаться в
           `llm_tokenizer_bpe_session::append` — 125 ложных рёбер на одном
           имени, плюс 27 у `load` и 8 у `insert` (замер 21.09.2026 обходом
           графа). Файл на `.py` не может вызывать функцию из `.cpp`.

        2. БЛИЖНИЙ ПЕРЕВЕШИВАЕТ. Имена вроде `run`, `save`, `load` есть в
           десятке файлов; брать первый попавшийся — значит бросить ребро
           наугад. Порядок: точное совпадение полного имени -> тот же файл ->
           тот же каталог -> где угодно.

        Ложное ребро хуже отсутствующего: обход по нему уводит, и по выдаче
        этого не видно.
        """
        nodes = self.engine.archive.nodes
        want_lang = self._lang_of(src_file) if src_file else None
        sep = "::" if want_lang == "cpp" else "."
        src_dir = os.path.dirname(src_file) if src_file else None
        local_only = name in self.COMMON_METHOD_NAMES

        same_file = exact = same_dir = anywhere = None
        for nid, node in nodes.items():
            if node.get("domain") != CODE_DOMAIN or node.get("status") != "active":
                continue
            if node.get("removed_from_source"):
                continue
            fp = node.get("file_path")
            if want_lang and self._lang_of(fp) != want_lang:
                continue
            qn = node.get("qualified_name", "")
            if qn != name and not qn.endswith(sep + name):
                continue
            if src_file and fp == src_file:
                same_file = same_file or nid
                break                       # ближе уже не бывает
            if qn == name:
                exact = exact or nid
            elif src_dir and os.path.dirname(str(fp)) == src_dir:
                same_dir = same_dir or nid
            else:
                anywhere = anywhere or nid
        if same_file:
            return same_file
        if local_only:
            return None                     # см. COMMON_METHOD_NAMES
        return exact or same_dir or anywhere

    #: Имена, которые почти всегда принадлежат стандартной библиотеке, а не
    #: проекту: `lst.append(...)`, `d.get(...)`, `f.write(...)`. Разрешать их
    #: в чужой файл — значит бросать ребро наугад: в графе от этого возникли
    #: 125 связей «PTG вызывает токенизатор llama.cpp». Внутри СВОЕГО файла
    #: такое имя разрешается как обычно (там оно действительно может быть
    #: методом проекта), за его пределами — нет.
    COMMON_METHOD_NAMES = frozenset({
        "append", "extend", "insert", "pop", "get", "items", "keys", "values",
        "update", "add", "remove", "discard", "sort", "join", "split", "strip",
        "format", "write", "read", "readlines", "close", "seek", "flush",
        "encode", "decode", "lower", "upper", "replace", "startswith",
        "endswith", "copy", "clear", "count", "index", "setdefault", "load",
        "dump", "loads", "dumps", "open", "print", "len", "range", "str",
        "int", "float", "list", "dict", "set", "tuple", "sorted", "max",
        "min", "sum", "abs", "round", "zip", "map", "filter", "enumerate",
        "isinstance", "getattr", "setattr", "hasattr", "super", "size",
        "reshape", "astype", "mean", "std", "shape", "push_back", "emplace_back",
        "begin", "end", "clone", "reset", "data", "empty", "resize", "at",
    })

    def get_active_card_by_qname(self, qualified_name: str) -> Optional[dict]:
        candidates = [
            n for n in self.engine.archive.nodes.values()
            if n.get("domain") == CODE_DOMAIN and n.get("qualified_name") == qualified_name
            and n.get("status") == "active" and not n.get("removed_from_source")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.get("index", 0))

    # ---------- инкрементальное обновление ----------

    def reindex_file(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        result = self._extract(file_path, source)

        new_qnames = set()
        changed = []
        for card in result.cards:
            qn = card.identity.qualified_name
            new_qnames.add(qn)
            existing = self.get_active_card_by_qname(qn)

            node = self.engine.graft_or_isolate(
                atom_id=card.identity.card_id,
                text=self._embed_text(card),
                file_path=card.identity.file_path,
            )
            node.update(card.to_node_payload())

            if existing is None:
                changed.append((qn, "new", None))
                continue

            if existing["signature_hash"] != card.identity.signature_hash:
                self.engine.add_structural_edge(card.identity.card_id, existing["id"], "supersedes")
                self.engine.mark_superseded(existing["id"])
                changed.append((qn, "supersedes", existing["id"]))
            elif existing["content_hash"] != card.identity.content_hash:
                self.engine.add_structural_edge(card.identity.card_id, existing["id"], "continues")
                changed.append((qn, "continues", existing["id"]))
            else:
                # ничего не изменилось — новый узел уже создан graft_or_isolate,
                # но он избыточен: откатываем (append-only не значит "плодить
                # идентичные копии на каждый reindex без изменений").
                self._rollback_unchanged(card.identity.card_id, existing["id"])
                changed.append((qn, "unchanged", existing["id"]))
                continue

        # старые активные карточки этого файла, исчезнувшие из исходника
        removed = []
        for nid, node in list(self.engine.archive.nodes.items()):
            if (node.get("domain") == CODE_DOMAIN and node.get("file_path") == file_path
                    and node.get("status") == "active" and not node.get("removed_from_source")
                    and node.get("qualified_name") not in new_qnames):
                node["removed_from_source"] = True
                removed.append(node.get("qualified_name"))

        for parent, child in result.containment_edges:
            self.engine.add_structural_edge(parent, child, "contains")
        for src, dst_name, edge_type in result.reference_edges:
            self._unresolved.append((src, dst_name, edge_type))
        self._resolve_refs()

        return {"changed": changed, "removed": removed}

    def _rollback_unchanged(self, new_id: str, keep_id: str):
        """Если ни content_hash, ни signature_hash не изменились — узел,
        созданный graft_or_isolate() для проверки, не нужен: удаляем его
        (единственное место, где мы физически убираем узел — это ДО того,
        как он стал частью истории, чисто техническая отмена holостого хода,
        а не нарушение append-only для реального знания)."""
        arc = self.engine.archive
        arc.nodes.pop(new_id, None)
        if new_id in arc.id_order:
            arc.id_order.remove(new_id)
        arc.branches.pop(new_id, None)
        arc.branch_states.pop(new_id, None)
        arc.branch_bodies.pop(new_id, None)
        arc.edges = [e for e in arc.edges if e["from"] != new_id and e["to"] != new_id]

    # ---------- производные операции ----------

    def get_dependencies(self, card_id: str) -> list[dict]:
        return [e for e in self.engine.outgoing(card_id) if e["type"] in STRUCTURAL_EDGE_TYPES]

    def get_dependents(self, card_id: str) -> list[dict]:
        return [e for e in self.engine.incoming(card_id) if e["type"] in STRUCTURAL_EDGE_TYPES]

    def get_risk_impact(self, card_id: str, max_hops: int = 3) -> list[dict]:
        """Обратный обход ТОЛЬКО по структурным рёбрам (кто вызывает/содержит/
        реализует) — семантические рёбра графтинга (continues/reinforces/...)
        сюда сознательно не входят, это другой граф с другим смыслом."""
        impacted = []
        frontier = {card_id}
        visited = {card_id}
        hops = 0
        nodes = self.engine.archive.nodes
        while frontier and hops < max_hops:
            next_frontier = set()
            for cid in frontier:
                for e in self.engine.incoming(cid):
                    if e["type"] not in STRUCTURAL_EDGE_TYPES:
                        continue
                    src = e["from"]
                    if src in visited:
                        continue
                    n = nodes.get(src)
                    if n and n.get("status") == "active":
                        impacted.append({
                            "card_id": src,
                            "qualified_name": n.get("qualified_name", src),
                            "via_edge": e["type"],
                            "hops": hops + 1,
                        })
                        next_frontier.add(src)
                        visited.add(src)
            frontier = next_frontier
            hops += 1
        return impacted

    def get_lineage(self, card_id: str) -> list[dict]:
        """История версий карточки через continues/supersedes-рёбра —
        Code-эквивалент того, что PTG называет branch_lineage, но по
        конкретной карточке, а не по ветке эмбеддингов."""
        lineage = []
        current = card_id
        nodes = self.engine.archive.nodes
        seen = set()
        while current and current not in seen:
            seen.add(current)
            n = nodes.get(current)
            if n is None:
                break
            lineage.append({"card_id": current, "status": n.get("status"),
                             "content_hash": n.get("content_hash"), "signature_hash": n.get("signature_hash")})
            nxt = None
            for e in self.engine.outgoing(current):
                if e["type"] in ("supersedes", "continues"):
                    nxt = e["to"]
                    break
            current = nxt
        return lineage

    # ---------- сохранение/загрузка ----------

    def save(self):
        self.engine.save()

    def load(self) -> bool:
        return self.engine.load()
