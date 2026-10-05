"""
gui_panels.py — виджеты CustomTkinter/ttk для PTG.

Иерархия дерева:
    📐 Проект
    └── 📂 Файл
        ├── 🔗 Связанные файлы
        └── 📁 Ветка [💤 dormant]
            └── 💬 Атом [path_coherence, status]
                └── ↩ Возврат

Превью атома включает:
    - Вопрос / Ответ раздельно
    - path_coherence, root_similarity
    - Состояние ветки: momentum, activation, entropy, dormant
    - Все типы рёбер: continues / returns_to / contradicts /
      fixes / refines / supersedes
"""

import time
import customtkinter as ctk
from tkinter import ttk


def _snip(text, n=60):
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n - 1] + "…"


def _short_path(path, max_len=70):
    """ПАТЧ 22 — фикс реального бага: раньше GUI показывал ГОЛОЕ ИМЯ файла
    (последний компонент пути), отбрасывая папку целиком. В проекте с
    множеством версионных подпапок (mycelium_v8.9, mycelium_v8.9.0.1, ...)
    одноимённые файлы (search.py, meta.txt) из РАЗНЫХ папок выглядели в
    дереве неотличимо друг от друга — визуально казалось, что архив плоский
    и видит только "корень", хотя данные собирались корректно на всех
    уровнях вложенности (проверено тестами).

    Первая версия фикса брала последние 2 компонента пути — оказалось
    недостаточно: в реальной структуре различающаяся папка
    (.../mycelium881/tools/search.py vs .../mycelium_v8.9/tools/search.py)
    лежит на 3 уровня выше самого файла, "tools/search.py" совпадает у
    обоих. Поэтому теперь показываем ВЕСЬ путь целиком (нормализуя
    разделители), обрезая с НАЧАЛА при превышении max_len — конец пути
    почти всегда самый информативный (сам файл и его ближайшая папка),
    так что обрезка спереди сохраняет то, что важнее для disambiguation."""
    norm = (path or "").replace("\\", "/").rstrip("/")
    if len(norm) <= max_len:
        return norm
    return "…" + norm[-(max_len - 1):]


_STATUS_ICON = {
    "active": "",
    "superseded": " ⚠ [вытеснен]",
    "deprecated": " ⚠ [устарел]",
}

_EDGE_ICON = {
    "continues": "→",
    "branches": "⑂",
    "returns_to": "↩",
    "contradicts": "✗",
    "fixes": "🔧",
    "refines": "⊕",
    "supersedes": "⇑",
}


# ---------------------------------------------------------------------------
# ПАТЧ 6 — глубокая иерархия TreePanel
# ---------------------------------------------------------------------------
class TreePanel(ctk.CTkFrame):
    """Структура дерева (ПАТЧ 6):

    📐 <Название проекта>
    ├── 📂 <Файл>
    │   └── 📁 <Ветка> (первый атом)
    │       ├── 💬 <Атом>  [статус]
    │       │   └── ↩ возврат к …
    │       └── 💬 <Атом>
    └── 📂 <Файл>
    """

    def __init__(self, master, on_select):
        super().__init__(master)
        self.on_select = on_select

        ctk.CTkLabel(self, text="Дерево мыслей",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=8, pady=(8, 0))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("PTG.Treeview", rowheight=22, font=("Segoe UI", 10))
        style.configure("PTG.Treeview.Heading", font=("Segoe UI", 10, "bold"))

        self.tree = ttk.Treeview(self, show="tree", style="PTG.Treeview")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        vsb.pack(side="right", fill="y", pady=8, padx=(0, 4))

        self.tree.bind("<<TreeviewSelect>>", self._on_click)
        self._node_iid_map = {}   # ttk item id -> ptg node_id

    def _on_click(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        node_id = self._node_iid_map.get(sel[0])
        if node_id:
            self.on_select(node_id)

    def refresh(self, archive):
        self.tree.delete(*self.tree.get_children())
        self._node_iid_map.clear()

        # ПАТЧ 14 — защита от None: main.py теперь может вызвать refresh(None),
        # когда в выбранной output-папке архива ещё нет (просто очищаем дерево).
        if archive is None or not archive.nodes:
            return

        # -- Корень проекта (ПАТЧ 3) ---
        proj_name = archive.root_project.get("project_name", "Проект")
        domains = archive.root_project.get("domains", [])
        dom_str = f" [{', '.join(domains[:4])}]" if domains else ""
        root_iid = self.tree.insert("", "end",
                                    text=f"📐 {proj_name}{dom_str}",
                                    open=True, tags=("root",))

        # -- Файлы как семантические объекты (ПАТЧ 4) ---
        # файлы с атомами в этом архиве
        file_to_nodes = {}
        for nid in archive.id_order:
            fpath = archive.nodes[nid]["file"]
            file_to_nodes.setdefault(fpath, []).append(nid)

        # возвратные цели — нужны для пометок на атомах
        returns_to_targets = {e["to"] for e in archive.edges if e["type"] == "returns_to"}

        # предвычислить по edge_type -> set(from_id) — чтобы не фильтровать весь список для каждого атома
        from_by_type = {}
        for e in archive.edges:
            from_by_type.setdefault(e["type"], {})[e["from"]] = e["to"]

        for fpath in sorted(file_to_nodes.keys()):
            fname = _short_path(fpath)
            fmeta = archive.files.get(fpath, {})
            concepts = fmeta.get("dominant_concepts", [])
            c_str = f" · {', '.join(concepts[:3])}" if concepts else ""
            count = len(file_to_nodes[fpath])
            file_iid = self.tree.insert(root_iid, "end",
                                        text=f"📂 {fname}  ({count} атомов{c_str})",
                                        open=False, tags=("file",))

            # ПАТЧ 1 — Connected Files ПЕРЕД ветками, отсортированные по убыванию sim
            # Строим словарь sim из file_edges один раз для этого файла
            _fe_sim = {}
            for _fe in getattr(archive, "file_edges", []):
                _s, _t = _fe["source_file"], _fe["target_file"]
                if _s == fpath:
                    _fe_sim[_t] = _fe["similarity"]
                elif _t == fpath:
                    _fe_sim[_s] = _fe["similarity"]
            connected = archive.files.get(fpath, {}).get("connected_files", [])
            # сортируем по убыванию похожести
            connected_sorted = sorted(connected, key=lambda p: _fe_sim.get(p, 0.0), reverse=True)
            if connected_sorted:
                cf_iid = self.tree.insert(file_iid, "end",
                                          text="🔗 Связанные файлы",
                                          open=False, tags=("cf_header",))
                for cf_path in connected_sorted:
                    cf_name = _short_path(cf_path)
                    sim_val = _fe_sim.get(cf_path)
                    sim_str = f" ({sim_val:.2f})" if sim_val is not None else ""
                    self.tree.insert(cf_iid, "end",
                                     text=f"📄 {cf_name}{sim_str}",
                                     tags=("cf_link",))

            # -- Ветки внутри файла ---
            file_branches = {}
            for nid in file_to_nodes[fpath]:
                bid = archive.nodes[nid].get("branch", nid)
                file_branches.setdefault(bid, []).append(nid)

            for bid, node_ids in sorted(file_branches.items(),
                                        key=lambda kv: archive.nodes[kv[1][0]]["index"]):
                root_node = archive.nodes.get(archive.branches[bid]["root"]) if bid in archive.branches else archive.nodes.get(node_ids[0])
                if root_node is None:
                    continue
                bs = getattr(archive, "branch_states", {}).get(bid, {})
                dormant_str = "  💤" if bs.get("dormant") else ""
                act = bs.get("activation", 1.0)
                act_str = f"  [{act:.2f}]" if bs else ""
                branch_label = (f"📁 {_snip(root_node.get('question') or root_node['text'], 38)}"
                                f"{dormant_str}{act_str}")
                branch_iid = self.tree.insert(file_iid, "end",
                                              text=branch_label,
                                              open=False, tags=("branch",))

                # -- Атомы в ветке, в порядке цепочки ---
                for nid in node_ids:
                    self._insert_atom(archive, nid, branch_iid,
                                      returns_to_targets, from_by_type)


        # раскрыть только корень — файлы и ветки закрыты по умолчанию
        self.tree.item(root_iid, open=True)

    def _insert_atom(self, archive, node_id, parent_iid, returns_to_targets, from_by_type):
        node = archive.nodes[node_id]
        status = node.get("status", "active")
        status_str = _STATUS_ICON.get(status, "")

        # первые слова вопроса как метка
        q_snip = _snip(node.get("question") or node["text"], 42)
        pc = node.get("path_coherence")
        pc_str = f"  ~{pc:.2f}" if pc is not None else ""
        label = f"💬 {q_snip}{pc_str}{status_str}"

        # пометки входящих/исходящих связей
        badges = []
        if node_id in returns_to_targets:
            badges.append("↩вх")
        if node_id in from_by_type.get("returns_to", {}):
            badges.append("↪ух")
        if node_id in from_by_type.get("contradicts", {}):
            badges.append("✗")
        if node_id in from_by_type.get("fixes", {}):
            badges.append("🔧")
        if node_id in from_by_type.get("supersedes", {}):
            badges.append("⇑")
        if badges:
            label += "  " + " ".join(badges)

        item_iid = self.tree.insert(parent_iid, "end", text=label, open=False,
                                    tags=(status, node_id))
        self._node_iid_map[item_iid] = node_id

        # дочерние возвраты — показываем явно вложенным уровнем
        for e in archive.edges:
            if e["type"] == "returns_to" and e["from"] == node_id:
                target = archive.nodes.get(e["to"])
                if target:
                    ret_iid = self.tree.insert(item_iid, "end",
                                               text=f"  ↩ возврат: {_snip(target['text'], 44)}",
                                               tags=(e["to"],))
                    self._node_iid_map[ret_iid] = e["to"]


# ---------------------------------------------------------------------------
# SearchPanel — без изменений по структуре, уточнены надписи
# ---------------------------------------------------------------------------
class SearchPanel(ctk.CTkFrame):
    def __init__(self, master, on_search, on_select):
        super().__init__(master)
        self.on_search = on_search
        self.on_select = on_select

        ctk.CTkLabel(self, text="Поиск по архиву",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=8, pady=(8, 0))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=4)
        self.entry = ctk.CTkEntry(row, placeholder_text="Запрос встроится через LM Studio…")
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _e: self._run())
        ctk.CTkButton(row, text="Найти", width=60, command=self._run).pack(side="left", padx=(6, 0))

        style = ttk.Style()
        style.configure("PTGSearch.Treeview", rowheight=22, font=("Segoe UI", 10))
        self.results = ttk.Treeview(self, show="tree", style="PTGSearch.Treeview")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.results.yview)
        self.results.configure(yscrollcommand=vsb.set)
        self.results.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 8))
        vsb.pack(side="right", fill="y", pady=(0, 8), padx=(0, 4))
        self.results.bind("<<TreeviewSelect>>", self._on_click)
        self._map = {}

    def _run(self):
        query = self.entry.get().strip()
        if query:
            self.on_search(query)

    def show_results(self, results, nodes):
        self.results.delete(*self.results.get_children())
        self._map.clear()
        for sim, node_id in results:
            node = nodes[node_id]
            fname = _short_path(node["file"])
            q_snip = _snip(node.get("question") or node["text"], 46)
            label = f"{sim:.3f}  📄 {fname}  {q_snip}"
            iid = self.results.insert("", "end", text=label, tags=(node_id,))
            self._map[iid] = node_id

    def _on_click(self, _event):
        sel = self.results.selection()
        if not sel:
            return
        node_id = self._map.get(sel[0])
        if node_id:
            self.on_select(node_id)


# ---------------------------------------------------------------------------
# PreviewPanel — показывает Q+A раздельно, расширенные рёбра (Патч 5)
# ---------------------------------------------------------------------------
class PreviewPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        ctk.CTkLabel(self, text="Просмотр атома",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=8, pady=(8, 0))
        self.box = ctk.CTkTextbox(self, wrap="word")
        self.box.pack(fill="both", expand=True, padx=8, pady=8)
        self.show_empty()

    def show_empty(self):
        self._write("Выберите атом в дереве или в результатах поиска.")

    def _write(self, full_text):
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        self.box.insert("end", full_text)
        self.box.configure(state="disabled")

    def show(self, preview):
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")

        n = preview["node"]
        status = preview.get("status", "active")

        # -- Заголовок ---
        self.box.insert("end", f"ФАЙЛ:  {n['file']}\n")
        pc   = n.get("path_coherence")
        rs   = n.get("root_similarity")
        pc_s = f"  связность пути: {pc:.3f}" if pc is not None else ""
        rs_s = f"  root_sim: {rs:.3f}" if rs is not None else ""
        self.box.insert("end", f"ПАРСЕР: {n.get('confidence','?')}   "
                                f"СТАТУС: {status}   "
                                f"ГЛУБИНА: {n.get('depth', 0)}\n")
        self.box.insert("end", f"off_root: {n.get('off_root','?')}"
                                f"{rs_s}{pc_s}\n\n")
        self.box.insert("end", "─" * 52 + "\n\n")

        # -- Q+A раздельно (ПАТЧ 1) ---
        q = n.get("question", "").strip()
        a = n.get("answer", "").strip()
        if q:
            self.box.insert("end", "[ ВОПРОС ]\n")
            self.box.insert("end", q + "\n\n")
        if a:
            self.box.insert("end", "[ ОТВЕТ ]\n")
            self.box.insert("end", a + "\n\n")
        if not q and not a:
            self.box.insert("end", n["text"] + "\n\n")

        self.box.insert("end", "─" * 52 + "\n\n")

        # -- Источник ---
        self.box.insert("end", "📐 ИСТОЧНИК (ORIGIN)\n")
        orig = preview["origin"]
        self.box.insert("end", f"  {orig['file']}\n  {_snip(orig['text'], 120)}\n\n")

        # -- Продолжение ---
        self.box.insert("end", "→ ПРОДОЛЖЕНИЕ (CONTINUATION)\n")
        if preview["continuation"]:
            for c in preview["continuation"]:
                self.box.insert("end", f"  → {_snip(c['text'], 90)}\n")
        else:
            self.box.insert("end", "  (пока нет)\n")
        self.box.insert("end", "\n")

        # -- Возвраты ---
        r = preview["returns"]
        self.box.insert("end", "↩ ВОЗВРАТЫ (RETURNS)\n")
        if r["out"]:
            for e in r["out"]:
                self.box.insert("end", f"  ↪ → {_snip(self._resolve_text(e), 90)}\n")
        if r["in"]:
            for e in r["in"]:
                self.box.insert("end", f"  ↩ ← {_snip(self._resolve_text(e), 90)}\n")
        if not r["out"] and not r["in"]:
            self.box.insert("end", "  (нет)\n")
        self.box.insert("end", "\n")

        # -- Расширенные рёбра (ПАТЧ 5) + reinforces (ПАТЧ 9/10) ---
        rel = preview.get("relations", {})
        for rtype, icon, label in [
            ("contradicts", "✗", "ПРОТИВОРЕЧИТ (CONTRADICTS)"),
            ("fixes",       "🔧", "ИСПРАВЛЯЕТ (FIXES)"),
            ("refines",     "⊕", "УТОЧНЯЕТ (REFINES)"),
            ("supersedes",  "⇑", "ВЫТЕСНЯЕТ (SUPERSEDES)"),
            ("reinforces",  "🔁", "РЕИНФОРСИТ (REINFORCES)"),
        ]:
            items = rel.get(rtype, [])
            self.box.insert("end", f"{icon} {label}\n")
            if items:
                for item in items:
                    self.box.insert("end", f"  {icon} {_snip(item.get('text',''), 90)}\n")
            else:
                self.box.insert("end", "  (нет)\n")
            self.box.insert("end", "\n")

        # -- Кем реинфорсирован (обратное направление, ПАТЧ 9/10) ---
        reinforced_by = preview.get("reinforced_by", [])
        self.box.insert("end", "🔁 РЕИНФОРСИРОВАН (REINFORCED BY)\n")
        if reinforced_by:
            for item in reinforced_by:
                self.box.insert("end", f"  🔁 ← {_snip(item.get('text',''), 90)}\n")
            epoch = n.get("semantic_epoch")
            self.box.insert("end", f"  (эпох повторения: {n.get('repetition_epochs', [])})\n")
        else:
            self.box.insert("end", "  (нет)\n")
        self.box.insert("end", "\n")

        # -- Незавершённое ---
        self.box.insert("end", "⚠ НЕЗАВЕРШЁННОЕ (UNRESOLVED)\n")
        self.box.insert("end",
            "  Открытая линия — продолжения нет.\n" if preview["is_unresolved"]
            else "  (завершено / продолжено)\n")
        self.box.insert("end", "\n")

        # -- Состояние ветки ---
        bs = preview.get("branch_state")
        bb = preview.get("branch_body")
        self.box.insert("end", "🌿 СОСТОЯНИЕ ВЕТКИ (BRANCH STATE)\n")
        if bs:
            dormant_s = "  💤 ДРЕМЛЮЩАЯ" if bs.get("dormant") else ""
            self.box.insert("end",
                f"  активация: {bs.get('activation',0):.3f}  "
                f"импульс: {bs.get('momentum',0):.2f}  "
                f"энтропия: {bs.get('entropy',0):.3f}\n")
            self.box.insert("end",
                f"  вес: {bs.get('weight',0):.3f}  "
                f"атомов: {bs.get('atom_count',0)}{dormant_s}\n")
        else:
            self.box.insert("end", "  (нет данных)\n")
        # ПАТЧ 7: тело ветки (centroid size, variance, true entropy)
        self.box.insert("end", "📐 ТЕЛО ВЕТКИ (BRANCH BODY)\n")
        if bb:
            self.box.insert("end",
                f"  атомов в теле: {bb.get('atom_count',0)}  "
                f"dim центроида: {bb.get('centroid_dim',0)}\n")
            self.box.insert("end",
                f"  вариация: {bb.get('variance',0):.4f}  "
                f"истинная энтропия: {bb.get('entropy',0):.4f}\n")
        else:
            self.box.insert("end", "  (тело ещё не построено)\n")

        self.box.configure(state="disabled")

    @staticmethod
    def _resolve_text(e):
        # e может быть словарём с ключом 'text' или без
        return e.get("text", "") if isinstance(e, dict) else str(e)


# ---------------------------------------------------------------------------
# ProgressLog — журнал хода построения
# ---------------------------------------------------------------------------
class ProgressLog(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        ctk.CTkLabel(self, text="Прогресс",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=8, pady=(8, 0))
        self.box = ctk.CTkTextbox(self, height=140, wrap="word")
        self.box.pack(fill="x", padx=8, pady=8)

    def write(self, msg):
        self.box.configure(state="normal")
        self.box.insert("end", msg + "\n")
        self.box.see("end")
        self.box.configure(state="disabled")

    def clear(self):
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        self.box.configure(state="disabled")


# ---------------------------------------------------------------------------
# ПАТЧ 9, часть XI — FinalAgentInputPanel
# ---------------------------------------------------------------------------
# НОВЫЙ КЛАСС. Ничего из TreePanel/SearchPanel/PreviewPanel/ProgressLog не
# меняется — панель добавляется в main.py как ещё один виджет в сетке,
# аналогично существующим (см. main.py: body.grid_columnconfigure/...).
#
# Это главная debug-поверхность из аудита (часть IX-X): не "как проходила
# навигация по графу", а "что РОВНО ушло агенту". Показывает: точный текст
# payload, упорядоченные context-блоки, provenance map, распределение
# токенов, entropy, временной и файловый разброс включённых чанков.
class FinalAgentInputPanel(ctk.CTkFrame):
    """Отображает результат Archive.assemble_context_snapshot() (+ то, что
    реально было сохранено ptg_snapshot_store.save_agent_snapshot()).
    Панель PASSIVE: она не строит снапшот сама — main.py передаёт ей
    готовый snapshot-dict через show(). Это сохраняет разделение
    ответственности: GUI не должен знать про формат MCP-инструментов,
    только про уже собранный словарь snapshot."""

    def __init__(self, master, on_export_payload=None, on_export_provenance=None,
                 on_toggle_snapshot=None):
        super().__init__(master)
        self.on_export_payload = on_export_payload
        self.on_export_provenance = on_export_provenance
        self.on_toggle_snapshot = on_toggle_snapshot
        self._last_snapshot = None

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(8, 0))
        ctk.CTkLabel(header, text="FINAL AGENT INPUT",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")

        self.snapshot_enabled = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(header, text="Enable Snapshot", variable=self.snapshot_enabled,
                         command=self._on_toggle).pack(side="right", padx=(6, 0))
        ctk.CTkButton(header, text="Export Provenance", width=130,
                      command=self._export_provenance).pack(side="right", padx=(6, 0))
        ctk.CTkButton(header, text="Export Payload", width=110,
                      command=self._export_payload).pack(side="right", padx=(6, 0))

        # -- сводная строка: токены / энтропия / разброс -----------------
        self.summary_label = ctk.CTkLabel(self, text="(снапшот ещё не собирался)",
                                          anchor="w", text_color="gray")
        self.summary_label.pack(fill="x", padx=8, pady=(4, 0))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=8, pady=8)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        # -- точный текст payload -----------------------------------------
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ctk.CTkLabel(left, text="Точный payload (без сокращений)",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w")
        self.payload_box = ctk.CTkTextbox(left, wrap="word")
        self.payload_box.pack(fill="both", expand=True, pady=(2, 0))

        # -- provenance map -------------------------------------------------
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(right, text="Provenance map",
                     font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w")
        style = ttk.Style()
        style.configure("PTGProv.Treeview", rowheight=20, font=("Segoe UI", 9))
        self.prov_tree = ttk.Treeview(
            right, show="headings", style="PTGProv.Treeview",
            columns=("pos", "tag", "file", "depth", "weight", "tokens"),
        )
        for col, txt, w in [("pos", "#", 28), ("tag", "tag", 90), ("file", "файл", 110),
                             ("depth", "depth", 45), ("weight", "weight", 55), ("tokens", "tok", 45)]:
            self.prov_tree.heading(col, text=txt)
            self.prov_tree.column(col, width=w, anchor="w")
        self.prov_tree.pack(fill="both", expand=True, pady=(2, 0))

        self.show_empty()

    # ------------------------------------------------------------------
    def _on_toggle(self):
        if self.on_toggle_snapshot:
            self.on_toggle_snapshot(self.snapshot_enabled.get())

    def _export_payload(self):
        if self.on_export_payload and self._last_snapshot:
            self.on_export_payload(self._last_snapshot)

    def _export_provenance(self):
        if self.on_export_provenance and self._last_snapshot:
            self.on_export_provenance(self._last_snapshot.get("provenance", []))

    def show_empty(self):
        self.summary_label.configure(text="(снапшот ещё не собирался)")
        self.payload_box.configure(state="normal")
        self.payload_box.delete("1.0", "end")
        self.payload_box.insert("end", "Соберите снапшот (поиск → выбрать атомы → «Собрать контекст»).")
        self.payload_box.configure(state="disabled")
        self.prov_tree.delete(*self.prov_tree.get_children())

    def show(self, snapshot: dict, final_agent_payload: str = None):
        """snapshot — результат Archive.assemble_context_snapshot() (можно
        напрямую то, что вернул ptg_assemble_snapshot). final_agent_payload —
        опционально точная склейка system+context+query, если она уже
        была посчитана (например, ptg_snapshot_store.save_agent_snapshot
        сохранил её на диск) — если не передано, панель сама склеивает
        только snapshot_text (без system_prompt/user_query, которых у GUI
        может не быть в контексте desktop-сессии)."""
        self._last_snapshot = snapshot

        prov = snapshot.get("provenance", [])
        files = sorted({p.get("filename") for p in prov if p.get("filename")})
        created_ats = [p.get("created_at") for p in prov if p.get("created_at")]
        temporal_spread_days = (
            (max(created_ats) - min(created_ats)) / 86400.0 if len(created_ats) >= 2 else 0.0
        )
        self.summary_label.configure(
            text=(f"Токенов: {snapshot.get('token_total', 0)}   "
                  f"Entropy: {snapshot.get('context_entropy', 0):.3f}   "
                  f"Файлов: {len(files)}   "
                  f"Временной разброс: {temporal_spread_days:.1f} дн.   "
                  f"Бюджет: {snapshot.get('char_used', 0)}/{snapshot.get('char_budget', 0)} симв."
                  f"{'  ⚠ ОБРЕЗАНО' if snapshot.get('truncated_by_budget') else ''}")
        )

        self.payload_box.configure(state="normal")
        self.payload_box.delete("1.0", "end")
        self.payload_box.insert("end", final_agent_payload or snapshot.get("snapshot_text", ""))
        self.payload_box.configure(state="disabled")

        self.prov_tree.delete(*self.prov_tree.get_children())
        for p in prov:
            self.prov_tree.insert("", "end", values=(
                p.get("position", ""),
                p.get("tag", ""),
                _short_path(p.get("source_path") or p.get("filename", "")),
                p.get("tree_depth", ""),
                p.get("final_weight", ""),
                p.get("token_count", ""),
            ))


# ---------------------------------------------------------------------------
# ПАТЧ 19, часть II — StatusBar ("таскбар")
# ---------------------------------------------------------------------------
# НОВЫЙ КЛАСС. Ничего из существующих панелей не меняется — StatusBar
# добавляется в main.py как отдельная строка внизу окна (pack side="bottom"),
# под сеткой основных панелей.
#
# Отличие от ProgressLog: тот — прокручиваемая ИСТОРИЯ сообщений (нужно
# листать, чтобы увидеть, что произошло). StatusBar — это "здесь и сейчас":
# без прокрутки, одним взглядом видно, идёт ли сейчас фоновая операция,
# сколько узлов/веток/файлов в архиве, подключён ли LM Studio и когда было
# последнее действие. Оба виджета дополняют друг друга, ни один не заменяет.
class StatusBar(ctk.CTkFrame):
    """Нижняя строка состояния приложения. Показывает:
    - индикатор состояния (● цвет + текст): idle/busy/error;
    - сводку архива (узлов/веток/файлов);
    - краткий статус LM Studio (компактнее, чем в верхней панели);
    - время последнего действия.

    Виджет PASSIVE — сам ничего не вычисляет и не опрашивает, только
    отображает то, что ему явно передали через set_*()."""

    _STATE_COLORS = {"idle": "#2fa84f", "busy": "#e8a33d", "error": "#e5484d"}
    _STATE_TEXT = {"idle": "Готово", "busy": "Выполняется…", "error": "Ошибка"}

    def __init__(self, master):
        super().__init__(master, height=28, corner_radius=0)
        self.grid_propagate(False)
        self.pack_propagate(False)

        self.state_dot = ctk.CTkLabel(self, text="●", text_color=self._STATE_COLORS["idle"], width=14)
        self.state_dot.pack(side="left", padx=(10, 2), pady=4)
        self.state_label = ctk.CTkLabel(self, text=self._STATE_TEXT["idle"], anchor="w", width=140)
        self.state_label.pack(side="left", padx=(0, 16), pady=4)

        sep1 = ctk.CTkLabel(self, text="│", text_color="gray40")
        sep1.pack(side="left", padx=(0, 12))

        self.archive_label = ctk.CTkLabel(self, text="Архив не загружен", anchor="w", text_color="gray")
        self.archive_label.pack(side="left", padx=(0, 16), pady=4)

        sep2 = ctk.CTkLabel(self, text="│", text_color="gray40")
        sep2.pack(side="left", padx=(0, 12))

        self.lm_label = ctk.CTkLabel(self, text="LM Studio: —", anchor="w", text_color="gray")
        self.lm_label.pack(side="left", padx=(0, 16), pady=4)

        self.last_action_label = ctk.CTkLabel(self, text="", anchor="e", text_color="gray")
        self.last_action_label.pack(side="right", padx=10, pady=4)

        self.current_state = "idle"  # ПАТЧ 19 — явный флаг состояния (не парсить displayed text)

    def set_state(self, state: str, detail: str = ""):
        """state: 'idle' | 'busy' | 'error'. detail — короткая приписка
        (например, «строю архив», «собираю снапшот»)."""
        self.current_state = state
        color = self._STATE_COLORS.get(state, "gray")
        text = self._STATE_TEXT.get(state, state)
        if detail:
            text = f"{text} — {detail}"
        self.state_dot.configure(text_color=color)
        self.state_label.configure(text=text)
        self._touch()

    def set_archive_info(self, nodes: int = 0, branches: int = 0, files: int = 0):
        self.archive_label.configure(
            text=f"Узлов: {nodes}  Веток: {branches}  Файлов: {files}",
            text_color=("gray10", "gray90"),
        )

    def set_lm_status(self, connected: bool, model: str = None):
        if connected:
            label = f"LM Studio: ● {model}" if model else "LM Studio: ● подключён"
            self.lm_label.configure(text=label, text_color=("gray10", "gray90"))
        else:
            self.lm_label.configure(text="LM Studio: ○ не подключён", text_color="gray")

    def _touch(self):
        self.last_action_label.configure(text=time.strftime("Обновлено: %H:%M:%S"))
