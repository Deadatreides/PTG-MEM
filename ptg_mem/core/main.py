"""
main.py — Personal Thought Graph (PTG) desktop-приложение.

Запуск:
    python main.py

Архитектура GUI:
- Индикатор «LM Studio: Подключён / Отключён» в верхней панели
- Кнопка «Построить архив» блокируется при отсутствии LM Studio
- Кнопка «⟳ LM Studio» — ручная повторная проверка соединения
- Дерево: Проект → Файлы → Связанные файлы → Ветки → Атомы → Возвраты
- Поиск: иерархический (файловые центроиды → file_edges → атомы)
- Просмотр: Q+A раздельно, статус, все типы рёбер включая root-aware

Исправлен баг: _poll_queue был определён дважды (мёртвый первый удалён).
"""

import os
import sys
import threading
import queue
import tkinter.filedialog as filedialog
import customtkinter as ctk

from ptg_core import Archive, Embedder
from gui_panels import TreePanel, SearchPanel, PreviewPanel, ProgressLog, FinalAgentInputPanel, StatusBar
from ptg_snapshot_store import save_agent_snapshot, save_full_context


class PTGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Личный граф мыслей — PTG")
        self.geometry("1280x800")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.archive = None
        self.folder = None
        self.output_dir = None   # ПАТЧ 14 — отдельная папка для векторной базы (None = рядом с folder)
        self.msg_queue = queue.Queue()
        self._last_search_results = []   # ПАТЧ 9 — seed-узлы для снапшота
        # ПАТЧ 20 — реальный инцидент: несколько фоновых операций GUI
        # (проверка LM Studio, сборка архива, поиск) могли одновременно
        # слать запросы на один и тот же LM Studio backend, который
        # обрабатывает их последовательно — это усиливало и без того
        # долгую обработку больших батчей на слабом железе. Лок не даёт
        # им стартовать параллельно; если LM Studio уже занят другой
        # операцией, новая проверка просто тихо пропускается (не встаёт
        # в очередь поверх уже идущей).
        self._lm_studio_lock = threading.Lock()

        self._build_layout()
        self.after(200, self._poll_queue)
        # Проверить LM Studio при старте
        self.after(400, self._check_lm_studio)

    # ------------------------------------------------------------------
    # Построение интерфейса
    # ------------------------------------------------------------------
    def _build_layout(self):
        # === Верхняя панель ===
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=10, pady=(10, 0))

        ctk.CTkButton(top, text="Выбрать папку…",
                      command=self._choose_folder, width=140).pack(side="left", padx=4, pady=8)

        self.build_btn = ctk.CTkButton(top, text="Построить / обновить архив",
                                       command=self._start_build, state="disabled", width=200)
        self.build_btn.pack(side="left", padx=4, pady=8)

        ctk.CTkButton(top, text="⟳ LM Studio", command=self._check_lm_studio,
                      width=110).pack(side="left", padx=4, pady=8)

        # ПАТЧ 9, часть XI — сборка FINAL AGENT INPUT из текущих
        # результатов поиска (seed-узлы для assemble_context_snapshot)
        ctk.CTkButton(top, text="🧠 Собрать снапшот", command=self._assemble_snapshot,
                      width=160).pack(side="left", padx=4, pady=8)

        # ПАТЧ 16 — главный запрошенный функционал: полный контекст ВСЕГО
        # проекта в 2 клика (выбрать папку + эта кнопка), без поиска и
        # без выбора seed-узлов — для возобновления работы в новом чате.
        ctk.CTkButton(top, text="📋 Полный контекст проекта", command=self._export_full_context,
                      width=200, fg_color="#2d6a4f", hover_color="#1b4332").pack(side="left", padx=4, pady=8)

        # Индикатор LM Studio (ПАТЧ 2)
        self.lm_indicator = ctk.CTkLabel(top, text="● LM Studio: проверка…",
                                          text_color="gray", width=220, anchor="w")
        self.lm_indicator.pack(side="left", padx=(10, 0), pady=8)

        self.folder_label = ctk.CTkLabel(top, text="Папка не выбрана",
                                          anchor="w", text_color="gray")
        self.folder_label.pack(side="left", padx=10, pady=8, fill="x", expand=True)

        # === Вторая строка верхней панели — ПАТЧ 14: отдельная папка для
        # векторной базы (.ptg/), независимая от папки с исходниками ===
        top2 = ctk.CTkFrame(self)
        top2.pack(fill="x", padx=10, pady=(4, 0))

        ctk.CTkButton(top2, text="Папка для векторной базы…",
                      command=self._choose_output_dir, width=200).pack(side="left", padx=4, pady=6)

        ctk.CTkButton(top2, text="✕ Сбросить (по умолчанию)",
                      command=self._reset_output_dir, width=190).pack(side="left", padx=4, pady=6)

        self.output_dir_label = ctk.CTkLabel(
            top2, text="Векторная база: рядом с исходниками (.ptg внутри выбранной папки)",
            anchor="w", text_color="gray",
        )
        self.output_dir_label.pack(side="left", padx=10, pady=6, fill="x", expand=True)

        # ПАТЧ 17 — выключатель обработки .py (только докстринги/#-комментарии,
        # см. ptg_py_comments.py). Включён по умолчанию — обратная
        # совместимость с ПАТЧ 16. Влияет только на СЛЕДУЮЩУЮ сборку —
        # уже добавленные .py-атомы не удаляются при выключении (append-only).
        self.py_comments_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(top2, text="Обрабатывать .py (только докстринги/комментарии)",
                         variable=self.py_comments_var).pack(side="right", padx=(6, 10), pady=6)

        # ПАТЧ 19 — StatusBar ("таскбар"): всегда видимая строка состояния
        # внизу окна. ВАЖЕН ПОРЯДОК: pack(side="bottom") должен произойти
        # ДО body.pack(expand=True) — иначе body заберёт себе всё
        # доступное место и статус-бару не останется пространства снизу
        # (это особенность geometry manager pack(), не опечатка порядка).
        self.status_bar = StatusBar(self)
        self.status_bar.pack(side="bottom", fill="x")

        # === Основное тело ===
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=10)
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=2)
        body.grid_columnconfigure(2, weight=3)
        body.grid_rowconfigure(0, weight=3)
        body.grid_rowconfigure(1, weight=2)
        body.grid_rowconfigure(2, weight=3)   # ПАТЧ 9 — строка под FINAL AGENT INPUT

        # Панель дерева (ПАТЧ 6 — глубокая иерархия)
        self.tree_panel = TreePanel(body, on_select=self._show_preview)
        self.tree_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 6))

        # Панель поиска
        self.search_panel = SearchPanel(body, on_search=self._do_search, on_select=self._show_preview)
        self.search_panel.grid(row=0, column=1, sticky="nsew", padx=6, pady=(0, 6))

        # Журнал прогресса
        self.progress_log = ProgressLog(body)
        self.progress_log.grid(row=1, column=1, sticky="nsew", padx=6, pady=(6, 0))

        # Панель просмотра
        self.preview_panel = PreviewPanel(body)
        self.preview_panel.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(6, 0))

        # ПАТЧ 9, часть XI — FINAL AGENT INPUT (главная debug-поверхность)
        self.agent_input_panel = FinalAgentInputPanel(
            body,
            on_export_payload=self._export_payload,
            on_export_provenance=self._export_provenance,
        )
        self.agent_input_panel.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(6, 0))

    # ------------------------------------------------------------------
    # ПАТЧ 2 — проверка LM Studio
    # ------------------------------------------------------------------
    def _check_lm_studio(self):
        def _worker():
            if not self._lm_studio_lock.acquire(blocking=False):
                # LM Studio уже занят другой операцией (build/search) —
                # не шлём конкурирующий запрос, который лишь добавит
                # очередь поверх уже идущей (см. ПАТЧ 20).
                return
            try:
                e = Embedder()
                ok = e.test_connection()
                self.msg_queue.put(("__LM_STATUS__", ok, e.model))
            finally:
                self._lm_studio_lock.release()
        threading.Thread(target=_worker, daemon=True).start()

    def _apply_lm_status(self, ok, model):
        if ok:
            self.lm_indicator.configure(
                text=f"● LM Studio: Подключён  [{model}]",
                text_color="#22c55e",
            )
            # Разблокировать «Построить» только если папка уже выбрана
            if self.folder:
                self.build_btn.configure(state="normal")
        else:
            self.lm_indicator.configure(
                text="● LM Studio: Отключён",
                text_color="#ef4444",
            )
            # Блокировать построение — без LM Studio нельзя (ПАТЧ 2)
            self.build_btn.configure(state="disabled")
        # ПАТЧ 19 — StatusBar
        self.status_bar.set_lm_status(ok, model if ok else None)

    # ------------------------------------------------------------------
    # Выбор папки
    # ------------------------------------------------------------------
    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с чат-логами")
        if not folder:
            return
        self.folder = folder
        # Фикс: в новых версиях customtkinter text_color=None вызывает
        # ValueError ("color is None, for transparency set color='transparent'")
        # вместо сброса к цвету темы по умолчанию — раньше это работало.
        # Ставим явный default text color CTkLabel (light, dark).
        self.folder_label.configure(text=folder, text_color=("gray10", "gray90"))

        # Загрузить существующий архив сразу, если есть (ПАТЧ 14: с учётом
        # уже выбранной отдельной папки для векторной базы, если она есть)
        archive = Archive(self.folder, progress_cb=self.msg_queue.put, output_dir=self.output_dir,
                             extract_py_comments=self.py_comments_var.get())
        if archive.load_if_exists():
            self.archive = archive
            self.tree_panel.refresh(self.archive)
            proj = archive.root_project.get("project_name", "")
            nodes_n = len(archive.nodes)
            files_n = len(archive.files)
            fe_n = len(getattr(archive, 'file_edges', []))
            bs_all_l = getattr(archive, "branch_states", {})
            dormant_l = sum(1 for b in bs_all_l.values() if b.get("dormant"))
            # ПАТЧ 10, фикс бага #7: archive.current_path — мёртвый атрибут,
            # которого никогда не было в Archive (реальное поле —
            # path_memory["short"], см. ptg_core.py). Раньше path_len
            # всегда молча равнялся 0 через getattr(..., []) fallback.
            path_len  = len(getattr(archive, "path_memory", {}).get("short", []))
            self.progress_log.write(
                f"Загружен архив: «{proj}»  —  {nodes_n} узлов, "
                f"{files_n} файлов, {fe_n} рёбер файл-графа.\n"
                f"  Веток: {len(bs_all_l)} "
                f"({dormant_l} дремлющих)  "
                f"Траектория: {path_len} атомов в памяти."
            )
            self.status_bar.set_archive_info(nodes_n, len(bs_all_l), files_n)
        else:
            self.progress_log.write(f"Архив не найден в {archive.ptg_dir} — готов к первой сборке.")
            self.status_bar.set_archive_info(0, 0, 0)

        # Разблокировать кнопку только при наличии LM Studio
        self._check_lm_studio()

    # ------------------------------------------------------------------
    # ПАТЧ 14 — отдельная папка для векторной базы (.ptg/)
    # ------------------------------------------------------------------
    def _choose_output_dir(self):
        folder = filedialog.askdirectory(title="Выберите папку для векторной базы (.ptg)")
        if not folder:
            return
        self.output_dir = folder
        self.output_dir_label.configure(
            text=f"Векторная база: {folder}\\.ptg", text_color=("gray10", "gray90")
        )
        self.progress_log.write(f"Папка для векторной базы установлена: {folder}")
        # Если папка с исходниками уже выбрана — сразу попробовать
        # подхватить архив из НОВОЙ output-папки (а не из старой, рядом
        # с исходниками), чтобы не путать пользователя устаревшим деревом.
        if self.folder:
            self._reload_with_current_paths()

    def _reset_output_dir(self):
        self.output_dir = None
        self.output_dir_label.configure(
            text="Векторная база: рядом с исходниками (.ptg внутри выбранной папки)",
            text_color="gray",
        )
        self.progress_log.write("Папка для векторной базы сброшена к значению по умолчанию.")
        if self.folder:
            self._reload_with_current_paths()

    def _reload_with_current_paths(self):
        """Перезагрузить (без сборки) архив из текущей пары folder/output_dir —
        общая логика для случаев смены output_dir уже после выбора папки
        с исходниками, чтобы дерево/статус в GUI не показывали устаревший
        архив из другого места."""
        archive = Archive(self.folder, progress_cb=self.msg_queue.put, output_dir=self.output_dir,
                             extract_py_comments=self.py_comments_var.get())
        if archive.load_if_exists():
            self.archive = archive
            self.tree_panel.refresh(self.archive)
            self.progress_log.write(
                f"Подхвачен существующий архив из {archive.ptg_dir} "
                f"({len(archive.nodes)} узлов)."
            )
            self.status_bar.set_archive_info(len(archive.nodes), len(archive.branches), len(archive.files))
        else:
            self.archive = None
            self.tree_panel.refresh(None)
            self.progress_log.write(f"В {archive.ptg_dir} архива ещё нет — будет создан при сборке.")
            self.status_bar.set_archive_info(0, 0, 0)
        self._check_lm_studio()

    # ------------------------------------------------------------------
    # Построение архива
    # ------------------------------------------------------------------
    def _start_build(self):
        if not self.folder:
            return
        self.build_btn.configure(state="disabled")
        self.progress_log.clear()
        self.status_bar.set_state("busy", "строю архив")
        threading.Thread(target=self._build_worker, daemon=True).start()

    def _build_worker(self):
        archive = Archive(self.folder, progress_cb=self.msg_queue.put, output_dir=self.output_dir,
                             extract_py_comments=self.py_comments_var.get())
        # ПАТЧ 20 — ждём (blocking=True), если LM Studio сейчас занят другой
        # операцией: сборка должна ДОЖДАТЬСЯ своей очереди, а не молча
        # пропуститься, как «просто health-check» в _check_lm_studio().
        with self._lm_studio_lock:
            try:
                summary = archive.build()
                self.archive = archive
                edges_str = "  ".join(
                    f"{t}: {n}"
                    for t, n in sorted(summary.get("edges_by_type", {}).items())
                )
                fe_count = len(getattr(self.archive, 'file_edges', []))
                re_str = ""
                if self.archive and self.archive.root_embedding is not None:
                    re_str = f"  root_embedding: dim={self.archive.root_embedding.shape[0]}\n"
                # branch_states статистика
                bs_all = getattr(self.archive, "branch_states", {})
                dormant_n = sum(1 for bs in bs_all.values() if bs.get("dormant"))
                active_n  = len(bs_all) - dormant_n
                bs_str = f"  ветки: {active_n} активных, {dormant_n} дремлющих\n" if bs_all else ""
                self.msg_queue.put(
                    f"Архив готов:\n"
                    f"  узлов: {summary['total_nodes']}   "
                    f"веток: {summary['total_branches']}   "
                    f"файлов: {summary['total_files']}   "
                    f"рёбер файл-граф: {fe_count}\n"
                    f"{re_str}"
                    f"{bs_str}"
                    f"  рёбра атомов — {edges_str}\n"
                    f"  эмбеддер: {summary['embedder_model']}"
                )
            except ConnectionError as ce:
                self.msg_queue.put(f"ОШИБКА СОЕДИНЕНИЯ: {ce}")
            except Exception as ex:
                self.msg_queue.put(f"ОШИБКА: {ex}")
            finally:
                self.msg_queue.put("__REFRESH_TREE__")
                self.msg_queue.put("__BUILD_DONE__")

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------
    def _do_search(self, query):
        if not self.archive:
            self.progress_log.write("Сначала постройте или загрузите архив.")
            return
        self.status_bar.set_state("busy", "ищу")

        def _worker():
            with self._lm_studio_lock:
                try:
                    # ПАТЧ 10 — миграция на branch-first (см. ptg_core.search_branch_first).
                    # Раньше здесь вызывался archive.search() (file-first) — то же самое
                    # уже мигрировано в ptg_mcp_server.ptg_search; GUI отставал, что
                    # давало разные результаты поиска человеку и агенту на одном архиве.
                    # Старый file-first поиск не удалён (archive.search()) — доступен
                    # программно для случаев, где важна файловая, а не траекторная
                    # локальность, но больше не вызывается из этой панели.
                    results = self.archive.search_branch_first(query)
                    self.msg_queue.put(("__SEARCH_RESULTS__", results))
                except Exception as ex:
                    self.msg_queue.put(f"ОШИБКА ПОИСКА: {ex}")

        threading.Thread(target=_worker, daemon=True).start()

    def _export_full_context(self):
        """ПАТЧ 16 — «2 клика»: выбрать папку (уже сделано к этому моменту)
        + нажать эту кнопку. Никакого поиска, никакого выбора seed-узлов —
        полный автоматический сбор всего проекта."""
        if not self.archive:
            self.progress_log.write("Сначала постройте или загрузите архив.")
            return
        self.status_bar.set_state("busy", "собираю полный контекст проекта")

        def _worker():
            try:
                result = self.archive.export_full_context()
                saved_path = save_full_context(self.archive.output_root, result)
                self.msg_queue.put(("__FULL_CONTEXT_READY__", result, saved_path))
            except Exception as ex:
                self.msg_queue.put(f"ОШИБКА СБОРКИ ПОЛНОГО КОНТЕКСТА: {ex}")

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # ПАТЧ 9, часть XI — FINAL AGENT INPUT: сборка и экспорт
    # ------------------------------------------------------------------
    def _assemble_snapshot(self):
        if not self.archive:
            self.progress_log.write("Сначала постройте или загрузите архив.")
            return
        if not self._last_search_results:
            self.progress_log.write("Сначала выполните поиск — его результаты станут seed-узлами снапшота.")
            return
        self.status_bar.set_state("busy", "собираю снапшот")

        def _worker():
            try:
                snapshot = self.archive.assemble_context_snapshot(self._last_search_results)
                saved_path = None
                if self.agent_input_panel.snapshot_enabled.get():
                    saved_path = save_agent_snapshot(self.archive.output_root, snapshot)
                self.msg_queue.put(("__SNAPSHOT_READY__", snapshot, saved_path))
            except Exception as ex:
                self.msg_queue.put(f"ОШИБКА СБОРКИ СНАПШОТА: {ex}")

        threading.Thread(target=_worker, daemon=True).start()

    def _export_payload(self, snapshot):
        path = filedialog.asksaveasfilename(
            title="Экспорт payload", defaultextension=".txt",
            initialfile="agent_input_payload.txt",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(snapshot.get("snapshot_text", ""))
        self.progress_log.write(f"Payload экспортирован: {path}")

    def _export_provenance(self, provenance):
        import json
        path = filedialog.asksaveasfilename(
            title="Экспорт provenance", defaultextension=".json",
            initialfile="provenance_map.json",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, ensure_ascii=False, indent=2)
        self.progress_log.write(f"Provenance map экспортирована: {path}")

    # ------------------------------------------------------------------
    # Просмотр атома
    # ------------------------------------------------------------------
    def _show_preview(self, node_id):
        if not self.archive:
            return
        preview = self.archive.preview(node_id)
        if preview:
            self.preview_panel.show(preview)

    # ------------------------------------------------------------------
    # Перехват результатов поиска (приходят из фонового потока)
    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()

                if isinstance(msg, tuple):
                    tag = msg[0]
                    if tag == "__LM_STATUS__":
                        _, ok, model = msg
                        self._apply_lm_status(ok, model)
                    elif tag == "__SEARCH_RESULTS__":
                        _, results = msg
                        self._last_search_results = [nid for _sim, nid in results]  # ПАТЧ 9
                        if self.archive:
                            self.search_panel.show_results(results, self.archive.nodes)
                        self.status_bar.set_state("idle")
                    elif tag == "__SNAPSHOT_READY__":
                        _, snapshot, saved_path = msg
                        self.agent_input_panel.show(snapshot)
                        note = f"  (сохранён: {saved_path})" if saved_path else "  (сохранение выключено)"
                        self.progress_log.write(
                            f"Снапшот собран: {len(snapshot.get('included_node_ids', []))} атомов, "
                            f"{snapshot.get('token_total', 0)} токенов, "
                            f"entropy={snapshot.get('context_entropy', 0):.3f}{note}"
                        )
                        self.status_bar.set_state("idle")
                    elif tag == "__FULL_CONTEXT_READY__":
                        _, result, saved_path = msg
                        # ПАТЧ 16 — export_full_context() возвращает другую форму
                        # (full_text/char_count/approx_tokens), чем
                        # assemble_context_snapshot() (snapshot_text/token_total/
                        # context_entropy/provenance). Адаптируем под уже
                        # существующий FinalAgentInputPanel.show(), не меняя
                        # саму панель — provenance здесь пуст (полный контекст
                        # уже несёт файловую/структурную разметку прямо в тексте).
                        adapted = {
                            "snapshot_text": result.get("full_text", ""),
                            "provenance": [],
                            "token_total": result.get("approx_tokens", 0),
                            "context_entropy": 0.0,
                            "char_budget": result.get("char_count", 0),
                            "char_used": result.get("char_count", 0),
                            "truncated_by_budget": False,
                        }
                        self.agent_input_panel.show(adapted)
                        self.progress_log.write(
                            f"Полный контекст проекта собран: {result.get('file_count', 0)} файлов, "
                            f"{result.get('atom_count', 0)} атомов, "
                            f"{result.get('unresolved_count', 0)} незакрытых линий, "
                            f"{result.get('contradiction_count', 0)} противоречий, "
                            f"~{result.get('approx_tokens', 0)} токенов.  (сохранён: {saved_path})"
                        )
                        self.status_bar.set_state("idle")
                    continue

                if msg == "__REFRESH_TREE__":
                    if self.archive:
                        self.tree_panel.refresh(self.archive)
                elif msg == "__BUILD_DONE__":
                    self._check_lm_studio()
                    # ПАТЧ 19 — StatusBar: если ошибка уже была выставлена
                    # чуть выше (обработкой строки "ОШИБКА...") в ЭТОМ ЖЕ
                    # проходе очереди — не затирать её на "idle".
                    if self.status_bar.current_state != "error":
                        self.status_bar.set_state("idle")
                    if self.archive:
                        self.status_bar.set_archive_info(
                            len(self.archive.nodes), len(self.archive.branches), len(self.archive.files)
                        )
                else:
                    text = str(msg)
                    self.progress_log.write(text)
                    if text.startswith("ОШИБКА"):
                        self.status_bar.set_state("error", text[:60])

        except Exception:
            pass
        self.after(150, self._poll_queue)


if __name__ == "__main__":
    if sys.platform == "darwin":
        os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")
    app = PTGApp()
    app.mainloop()
