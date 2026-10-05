# -*- coding: utf-8 -*-
"""MemoryEngine — one project's memory, kept current while you work.

What it adds on top of the research engine (``core/ptg_core.py``):

ONE PIPELINE. The budgeted graft (V5) and the vectorised graft (V4) are installed
on the live archive at load time, so every new atom is attached by the V5 rule the
moment it arrives. There is no separate "build with the old graft, then replay"
step any more: that two-stage cycle existed only because the old archive had been
built before V5. After one replay (``ptg migrate``) a store stays V5 forever.

EDITS ARE NOT DUPLICATES. ptg_core re-appended a changed file in full, so every
save of a note added a second copy of all its paragraphs, and deleting a file
changed nothing. Here a changed file is diffed atom by atom (atom text is
deterministic for the same content): unchanged atoms are kept, vanished atoms
are *retracted* (kept for history, never searched or injected), and only new
text is embedded. A new atom at the position of a retracted one gets a
``revises`` edge to it, so the graph keeps the edit history.

RECALL IS FLAT, RELATIONS ARE THE GRAPH'S JOB. Measured four times on this
project: flat cosine recall beats branch-first and graph expansion (70.3 % vs
64.9 %). So recall ranks by cosine over live atoms and then uses the graph for
what it is good at — telling the agent that a hit was superseded (and by what)
or is contradicted.
"""
from __future__ import annotations

import collections
import hashlib
import os
import re
import threading
import time
import uuid

import numpy as np

import ptg_mem  # noqa: F401  — puts core/v4/v5 on sys.path
import budget_graft
import graft_fast
import ptg_core
import ptg_logging
import ptg_py_comments

from . import store, transcripts
from .filters import FileFilter, scrub_secrets

# ptg_core keeps its thresholds as module globals, and a daemon may hold several
# projects: every engine operation runs under this lock with its own thresholds.
ENGINE_LOCK = threading.RLock()

TYPED = ("supersedes", "contradicts", "fixes", "refines", "revises")
NOTES_FILE = "memory://notes"
# ptg_core marks "A supersedes B" when A contains a marker ("instead of", "вместо",
# "deprecated"...) and B is A's nearest neighbour above ~0.57. On a small project that
# links unrelated notes of the same domain: "instead of an LRU we use a TTL" superseded
# a note about batching. Recall must never swap a hit for an unrelated "newer
# version", so a supersession counts only when the two texts are close for real.
SUPERSEDE_MIN_SIM = 0.75
# Swapping a recalled hit for "its newer version" is a strong act; below this the newer
# atom is only mentioned. Measured on the reference archive: a conversational atom with
# "вместо" in a long answer still reached 0.75 against an unrelated earlier turn.
REPLACE_IN_RECALL_SIM = 0.82
FILE_GRAPH_THRESH = ptg_core.FILE_GRAPH_THRESH


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _fmt_day(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError):
        return "?"


class MemoryEngine:
    def __init__(self, project: dict, embedder, log=None):
        self.p = project
        self.root = project["root"]
        self.store_dir = project["store"]
        self.embedder = embedder
        self.log = log or (lambda m: None)
        self.filter = FileFilter(self.root, exclude_dirs=project.get("exclude_dirs", ()),
                                 include_exts=project.get("include_exts"),
                                 store_dir=self.store_dir)
        self.arc = None
        self.fg = None
        self.bg = None
        self.dirty = False
        self.last_change = 0.0
        self.last_save = 0.0
        self.file_graph_dirty = False
        self.extra = {}
        self.file_index: dict[str, list[str]] = {}
        self.sup_newer: dict[str, str] = {}               # old -> newer (supersedes)
        self.rel: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
        self._edges_seen = 0
        self._fe_index: dict[tuple[str, str], float] = {}
        self.events = collections.deque(maxlen=400)
        self.loaded_at = 0.0
        self._compact = transcripts.CompactTracker()

    # ------------------------------------------------------------------ setup
    def _apply_thresholds(self):
        t = self.p.get("thresholds") or {}
        ptg_core.BRANCH_THRESH = float(t.get("branch", 0.47))
        ptg_core.CONTINUE_THRESH = float(t.get("continue", 0.57))
        ptg_core.RETURN_THRESH = float(t.get("return", 0.65))
        ptg_core.CONTRADICT_SIM_THRESH = float(t.get("contradict", 0.62))
        ptg_core.SUPERSEDE_SIM_THRESH = float(t.get("supersede", 0.57))
        ptg_core.MMAP_THRESHOLD_BYTES = 1 << 62

    def _new_archive(self):
        # ptg_core opens a log file inside its store folder; keep it out of the
        # data folder that saves swap, or Windows refuses the rename.
        logs = os.path.join(self.store_dir, "logs")
        orig = ptg_logging.setup_logging
        ptg_core.setup_logging = lambda _d: orig(logs)
        try:
            arc = ptg_core.Archive(folder=self.root, output_dir=os.path.join(self.store_dir, "engine"),
                                   progress_cb=lambda m: None, extract_py_comments=True)
        finally:
            ptg_core.setup_logging = orig
        arc.embedder = self.embedder
        return arc

    def open(self):
        """Load the store (format 6), or migrate a legacy one, or start empty."""
        with ENGINE_LOCK:
            self._apply_thresholds()
            t0 = time.time()
            arc = self._new_archive()
            legacy = self.p.get("legacy_store")
            if store.exists(self.store_dir):
                self.extra = store.load(arc, self.store_dir)
                how = "store"
            elif legacy:
                store.load_legacy(arc, legacy)
                how = "legacy store %s" % legacy
                self.dirty = True
            else:
                how = "new"
            if arc.embedder_dim_used and not self.embedder.dim:
                self.embedder.dim = int(arc.embedder_dim_used)
            if arc.nodes and arc.embedder_model_used and \
                    os.path.basename(str(arc.embedder_model_used)).lower().replace(".gguf", "") != \
                    os.path.basename(str(self.embedder.model)).lower().replace(".gguf", "") and \
                    not self.p.get("embedder_alias_ok"):
                raise RuntimeError(
                    "this store was built with %r, the project is configured for %r; vectors "
                    "from different models are not comparable — use the same model or a new "
                    "store" % (arc.embedder_model_used, self.embedder.model))
            self.arc = arc
            self._install()
            self._index_all()
            self.loaded_at = time.time()
            self.last_save = time.time()
            self.log("opened %s (%s): %d atoms, %d branches, %d edges in %.1f s"
                     % (self.root, how, len(arc.nodes), len(arc.branches), len(arc.edges),
                        time.time() - t0))
            return self

    def _install(self):
        arc = self.arc
        fg = graft_fast.install(arc, ptg_core, verify=0)
        self.bg = budget_graft.install(arc, ptg_core, fg, margin=float(self.p.get("margin", 5.0)))
        self.fg = fg
        eng = self

        def _file_aff(a, b):                         # was O(|file_edges|) per call
            if a == b:
                return 1.0
            return eng._fe_index.get((a, b), 0.0)

        arc._file_origin_affinity = _file_aff

        def _is_unresolved_head(node_id):            # was O(N+E) per call
            key = (len(arc.nodes), len(arc.edges))
            cache = getattr(arc, "_unres_cache", None)
            if cache is None or cache[0] != key:
                parents = {n.get("parent") for n in arc.nodes.values() if n.get("parent")}
                returns = set()
                for e in arc.edges:
                    if e["type"] == "returns_to":
                        returns.add(e["from"])
                        returns.add(e["to"])
                cache = (key, parents, returns)
                arc._unres_cache = cache
            _k, parents, returns = cache
            node = arc.nodes.get(node_id)
            if node is None or arc.branches.get(node.get("branch"), {}).get("head") != node_id:
                return False
            return node_id not in parents and node_id not in returns

        arc._is_unresolved_head = _is_unresolved_head

    def _index_all(self):
        arc = self.arc
        self.file_index = collections.defaultdict(list)
        for nid in arc.id_order:
            n = arc.nodes[nid]
            self.file_index[n.get("file")].append(nid)
            if n.get("status") == "retracted":
                self._zero_row(nid)
        self.sup_newer, self.rel = {}, collections.defaultdict(list)
        self._edges_seen = 0
        self._index_new_edges()
        self._rebuild_fe_index()

    def _index_new_edges(self):
        edges = self.arc.edges
        nodes = self.arc.nodes
        for e in edges[self._edges_seen:]:
            t = e["type"]
            if t not in TYPED:
                continue
            if t in ("supersedes", "contradicts"):
                # both come from word markers ("instead of", "нет, ", "однако"...) next to a
                # merely similar atom; in conversation that fires constantly. Keep the pair only
                # when the two texts are about the same thing for real.
                a, b = nodes.get(e["from"]), nodes.get(e["to"])
                if a is None or b is None or float(np.dot(a["vec"], b["vec"])) < SUPERSEDE_MIN_SIM:
                    continue
                if t == "supersedes":
                    self.sup_newer[e["to"]] = e["from"]
            self.rel[e["from"]].append(("out", t, e["to"]))
            self.rel[e["to"]].append(("in", t, e["from"]))
        self._edges_seen = len(edges)

    def _rebuild_fe_index(self):
        idx = {}
        for fe in self.arc.file_edges:
            a, b, s = fe["source_file"], fe["target_file"], float(fe["similarity"])
            idx[(a, b)] = s
            idx[(b, a)] = s
        self._fe_index = idx

    def _zero_row(self, nid):
        i = self.fg.npos.get(nid) if self.fg else None
        if i is not None:
            self.fg.M[i] = 0.0

    # ------------------------------------------------------------------ ingest
    def _event(self, **kw):
        kw.setdefault("t", time.time())
        self.events.append(kw)

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        """One text per call, outside the engine lock: a recall arriving mid-ingest
        waits for at most one text, not for a whole file."""
        out = []
        for t in texts:
            out.append(self.embedder.embed([t])[0])
        return out

    def _root_setup(self):
        """First ingest of a new project: the root prior (ptg_core patch 1)."""
        if self.arc.root_embedding is not None:
            return
        rp = self._root_project()
        text = ptg_core._root_project_to_text(rp)
        v = self._embed([text])[0]
        with ENGINE_LOCK:
            self.arc.root_project = rp
            self.arc.root_embedding = v

    def _root_project(self) -> dict:
        """ptg_core._build_root_project, but over the filtered file set (the
        original walks everything, weights and run dumps included)."""
        headings, concepts = [], set()
        name = os.path.basename(self.root) or "project"
        n = 0
        for path in self.filter.walk():
            base = os.path.basename(path)
            if not (base.lower().endswith(".md") or base.upper().startswith(ptg_core.ROOT_DOC_PREFIXES)):
                continue
            text = ptg_core._read_text_file(path) or ""
            for line in text.splitlines():
                m = re.match(r"^(#{1,2})\s+(.+)", line.strip())
                if m:
                    headings.append(m.group(2).strip())
            concepts.update(re.findall(r"`([a-zA-Z0-9_./-]{3,40})`", text))
            if base.upper().startswith("README") and os.path.dirname(path) == self.root:
                m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
                if m:
                    name = m.group(1).strip()
            n += 1
            if n >= 400:
                break
        try:
            domains = sorted(d for d in os.listdir(self.root)
                             if os.path.isdir(os.path.join(self.root, d)) and self.filter.dir_ok(d))
        except OSError:
            domains = []
        return {"project_name": name, "domains": domains,
                "root_modules": sorted(set(headings))[:50], "known_concepts": sorted(concepts)[:100]}

    def parse(self, path: str, kind: str) -> list[dict] | None:
        """Atoms of a source, or None if the source is gone."""
        if not os.path.exists(path):
            return None
        if kind == "claude":
            atoms = transcripts.parse_claude(path)
        elif kind == "codex":
            atoms = transcripts.parse_codex(path)
        else:
            # section labels of .py atoms are part of the atom text: keep the project's
            # language, or every .py file would count as edited (see ptg_py_comments.LANG)
            ptg_py_comments.LANG = self.p.get("py_comment_lang", "en")
            atoms = ptg_core.parse_file_to_atoms(path)
        removed = 0
        for a in atoms:
            for k in ("text", "question", "answer"):
                if a.get(k):
                    a[k], n = scrub_secrets(a[k])
                    removed += n
        if removed:
            self._event(kind="secrets", file=path, removed=removed)
        return atoms

    def ingest(self, path: str, kind: str = "file") -> dict:
        """Bring one source up to date: diff, embed only new text, graft, retract."""
        t0 = time.time()
        path = os.path.normpath(path)
        atoms = self.parse(path, kind)
        if atoms is None:
            return self.retract_file(path, reason="deleted")
        if self.arc.root_embedding is None:
            self._root_setup()

        with ENGINE_LOCK:
            old = [nid for nid in self.file_index.get(path, ())
                   if self.arc.nodes[nid].get("status") != "retracted"]
            by_hash = collections.defaultdict(list)
            for nid in old:
                by_hash[_h(self.arc.nodes[nid]["text"])].append(nid)
            keep, fresh = set(), []
            for a in atoms:
                lst = by_hash.get(_h(a["text"]))
                if lst:
                    keep.add(lst.pop(0))
                else:
                    fresh.append(a)
            gone = [nid for nid in old if nid not in keep]
        if not fresh and not gone:
            self._mark_processed(path)
            return {"file": path, "added": 0, "retracted": 0, "kept": len(keep)}

        vecs = self._embed([a["text"] for a in fresh])

        with ENGINE_LOCK:
            self._apply_thresholds()
            gone_by_order = {}
            for nid in gone:
                n = self.arc.nodes[nid]
                n["status"] = "retracted"
                n["retracted_at"] = time.time()
                self._zero_row(nid)
                gone_by_order.setdefault(n.get("order"), nid)
            added = []
            for a, v in zip(fresh, vecs):
                a["id"] = str(uuid.uuid4())
                a["vec"] = np.asarray(v, dtype="float32")
                a.setdefault("timestamp", os.path.getmtime(path) if os.path.exists(path) else time.time())
                node = self.arc._add_atom(a)
                self.file_index[path].append(node["id"])
                prev = gone_by_order.pop(a.get("order"), None)
                if prev is not None:
                    self.arc.edges.append({"from": node["id"], "to": prev, "type": "revises"})
                added.append(node["id"])
            # ptg_core marks the target of a `supersedes` edge "superseded" — which would
            # resurrect an atom we just retracted into search. Retraction wins.
            for nid in gone:
                self.arc.nodes[nid]["status"] = "retracted"
            self._index_new_edges()
            self._update_file(path)
            self._mark_processed(path)
            self.dirty = True
            self.file_graph_dirty = True
            self.last_change = time.time()
        r = {"file": path, "src": kind, "added": len(added), "retracted": len(gone),
             "kept": len(keep), "seconds": round(time.time() - t0, 2)}
        self._event(kind="ingest", **r)
        return r

    def retract_file(self, path: str, reason: str = "deleted") -> dict:
        path = os.path.normpath(path)
        with ENGINE_LOCK:
            n = 0
            for nid in self.file_index.get(path, ()):
                node = self.arc.nodes[nid]
                if node.get("status") != "retracted":
                    node["status"] = "retracted"
                    node["retracted_at"] = time.time()
                    self._zero_row(nid)
                    n += 1
            self.arc.processed_files.pop(path, None)
            self.arc.files.pop(path, None)
            if n:
                self.dirty = True
                self.file_graph_dirty = True
                self.last_change = time.time()
        if n:
            self._event(kind="retract", file=path, retracted=n, reason=reason)
        return {"file": path, "added": 0, "retracted": n, "kept": 0}

    def remember(self, text: str, session_id: str | None = None, source: str = "agent") -> dict:
        """An explicit note or decision, stored immediately."""
        text, _ = scrub_secrets(text.strip())
        if len(text) < 8:
            raise ValueError("note too short")
        if self.arc.root_embedding is None:
            self._root_setup()
        v = self._embed([text])[0]
        now = time.time()
        with ENGINE_LOCK:
            self._apply_thresholds()
            order = len(self.file_index.get(NOTES_FILE, ()))
            atom = {"id": str(uuid.uuid4()), "file": NOTES_FILE, "order": order,
                    "question": "Note (%s)" % source, "answer": text, "text": text,
                    "confidence": "note", "timestamp": now, "vec": np.asarray(v, dtype="float32"),
                    "source_path": NOTES_FILE, "filename": "notes", "folder_path": "memory://",
                    "created_at": now, "modified_at": now, "session_id": session_id, "source": source}
            node = self.arc._add_atom(atom)
            self.file_index[NOTES_FILE].append(node["id"])
            self._index_new_edges()
            self.dirty = True
            self.last_change = now
        self._event(kind="remember", id=node["id"], text=text[:200])
        return self.card(node["id"])

    def _mark_processed(self, path):
        try:
            self.arc.processed_files[path] = os.path.getmtime(path)
        except OSError:
            pass

    def _update_file(self, path):
        live = [self.arc.nodes[nid]["vec"] for nid in self.file_index.get(path, ())
                if self.arc.nodes[nid].get("status") != "retracted"]
        fd = self.arc.files.get(path) or {"file_id": path}
        if live:
            c = np.mean(np.asarray(live, dtype="float32"), axis=0)
            nrm = float(np.linalg.norm(c))
            fd["centroid"] = (c / nrm if nrm else c).astype("float32")
        else:
            fd["centroid"] = None
        fd["main_branches"] = sorted({self.arc.nodes[nid]["branch"] for nid in self.file_index.get(path, ())
                                      if self.arc.nodes[nid].get("status") != "retracted"})
        self.arc.files[path] = fd

    def rebuild_file_graph(self):
        """ptg_core._build_file_graph with one matrix product instead of a double loop."""
        with ENGINE_LOCK:
            paths = [p for p, fd in self.arc.files.items()
                     if fd.get("centroid") is not None and len(fd["centroid"])]
            if len(paths) < 2:
                self.file_graph_dirty = False
                return
            m = np.asarray([self.arc.files[p]["centroid"] for p in paths], dtype="float32")
            m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
            sims = m @ m.T
            iu, ju = np.nonzero(np.triu(sims >= FILE_GRAPH_THRESH, k=1))
            fe, conn = [], collections.defaultdict(list)
            for i, j in zip(iu.tolist(), ju.tolist()):
                fe.append({"source_file": paths[i], "target_file": paths[j],
                           "similarity": round(float(sims[i, j]), 4), "relation": "related"})
                conn[paths[i]].append(paths[j])
                conn[paths[j]].append(paths[i])
            for p in paths:
                self.arc.files[p]["connected_files"] = conn.get(p, [])
            self.arc.file_edges = fe
            self._rebuild_fe_index()
            self.file_graph_dirty = False
            self.dirty = True

    # ------------------------------------------------------------------ plan
    def plan(self, include_transcripts: bool = True):
        """What is out of date: [(path, kind)] to ingest, [path] to retract."""
        todo, seen = [], set()
        src = self.p.get("sources") or {}
        if src.get("folder", True):
            for path in self.filter.walk():
                seen.add(path)
                try:
                    mt = os.path.getmtime(path)
                except OSError:
                    continue
                if self.arc.processed_files.get(path) != mt:
                    todo.append((path, "file"))
        if include_transcripts:
            if src.get("claude", True):
                for path in transcripts.claude_transcripts(self.root):
                    seen.add(path)
                    if self.arc.processed_files.get(path) != os.path.getmtime(path):
                        todo.append((path, "claude"))
            if src.get("codex", True):
                for path in transcripts.codex_transcripts(self.root):
                    seen.add(path)
                    if self.arc.processed_files.get(path) != os.path.getmtime(path):
                        todo.append((path, "codex"))
        gone = [p for p in list(self.arc.processed_files)
                if p not in seen and not p.startswith(("memory://", ptg_core.STRUCTURE_FILE_MARKER))
                and not os.path.exists(p)]
        return todo, gone

    # ------------------------------------------------------------------ save
    def save(self) -> dict:
        with ENGINE_LOCK:
            if self.file_graph_dirty:
                self.rebuild_file_graph()
            snap = _Snapshot(self.arc)
            self.dirty = False
        r = store.save(snap, self.store_dir, extra=self.extra)
        self.last_save = time.time()
        self._event(kind="save", **r)
        return r

    # ------------------------------------------------------------------ read
    def _q(self, text: str) -> np.ndarray:
        return np.asarray(self.embedder.embed([text])[0], dtype="float32")

    def _scores(self, qv) -> np.ndarray:
        n = len(self.fg.nid_list)
        return self.fg.M[:n] @ qv

    _TURN_RE = re.compile(r"^(?:Пользователь|Claude|User|Assistant)\s*:\s*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]")

    def _turn_ts(self, n) -> float:
        """When this exchange happened. Transcript atoms carry it; atoms of transcripts
        converted to documents carry it in the text ("User: [2026-09-22 14:57] ...")."""
        m = self._TURN_RE.match(n.get("text") or "")
        if m:
            try:
                return time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M"))
            except (ValueError, OverflowError):
                pass
        return float(n.get("timestamp") or 0)

    def _is_noise(self, n) -> bool:
        f = n.get("file") or ""
        if f.startswith(ptg_core.STRUCTURE_FILE_MARKER):
            return True                                  # folder listings, not thought
        head = (n.get("question") or n.get("text") or "")[:120]
        head = self._TURN_RE.sub("", head).lstrip()
        return head.startswith(transcripts.AUTOMATED)

    def search(self, query: str, k: int = 8, exclude_files=(), exclude_ids=(),
               with_retracted: bool = False, keep=None) -> list[dict]:
        qv = self._q(query)
        with ENGINE_LOCK:
            s = self._scores(qv).copy()
            ex = set(exclude_ids)
            for f in exclude_files:
                for nid in self.file_index.get(f, ()):
                    i = self.fg.npos.get(nid)
                    if i is not None:
                        s[i] = -1.0
            kk = min(len(s), max(k * 8, 60))
            if kk <= 0:
                return []
            idx = np.argpartition(-s, kk - 1)[:kk]
            idx = idx[np.argsort(-s[idx])]
            out, seen_text = [], set()
            for i in idx.tolist():
                nid = self.fg.nid_list[i]
                n = self.arc.nodes[nid]
                if nid in ex or (n.get("status") == "retracted" and not with_retracted):
                    continue
                if keep is not None and not keep(n):
                    continue
                th = _h(n["text"])
                if th in seen_text:
                    continue
                seen_text.add(th)
                out.append(self.card(nid, score=float(s[i])))
                if len(out) >= k:
                    break
            return out

    # one pair of atoms often carries several typed edges at once (an edited answer is
    # `revises` + `supersedes` + `fixes`); show the pair once, most telling type first
    _TYPE_ORDER = ("supersedes", "revises", "contradicts", "fixes", "refines")

    def card(self, nid: str, score: float | None = None, chars: int = 700) -> dict:
        n = self.arc.nodes[nid]
        pairs = collections.OrderedDict()
        for d, t, other in self.rel.get(nid, ()):
            pairs.setdefault((d, other), set()).add(t)
        rels = []
        for (d, other), ts in pairs.items():
            o = self.arc.nodes.get(other)
            if o is None:
                continue
            types = [t for t in self._TYPE_ORDER if t in ts]
            rels.append({"dir": d, "type": types[0], "types": types, "id": other,
                         "status": self._status(other),
                         "date": _fmt_day(o.get("timestamp")), "text": o["text"][:240]})
        rels.sort(key=lambda r: self._TYPE_ORDER.index(r["type"]))
        newer = self._newest(nid)
        status = self._status(nid)
        return {"id": nid, "score": None if score is None else round(score, 4),
                "status": status, "date": _fmt_day(n.get("timestamp")),
                "file": n.get("file"), "rel_file": self._rel(n.get("file")),
                "branch": n.get("branch"), "question": (n.get("question") or "")[:chars],
                "text": n["text"][:chars], "chars": len(n["text"]), "relations": rels,
                "current_id": newer if newer != nid else None,
                "session_id": n.get("session_id")}

    def _status(self, nid: str) -> str:
        """Status as the product shows it: ptg_core's "superseded" counts only when the
        supersession is real (see SUPERSEDE_MIN_SIM)."""
        n = self.arc.nodes.get(nid) or {}
        st = n.get("status", "active")
        if st == "superseded" and nid not in self.sup_newer:
            return "active"
        return st

    def _newest(self, nid: str) -> str:
        cur, seen = nid, {nid}
        while cur in self.sup_newer:
            nxt = self.sup_newer[cur]
            if nxt in seen:
                break
            seen.add(nxt)
            cur = nxt
        return cur

    def _rel(self, path):
        if not path or path.startswith("memory://"):
            return path
        try:
            r = os.path.relpath(path, self.root)
            return path if r.startswith("..") else r
        except ValueError:
            return path

    def recall(self, prompt: str, session_id: str | None = None, transcript_path: str | None = None,
               budget_chars: int = 3600, max_items: int = 5, min_score: float = 0.5,
               exclude_ids=()) -> dict:
        """The context injected into a prompt. Empty when nothing clears min_score."""
        prompt = (prompt or "").strip()
        if len(prompt) < 12:
            return {"context": "", "items": []}
        # The current session: what the agent said after its last compaction is still in
        # its context window — injecting it again is noise. What came before the
        # compaction is gone from the window, and that is exactly what memory is for.
        session_files = set()
        if transcript_path:
            session_files.add(os.path.normpath(transcript_path))
        if session_id:
            short = session_id[:8]
            for f in list(self.file_index):
                if f and short in os.path.basename(f):
                    session_files.add(f)
        cut = self._compact.last_compact(transcript_path) if transcript_path else None

        def keep(n):
            if self._is_noise(n):
                return False
            if n.get("file") in session_files:
                return cut is not None and self._turn_ts(n) < cut
            return True

        hits = self.search(prompt, k=max_items * 3, exclude_ids=exclude_ids, keep=keep)
        if not hits:
            return {"context": "", "items": []}
        top = hits[0]["score"] or 0.0
        chosen, ids = [], set()
        for h in hits:
            sc = h["score"] or 0
            if sc < min_score or sc < top - 0.12:
                continue
            if h["current_id"]:
                cur_node = self.arc.nodes.get(h["current_id"])
                sim = float(np.dot(cur_node["vec"], self.arc.nodes[h["id"]]["vec"])) if cur_node else 0.0
                if (cur_node is None or cur_node.get("status") == "retracted" or not keep(cur_node)):
                    pass                                 # the "newer" atom is noise: keep the hit as is
                elif sim >= REPLACE_IN_RECALL_SIM:
                    # a confidently superseded hit is shown as its current version, once
                    if h["current_id"] in ids:
                        continue
                    cur = self.card(h["current_id"])
                    cur["score"] = sc
                    cur["replaces"] = {"date": h["date"], "text": h["text"][:200]}
                    h = cur
                else:
                    cur = self.card(h["current_id"], chars=200)
                    h = dict(h, maybe_newer={"date": cur["date"], "text": cur["text"]})
            if h["id"] in ids:
                continue
            ids.add(h["id"])
            chosen.append(h)
            if len(chosen) >= max_items:
                break
        if not chosen:
            return {"context": "", "items": [], "top_score": top}
        lines = ["<ptg-memory>",
                 "Auto-recalled from this project's memory (PTG-MEM). Older notes may be "
                 "outdated — check the date and the markers before relying on them."]
        used = sum(len(x) for x in lines)
        items = []
        per_item = max(300, (budget_chars - used) // max(1, len(chosen)) - 120)
        for i, h in enumerate(chosen, 1):
            body = " ".join(h["text"].split())
            if len(body) > per_item:
                body = body[:per_item].rstrip() + " …"
            entry = ["%d. [%s · %s · %.2f] %s" % (i, h["date"], h["rel_file"], h["score"], body)]
            if h.get("replaces"):
                entry.append("   ↳ this REPLACES an earlier note (%s): %s"
                             % (h["replaces"]["date"], " ".join(h["replaces"]["text"].split())[:160]))
            elif h.get("maybe_newer"):
                entry.append("   ↳ possibly updated later (%s): %s"
                             % (h["maybe_newer"]["date"], " ".join(h["maybe_newer"]["text"].split())[:160]))
            else:
                # the current version of something that was edited or replaced: say what it
                # replaced, so the agent does not fall back to the old answer from habit
                for r in h["relations"]:
                    if r["dir"] == "out" and r["type"] in ("revises", "supersedes"):
                        entry.append("   ↳ replaces an earlier version (%s): %s"
                                     % (r["date"], " ".join(r["text"].split())[:160]))
                        break
            for r in h["relations"]:
                if r["type"] == "contradicts":
                    entry.append("   ↳ %s (%s): %s" % (
                        "contradicts" if r["dir"] == "out" else "contradicted by",
                        r["date"], " ".join(r["text"].split())[:200]))
                    break
            block = "\n".join(entry)
            if used + len(block) > budget_chars:
                break
            lines.append(block)
            used += len(block)
            items.append(h)
        lines.append("</ptg-memory>")
        ctx = "\n".join(lines) if items else ""
        return {"context": ctx, "items": items, "top_score": top}

    def brief(self, max_chars: int = 1600) -> str:
        """SessionStart: what this memory is and what was left open."""
        with ENGINE_LOCK:
            live = sum(1 for n in self.arc.nodes.values() if n.get("status") != "retracted")
            now = time.time()
            recent_open = []
            for b in self.arc.branches.values():
                hid = b.get("head")
                n = self.arc.nodes.get(hid)
                if not n or n.get("status") == "retracted" or now - float(n.get("timestamp") or 0) > 14 * 86400:
                    continue
                if n.get("file", "").startswith(ptg_core.STRUCTURE_FILE_MARKER):
                    continue
                if self.arc._is_unresolved_head(hid):
                    recent_open.append(n)
            recent_open.sort(key=lambda n: -float(n.get("timestamp") or 0))
        lines = ["<ptg-memory>",
                 "PTG-MEM is active for this project: %d memory atoms. Relevant memory is "
                 "attached to each prompt automatically. Tools: ptg_search, ptg_node, "
                 "ptg_decisions, ptg_remember (store a decision the moment it is made)." % live]
        if recent_open:
            lines.append("Open threads from the last two weeks:")
            for n in recent_open[:5]:
                q = " ".join((n.get("question") or n["text"]).split())[:160]
                lines.append("- [%s · %s] %s" % (_fmt_day(n.get("timestamp")), self._rel(n.get("file")), q))
        lines.append("</ptg-memory>")
        text = "\n".join(lines)
        return text[:max_chars]

    def decisions(self, limit: int = 50, primary=("supersedes", "revises", "contradicts")) -> list:
        """Pairs where something was replaced, edited or contradicted. `fixes` and
        `refines` come from keyword heuristics (8 308 `fixes` on the reference archive:
        any paragraph that says "bug" next to a similar one) — shown only as extra
        badges on a pair that already qualifies, never on their own."""
        with ENGINE_LOCK:
            pairs = collections.OrderedDict()
            for e in self.arc.edges:
                if e["type"] not in TYPED:
                    continue
                if e["type"] == "supersedes" and self.sup_newer.get(e["to"]) != e["from"]:
                    continue                             # weak marker match, see SUPERSEDE_MIN_SIM
                pairs.setdefault((e["from"], e["to"]), set()).add(e["type"])
            out = []
            for (fa, fb), ts in pairs.items():
                if not ts.intersection(primary):
                    continue
                a, b = self.arc.nodes.get(fa), self.arc.nodes.get(fb)
                if not a or not b:
                    continue
                tl = [t for t in self._TYPE_ORDER if t in ts]
                out.append({"type": tl[0], "types": tl, "date": _fmt_day(a.get("timestamp")),
                            "ts": float(a.get("timestamp") or 0),
                            "new": {"id": a["id"], "text": a["text"][:280], "file": self._rel(a.get("file")),
                                    "status": self._status(fa)},
                            "old": {"id": b["id"], "text": b["text"][:280], "file": self._rel(b.get("file")),
                                    "status": self._status(fb), "date": _fmt_day(b.get("timestamp"))}})
            out.sort(key=lambda x: -x["ts"])
            return out[:limit]

    def node(self, nid: str) -> dict | None:
        with ENGINE_LOCK:
            n = self.arc.nodes.get(nid)
            if n is None:
                return None
            c = self.card(nid, chars=20000)
            c["answer"] = n.get("answer", "")
            b = self.arc.branch_bodies.get(n.get("branch"), {})
            ids = b.get("atom_ids", [])
            pos = ids.index(nid) if nid in ids else -1
            c["lineage"] = [{"id": x, "date": _fmt_day(self.arc.nodes[x].get("timestamp")),
                             "status": self._status(x),
                             "text": self.arc.nodes[x]["text"][:160]}
                            for x in ids[max(0, pos - 4):pos + 5] if x in self.arc.nodes]
            return c

    def graph(self, nid: str, depth: int = 2, limit: int = 60) -> dict:
        """Ego network for the GUI: typed edges first, then structural ones."""
        with ENGINE_LOCK:
            if nid not in self.arc.nodes:
                return {"nodes": [], "edges": []}
            adj = collections.defaultdict(list)
            want = {nid}
            frontier = {nid}
            for _ in range(max(1, depth)):
                nxt = set()
                for e in self.arc.edges:
                    a, b = e["from"], e["to"]
                    if a in frontier or b in frontier:
                        adj[a].append(e)
                        adj[b].append(e)
                        for x in (a, b):
                            if x not in want and len(want) < limit:
                                want.add(x)
                                nxt.add(x)
                frontier = nxt
                if not frontier:
                    break
            edges, seen = [], set()
            for x in want:
                for e in adj.get(x, ()):
                    key = (e["from"], e["to"], e["type"])
                    if e["from"] in want and e["to"] in want and key not in seen:
                        seen.add(key)
                        edges.append({"from": e["from"], "to": e["to"], "type": e["type"]})
            nodes = []
            for x in want:
                n = self.arc.nodes[x]
                st = self._status(x)
                nodes.append({"id": x, "label": " ".join((n.get("question") or n["text"]).split())[:60],
                              "status": st, "date": _fmt_day(n.get("timestamp")),
                              "file": self._rel(n.get("file")), "center": x == nid})
            return {"nodes": nodes, "edges": edges}

    def stats(self) -> dict:
        with ENGINE_LOCK:
            arc = self.arc
            st = collections.Counter(n.get("status", "active") for n in arc.nodes.values())
            n = max(1, len(arc.nodes))
            deg = 2.0 * len(arc.edges) / n
            return {"root": self.root, "store": self.store_dir, "atoms": len(arc.nodes),
                    "live": st.get("active", 0) + st.get("superseded", 0),
                    "retracted": st.get("retracted", 0), "superseded": st.get("superseded", 0),
                    "branches": len(arc.branches), "edges": len(arc.edges),
                    "files": len(arc.processed_files), "mean_degree": round(deg, 2),
                    "budget_d": budget_graft.budget_for(len(arc.id_order), float(self.p.get("margin", 5.0))),
                    "dirty": self.dirty, "last_save": self.last_save, "last_change": self.last_change,
                    "loaded_at": self.loaded_at, "embedder": self.embedder.status()}


class _Snapshot:
    """A consistent copy of what store.save reads, taken under the engine lock so
    the slow part (serialising) runs without blocking recall."""

    def __init__(self, arc):
        self.id_order = list(arc.id_order)
        self.nodes = {nid: dict(arc.nodes[nid]) for nid in self.id_order}
        self.edges = list(arc.edges)
        self.branches = {k: dict(v) for k, v in arc.branches.items()}
        self.branch_states = {k: dict(v) for k, v in arc.branch_states.items()}
        self.branch_bodies = {k: dict(v, atom_ids=list(v.get("atom_ids", ())))
                              for k, v in arc.branch_bodies.items()}
        self.files = {k: dict(v) for k, v in arc.files.items()}
        self.file_edges = list(arc.file_edges)
        self.root_project = dict(arc.root_project or {})
        self.root_embedding = arc.root_embedding
        self.processed_files = dict(arc.processed_files)
        self.last_structure_hash = arc.last_structure_hash
        self.path_memory = {"short": list(arc.path_memory.get("short", [])),
                            "long": list(arc.path_memory.get("long", []))}
        self._atoms_since_decay = arc._atoms_since_decay
        self._prev_branch_id = arc._prev_branch_id

        class _E:
            pass
        e = _E()
        e.backend = arc.embedder.backend
        e.model = arc.embedder.model
        e.dim = arc.embedder.dim
        self.embedder = e
