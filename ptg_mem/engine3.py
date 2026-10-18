# -*- coding: utf-8 -*-
"""MemoryEngine3 — PTG-MEM 0.3: the same interface the daemon uses (``engine.MemoryEngine``), rebuilt on ready
parts and on two packages written through the rptg kit.

* decisions — ``memcore`` (PTG-MEM/rev): what a re-read file keeps, retracts and adds; how to search; what goes into
  the prompt; how a relation is recorded (Graphiti-style: a new claim closes the old edge with a date, nothing is
  deleted); secret scrubbing (detect-secrets + vendor formats).
* the text path — ``memshell`` (PTG-MEM/rev-shell): file → Markdown (MarkItDown) → paragraph atoms (Chonkie only for
  long paragraphs); the recall window, the current-session rule, harness noise; the ``<ptg-memory>`` block and the
  session brief, strictly inside their budgets; the decision feed.
* storage — LanceDB: one table of atoms with a vector column, one of relations, one of processed files.

What stays our own: the transcript parsers (``transcripts``), the file walk (``filters.FileFilter``), the daemon,
hooks and MCP. This module is the stateful glue the kit cannot check card by card (several calls on one store);
its behaviour is pinned by ``tests/test_engine3.py``.

The libraries are imported lazily, inside the engine: the 0.2 daemon on the system Python imports ``ptg_mem`` and
must not need them.
"""
from __future__ import annotations

import collections
import os
import threading
import time
import uuid

import numpy as np

from . import transcripts
from .filters import FileFilter

LOCK = threading.RLock()
NOTES_FILE = "memory://notes"
LIMIT = 1200                    # paragraph atoms; longer paragraphs are cut by Chonkie (bench/chunk_locality.py)
WINDOW = 0.12                   # recall: hits further than this from the best one are dropped (PTG-MEM 0.2)
TYPE_ORDER = ("supersedes", "contradicts", "fixes", "refines", "continues", "returns_to", "reinforces")


def _day(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError):
        return "?"


class MemoryEngine3:
    def __init__(self, project: dict, embedder, log=None):
        self.p = project
        self.root = project["root"]
        self.store_dir = project["store"]
        self.embedder = embedder
        self.log = log or (lambda m: None)
        self.filter = FileFilter(self.root, exclude_dirs=project.get("exclude_dirs", ()),
                                 include_exts=project.get("include_exts"), store_dir=self.store_dir)
        self.atoms: dict[str, dict] = {}                 # id → row without the vector
        self.file_index: dict[str, list[str]] = collections.defaultdict(list)
        self.edges: list = []                            # memcore.types.Edge
        self.processed: dict[str, float] = {}
        self.events = collections.deque(maxlen=400)
        self.dirty = False
        self.last_change = self.last_save = self.loaded_at = 0.0
        self.db = self.t_atoms = None
        self.embedder_failed = False
        self._compact = transcripts.CompactTracker()

    # ------------------------------------------------------------------ store
    def open(self):
        import lancedb
        with LOCK:
            t0 = time.time()
            os.makedirs(self.store_dir, exist_ok=True)
            self.db = lancedb.connect(os.path.join(self.store_dir, "lance"))
            names = set(self.db.list_tables().tables) if hasattr(self.db.list_tables(), "tables") else set(self.db.list_tables())
            meta = self._read("meta") if "meta" in names else {}
            model = meta.get("model")
            if model and model != str(self.embedder.model) and not self.p.get("embedder_alias_ok"):
                raise RuntimeError("this store was built with %r, the project is configured for %r; vectors from "
                                   "different models are not comparable — use the same model or a new store"
                                   % (model, self.embedder.model))
            if "atoms" in names:
                self.t_atoms = self.db.open_table("atoms")
                for r in self.t_atoms.search().select([c for c in self._atom_columns()]).limit(None).to_list():
                    self.atoms[r["id"]] = r
                    self.file_index[r["file"]].append(r["id"])
                for ids in self.file_index.values():
                    ids.sort(key=lambda i: (self.atoms[i]["position"], self.atoms[i]["timestamp"]))
            if "edges" in names:
                from memcore.types import Edge
                self.edges = [Edge(r["src"], r["dst"], r["kind"], r["valid_at"], r["invalid_at"], r["source"])
                              for r in self.db.open_table("edges").search().limit(None).to_list()]
                self.edges.sort(key=lambda e: e.valid_at)
            if "files" in names:
                self.processed = {r["path"]: r["mtime"] for r in self.db.open_table("files").search().limit(None).to_list()}
            self.loaded_at = self.last_save = time.time()
            self.log("opened %s (v3): %d atoms, %d edges in %.1f s" % (self.root, len(self.atoms), len(self.edges), time.time() - t0))
            return self

    @staticmethod
    def _atom_columns():
        return ("id", "file", "position", "digest", "text", "question", "answer", "timestamp", "status", "retracted_at",
                "session_id", "source")

    def _read(self, name) -> dict:
        return {r["key"]: r["value"] for r in self.db.open_table(name).search().limit(None).to_list()}

    def _put_table(self, name, rows, schema=None):
        """Small tables (meta, files, edges) are rewritten whole: a few thousand rows at most."""
        import pyarrow as pa
        if not rows and schema is None:
            return
        data = pa.Table.from_pylist(rows, schema=schema) if schema is not None else rows
        self.db.create_table(name, data=data, mode="overwrite")

    def _save_meta(self):
        import pyarrow as pa
        schema = pa.schema([pa.field("key", pa.string()), pa.field("value", pa.string())])
        self._put_table("meta", [{"key": "model", "value": str(self.embedder.model)}, {"key": "format", "value": "ptg-mem-0.3"}], schema)

    def _save_edges(self):
        import pyarrow as pa
        schema = pa.schema([pa.field("src", pa.string()), pa.field("dst", pa.string()), pa.field("kind", pa.string()),
                            pa.field("valid_at", pa.float64()), pa.field("invalid_at", pa.float64()), pa.field("source", pa.string())])
        self._put_table("edges", [e.__dict__ for e in self.edges], schema)

    def _save_files(self):
        import pyarrow as pa
        schema = pa.schema([pa.field("path", pa.string()), pa.field("mtime", pa.float64())])
        self._put_table("files", [{"path": k, "mtime": float(v)} for k, v in self.processed.items()], schema)

    def _add_rows(self, rows, vecs):
        import pyarrow as pa
        dim = int(vecs.shape[1])
        data = [{**r, "vector": v.astype("float32").tolist()} for r, v in zip(rows, vecs)]
        if self.t_atoms is None:
            schema = pa.schema([pa.field("id", pa.string()), pa.field("file", pa.string()), pa.field("position", pa.int64()),
                                pa.field("digest", pa.string()), pa.field("text", pa.string()), pa.field("question", pa.string()),
                                pa.field("answer", pa.string()), pa.field("timestamp", pa.float64()), pa.field("status", pa.string()),
                                pa.field("retracted_at", pa.float64()), pa.field("session_id", pa.string()), pa.field("source", pa.string()),
                                pa.field("vector", pa.list_(pa.float32(), dim))])
            self.t_atoms = self.db.create_table("atoms", data=pa.Table.from_pylist(data, schema=schema))
            self._save_meta()
        else:
            self.t_atoms.add(data)
        for r in rows:
            self.atoms[r["id"]] = r
            self.file_index[r["file"]].append(r["id"])

    def _retract(self, ids, at):
        if not ids:
            return
        quoted = ", ".join("'%s'" % i.replace("'", "''") for i in ids)
        self.t_atoms.update(where="id IN (%s)" % quoted, values={"status": "retracted", "retracted_at": float(at)})
        for i in ids:
            self.atoms[i]["status"] = "retracted"
            self.atoms[i]["retracted_at"] = float(at)

    # ------------------------------------------------------------------ ingest
    def _event(self, **kw):
        kw.setdefault("t", time.time())
        self.events.append(kw)

    def _embed(self, texts):
        out = [self.embedder.embed([t])[0] for t in texts]   # one text per call: a recall waits for one text at most
        return np.asarray(out, dtype="float32").reshape(len(out), -1)

    def parse(self, path: str, kind: str) -> list[dict] | None:
        """Atoms of a source as dicts (text, question, answer, order, timestamp, session_id, source), or None if gone."""
        import memcore
        import memshell
        if not os.path.exists(path):
            return None
        if kind == "claude":
            atoms = transcripts.parse_claude(path)
        elif kind == "codex":
            atoms = transcripts.parse_codex(path)
        else:
            try:
                text = memshell.convert(path)
            except FileNotFoundError:
                return None
            ts = os.path.getmtime(path)
            atoms = [{"order": d.position, "text": d.text, "question": "", "answer": d.text, "timestamp": ts, "session_id": None,
                      "source": "file"} for d in memshell.atomize(path, text, LIMIT)]
        removed = 0
        for a in atoms:
            for k in ("text", "question", "answer"):
                if a.get(k):
                    s = memcore.scrub(a[k])
                    a[k] = s.text
                    removed += s.found
        if removed:
            self._event(kind="secrets", file=path, removed=removed)
        return atoms

    def ingest(self, path: str, kind: str = "file") -> dict:
        """Bring one source up to date: memcore.plan_sync decides, only new text is embedded, an atom written in the
        place of a retracted one supersedes it (source "edit")."""
        import memcore
        import memshell
        from memcore.types import Atom, Claim, Stored
        t0 = time.time()
        path = os.path.normpath(path)
        atoms = self.parse(path, kind)
        if atoms is None:
            return self.retract_file(path, reason="deleted")
        with LOCK:
            stored = [Stored(i, path, self.atoms[i]["position"], self.atoms[i]["digest"], self.atoms[i]["status"] != "retracted")
                      for i in self.file_index.get(path, ())]
            fresh, by_id = [], {}
            for pos, a in enumerate(atoms):
                aid = str(uuid.uuid4())
                fresh.append(Atom(aid, path, pos, memshell.digest(a["text"])))
                by_id[aid] = a
            plan = memcore.plan_sync(stored, fresh)
        if not plan.add and not plan.retract:
            self._mark(path)
            return {"file": path, "added": 0, "retracted": 0, "kept": len(plan.keep)}
        vecs = self._embed([by_id[i]["text"] for i in plan.add]) if plan.add else None
        now = time.time()
        with LOCK:
            self._retract(list(plan.retract), now)
            if plan.add:
                pos = {a.id: a.position for a in fresh}
                rows = []
                for i in plan.add:
                    a = by_id[i]
                    rows.append({"id": i, "file": path, "position": pos[i], "digest": memshell.digest(a["text"]), "text": a["text"],
                                 "question": a.get("question") or "", "answer": a.get("answer") or "",
                                 "timestamp": float(a.get("timestamp") or now), "status": "active", "retracted_at": 0.0,
                                 "session_id": a.get("session_id") or "", "source": a.get("source") or kind})
                self._add_rows(rows, vecs)
            for new, old in plan.revises:
                self.edges = memcore.record_relation(self.edges, Claim(new, old, "supersedes", now, "edit"))
            if plan.revises:
                self._save_edges()
            self._mark(path)
            self.dirty, self.last_change = True, now
        r = {"file": path, "src": kind, "added": len(plan.add), "retracted": len(plan.retract), "kept": len(plan.keep),
             "seconds": round(time.time() - t0, 2)}
        self._event(kind="ingest", **r)
        return r

    def _mark(self, path):
        try:
            self.processed[path] = os.path.getmtime(path)
        except OSError:
            return
        self._save_files()

    def retract_file(self, path: str, reason: str = "deleted") -> dict:
        path = os.path.normpath(path)
        with LOCK:
            live = [i for i in self.file_index.get(path, ()) if self.atoms[i]["status"] != "retracted"]
            self._retract(live, time.time())
            if self.processed.pop(path, None) is not None:
                self._save_files()
            if live:
                self.dirty, self.last_change = True, time.time()
        if live:
            self._event(kind="retract", file=path, retracted=len(live), reason=reason)
        return {"file": path, "added": 0, "retracted": len(live), "kept": 0}

    def remember(self, text: str, session_id: str | None = None, source: str = "agent") -> dict:
        """An explicit note or decision, stored immediately."""
        import memcore
        import memshell
        text = memcore.scrub(text.strip()).text
        if len(text) < 8:
            raise ValueError("note too short")
        vecs = self._embed([text])
        now = time.time()
        with LOCK:
            aid = str(uuid.uuid4())
            row = {"id": aid, "file": NOTES_FILE, "position": len(self.file_index.get(NOTES_FILE, ())), "digest": memshell.digest(text),
                   "text": text, "question": "Note (%s)" % source, "answer": text, "timestamp": now, "status": "active",
                   "retracted_at": 0.0, "session_id": session_id or "", "source": source}
            self._add_rows([row], vecs)
            self.dirty, self.last_change = True, now
        self._event(kind="remember", id=aid, text=text[:200])
        return self.card(aid)

    def relate(self, src: str, dst: str, kind: str, source: str = "agent") -> list:
        """A relation claimed by an agent or a model (never guessed from a cosine threshold)."""
        import memcore
        from memcore.types import Claim
        with LOCK:
            if src not in self.atoms or dst not in self.atoms:
                raise KeyError("unknown atom")
            self.edges = memcore.record_relation(self.edges, Claim(src, dst, kind, time.time(), source))
            self._save_edges()
            self.dirty = True
            return [e for e in self.edges if {e.src, e.dst} == {src, dst}]

    # ------------------------------------------------------------------ plan / save
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
                if self.processed.get(path) != mt:
                    todo.append((path, "file"))
        if include_transcripts:
            for key, lister in (("claude", transcripts.claude_transcripts), ("codex", transcripts.codex_transcripts)):
                if src.get(key, True):
                    for path in lister(self.root):
                        seen.add(path)
                        if self.processed.get(path) != os.path.getmtime(path):
                            todo.append((path, key))
        gone = [p for p in list(self.processed) if p not in seen and not p.startswith("memory://") and not os.path.exists(p)]
        return todo, gone

    def save(self) -> dict:
        """Writes go to LanceDB as they happen; saving compacts the atom table's fragments."""
        with LOCK:
            if self.t_atoms is not None:
                self.t_atoms.optimize()
            self.dirty = False
            self.last_save = time.time()
        r = {"atoms": len(self.atoms), "edges": len(self.edges), "files": len(self.processed)}
        self._event(kind="save", **r)
        return r

    # ------------------------------------------------------------------ read
    def _successor(self, nid: str) -> str:
        import memcore
        cur = memcore.current_version(nid, self.edges)
        return "" if cur == nid else cur

    def _rel(self, path):
        if not path or path.startswith("memory://"):
            return path
        try:
            r = os.path.relpath(path, self.root)
            return path if r.startswith("..") else r
        except ValueError:
            return path

    def _status(self, nid: str) -> str:
        row = self.atoms.get(nid) or {}
        if row.get("status") == "retracted":
            return "retracted"
        return "superseded" if self._successor(nid) else "active"

    def search(self, query: str, k: int = 8, exclude_files=(), exclude_ids=(), with_retracted: bool = False, keep=None) -> list[dict]:
        """memcore.choose_search picks the way: the project's vector column while the embedder works, words (LanceDB
        full-text index) when it does not."""
        import memcore
        from memcore.types import Engine
        if self.t_atoms is None:
            return []
        model = str(self.embedder.model)
        plan = memcore.choose_search([Engine(model, not self.embedder_failed, len(self.atoms), ())], model)
        where = None if with_retracted else "status = 'active'"
        n = max(k * 8, 60)
        rows = []
        if plan.mode != "lexical":
            try:
                qv = self._embed([query])[0]
                q = self.t_atoms.search(qv).metric("cosine")
                q = q.where(where, prefilter=True) if where else q
                rows = [(r["id"], 1.0 - float(r["_distance"])) for r in q.select(["id"]).limit(n).to_list()]
            except Exception as ex:                       # noqa: BLE001 — embedder down: fall back to words
                self.embedder_failed = True
                self._event(kind="embedder", error=str(ex)[:200])
                plan = memcore.choose_search([Engine(model, False, len(self.atoms), ())], model)
        if plan.mode == "lexical":
            rows = self._lexical(query, n, where)
        ex = set(exclude_ids)
        files = set(exclude_files)
        out, seen = [], set()
        for nid, score in rows:
            a = self.atoms.get(nid)
            if a is None or nid in ex or a["file"] in files or (keep is not None and not keep(a)):
                continue
            if a["digest"] in seen:
                continue
            seen.add(a["digest"])
            out.append(self.card(nid, score=score))
            if len(out) >= k:
                break
        return out

    def _lexical(self, query, n, where):
        from lancedb.index import FTS
        try:
            self.t_atoms.create_index("text", config=FTS(), replace=True)
            q = self.t_atoms.search(query, query_type="fts")
            q = q.where(where, prefilter=True) if where else q
            hits = q.select(["id"]).limit(n).to_list()
        except Exception:                                 # noqa: BLE001 — no index or no words: nothing found
            return []
        top = max((float(r["_score"]) for r in hits), default=1.0) or 1.0
        return [(r["id"], float(r["_score"]) / top) for r in hits]

    def card(self, nid: str, score: float | None = None, chars: int = 700) -> dict:
        a = self.atoms[nid]
        rels = []
        for e in self.edges:
            if nid not in (e.src, e.dst) or e.invalid_at != 0.0:
                continue
            other = e.dst if e.src == nid else e.src
            o = self.atoms.get(other)
            if o is None:
                continue
            rels.append({"dir": "out" if e.src == nid else "in", "type": e.kind, "types": [e.kind], "id": other, "source": e.source,
                         "status": self._status(other), "date": _day(o["timestamp"]), "text": o["text"][:240]})
        rels.sort(key=lambda r: TYPE_ORDER.index(r["type"]) if r["type"] in TYPE_ORDER else 99)
        succ = self._successor(nid)
        return {"id": nid, "score": None if score is None else round(score, 4), "status": self._status(nid),
                "date": _day(a["timestamp"]), "file": a["file"], "rel_file": self._rel(a["file"]), "branch": None,
                "question": (a.get("question") or "")[:chars], "text": a["text"][:chars], "chars": len(a["text"]),
                "relations": rels, "current_id": succ or None, "session_id": a.get("session_id") or None}

    def _turn_at(self, a) -> float:
        return float(a.get("timestamp") or 0.0)

    def recall(self, prompt: str, session_id: str | None = None, transcript_path: str | None = None,
               budget_chars: int = 3600, max_items: int = 5, min_score: float = 0.5, exclude_ids=()) -> dict:
        """The context injected into a prompt. Empty when nothing clears min_score."""
        import memcore
        import memshell
        from memcore.types import Budget, Hit
        from memshell.types import AUTOMATED, Entry
        prompt = (prompt or "").strip()
        if len(prompt) < 12:
            return {"context": "", "items": []}
        session = set()
        if transcript_path:
            session.add(os.path.normpath(transcript_path))
        if session_id:
            short = session_id[:8]
            session |= {f for f in self.file_index if f and short in os.path.basename(f)}
        cut = (self._compact.last_compact(transcript_path) if transcript_path else None) or 0.0
        files = tuple(sorted(session))

        def keep(a):
            head = (a.get("question") or a.get("text") or "")[:200]
            return not memshell.is_noise(head, AUTOMATED) and memshell.session_ok(a["file"], self._turn_at(a), files, float(cut))

        cards = self.search(prompt, k=max_items * 3, exclude_ids=exclude_ids, keep=keep)
        if not cards:
            return {"context": "", "items": []}
        top = cards[0]["score"] or 0.0
        by_id = {c["id"]: c for c in cards}
        hits = memshell.in_window([Hit(c["id"], c["file"], c["text"], float(c["score"] or 0.0), c["current_id"] or "") for c in cards],
                                  min_score, WINDOW)
        chosen = memcore.select_context(hits, Budget(10 ** 9, max_items, min_score, max_items))
        entries, items = [], []
        for it in chosen:
            c = by_id[it.id]
            notes = []
            if c["current_id"] and c["current_id"] in self.atoms:
                n = self.atoms[c["current_id"]]
                notes.append("replaced later (%s): %s" % (_day(n["timestamp"]), n["text"][:160]))
            for r in c["relations"]:
                if r["dir"] == "out" and r["type"] == "supersedes" and not c["current_id"]:
                    notes.append("replaces an earlier version (%s): %s" % (r["date"], r["text"][:160]))
                    break
            for r in c["relations"]:
                if r["type"] == "contradicts":
                    notes.append("%s (%s): %s" % ("contradicts" if r["dir"] == "out" else "contradicted by", r["date"], r["text"][:200]))
                    break
            entries.append(Entry(c["date"], c["rel_file"], float(c["score"] or 0.0), c["text"], tuple(notes)))
            items.append(c)
        ctx = memshell.render_recall(entries, budget_chars)
        shown = sum(1 for i in range(1, len(entries) + 1) if "\n%d. [" % i in ctx)   # render_recall stops at the budget
        return {"context": ctx, "items": items[:shown], "top_score": top}

    def brief(self, max_chars: int = 1600) -> str:
        """SessionStart: how much is in memory and the latest notes and decisions (two weeks)."""
        import memshell
        from memshell.types import Entry
        with LOCK:
            live = sum(1 for a in self.atoms.values() if a["status"] != "retracted")
            since = time.time() - 14 * 86400
            notes = sorted((self.atoms[i] for i in self.file_index.get(NOTES_FILE, ())
                            if self.atoms[i]["status"] != "retracted" and self.atoms[i]["timestamp"] >= since),
                           key=lambda a: -a["timestamp"])[:5]
        return memshell.render_brief(live, [Entry(_day(a["timestamp"]), a["file"], 0.0, a["text"], ()) for a in notes], max_chars)

    def decisions(self, limit: int = 50, primary=None) -> list:
        """What replaced what and what contradicts what — newest first, closed relations too."""
        import memshell
        out = []
        for e in memshell.decision_pairs(self.edges, limit):
            a, b = self.atoms.get(e.src), self.atoms.get(e.dst)
            if not a or not b:
                continue
            out.append({"type": e.kind, "types": [e.kind], "date": _day(e.valid_at), "ts": e.valid_at, "source": e.source,
                        "closed": e.invalid_at != 0.0,
                        "new": {"id": a["id"], "text": a["text"][:280], "file": self._rel(a["file"]), "status": self._status(a["id"])},
                        "old": {"id": b["id"], "text": b["text"][:280], "file": self._rel(b["file"]), "status": self._status(b["id"]),
                                "date": _day(b["timestamp"])}})
        return out

    def node(self, nid: str) -> dict | None:
        with LOCK:
            a = self.atoms.get(nid)
            if a is None:
                return None
            c = self.card(nid, chars=20000)
            c["answer"] = a.get("answer", "")
            ids = self.file_index.get(a["file"], [])
            pos = ids.index(nid) if nid in ids else -1
            c["lineage"] = [{"id": x, "date": _day(self.atoms[x]["timestamp"]), "status": self._status(x), "text": self.atoms[x]["text"][:160]}
                            for x in ids[max(0, pos - 4):pos + 5]]
            return c

    def graph(self, nid: str, depth: int = 2, limit: int = 60) -> dict:
        """Ego network over the relations, for the GUI."""
        with LOCK:
            if nid not in self.atoms:
                return {"nodes": [], "edges": []}
            want, frontier = {nid}, {nid}
            for _ in range(max(1, depth)):
                nxt = set()
                for e in self.edges:
                    for x, y in ((e.src, e.dst), (e.dst, e.src)):
                        if x in frontier and y not in want and len(want) < limit and y in self.atoms:
                            want.add(y)
                            nxt.add(y)
                frontier = nxt
                if not frontier:
                    break
            edges = [{"from": e.src, "to": e.dst, "type": e.kind} for e in self.edges if e.src in want and e.dst in want]
            nodes = [{"id": x, "label": " ".join((self.atoms[x].get("question") or self.atoms[x]["text"]).split())[:60],
                      "status": self._status(x), "date": _day(self.atoms[x]["timestamp"]), "file": self._rel(self.atoms[x]["file"]),
                      "center": x == nid} for x in want]
            return {"nodes": nodes, "edges": edges}

    def stats(self) -> dict:
        with LOCK:
            st = collections.Counter(self._status(i) for i in self.atoms)
            n = max(1, len(self.atoms))
            return {"root": self.root, "store": self.store_dir, "atoms": len(self.atoms),
                    "live": st.get("active", 0) + st.get("superseded", 0), "retracted": st.get("retracted", 0),
                    "superseded": st.get("superseded", 0), "branches": 0, "edges": len(self.edges), "files": len(self.processed),
                    "mean_degree": round(2.0 * len(self.edges) / n, 2), "dirty": self.dirty, "last_save": self.last_save,
                    "last_change": self.last_change, "loaded_at": self.loaded_at, "engine": "v3", "embedder": self.embedder.status()}
