# -*- coding: utf-8 -*-
"""MemoryEngineFull — the full PTG behind the daemon's engine interface (``engine.MemoryEngine``): text, Claude Code
and Codex sessions, chat exports, agent notes, relations with validity dates, RRF search over words and embedding
spaces, the prompt block and the session brief, the decision feed, the code graph (codebase-memory-mcp) and the
metatree, one LanceDB store.

All the decisions live in ``ptgcore`` (PTG-MEM/rev-core), written through the rptg kit card by card. This class is
the glue the kit cannot check: the daemon's embedder (vectors for new atoms, notes and queries), the store's space
name, one writer at a time, and the dict shapes the daemon, hooks, MCP and the window expect (as in ``engine3``).

The libraries are imported lazily: the daemon on the system Python imports ``ptg_mem`` and must not need them.
"""
from __future__ import annotations

import collections
import dataclasses
import os
import threading
import time
import types

from .filters import FileFilter

LOCK = threading.RLock()
CODE_EXTS = (".py", ".pyi", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu")


def _link(x) -> dict:
    return {"dir": x.dir, "type": x.kind, "types": [x.kind], "id": x.id, "source": x.source, "status": x.status,
            "date": x.date, "text": x.text}


def _card(c, chars: int | None = None) -> dict:
    return {"id": c.id, "score": None if c.score is None else round(float(c.score), 4), "status": c.status, "date": c.date,
            "file": c.file, "rel_file": c.rel_file, "branch": None, "question": c.question[:chars] if chars else c.question,
            "text": c.text[:chars] if chars else c.text, "chars": c.chars, "relations": [_link(x) for x in c.links],
            "current_id": c.current or None, "session_id": c.session or None}


def _side(s, with_date: bool) -> dict:
    d = {"id": s.id, "text": s.text, "file": s.file, "status": s.status}
    if with_date:
        d["date"] = s.date
    return d


def _report(r, **extra) -> dict:
    return {"file": r.file, "added": r.added, "retracted": r.retracted, "kept": r.kept, **extra}


class MemoryEngineFull:
    def __init__(self, project: dict, embedder, log=None):
        if not os.environ.get("PTG_MEM_CONTRACTS"):
            # the conditions are checked by the kit's tests; checked on every call in the daemon they are nearly all
            # of its time (14.10, 27 816 atoms: recall 316 s with them, 3.2 s without). PTG_MEM_CONTRACTS=1 keeps them.
            import deal
            deal.disable()
        self.p = project
        self.root = project["root"]
        self.store_dir = project["store"]
        self.home = os.path.join(self.store_dir, "full")
        self.embedder = embedder
        self.log = log or (lambda m: None)
        self.filter = FileFilter(self.root, exclude_dirs=project.get("exclude_dirs", ()),
                                 include_exts=project.get("include_exts"), store_dir=self.store_dir)
        self.space = ""
        self.events = collections.deque(maxlen=400)
        self.dirty = False
        self.last_change = self.last_save = self.loaded_at = 0.0
        self.embedder_failed = False
        self.code_dirty = False

    # ------------------------------------------------------------------ store
    def _spec(self) -> str:
        """The store's space for the project's embedder: «name=provider:model@dim». The name is the project's
        ``space`` (the migrated store says which), else ``main``; the dimension is the store's, or the model's."""
        import ptgfull.store
        name = self.p.get("space") or "main"
        provider = getattr(self.embedder, "backend", "hash")
        model = str(self.embedder.model)
        if provider == "gguf":
            model = os.path.splitext(os.path.basename(model))[0]
        known = {s.name: s for s in ptgfull.store.read_spaces(self.home)}
        if name in known:
            s = known[name]
            if (s.provider, s.model) != (provider, model) and not self.p.get("embedder_alias_ok"):
                raise RuntimeError("this store was built with %r, the project is configured for %r; vectors from "
                                   "different models are not comparable — use the same model or a new store"
                                   % ("%s:%s" % (s.provider, s.model), "%s:%s" % (provider, model)))
            return "%s=%s:%s@%d" % (name, s.provider, s.model, s.dim)
        dim = self.embedder.dim or int(self.embedder.embed(["dimension probe"]).shape[1])
        return "%s=%s:%s@%d" % (name, provider, model, dim)

    def open(self):
        import ptgcore
        with LOCK:
            t0 = time.time()
            os.makedirs(self.home, exist_ok=True)
            spec = self._spec()
            try:
                st = ptgcore.open_home(self.home, spec)
            except ValueError as ex:
                raise RuntimeError("%s — vectors from different models are not comparable" % ex) from ex
            self.space = spec.split("=", 1)[0]
            self.loaded_at = self.last_save = time.time()
            self.log("opened %s (full): %d atoms, %d relations, %d code symbols, %d metatree edges in %.1f s"
                     % (self.root, st.atoms, st.relations, st.code, st.edges, time.time() - t0))
            return self

    @property
    def edges(self) -> list:
        """Relations (memcore.types.Edge), by the time they were recorded."""
        import ptgcore.relations
        return list(ptgcore.relations.claims(self.home))

    @property
    def arc(self):
        """The daemon asks ``arc.processed_files`` whether a vanished path was ingested."""
        import ptgcore.ledger
        return types.SimpleNamespace(processed_files=dict(ptgcore.ledger.processed(self.home)))

    # ------------------------------------------------------------------ vectors
    def _embed(self, texts) -> tuple:
        """One text per call: a recall waits for one text at most."""
        return tuple(tuple(float(x) for x in self.embedder.embed([t])[0]) for t in texts)

    def _query(self, text: str) -> tuple:
        """Query vectors for the store's routes; the embedder down — words only (memcore P1)."""
        import ptgcore
        if not ptgcore.query_routes(self.home, (self.space,)):
            return ()
        try:
            v = self._embed([text])[0]
        except Exception as ex:                          # noqa: BLE001 — GPU busy, model gone: search by words
            self.embedder_failed = True
            self._event(kind="embedder", error=str(ex)[:200])
            return ()
        self.embedder_failed = False
        return ((self.space, v),)

    def _event(self, **kw):
        kw.setdefault("t", time.time())
        self.events.append(kw)

    def _settings(self):
        from ptgcore.types import Settings
        src = self.p.get("sources") or {}
        return Settings(self.root, self.store_dir, bool(src.get("folder", True)), bool(src.get("claude", True)),
                        bool(src.get("codex", True)), tuple(self.p.get("exclude_dirs") or ()),
                        tuple(self.p.get("include_exts") or ()), self.p.get("py_comment_lang", "en"))

    def _touched(self):
        self.dirty, self.last_change = True, time.time()

    # ------------------------------------------------------------------ ingest
    def ingest(self, path: str, kind: str = "file") -> dict:
        """Bring one source up to date: ptgcore.prepare decides, only new atoms are embedded. An embedder failure
        raises — the source stays unmarked and the next catch-up tries again, so no atom is left without a vector."""
        import ptgcore
        from ptgcore.types import Source
        t0 = time.time()
        path = os.path.normpath(path)
        batch = ptgcore.prepare(self.home, Source(path, kind), self.p.get("py_comment_lang", "en"))
        vectors = self._embed([e.text for e in batch.fresh]) if batch.fresh else ()
        with LOCK:
            r = ptgcore.commit(self.home, batch, self.space, vectors, time.time())
        if batch.secrets:
            self._event(kind="secrets", file=path, removed=batch.secrets)
        if r.added or r.retracted:
            self._touched()
            if kind == "file" and path.lower().endswith(CODE_EXTS) and self._in_code_repo(path):
                self.code_dirty = True
        out = _report(r, src=kind, seconds=round(time.time() - t0, 2))
        if r.added or r.retracted:
            self._event(kind="ingest", **out)
        return out

    def _in_code_repo(self, path: str) -> bool:
        low = os.path.normcase(os.path.abspath(path))
        return any(low.startswith(os.path.normcase(os.path.abspath(r)) + os.sep) for r in self.p.get("code_repos") or ())

    def retract_file(self, path: str, reason: str = "deleted") -> dict:
        import ptgcore
        path = os.path.normpath(path)
        with LOCK:
            r = ptgcore.retract_file(self.home, path, time.time())
        if r.retracted:
            self._touched()
            self._event(kind="retract", file=path, retracted=r.retracted, reason=reason)
        return _report(r)

    def remember(self, text: str, session_id: str | None = None, source: str = "agent") -> dict:
        """An explicit note or decision, stored immediately; without the embedder — by words until re-embedded."""
        import ptgcore
        clean = ptgcore.note_text((text or "").strip())
        if not clean:
            raise ValueError("note too short")
        try:
            vector = self._embed([clean])[0]
        except Exception as ex:                          # noqa: BLE001
            self.embedder_failed = True
            self._event(kind="embedder", error=str(ex)[:200])
            vector = ()
        with LOCK:
            c = ptgcore.remember(self.home, clean, session_id or "", source, self.space, vector, time.time(), self.root)
        self._touched()
        self._event(kind="remember", id=c.id, text=clean[:200])
        return _card(c)

    def relate(self, src: str, dst: str, kind: str, source: str = "agent") -> list:
        """A relation claimed by an agent or a model (never guessed from a cosine threshold)."""
        import ptgcore
        with LOCK:
            try:
                out = ptgcore.relate(self.home, src, dst, kind, source, time.time())
            except KeyError as ex:
                raise KeyError("unknown atom") from ex
        self._touched()
        return list(out)

    # ------------------------------------------------------------------ plan / save
    def plan(self, include_transcripts: bool = True):
        """What is out of date: [(path, kind)] to ingest, [path] to retract."""
        import ptgcore
        s = self._settings()
        if not include_transcripts:
            s = dataclasses.replace(s, claude=False, codex=False)
        p = ptgcore.plan(self.home, s)
        return [(x.path, x.kind) for x in p.todo], list(p.gone)

    def index_code(self) -> int:
        """The code graph of the project's chosen repositories (``code_repos``) — codebase-memory-mcp, ``code_binary``."""
        import ptgcore
        repos = tuple(self.p.get("code_repos") or ())
        if not repos:
            return 0
        with LOCK:
            n = ptgcore.index_code(self.home, repos, self.p.get("code_binary") or "codebase-memory-mcp")
        self.code_dirty = False
        self._event(kind="code", symbols=n)
        return n

    def save(self) -> dict:
        """Rebuild the metatree over the live atoms and the code graph (re-indexed when code changed), compact."""
        import ptgcore
        t0 = time.time()
        if self.code_dirty:
            try:
                self.index_code()
            except (FileNotFoundError, RuntimeError) as ex:
                self._event(kind="code", error=str(ex)[:200])
        with LOCK:
            st = ptgcore.save(self.home)
            self.dirty = False
            self.last_save = time.time()
        r = {"atoms": st.atoms, "edges": st.relations, "files": st.files, "code": st.code, "metatree": st.edges,
             "seconds": round(time.time() - t0, 1)}
        self._event(kind="save", **r)
        return r

    # ------------------------------------------------------------------ read
    def search(self, query: str, k: int = 8, exclude_files=(), exclude_ids=(), with_retracted: bool = False, keep=None) -> list[dict]:
        """RRF over words and the store's spaces; cards with their relations."""
        import ptgcore
        files = set(exclude_files)
        extra = k * 4 if files or keep is not None else k
        found = ptgcore.search(self.home, query, self._query(query), max(1, extra), tuple(exclude_ids), with_retracted, self.root)
        out = [_card(c) for c in found if c.file not in files]
        if keep is not None:
            out = [c for c in out if keep(c)]
        return out[:k]

    def card(self, nid: str, score: float | None = None, chars: int = 700) -> dict:
        import ptgcore
        n = ptgcore.node(self.home, nid, self.root)
        if n is None:
            raise KeyError(nid)
        d = _card(n.card, chars)
        d["score"] = None if score is None else round(float(score), 4)
        return d

    def recall(self, prompt: str, session_id: str | None = None, transcript_path: str | None = None,
               budget_chars: int = 3600, max_items: int = 5, min_score: float = 0.5, exclude_ids=()) -> dict:
        """The context injected into a prompt: RRF order, admitted by cosine (ptgcore.context.admit)."""
        import ptgcore
        prompt = (prompt or "").strip()
        if len(prompt) < 12:
            return {"context": "", "items": []}
        r = ptgcore.recall(self.home, prompt, self._query(prompt), session_id or "", transcript_path or "", int(budget_chars),
                           int(max_items), float(min_score), self.root)
        ex = set(exclude_ids)
        return {"context": r.context, "items": [_card(c) for c in r.items if c.id not in ex], "top_score": r.top}

    def brief(self, max_chars: int = 1600) -> str:
        """SessionStart: how much is in memory and the latest notes and decisions (two weeks)."""
        import ptgcore
        return ptgcore.brief(self.home, time.time(), int(max_chars))

    def decisions(self, limit: int = 50, primary=None) -> list:
        """What replaced what and what contradicts what — newest first, closed relations too."""
        import ptgcore
        return [{"type": d.kind, "types": [d.kind], "date": d.date, "ts": d.at, "source": d.source, "closed": d.closed,
                 "new": _side(d.new, False), "old": _side(d.old, True)}
                for d in ptgcore.decisions(self.home, int(limit), self.root)]

    def node(self, nid: str) -> dict | None:
        import ptgcore
        n = ptgcore.node(self.home, nid, self.root)
        if n is None:
            return None
        d = _card(n.card)
        d["answer"] = n.answer
        d["lineage"] = [{"id": s.id, "date": s.date, "status": s.status, "text": s.text} for s in n.lineage]
        return d

    def graph(self, nid: str, depth: int = 2, limit: int = 60) -> dict:
        """Ego network over the relations, then the metatree (code seams, neighbours), for the window."""
        import ptgcore
        g = ptgcore.graph(self.home, nid, max(1, int(depth)), int(limit), self.root)
        return {"nodes": [{"id": x.id, "label": x.label, "status": x.status, "date": x.date, "file": x.file, "center": x.center}
                          for x in g.nodes],
                "edges": [{"from": e.src, "to": e.dst, "type": e.kind} for e in g.edges]}

    def stats(self) -> dict:
        import ptgcore
        st = ptgcore.stats(self.home)
        n = max(1, st.atoms)
        return {"root": self.root, "store": self.store_dir, "atoms": st.atoms, "live": st.live, "retracted": st.retracted,
                "superseded": st.superseded, "branches": 0, "edges": st.relations, "files": st.files, "code": st.code,
                "metatree": st.edges, "spaces": list(st.spaces), "mean_degree": round(2.0 * st.relations / n, 2),
                "dirty": self.dirty, "last_save": self.last_save, "last_change": self.last_change, "loaded_at": self.loaded_at,
                "engine": "full", "embedder": self.embedder.status()}
