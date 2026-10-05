# -*- coding: utf-8 -*-
"""The PTG-MEM daemon: one process per user that owns every project's memory.

Why one owner. Before this, the MCP server, the builder and the replay each
opened the same archive files: a server holding the vector file memory-mapped
broke the builder's save (Windows ERROR_USER_MAPPED_FILE, an hour of embeddings
lost), and every MCP start paid 54 s of JSON parsing. Now the daemon loads a
project once, keeps it current from file-system events, and everything else —
Claude Code hooks, the MCP server, the CLI, the GUI — is a thin HTTP client.

Listens on 127.0.0.1 only. Every API call needs the token from daemon.json (the
GUI gets it through the URL fragment, which never reaches the server logs), and
requests whose Host header is not local are refused (DNS-rebinding guard).
"""
from __future__ import annotations

import collections
import json
import mimetypes
import os
import secrets
import socket
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ptg_mem
from . import license as lic
from . import paths, transcripts
from .embedders import make_embedder
from .watcher import DebounceQueue, Watcher

GUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui")
FILE_DELAY = 2.0          # quiet period before a changed file is ingested
TRANSCRIPT_DELAY = 8.0    # transcripts grow on every turn; wait for the turn to end


def _log_line(msg: str):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        print(line, flush=True)
    except Exception:                                # noqa: BLE001 — no console under pythonw
        pass
    try:
        with open(os.path.join(paths.LOGS, "daemon.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


class Project:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = cfg["root"]
        self.engine = None
        self.state = "idle"          # idle | loading | ready | error
        self.error = None
        self.watcher = None
        self.feed = collections.deque(maxlen=300)
        self.catchup_total = 0
        self.catchup_done = 0
        # a project may prepare its own intake (e.g. a script that turns agent
        # transcripts, notes and logs into documents); it runs when transcripts
        # changed, at most every intake_every_seconds
        self.intake_due = False
        self.intake_last = 0.0
        self.intake_running = False
        self.transcripts_seen = None          # newest transcript mtime seen by polling


class Daemon:
    def __init__(self):
        paths.ensure_home()
        self.cfg = paths.load_config()
        self.port = int(self.cfg.get("port") or paths.DEFAULT_PORT)
        self.token = secrets.token_hex(16)
        self.projects: dict[str, Project] = {}
        self.embedders: dict[str, object] = {}
        self.queue = DebounceQueue()
        self.stop = threading.Event()
        self.started = time.time()
        self.lock = threading.RLock()

    # ------------------------------------------------------------ projects
    def embedder_for(self, spec: dict):
        key = json.dumps(spec, sort_keys=True)
        with self.lock:
            if key not in self.embedders:
                self.embedders[key] = make_embedder(spec, float(self.cfg.get("idle_unload_seconds", 600)))
            return self.embedders[key]

    def project_for(self, cwd: str | None, root: str | None = None, autoload: bool = True):
        if root:
            cfg = paths.load_projects().get(paths.norm_root(root))
        else:
            cfg = paths.find_project(cwd or "")
        if not cfg:
            return None
        with self.lock:
            pr = self.projects.get(cfg["root"])
            if pr is None:
                pr = Project(cfg)
                self.projects[cfg["root"]] = pr
        if autoload and pr.state == "idle":
            self.load(pr)
        return pr

    def load(self, pr: Project):
        with self.lock:
            if pr.state in ("loading", "ready"):
                return
            pr.state = "loading"
        threading.Thread(target=self._load, args=(pr,), daemon=True, name="ptg-load").start()

    def _load(self, pr: Project):
        from .engine import MemoryEngine                 # heavy imports only when needed
        try:
            emb = self.embedder_for(pr.cfg["embedder"])
            eng = MemoryEngine(pr.cfg, emb, log=_log_line)
            eng.open()
            pr.engine = eng
            pr.state = "ready"
            if pr.cfg.get("read_only"):
                return
            self._catch_up(pr)
            folders = [pr.root] if (pr.cfg.get("sources") or {}).get("folder", True) else []
            cdir = transcripts.claude_project_dir(pr.root)
            if ((pr.cfg.get("sources") or {}).get("claude", True) or pr.cfg.get("intake_command")) \
                    and os.path.isdir(cdir):
                folders.append(cdir)
            codex = os.path.join(transcripts.codex_home(), "sessions")
            if (pr.cfg.get("sources") or {}).get("codex", True) and os.path.isdir(codex):
                folders.append(codex)
            pr.watcher = Watcher(folders, lambda p, pr=pr: self.on_path(pr, p),
                                 poll_fn=lambda pr=pr: self._catch_up(pr), log=_log_line).start()
            _log_line("watching %s (%s): %s" % (pr.root, pr.watcher.mode, folders))
        except Exception as ex:                          # noqa: BLE001
            pr.state = "error"
            pr.error = str(ex)
            _log_line("load failed for %s: %s\n%s" % (pr.root, ex, traceback.format_exc()))

    def _catch_up(self, pr: Project):
        if pr.cfg.get("intake_command"):
            # without file-system events (polling mode) transcript growth is noticed here
            cdir = transcripts.claude_project_dir(pr.root)
            try:
                newest = max((e.stat().st_mtime for e in os.scandir(cdir) if e.name.endswith(".jsonl")),
                             default=0.0)
            except OSError:
                newest = 0.0
            if pr.transcripts_seen is not None and newest > pr.transcripts_seen:
                pr.intake_due = True
            pr.transcripts_seen = max(newest, pr.transcripts_seen or 0.0)
        todo, gone = pr.engine.plan()
        for path in gone:
            self.queue.put((pr.root, path, "gone"), 0.1)
        for path, kind in todo:
            self.queue.put((pr.root, path, kind), 0.1)
        if todo or gone:
            # the queue de-duplicates, so what is pending is exactly this plan
            pr.catchup_total = pr.catchup_done + len(todo) + len(gone)
            _log_line("%s: %d to ingest, %d to retract" % (pr.root, len(todo), len(gone)))

    def on_path(self, pr: Project, path: str):
        eng = pr.engine
        if eng is None:
            return
        p = os.path.normpath(path)
        low = p.lower()
        if low.endswith(".jsonl"):
            cdir = os.path.normpath(transcripts.claude_project_dir(pr.root)).lower()
            if low.startswith(cdir + os.sep):
                if (pr.cfg.get("sources") or {}).get("claude", True):
                    self.queue.put((pr.root, p, "claude"), TRANSCRIPT_DELAY)
                elif pr.cfg.get("intake_command"):
                    pr.intake_due = True
                return
            if os.sep + "sessions" + os.sep in low and "rollout-" in low:
                if (pr.cfg.get("sources") or {}).get("codex", True):
                    meta = transcripts.codex_session_meta(p)
                    cwd = meta.get("cwd") or ""
                    if cwd and paths.norm_root(cwd).lower().startswith(pr.root.lower()):
                        self.queue.put((pr.root, p, "codex"), TRANSCRIPT_DELAY)
                return
        if not (pr.cfg.get("sources") or {}).get("folder", True):
            return
        if os.path.exists(p):
            if eng.filter.path_ok(p):
                self.queue.put((pr.root, p, "file"), FILE_DELAY)
        elif p in eng.arc.processed_files:
            self.queue.put((pr.root, p, "gone"), FILE_DELAY)

    # ------------------------------------------------------------ worker
    def worker(self):
        while not self.stop.is_set():
            key = self.queue.get(timeout=1.0)
            if key is None:
                continue
            root, path, kind = key
            pr = self.projects.get(root)
            if not pr or not pr.engine:
                continue
            try:
                if kind == "gone":
                    r = pr.engine.retract_file(path)
                else:
                    r = pr.engine.ingest(path, kind)
                if r.get("added") or r.get("retracted"):
                    pr.feed.append({"t": time.time(), "kind": "ingest", **r})
            except Exception as ex:                      # noqa: BLE001
                pr.feed.append({"t": time.time(), "kind": "error", "file": path, "error": str(ex)})
                _log_line("ingest failed %s: %s" % (path, ex))
            finally:
                if pr.catchup_total:
                    pr.catchup_done += 1
                    if pr.catchup_done >= pr.catchup_total:
                        pr.catchup_total = pr.catchup_done = 0

    def _run_intake(self, pr: Project):
        import subprocess
        cmd = pr.cfg["intake_command"]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=pr.cfg.get("intake_cwd") or pr.root, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=900,
                               env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            tail = (r.stdout or "")[-600:]
            pr.feed.append({"t": time.time(), "kind": "intake", "code": r.returncode,
                            "seconds": round(time.time() - t0, 1), "tail": tail})
            _log_line("intake for %s: code %s in %.1f s" % (pr.root, r.returncode, time.time() - t0))
            if r.returncode == 0:
                self._catch_up(pr)                       # pick up what it wrote right away
        except Exception as ex:                          # noqa: BLE001
            pr.feed.append({"t": time.time(), "kind": "error", "file": "intake", "error": str(ex)})
            _log_line("intake failed for %s: %s" % (pr.root, ex))
        finally:
            pr.intake_last = time.time()
            pr.intake_running = False

    def maintenance(self):
        while not self.stop.wait(5.0):
            now = time.time()
            for pr in list(self.projects.values()):
                if (pr.intake_due and not pr.intake_running and pr.state == "ready"
                        and now - pr.intake_last >= float(pr.cfg.get("intake_every_seconds", 600))):
                    pr.intake_due = False
                    pr.intake_running = True
                    threading.Thread(target=self._run_intake, args=(pr,), daemon=True).start()
            for emb in list(self.embedders.values()):
                try:
                    if emb.maybe_unload():
                        _log_line("embedder %s unloaded after idle" % emb.model)
                except Exception:                        # noqa: BLE001
                    pass
            now = time.time()
            for pr in list(self.projects.values()):
                eng = pr.engine
                if not eng or not eng.dirty:
                    continue
                quiet = now - eng.last_change >= float(self.cfg.get("save_after_seconds", 45))
                overdue = now - eng.last_save >= float(self.cfg.get("save_every_seconds", 600))
                if (quiet and len(self.queue) == 0) or overdue:
                    try:
                        r = eng.save()
                        _log_line("saved %s: %s" % (pr.root, r))
                    except Exception as ex:              # noqa: BLE001
                        _log_line("save failed %s: %s\n%s" % (pr.root, ex, traceback.format_exc()))

    def shutdown(self):
        self.stop.set()
        for pr in self.projects.values():
            if pr.watcher:
                pr.watcher.stop()
            if pr.engine and pr.engine.dirty:
                try:
                    pr.engine.save()
                except Exception as ex:                  # noqa: BLE001
                    _log_line("final save failed %s: %s" % (pr.root, ex))

    # ------------------------------------------------------------ status
    def project_status(self, pr: Project) -> dict:
        d = {"root": pr.root, "name": pr.cfg.get("name"), "state": pr.state, "error": pr.error,
             "read_only": bool(pr.cfg.get("read_only")),
             "watch": pr.watcher.mode if pr.watcher else None,
             "catchup": [pr.catchup_done, pr.catchup_total]}
        if pr.engine:
            d.update(pr.engine.stats())
        return d


# ================================================================ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "ptg-mem"
    protocol_version = "HTTP/1.1"
    daemon: Daemon = None                                # set in serve()

    def log_message(self, *a):
        pass

    def _send(self, code, obj=None, body: bytes | None = None, ctype="application/json"):
        if body is None:
            body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text/", "application/json", "application/javascript")) else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1")

    def _auth(self) -> bool:
        return secrets.compare_digest(self.headers.get("X-PTG-Token", ""), self.daemon.token)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def _route(self, method):
        try:
            if not self._host_ok():
                return self._send(403, {"error": "bad host"})
            u = urllib.parse.urlsplit(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            if u.path == "/api/health":
                return self._send(200, {"ok": True, "version": ptg_mem.__version__, "pid": os.getpid()})
            if not u.path.startswith("/api/"):
                return self._static(u.path)
            if not self._auth():
                return self._send(401, {"error": "token required"})
            body = self._body() if method == "POST" else {}
            fn = getattr(self, "api_" + u.path[5:].replace("/", "_"), None)
            if fn is None:
                return self._send(404, {"error": "no such endpoint"})
            return self._send(200, fn(q, body))
        except PermissionError as ex:
            return self._send(402, {"error": str(ex), "pro": True})
        except LookupError as ex:
            return self._send(404, {"error": str(ex)})
        except (RuntimeError, ValueError) as ex:
            return self._send(409, {"error": str(ex)})
        except Exception as ex:                          # noqa: BLE001
            _log_line("api error %s: %s\n%s" % (self.path, ex, traceback.format_exc()))
            return self._send(500, {"error": str(ex)})

    def _static(self, path):
        if path in ("", "/"):
            path = "/index.html"
        full = os.path.normpath(os.path.join(GUI_DIR, path.lstrip("/")))
        if not full.startswith(GUI_DIR) or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if full.endswith(".js"):
            ctype = "application/javascript"
        with open(full, "rb") as f:
            return self._send(200, body=f.read(), ctype=ctype)

    # --- helpers -----------------------------------------------------------
    def _project(self, q, body, need_ready=True):
        d = self.daemon
        pr = d.project_for(body.get("cwd") or q.get("cwd"), body.get("root") or q.get("root"))
        if pr is None:
            raise LookupError("no registered project for this folder — run: ptg init")
        if need_ready and pr.state != "ready":
            raise RuntimeError("project is %s%s" % (pr.state, (": " + pr.error) if pr.error else ""))
        return pr

    # --- endpoints -----------------------------------------------------------
    def api_status(self, q, body):
        d = self.daemon
        return {"version": ptg_mem.__version__, "pid": os.getpid(), "uptime": time.time() - d.started,
                "queue": len(d.queue), "pending": [list(k) for k in d.queue.pending()[:20]],
                "projects": [d.project_status(pr) for pr in d.projects.values()],
                "registered": list(paths.load_projects()),
                "embedders": [e.status() for e in d.embedders.values()],
                "inject": d.cfg.get("inject"), "license": lic.status()}

    def api_open(self, q, body):
        pr = self._project(q, body, need_ready=False)
        return self.daemon.project_status(pr)

    def api_recall(self, q, body):
        d = self.daemon
        inj = d.cfg.get("inject") or {}
        if not inj.get("enabled", True):
            return {"context": "", "items": [], "disabled": True}
        pr = d.project_for(body.get("cwd"))
        if pr is None:
            return {"context": "", "items": [], "status": "unregistered"}
        if pr.state != "ready":
            return {"context": "", "items": [], "status": pr.state}
        r = pr.engine.recall(body.get("prompt", ""), session_id=body.get("session_id"),
                             transcript_path=body.get("transcript_path"),
                             budget_chars=int(inj.get("budget_chars", 3600)),
                             max_items=int(inj.get("max_items", 5)),
                             min_score=float(inj.get("min_score", 0.5)))
        def _replaces(i):
            if i.get("replaces"):
                return i["replaces"]
            for rel in i.get("relations", ()):
                if rel["dir"] == "out" and rel["type"] in ("revises", "supersedes"):
                    return {"date": rel["date"], "text": rel["text"][:160]}
            return None

        pr.feed.append({"t": time.time(), "kind": "recall", "prompt": (body.get("prompt") or "")[:300],
                        "session": (body.get("session_id") or "")[:8], "top": r.get("top_score"),
                        "items": [{"id": i["id"], "score": i["score"], "file": i["rel_file"],
                                   "date": i["date"], "text": i["text"][:220],
                                   "replaces": _replaces(i)} for i in r["items"]],
                        "chars": len(r["context"])})
        return {"context": r["context"], "items": [i["id"] for i in r["items"]], "status": "ready"}

    def api_brief(self, q, body):
        d = self.daemon
        pr = d.project_for(body.get("cwd"))
        if pr is None:
            return {"context": "", "status": "unregistered"}
        if pr.state != "ready" or not (d.cfg.get("inject") or {}).get("session_brief", True):
            return {"context": "", "status": pr.state}
        return {"context": pr.engine.brief(), "status": "ready"}

    def api_search(self, q, body):
        query = body.get("query") or q.get("query") or ""
        k = int(body.get("k") or q.get("k") or 10)
        if body.get("all_projects"):
            if not lic.has("cross_project"):
                raise PermissionError("cross-project search is a Pro feature")
            out = []
            for pr in list(self.daemon.projects.values()):
                if pr.state == "ready":
                    for it in pr.engine.search(query, k=k):
                        it["project"] = pr.root
                        out.append(it)
            out.sort(key=lambda x: -(x["score"] or 0))
            return {"items": out[:k]}
        pr = self._project(q, body)
        return {"items": pr.engine.search(query, k=k), "project": pr.root}

    def api_node(self, q, body):
        pr = self._project(q, body)
        n = pr.engine.node(body.get("id") or q.get("id"))
        if n is None:
            raise LookupError("no such node")
        return n

    def api_graph(self, q, body):
        pr = self._project(q, body)
        return pr.engine.graph(body.get("id") or q.get("id"), depth=int(q.get("depth") or body.get("depth") or 2))

    def api_decisions(self, q, body):
        pr = self._project(q, body)
        return {"items": pr.engine.decisions(limit=int(q.get("limit") or body.get("limit") or 60))}

    def api_decisions_export(self, q, body):
        if not lic.has("decision_export"):
            raise PermissionError("decision log export is a Pro feature")
        pr = self._project(q, body)
        items = pr.engine.decisions(limit=100000)
        lines = ["# Decision log — %s" % pr.cfg.get("name"), "",
                 "Generated by PTG-MEM on %s. Newest first." % time.strftime("%Y-%m-%d"), ""]
        names = {"supersedes": "replaced", "contradicts": "contradicts", "fixes": "fixes",
                 "revises": "revised"}
        for it in items:
            lines.append("## %s — %s" % (it["date"], names.get(it["type"], it["type"])))
            lines.append("**Now** (%s): %s" % (it["new"]["file"], " ".join(it["new"]["text"].split())))
            lines.append("")
            lines.append("**Before** (%s, %s): %s" % (it["old"]["date"], it["old"]["file"],
                                                        " ".join(it["old"]["text"].split())))
            lines.append("")
        return {"markdown": "\n".join(lines), "count": len(items)}

    def api_feed(self, q, body):
        d = self.daemon
        out = []
        for pr in d.projects.values():
            if q.get("root") and pr.root != paths.norm_root(q["root"]):
                continue
            for ev in list(pr.feed):
                out.append(dict(ev, project=pr.cfg.get("name")))
            if pr.engine:
                for ev in list(pr.engine.events):
                    if ev.get("kind") in ("remember", "retract", "secrets", "save"):
                        out.append(dict(ev, project=pr.cfg.get("name")))
        out.sort(key=lambda e: -e.get("t", 0))
        return {"items": out[:int(q.get("limit") or 120)]}

    def api_remember(self, q, body):
        pr = self._project(q, body)
        card = pr.engine.remember(body.get("text", ""), session_id=body.get("session_id"),
                                  source=body.get("source") or "agent")
        pr.feed.append({"t": time.time(), "kind": "remember", "id": card["id"], "text": card["text"][:220]})
        return card

    def api_ingest(self, q, body):
        pr = self._project(q, body)
        p = body.get("path")
        kind = body.get("kind") or "file"
        if p:
            src = pr.cfg.get("sources") or {}
            if kind in ("claude", "codex") and not src.get(kind, True):
                if kind == "claude" and pr.cfg.get("intake_command"):
                    pr.intake_due = True                 # this project digests transcripts itself
                return {"queued": 0}
            self.daemon.queue.put((pr.root, os.path.normpath(p), kind), 0.5)
            return {"queued": 1}
        before = len(self.daemon.queue)
        self.daemon._catch_up(pr)
        return {"queued": len(self.daemon.queue) - before}

    def api_config(self, q, body):
        d = self.daemon
        if body:
            inj = d.cfg.setdefault("inject", {})
            for k in ("enabled", "budget_chars", "max_items", "min_score", "session_brief"):
                if k in body:
                    inj[k] = body[k]
            if "idle_unload_seconds" in body:
                d.cfg["idle_unload_seconds"] = float(body["idle_unload_seconds"])
                for e in d.embedders.values():
                    e.idle_unload_seconds = float(body["idle_unload_seconds"])
            paths.save_config(d.cfg)
        return {"inject": d.cfg.get("inject"), "idle_unload_seconds": d.cfg.get("idle_unload_seconds")}

    def api_license(self, q, body):
        if body.get("key"):
            try:
                lic.install(body["key"])
            except ValueError as ex:
                return {"ok": False, "error": str(ex), **lic.status()}
            return {"ok": True, **lic.status()}
        return lic.status()

    def api_unload(self, q, body):
        for e in self.daemon.embedders.values():
            e.unload()
        return {"ok": True}

    def api_save(self, q, body):
        pr = self._project(q, body)
        return pr.engine.save()

    def api_shutdown(self, q, body):
        threading.Thread(target=self.server.shutdown, daemon=True).start()
        return {"ok": True}


class _Server(ThreadingHTTPServer):
    # HTTPServer turns SO_REUSEADDR on, and on Windows that lets a second process
    # bind a port that is already listening — two daemons would split the traffic.
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve():
    paths.ensure_home()
    d = Daemon()
    try:
        httpd = _Server(("127.0.0.1", d.port), Handler)
    except OSError:
        # someone already listens there: most likely a daemon started a moment ago
        _log_line("port %d busy — another daemon is running, exiting" % d.port)
        return 1
    httpd.daemon_threads = True
    Handler.daemon = d
    paths.save_daemon({"pid": os.getpid(), "port": d.port, "token": d.token,
                       "started": d.started, "version": ptg_mem.__version__,
                       "python": sys.executable})
    _log_line("daemon %s listening on 127.0.0.1:%d (pid %d)" % (ptg_mem.__version__, d.port, os.getpid()))
    threading.Thread(target=d.worker, daemon=True, name="ptg-worker").start()
    threading.Thread(target=d.maintenance, daemon=True, name="ptg-maint").start()
    # open every registered project in the background: watchers must run even
    # when no agent session is active, or memory is only as fresh as the last prompt
    for root, cfg in paths.load_projects().items():
        if cfg.get("autostart", True):
            d.project_for(None, root=root)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        d.shutdown()                 # final save while the port is still ours: it doubles as a lock
        httpd.server_close()
        try:
            info = paths.load_daemon()
            if info.get("pid") == os.getpid():
                os.remove(paths.DAEMON)
        except OSError:
            pass
        _log_line("daemon stopped")
    return 0


if __name__ == "__main__":
    socket.setdefaulttimeout(None)
    sys.exit(serve())
