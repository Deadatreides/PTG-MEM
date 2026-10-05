# -*- coding: utf-8 -*-
"""ptg — command line for PTG-MEM.

    ptg init [PATH]            register a project and start indexing it
    ptg install claude|codex   attach memory to your agent (hooks + MCP)
    ptg gui                    open the local GUI
    ptg status                 what the daemon holds, what is pending
    ptg search "query"         search memory from the terminal
    ptg remember "text"        store a decision now
    ptg start | stop           run / stop the daemon
    ptg license [KEY]          show or install a Pro licence
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

from . import client, paths


def _say(*a):
    print(*a, flush=True)


def cmd_init(a):
    root = paths.norm_root(a.path or os.getcwd())
    if not os.path.isdir(root):
        _say("not a folder:", root)
        return 2
    if a.gguf:
        emb = {"backend": "gguf", "model": os.path.abspath(a.gguf)}
    else:
        emb = {"backend": "st", "model": a.model}
    proj = paths.new_project(root, store=a.store, embedder=emb, margin=a.margin)
    proj["exclude_dirs"] = a.exclude or []
    proj["sources"] = {"folder": not a.no_folder, "claude": not a.no_claude, "codex": not a.no_codex}
    if a.legacy_store:
        proj["legacy_store"] = os.path.abspath(a.legacy_store)
    old = paths.load_projects().get(root)
    if old and not a.force:
        _say("already registered:", root, "(store %s). Use --force to overwrite the settings." % old["store"])
        return 1
    # say exactly what is about to happen before it happens
    from .filters import FileFilter
    n = 0
    if proj["sources"]["folder"]:
        for _ in FileFilter(root, exclude_dirs=proj["exclude_dirs"], store_dir=proj["store"]).walk():
            n += 1
    _say("PTG-MEM will index %d files under %s" % (n, root))
    _say("  plus Claude Code sessions: %s, Codex sessions: %s" % (proj["sources"]["claude"], proj["sources"]["codex"]))
    if emb["backend"] == "st":
        _say("  embedding model %s is downloaded from Hugging Face on first use "
             "(Qwen3-Embedding-0.6B is ~1.2 GB) and runs locally; nothing else leaves this machine" % emb["model"])
    if a.dry_run:
        return 0
    if not a.yes:
        try:
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes", "д", "да"):
                _say("cancelled")
                return 1
        except EOFError:
            _say("no terminal to confirm — rerun with --yes")
            return 1
    paths.register(proj)
    _say("registered", root)
    _say("  store    ", proj["store"])
    _say("  embedder ", emb)
    if client.ensure_daemon():
        client.call("/api/open", {"root": root}, timeout=10)
        _say("indexing started in the background — watch it: ptg status / ptg gui")
    else:
        _say("daemon did not start — see", os.path.join(paths.LOGS, "daemon.log"))
    return 0


def cmd_status(a):
    if not client.alive():
        _say("daemon is not running (ptg start)")
        return 1
    s = client.call("/api/status", {})
    _say("PTG-MEM %s, pid %s, queue %d, licence: %s" % (s["version"], s["pid"], s["queue"],
                                                      s["license"]["plan"]))
    for p in s["projects"]:
        _say("\n%s  [%s]" % (p["root"], p["state"] + (" — " + p["error"] if p.get("error") else "")))
        if p.get("atoms") is not None:
            _say("  atoms %d (live %d, retracted %d), files %d, edges %d, degree %.1f (budget %d)"
                 % (p["atoms"], p["live"], p["retracted"], p["files"], p["edges"], p["mean_degree"],
                    p["budget_d"]))
            e = p["embedder"]
            _say("  embedder %s %s — %s" % (e["backend"], os.path.basename(str(e["model"])),
                                            "loaded" if e["loaded"] else "unloaded"))
            if p["catchup"][1]:
                _say("  catching up: %d / %d" % tuple(p["catchup"]))
    return 0


def _cwd_payload(a):
    return {"cwd": paths.norm_root(getattr(a, "path", None) or os.getcwd())}


def cmd_search(a):
    client.ensure_daemon()
    r = client.call("/api/search", dict(_cwd_payload(a), query=a.query, k=a.k), timeout=60)
    for i, it in enumerate(r.get("items", []), 1):
        _say("%2d. %.3f  %s  %s%s" % (i, it["score"] or 0, it["date"], it["rel_file"],
                                      "  [superseded]" if it.get("current_id") else ""))
        _say("    " + " ".join(it["text"].split())[:300])
    return 0


def cmd_remember(a):
    client.ensure_daemon()
    r = client.call("/api/remember", dict(_cwd_payload(a), text=a.text, source="user"), timeout=60)
    _say("stored", r["id"])
    return 0


def cmd_gui(a):
    if not client.ensure_daemon():
        _say("daemon did not start")
        return 1
    info = paths.load_daemon()
    url = "http://127.0.0.1:%d/#token=%s" % (info["port"], info["token"])
    _say(url)
    if not a.no_browser:
        webbrowser.open(url)
    return 0


def cmd_start(a):
    if a.foreground:
        from .daemon import serve
        return serve()
    ok = client.ensure_daemon()
    _say("daemon running" if ok else "daemon did not start — see " + os.path.join(paths.LOGS, "daemon.log"))
    return 0 if ok else 1


def cmd_stop(a):
    if not client.alive():
        _say("not running")
        return 0
    pid = paths.load_daemon().get("pid")
    client.call("/api/shutdown", {})
    _say("stopping (the store is saved first)…")
    # the HTTP side stops at once, but the final save of a large store takes a while and the
    # port stays bound until it is done; daemon.json disappears only when the process is over
    import time
    end = time.time() + 180
    while time.time() < end:
        if paths.load_daemon().get("pid") != pid:
            _say("stopped")
            return 0
        time.sleep(0.5)
    _say("still saving after 3 minutes — see", os.path.join(paths.LOGS, "daemon.log"))
    return 1


def cmd_install(a):
    from . import install
    if a.target == "claude":
        root = paths.norm_root(a.path or os.getcwd())
        if a.scope == "project" and not paths.find_project(root) and not a.remove:
            _say("note: %s is not a registered project yet — run `ptg init` there" % root)
        for line in install.install_claude(root, scope=a.scope, remove=a.remove):
            _say(line)
        _say("restart Claude Code sessions to pick the hooks up")
    else:
        for line in install.install_codex(remove=a.remove):
            _say(line)
    return 0


def cmd_license(a):
    from . import license as lic
    if a.key:
        try:
            p = lic.install(a.key)
        except ValueError as ex:
            _say("key rejected:", ex)
            return 1
        _say("licensed to %s (%s)" % (p.get("name"), p.get("plan")))
        return 0
    _say(json.dumps(lic.status(), ensure_ascii=False, indent=1))
    return 0


def cmd_projects(a):
    for root, p in paths.load_projects().items():
        _say("%s\n  store %s\n  embedder %s" % (root, p["store"], p["embedder"]))
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="ptg", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("init", help="register a project and start indexing")
    p.add_argument("path", nargs="?")
    p.add_argument("--store", help="where the memory store lives (default ~/.ptg-mem/stores)")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="sentence-transformers model")
    p.add_argument("--gguf", help="use a GGUF embedding model through llama.cpp instead")
    p.add_argument("--margin", type=float, default=None, help="edge budget margin (default 5.0)")
    p.add_argument("--exclude", nargs="*", help="extra folder names to skip")
    p.add_argument("--no-folder", action="store_true", help="do not index the folder's files")
    p.add_argument("--no-claude", action="store_true", help="do not ingest Claude Code sessions")
    p.add_argument("--no-codex", action="store_true", help="do not ingest Codex sessions")
    p.add_argument("--legacy-store", help="migrate an existing ptg_core store (.ptg/ folder parent)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    p.add_argument("--dry-run", action="store_true", help="only show what would be indexed")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("status")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--path")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("remember")
    p.add_argument("text")
    p.add_argument("--path")
    p.set_defaults(fn=cmd_remember)

    p = sub.add_parser("gui")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_gui)

    p = sub.add_parser("start")
    p.add_argument("--foreground", action="store_true")
    p.set_defaults(fn=cmd_start)

    p = sub.add_parser("stop")
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser("install", help="attach memory to Claude Code or Codex")
    p.add_argument("target", choices=["claude", "codex"])
    p.add_argument("--path")
    p.add_argument("--scope", choices=["project", "user"], default="project")
    p.add_argument("--remove", action="store_true")
    p.set_defaults(fn=cmd_install)

    p = sub.add_parser("license")
    p.add_argument("key", nargs="?")
    p.set_defaults(fn=cmd_license)

    p = sub.add_parser("projects")
    p.set_defaults(fn=cmd_projects)

    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help()
        return 0
    try:
        return a.fn(a) or 0
    except client.DaemonDown as ex:
        _say("daemon unreachable:", ex)
        return 1
    except RuntimeError as ex:
        _say("error:", ex)
        return 1


if __name__ == "__main__":
    sys.exit(main())
