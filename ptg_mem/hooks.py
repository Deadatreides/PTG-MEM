# -*- coding: utf-8 -*-
"""Claude Code hooks: memory arrives without being asked for.

    session-start   brief (what this memory holds, open threads) + start the daemon
    prompt          relevant memory attached to every prompt (UserPromptSubmit)
    stop            nudge the daemon to ingest the turn that just ended

Rules a hook must keep:
  * never block the user: short timeouts, and every failure is silent (exit 0);
  * never import anything heavy: this file and client.py use the standard library;
  * print ASCII-escaped JSON, so a console code page cannot mangle it.
"""
from __future__ import annotations

import json
import os
import sys

from . import client, paths

PROMPT_TIMEOUT = 6.0


def _read_input() -> dict:
    try:
        raw = sys.stdin.buffer.read()
    except Exception:                                # noqa: BLE001
        return {}
    # a BOM in front of the JSON (PowerShell pipes add one) made json.loads fail,
    # and the hook then acted on the wrong folder — decode defensively
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            return json.loads(text) if text.strip() else {}
        except (UnicodeDecodeError, ValueError):
            continue
    return {}


def _emit(event: str, context: str) -> None:
    if not context:
        return
    out = {"hookSpecificOutput": {"hookEventName": event, "additionalContext": context}}
    sys.stdout.write(json.dumps(out, ensure_ascii=True))
    sys.stdout.flush()


def _auto_register(cwd: str) -> bool:
    """First session in a folder that is not yet a PTG-MEM project.

    OFF by default. Registering starts a model download and indexing of the whole
    folder, and that must be the user's explicit decision (`ptg init` or the
    /ptg-mem:init command) — never a side effect of opening a session."""
    cfg = paths.load_config()
    if not cfg.get("auto_register", False) or not cwd or not os.path.isdir(cwd):
        return False
    root = paths.norm_root(cwd)
    home = paths.norm_root(os.path.expanduser("~"))
    # never index a whole home folder or a drive: that is a machine, not a project
    if root == home or os.path.dirname(root) == root or len(root) <= 3:
        return False
    if paths.find_project(root):
        return False
    proj = paths.new_project(root)
    if cfg.get("default_embedder"):
        proj["embedder"] = cfg["default_embedder"]
    paths.register(proj)
    return True


def session_start(data: dict) -> None:
    cwd = data.get("cwd")
    if not cwd:
        return                                       # never guess the folder
    new = _auto_register(cwd)
    if not paths.find_project(cwd):
        return                                       # not a PTG-MEM project: stay silent
    if not client.ensure_daemon(wait=4.0):
        return
    try:
        r = client.call("/api/brief", {"cwd": cwd}, timeout=3.0)
    except Exception:                                # noqa: BLE001
        return
    ctx = r.get("context") or ""
    if not ctx and (new or r.get("status") in ("loading", "idle")):
        ctx = ("<ptg-memory>PTG-MEM is loading this project's memory in the background; it will "
               "be attached to prompts as soon as it is ready.</ptg-memory>")
    _emit("SessionStart", ctx)


def prompt(data: dict) -> None:
    cwd = data.get("cwd")
    if not cwd or not paths.find_project(cwd):
        return                                       # not a PTG-MEM project: no daemon, no cost
    if not client.alive(timeout=0.4):
        client.spawn_daemon()                        # ready by the next prompt
        return
    payload = {"cwd": cwd, "prompt": data.get("prompt") or "",
               "session_id": data.get("session_id"), "transcript_path": data.get("transcript_path")}
    try:
        r = client.call("/api/recall", payload, timeout=PROMPT_TIMEOUT)
    except Exception:                                # noqa: BLE001
        return
    _emit("UserPromptSubmit", r.get("context") or "")


def stop(data: dict) -> None:
    tp, cwd = data.get("transcript_path"), data.get("cwd")
    if not tp or not cwd or not paths.find_project(cwd) or not client.alive(timeout=0.3):
        return
    try:
        client.call("/api/ingest", {"cwd": cwd, "path": tp, "kind": "claude"}, timeout=1.0)
    except Exception:                                # noqa: BLE001
        pass


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else ""
    data = _read_input()
    try:
        {"session-start": session_start, "prompt": prompt, "stop": stop}.get(cmd, lambda d: None)(data)
    except Exception:                                # noqa: BLE001 — a hook never breaks the session
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
