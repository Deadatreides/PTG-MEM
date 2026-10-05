# -*- coding: utf-8 -*-
"""Talking to the daemon. Standard library only: hooks import this on every
prompt, and a hook that spends a second importing numpy is a hook people remove.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from . import paths


class DaemonDown(RuntimeError):
    pass


def _info():
    return paths.load_daemon()


def url(path: str, info: dict | None = None) -> str:
    info = info or _info()
    return "http://127.0.0.1:%d%s" % (int(info.get("port") or paths.DEFAULT_PORT), path)


def call(path: str, payload: dict | None = None, timeout: float = 5.0, method: str | None = None):
    info = _info()
    if not info.get("port"):
        raise DaemonDown("daemon not started")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    # a project path in the query string is often non-ASCII (Cyrillic folders): http.client
    # refuses it unencoded, so percent-encode whatever is not already safe
    path = urllib.parse.quote(path, safe="/?&=%:+,;@-._~")
    req = urllib.request.Request(url(path, info), data=data,
                                 method=method or ("POST" if data is not None else "GET"))
    req.add_header("Content-Type", "application/json")
    req.add_header("X-PTG-Token", info.get("token", ""))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as ex:
        try:
            body = json.loads(ex.read().decode("utf-8"))
        except Exception:                            # noqa: BLE001
            body = {"error": str(ex)}
        raise RuntimeError(body.get("error") or str(ex)) from ex
    except (urllib.error.URLError, OSError, TimeoutError) as ex:
        raise DaemonDown(str(ex)) from ex


def alive(timeout: float = 0.6) -> bool:
    info = _info()
    if not info.get("port"):
        return False
    try:
        with urllib.request.urlopen(url("/api/health", info), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")).get("ok") is True
    except Exception:                                # noqa: BLE001
        return False


def spawn_daemon() -> None:
    """Start the daemon detached from the caller (a hook or an MCP server must
    not own it: they come and go with the session, memory should not)."""
    paths.ensure_home()
    exe = sys.executable
    if os.name == "nt":
        w = os.path.join(os.path.dirname(exe), "pythonw.exe")
        if os.path.exists(w):
            exe = w
    log = open(os.path.join(paths.LOGS, "daemon.out"), "ab")
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": log, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                   | getattr(subprocess, "DETACHED_PROCESS", 0x8)
                                   | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    else:
        kwargs["start_new_session"] = True
    # the package may run from a checkout rather than an install: make it importable
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pp = os.environ.get("PYTHONPATH")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONPATH=pkg_parent + (os.pathsep + pp if pp else ""))
    subprocess.Popen([exe, "-m", "ptg_mem.daemon"], env=env, **kwargs)


def ensure_daemon(wait: float = 8.0) -> bool:
    if alive():
        return True
    spawn_daemon()
    end = time.time() + wait
    while time.time() < end:
        time.sleep(0.25)
        if alive():
            return True
    return False
