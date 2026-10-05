# -*- coding: utf-8 -*-
"""Where PTG-MEM keeps its state, and the project registry.

Everything lives under one home directory (``PTG_MEM_HOME`` or ``~/.ptg-mem``):

    config.json     global settings (port, injection budget, idle unload)
    projects.json   registered projects: root -> store, embedder, thresholds
    daemon.json     the running daemon: pid, port, access token
    license.key     Pro licence, if any
    logs/           daemon log
    stores/         default place for project stores

A store can live anywhere (``ptg init --store``): on a machine with a small
system disk the store belongs on the big one, and only this small registry
stays in the home directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time

HOME = os.path.abspath(os.path.expanduser(os.environ.get("PTG_MEM_HOME", "~/.ptg-mem")))
CONFIG = os.path.join(HOME, "config.json")
PROJECTS = os.path.join(HOME, "projects.json")
DAEMON = os.path.join(HOME, "daemon.json")
LICENSE = os.path.join(HOME, "license.key")
LOGS = os.path.join(HOME, "logs")
STORES = os.path.join(HOME, "stores")

DEFAULT_PORT = 7437

# Thresholds of the graft engine. Calibrated on Qwen3-Embedding (4B) against the
# similarity distribution of a real 28k-atom archive; ptg_core's own literals
# (0.60/0.78/0.83) were set for a different model and leave most atoms orphaned.
DEFAULT_THRESHOLDS = {"branch": 0.47, "continue": 0.57, "return": 0.65,
                      "contradict": 0.62, "supersede": 0.57}
DEFAULT_MARGIN = 5.0          # budget d = ceil(margin * ln n); 5.0 keeps d above 3 ln n

DEFAULT_CONFIG = {
    "port": DEFAULT_PORT,
    "idle_unload_seconds": 600,       # free the embedder (VRAM) after this much idleness
    "save_after_seconds": 45,         # save a dirty store once ingestion has been quiet this long
    "save_every_seconds": 600,        # ...and never keep it dirty longer than this
    "inject": {
        "enabled": True,
        "budget_chars": 3600,         # ~1k tokens per prompt, hard cap
        "max_items": 5,
        "min_score": 0.50,            # cosine; below this nothing is injected at all
        "session_brief": True,        # SessionStart: project brief + open threads
    },
}

_lock = threading.Lock()


def ensure_home() -> None:
    for d in (HOME, LOGS, STORES):
        os.makedirs(d, exist_ok=True)


def _read(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write(path: str, data) -> None:
    ensure_home()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def norm_root(path: str) -> str:
    """One spelling per project folder: absolute, normalised, no trailing slash.
    Case is kept — on Windows the drive letter is upper-cased only."""
    p = os.path.normpath(os.path.abspath(path))
    if os.name == "nt" and len(p) > 1 and p[1] == ":":
        p = p[0].upper() + p[1:]
    return p


def slug(root: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "-", os.path.basename(root)).strip("-").lower() or "project"
    return "%s-%s" % (base[:40], hashlib.sha1(root.encode("utf-8")).hexdigest()[:8])


# --- global config -----------------------------------------------------------
def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    user = _read(CONFIG, {})
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def save_config(cfg: dict) -> None:
    with _lock:
        _write(CONFIG, cfg)


# --- projects ----------------------------------------------------------------
def load_projects() -> dict:
    return _read(PROJECTS, {"projects": {}}).get("projects", {})


def save_projects(projects: dict) -> None:
    with _lock:
        _write(PROJECTS, {"projects": projects})


def find_project(path: str):
    """The registered project that contains ``path`` (deepest match), or None.
    Hooks receive the session's cwd, which may be a sub-folder of the project."""
    p = norm_root(path)
    best = None
    for root, cfg in load_projects().items():
        r = norm_root(root)
        if p == r or p.startswith(r.rstrip("\\/") + os.sep):
            if best is None or len(r) > len(best[0]):
                best = (r, cfg)
    return best[1] if best else None


def new_project(root: str, store: str | None = None, embedder: dict | None = None,
                thresholds: dict | None = None, margin: float | None = None) -> dict:
    root = norm_root(root)
    return {
        "root": root,
        "name": os.path.basename(root) or root,
        "store": os.path.abspath(store) if store else os.path.join(STORES, slug(root)),
        "embedder": embedder or {"backend": "st", "model": "Qwen/Qwen3-Embedding-0.6B"},
        "thresholds": dict(thresholds or DEFAULT_THRESHOLDS),
        "margin": float(margin or DEFAULT_MARGIN),
        "sources": {"folder": True, "claude": True, "codex": True},
        "exclude_dirs": [],
        "created": time.time(),
    }


def register(project: dict) -> None:
    projects = load_projects()
    projects[project["root"]] = project
    save_projects(projects)


# --- daemon --------------------------------------------------------------------
def load_daemon() -> dict:
    return _read(DAEMON, {})


def save_daemon(info: dict) -> None:
    _write(DAEMON, info)
