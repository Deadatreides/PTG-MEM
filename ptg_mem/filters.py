# -*- coding: utf-8 -*-
"""What gets into memory, and what never does.

The file filter comes from an inventory of a real 50 GB research folder
(43 152 files): raw run traces, weights and vendored upstream code are data, not
thought, and embedding them drowns the few hundred files where the reasoning
lives. Two tiers:

  * included and embedded: docs, notes, configs, scripts, source files
    (``.py`` contributes only docstrings and comments);
  * excluded whole: model weights, run dumps, vendored trees, caches, VCS.

Secrets are cut out of every text before it is embedded or stored. Memory has
no delete button by design (nodes are retracted, never erased), so the ingest
path is the last place where a pasted API key can still be stopped.
"""
from __future__ import annotations

import os
import re

EXCLUDE_DIRS = {
    ".ptg", ".ptg-mem", ".code_ptg", ".git", ".hg", ".svn", ".idea", ".vscode",
    "__pycache__", ".ipynb_checkpoints", "node_modules", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages",
    "dist", "build", "target", ".next", ".nuxt", "coverage",
    "_snapshot_run", "models", "traces", "runs", "raw", "out", "outputs", "logs", "log",
    "checkpoints", "ckpt", "wandb", "mlruns", "artifacts", "cache", "tmp", "temp",
    "vendor", "vendor_ptg", "third_party", "llama.cpp",
}
DOC_EXTS = {".md", ".markdown", ".mdx", ".rst", ".txt", ".org", ".adoc", ".tex"}
CODE_EXTS = {".py"}                                  # docstrings + comments only
NATIVE_EXTS = {".cpp", ".hpp", ".cc", ".c", ".h", ".cu", ".rs", ".go", ".java", ".kt",
               ".ts", ".tsx", ".js", ".jsx", ".cs", ".swift", ".rb", ".php", ".lua"}
CFG_EXTS = {".yaml", ".yml", ".toml", ".ini", ".cfg"}
SCRIPT_EXTS = {".sh", ".bat", ".ps1"}
OFFICE_EXTS = {".docx", ".doc"}
INCLUDE_EXTS = DOC_EXTS | CODE_EXTS | NATIVE_EXTS | CFG_EXTS | SCRIPT_EXTS | OFFICE_EXTS

EXCLUDE_NAMES = {"code_dump.txt", ".env", "poetry.lock", "package-lock.json", "yarn.lock",
                 "pnpm-lock.yaml", "cargo.lock", "uv.lock"}
EXCLUDE_SUFFIX = (".min.js", ".lock", ".orig", ".incomplete", ".map")

MAX_FILE_BYTES = 1_200_000
MIN_FILE_BYTES = 24


class FileFilter:
    def __init__(self, root: str, exclude_dirs=(), include_exts=None, store_dir: str | None = None):
        self.root = os.path.normpath(root)
        self.exclude_dirs = {d.lower() for d in EXCLUDE_DIRS} | {d.lower() for d in exclude_dirs}
        self.include_exts = set(include_exts) if include_exts else INCLUDE_EXTS
        self.store_dir = os.path.normpath(store_dir) if store_dir else None

    def dir_ok(self, name: str) -> bool:
        n = name.lower()
        return n not in self.exclude_dirs and not n.startswith(".")

    def path_ok(self, path: str) -> bool:
        """Full check for a single path, used for watcher events: every folder on
        the way from the project root must pass, and the file itself too."""
        p = os.path.normpath(path)
        if self.store_dir and (p == self.store_dir or p.startswith(self.store_dir + os.sep)):
            return False
        try:
            rel = os.path.relpath(p, self.root)
        except ValueError:
            return False
        if rel.startswith(".."):
            return False
        parts = rel.split(os.sep)
        if any(not self.dir_ok(d) for d in parts[:-1]):
            return False
        return self.file_ok(p)

    def file_ok(self, path: str) -> bool:
        name = os.path.basename(path).lower()
        if name in EXCLUDE_NAMES or name.endswith(EXCLUDE_SUFFIX):
            return False
        if os.path.splitext(name)[1] not in self.include_exts:
            return False
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        return MIN_FILE_BYTES <= size <= MAX_FILE_BYTES

    def walk(self):
        """All accepted files under the root, in a stable order."""
        for dirpath, dirs, files in os.walk(self.root):
            dirs[:] = sorted(d for d in dirs if self.dir_ok(d)
                             and not (self.store_dir and os.path.normpath(
                                 os.path.join(dirpath, d)) == self.store_dir))
            for f in sorted(files):
                p = os.path.join(dirpath, f)
                if self.file_ok(p):
                    yield p


# --- secrets -----------------------------------------------------------------
SECRET_RE = re.compile(
    r"\b("
    r"uak_[A-Za-z0-9_\-]{16,}"            # Composio user API key
    r"|ak_[A-Za-z0-9_\-]{20,}"            # Composio project key
    r"|ck_[A-Za-z0-9_\-]{16,}"            # Composio consumer key
    r"|sk-[A-Za-z0-9_\-]{20,}"            # OpenAI, Anthropic (sk-ant-...) and compatibles
    r"|gh[pousr]_[A-Za-z0-9]{20,}"        # GitHub tokens
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|glpat-[A-Za-z0-9_\-]{20,}"         # GitLab
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"     # Slack
    r"|AIza[A-Za-z0-9_\-]{30,}"           # Google
    r"|hf_[A-Za-z0-9]{20,}"               # Hugging Face
    r"|AKIA[0-9A-Z]{16}"                  # AWS access key id
    r"|eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"   # JWT
    r")\b"
)
PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)


def scrub_secrets(text: str) -> tuple[str, int]:
    """Replace anything that looks like a credential. Returns (text, replacements)."""
    n = 0

    def _sub(m):
        nonlocal n
        n += 1
        tok = m.group(1)
        head = tok.split("_")[0] if "_" in tok[:8] else tok[:4]
        return "<secret-removed:%s>" % head

    text = SECRET_RE.sub(_sub, text)

    def _pk(_m):
        nonlocal n
        n += 1
        return "<secret-removed:private-key>"

    text = PRIVATE_KEY_RE.sub(_pk, text)
    return text, n
