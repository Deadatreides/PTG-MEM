# -*- coding: utf-8 -*-
"""Agent sessions -> memory atoms.

An atom is one exchange: what the user asked, what the agent said back, and one
line per tool call. Reasoning blocks never enter memory — only what was said and
what was done.

Claude Code keeps sessions in ``~/.claude/projects/<cwd with non-alphanumerics
replaced by '-'>/<session-id>.jsonl``; Codex in
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`` with the cwd inside the file.

Atom ``text`` is deterministic for the same exchange, so re-reading a growing
transcript yields the same atoms plus new ones: the engine's per-file diff then
adds only the tail and never duplicates.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re

MIN_CHARS = 20
MAX_TOOL_LINES = 24            # a long agentic turn is mostly tool calls; keep the talk
SYSREM_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
# PTG-MEM's own injected context must not be re-ingested as if the user wrote it
PTG_BLOCK_RE = re.compile(r"<ptg-memory>.*?</ptg-memory>", re.S)


def claude_home() -> str:
    return os.path.join(os.path.expanduser("~"), ".claude")


def codex_home() -> str:
    return os.environ.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex")


def claude_project_dir(root: str) -> str:
    """Claude Code's folder name for a project: every non-alphanumeric -> '-'."""
    return os.path.join(claude_home(), "projects", re.sub(r"[^A-Za-z0-9]", "-", root))


def claude_transcripts(root: str) -> list[str]:
    d = claude_project_dir(root)
    return sorted(glob.glob(os.path.join(d, "*.jsonl"))) if os.path.isdir(d) else []


def _ts(s) -> float | None:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _one_line(s: str, n: int = 200) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


AUTOMATED = ("<local-command", "<command-", "Caveat:", "<task-notification", "<system-reminder",
             "[SYSTEM NOTIFICATION", "<bash-", "<user-memory")


def _clean_user(text: str) -> str:
    text = PTG_BLOCK_RE.sub("", SYSREM_RE.sub("", text)).strip()
    if not text or text.startswith(AUTOMATED):
        return ""                                    # the harness talking, not the user
    return text


class CompactTracker:
    """When was a Claude Code session last compacted? Everything the agent said
    after that is still in its context window; everything before it is only in
    memory. Scans each transcript incrementally (they grow to tens of MB)."""

    def __init__(self):
        self._state: dict[str, tuple[int, float | None]] = {}

    def last_compact(self, path: str) -> float | None:
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        off, ts = self._state.get(path, (0, None))
        if size < off:                                # rewritten: start over
            off, ts = 0, None
        if size > off:
            with open(path, "rb") as f:
                f.seek(off)
                data = f.read()
            end = data.rfind(b"\n") + 1               # only whole lines
            for line in data[:end].splitlines():
                if b"compact_boundary" not in line:
                    continue
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                if o.get("subtype") == "compact_boundary":
                    t = _ts(o.get("timestamp"))
                    if t and (ts is None or t > ts):
                        ts = t
            off += end
            self._state[path] = (off, ts)
        return ts


def _atom(path, order, question, answer_parts, tools, ts, session_id, source):
    if len(tools) > MAX_TOOL_LINES:
        keep = MAX_TOOL_LINES // 2
        tools = tools[:keep] + ["… %d more tool calls …" % (len(tools) - 2 * keep)] + tools[-keep:]
    answer = "\n\n".join(p for p in answer_parts if p.strip())
    if tools:
        answer = (answer + "\n\n" if answer else "") + "Tools: " + "\n".join(tools)
    question = question.strip()
    text = (question + "\n\n" + answer).strip()
    if len(text) < MIN_CHARS:
        return None
    return {"file": path, "order": order, "question": question, "answer": answer,
            "text": text, "confidence": "transcript", "timestamp": ts,
            "source_path": path, "filename": os.path.basename(path),
            "folder_path": os.path.dirname(path), "created_at": ts, "modified_at": ts,
            "session_id": session_id, "source": source}


# --- Claude Code -----------------------------------------------------------------
def _claude_tool_line(block: dict) -> str:
    inp = block.get("input") or {}
    desc = ""
    for key in ("description", "file_path", "pattern", "query", "url", "command", "prompt"):
        if inp.get(key):
            desc = inp[key]
            break
    return "[%s] %s" % (block.get("name", "?"), _one_line(desc))


def parse_claude(path: str) -> list[dict]:
    sid = os.path.splitext(os.path.basename(path))[0]
    atoms, order = [], 0
    q, parts, tools, ts = None, [], [], None
    mtime = os.path.getmtime(path)

    def flush():
        nonlocal order
        if q is None and not parts:
            return
        a = _atom(path, order, q or "", parts, tools, ts or mtime, sid, "claude")
        if a:
            atoms.append(a)
            order += 1

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            t = o.get("type")
            if t not in ("user", "assistant") or o.get("isSidechain") or o.get("isMeta"):
                continue
            msg = o.get("message") or {}
            content = msg.get("content")
            if t == "user" and o.get("isCompactSummary"):
                # the harness's own summary of everything before a compaction: a dense,
                # valuable memory — kept as a standalone atom, not as a user question
                text = content if isinstance(content, str) else "\n\n".join(
                    b.get("text", "") for b in content or [] if isinstance(b, dict))
                flush()
                q, parts, tools, ts = None, [], [], None
                a = _atom(path, order, "Session summary (context compaction)", [text.strip()], [],
                          _ts(o.get("timestamp")) or mtime, sid, "claude")
                if a:
                    atoms.append(a)
                    order += 1
                continue
            if t == "user":
                if isinstance(content, str):
                    text = content
                else:
                    text = "\n\n".join(b.get("text", "") for b in content or []
                                       if isinstance(b, dict) and b.get("type") == "text")
                text = _clean_user(text)
                if not text:
                    continue                      # tool results come back as "user"
                flush()
                q, parts, tools, ts = text, [], [], _ts(o.get("timestamp"))
                continue
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip():
                    parts.append(b["text"].strip())
                elif b.get("type") == "tool_use":
                    tools.append(_claude_tool_line(b))
    flush()
    return atoms


def claude_session_cwd(path: str) -> str | None:
    """The cwd recorded in a Claude Code transcript (first record that has one)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 200:
                    break
                try:
                    cwd = json.loads(line).get("cwd")
                except ValueError:
                    continue
                if cwd:
                    return cwd
    except OSError:
        pass
    return None


# --- Codex -----------------------------------------------------------------------
def codex_session_meta(path: str) -> dict:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            first = json.loads(f.readline() or "{}")
    except (OSError, ValueError):
        return {}
    return first.get("payload") or {} if first.get("type") == "session_meta" else {}


def codex_transcripts(root: str) -> list[str]:
    """Codex rollouts whose recorded cwd is inside ``root``. Sub-agent sessions
    (guardian and the like) are Codex talking to itself and are skipped."""
    base = os.path.join(codex_home(), "sessions")
    if not os.path.isdir(base):
        return []
    r = os.path.normcase(os.path.normpath(root))
    out = []
    for p in sorted(glob.glob(os.path.join(base, "*", "*", "*", "rollout-*.jsonl"))):
        meta = codex_session_meta(p)
        src = meta.get("source")
        if isinstance(src, dict) and "subagent" in src:
            continue
        cwd = meta.get("cwd")
        if not cwd:
            continue
        c = os.path.normcase(os.path.normpath(cwd))
        if c == r or c.startswith(r + os.sep):
            out.append(p)
    return out


def parse_codex(path: str) -> list[dict]:
    meta = codex_session_meta(path)
    sid = meta.get("id") or os.path.splitext(os.path.basename(path))[0]
    atoms, order = [], 0
    q, parts, tools, ts = None, [], [], None
    mtime = os.path.getmtime(path)

    def flush():
        nonlocal order
        if q is None and not parts:
            return
        a = _atom(path, order, q or "", parts, tools, ts or mtime, sid, "codex")
        if a:
            atoms.append(a)
            order += 1

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            p = o.get("payload") or {}
            kind = (o.get("type"), p.get("type"))
            if kind == ("event_msg", "user_message"):
                text = _clean_user(p.get("message") or "")
                if not text:
                    continue
                flush()
                q, parts, tools, ts = text, [], [], _ts(o.get("timestamp"))
            elif kind == ("event_msg", "agent_message"):
                if (p.get("message") or "").strip():
                    parts.append(p["message"].strip())
            elif kind in (("response_item", "function_call"), ("response_item", "custom_tool_call")):
                args = p.get("arguments") or p.get("input") or ""
                try:
                    a = json.loads(args) if isinstance(args, str) else args
                    desc = a.get("command") or a.get("path") or a.get("query") or args
                except (ValueError, AttributeError):
                    desc = args
                if isinstance(desc, list):
                    desc = " ".join(map(str, desc))
                tools.append("[%s] %s" % (p.get("name", "?"), _one_line(desc)))
    flush()
    return atoms
