# -*- coding: utf-8 -*-
"""Wire PTG-MEM into Claude Code and Codex.

Claude Code: three hooks (SessionStart / UserPromptSubmit / Stop) and an MCP
server. Codex: the same hooks (Codex accepts the same hook schema and the same
``additionalContext`` output) plus the MCP server in config.toml.

Every file is backed up before it is changed, previous PTG-MEM entries are
replaced rather than duplicated, and ``--remove`` takes them out again.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time

from .transcripts import codex_home

PKG = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(PKG)
HOOK = os.path.join(REPO, "bin", "ptg-hook.py")
MCP = os.path.join(REPO, "bin", "ptg-mcp.py")
# a checkout has bin/ scripts; an installed wheel has the package on sys.path instead
FROM_CHECKOUT = os.path.isfile(HOOK)
HOOK_ARGS = [HOOK] if FROM_CHECKOUT else ["-m", "ptg_mem.hooks"]
MCP_ARGS = [MCP] if FROM_CHECKOUT else ["-m", "ptg_mem.mcp_server"]
MARKS = ("ptg-hook.py", "ptg_mem.hooks")


def _python() -> str:
    exe = sys.executable
    if os.name == "nt" and exe.lower().endswith("pythonw.exe"):
        exe = exe[:-5] + ".exe"
    return exe


def _backup(path: str):
    if os.path.exists(path):
        shutil.copy2(path, "%s.bak-ptg-%s" % (path, time.strftime("%Y%m%d-%H%M%S")))


def _load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _strip_ours(hooks: dict) -> dict:
    """Remove PTG-MEM entries (any version), keep everything else untouched."""
    out = {}
    for event, groups in (hooks or {}).items():
        kept = []
        for g in groups or []:
            hs = [h for h in g.get("hooks", [])
                  if not any(m in (h.get("command", "") + " " + " ".join(h.get("args", []) or [])
                                   + " " + h.get("commandWindows", "")) for m in MARKS)]
            if hs:
                kept.append(dict(g, hooks=hs))
        if kept:
            out[event] = kept
    return out


def claude_hooks() -> dict:
    py = _python()

    def h(cmd, timeout, **kw):
        return {"type": "command", "command": py, "args": HOOK_ARGS + [cmd], "timeout": timeout, **kw}

    return {
        "SessionStart": [{"hooks": [h("session-start", 15)]}],
        "UserPromptSubmit": [{"hooks": [h("prompt", 10)]}],
        "Stop": [{"hooks": [h("stop", 5, **{"async": True})]}],
    }


def install_claude(root: str | None, scope: str = "project", remove: bool = False) -> list[str]:
    done = []
    if scope == "user":
        settings = os.path.join(os.path.expanduser("~"), ".claude", "settings.json")
    else:
        settings = os.path.join(root, ".claude", "settings.local.json")
    data = _load_json(settings)
    _backup(settings)
    hooks = _strip_ours(data.get("hooks") or {})
    if not remove:
        for event, groups in claude_hooks().items():
            hooks.setdefault(event, []).extend(groups)
    if hooks:
        data["hooks"] = hooks
    else:
        data.pop("hooks", None)
    _save_json(settings, data)
    done.append(("removed hooks from " if remove else "hooks -> ") + settings)

    # MCP server
    server = {"command": _python(), "args": list(MCP_ARGS)}
    if scope == "user":
        cli = shutil.which("claude")
        if cli:
            subprocess.run([cli, "mcp", "remove", "--scope", "user", "ptg-mem"], capture_output=True)
            if not remove:
                subprocess.run([cli, "mcp", "add", "--scope", "user", "ptg-mem", "--", _python()] + MCP_ARGS,
                               capture_output=True)
            done.append("MCP (user scope) via claude CLI")
        else:
            done.append("claude CLI not on PATH — add the MCP server by hand: claude mcp add --scope "
                        "user ptg-mem -- \"%s\" %s" % (_python(), " ".join(MCP_ARGS)))
    else:
        mcp_path = os.path.join(root, ".mcp.json")
        m = _load_json(mcp_path)
        _backup(mcp_path)
        servers = m.setdefault("mcpServers", {})
        servers.pop("ptg-mem", None)
        if not remove:
            servers["ptg-mem"] = dict(server, env={"PTG_MEM_PROJECT": root})
        _save_json(mcp_path, m)
        done.append(("removed MCP from " if remove else "MCP -> ") + mcp_path)
    return done


def _shell_cmd(*parts) -> str:
    return " ".join(('"%s"' % p) if (" " in p) else p for p in parts)


def install_codex(remove: bool = False) -> list[str]:
    home = codex_home()
    done = []
    hooks_path = os.path.join(home, "hooks.json")
    data = _load_json(hooks_path)
    _backup(hooks_path)
    hooks = _strip_ours(data.get("hooks") or {})
    if not remove:
        py = _python()
        for event, cmd, t in (("SessionStart", "session-start", 15), ("UserPromptSubmit", "prompt", 10),
                              ("Stop", "stop", 5)):
            c = _shell_cmd(py, *HOOK_ARGS, cmd)
            hooks.setdefault(event, []).append(
                {"hooks": [{"type": "command", "command": c, "commandWindows": c, "timeout": t}]})
    data["hooks"] = hooks
    _save_json(hooks_path, data)
    done.append(("removed hooks from " if remove else "hooks -> ") + hooks_path)

    cfg_path = os.path.join(home, "config.toml")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        text = ""
    _backup(cfg_path)
    # drop a previous block of ours (it is always written as one contiguous block)
    text = re.sub(r"\n?# >>> ptg-mem >>>.*?# <<< ptg-mem <<<\n?", "\n", text, flags=re.S)
    if not remove:
        block = ("\n# >>> ptg-mem >>>\n[mcp_servers.ptg-mem]\ncommand = '%s'\nargs = [%s]\n"
                 "startup_timeout_sec = 30\n# <<< ptg-mem <<<\n"
                 % (_python(), ", ".join("'%s'" % a for a in MCP_ARGS)))
        text = text.rstrip("\n") + "\n" + block
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(text)
    done.append(("removed MCP from " if remove else "MCP -> ") + cfg_path)
    return done
