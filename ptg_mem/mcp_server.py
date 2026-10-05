# -*- coding: utf-8 -*-
"""MCP server: the same memory, for agents that ask.

Six tools instead of the research server's seventeen. Every call goes to the
daemon, so the server starts instantly and holds nothing in memory; if the
daemon is not running it is started.

The project is the folder the agent works in: ``PTG_MEM_PROJECT``, else
``CLAUDE_PROJECT_DIR``, else the current directory.
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from . import client

mcp = FastMCP("ptg-mem")


def _cwd() -> str:
    return os.environ.get("PTG_MEM_PROJECT") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def _call(path, payload, timeout=30.0):
    if not client.alive():
        client.ensure_daemon(wait=10.0)
    payload = dict(payload, cwd=_cwd())
    return client.call(path, payload, timeout=timeout)


def _fmt(items):
    out = []
    for it in items:
        line = {"id": it["id"], "score": it.get("score"), "date": it.get("date"),
                "file": it.get("rel_file"), "status": it.get("status"), "text": it.get("text")}
        if it.get("current_id"):
            line["superseded_by"] = it["current_id"]
        rel = [r for r in it.get("relations", []) if r["type"] in ("supersedes", "contradicts", "fixes")]
        if rel:
            line["relations"] = [{"type": r["type"], "dir": r["dir"], "id": r["id"], "date": r["date"]}
                                 for r in rel[:6]]
        out.append(line)
    return out


@mcp.tool()
def ptg_search(query: str, k: int = 8) -> list:
    """Search this project's memory (notes, docs, past agent sessions) by meaning.
    Each hit carries its date and file; `superseded_by` means a newer atom replaced
    it — open that one with ptg_node before relying on the old text."""
    return _fmt(_call("/api/search", {"query": query, "k": k}).get("items", []))


@mcp.tool()
def ptg_node(node_id: str) -> dict:
    """Full text of one memory atom, its typed relations (supersedes, contradicts,
    fixes, revises) and its neighbourhood in the line of thought it belongs to."""
    return _call("/api/node", {"id": node_id})


@mcp.tool()
def ptg_decisions(limit: int = 20) -> list:
    """The decision log: what replaced, contradicted, fixed or revised what, newest
    first. Use it before re-proposing something that may already have been tried."""
    return _call("/api/decisions", {"limit": limit}).get("items", [])


@mcp.tool()
def ptg_remember(text: str) -> dict:
    """Store a decision, a finding or a constraint in memory now, in one or two
    self-contained sentences (what was decided and why). It will be recalled in
    later sessions automatically."""
    r = _call("/api/remember", {"text": text, "source": "agent"})
    return {"stored": r.get("id"), "date": r.get("date")}


@mcp.tool()
def ptg_recall(task: str) -> str:
    """The context PTG-MEM would attach for this task: the few most relevant
    memories, with supersession and contradiction markers. Empty if nothing in
    memory is relevant enough."""
    return _call("/api/recall", {"prompt": task}).get("context") or "(nothing relevant in memory)"


@mcp.tool()
def ptg_status() -> dict:
    """Is memory up to date: atoms, files, pending ingestion, embedder state."""
    r = _call("/api/status", {})
    return {"projects": r.get("projects"), "queue": r.get("queue"), "version": r.get("version")}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
