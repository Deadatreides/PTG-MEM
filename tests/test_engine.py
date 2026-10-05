# -*- coding: utf-8 -*-
"""Engine behaviour without a model: diff, retraction, revises, recall, save/load.

Runs on every OS in CI with the deterministic hash embedder, so nothing is
downloaded. Real-model quality is measured separately (bench/)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PTG_MEM_HOME", str(tmp_path / "home"))
    import importlib
    import ptg_mem.paths
    importlib.reload(ptg_mem.paths)
    proj = tmp_path / "proj"
    (proj / "docs").mkdir(parents=True)
    return tmp_path, proj


def _engine(tmp_path, proj):
    from ptg_mem import paths
    from ptg_mem.embedders import make_embedder
    from ptg_mem.engine import MemoryEngine
    p = paths.new_project(str(proj), store=str(tmp_path / "store"), embedder={"backend": "hash"})
    p["sources"] = {"folder": True, "claude": False, "codex": False}
    return MemoryEngine(p, make_embedder(p["embedder"])).open(), p


def _w(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_ingest_edit_delete_and_reload(env):
    tmp_path, proj = env
    eng, p = _engine(tmp_path, proj)
    f = _w(proj / "docs" / "decisions.md",
           "Question: which database for vectors?\n\nAnswer: SQLite with a flat index.\n\n"
           "Question: how do we batch?\n\nAnswer: one text per call.\n")
    _w(proj / "README.md", "# Demo\n\nA tiny service that caches embeddings in SQLite.\n")
    todo, gone = eng.plan()
    assert len(todo) == 2 and not gone
    for path, kind in todo:
        eng.ingest(path, kind)
    s = eng.stats()
    assert s["atoms"] == s["live"] >= 3

    # edit one answer: exactly one atom retracted, one added, a revises edge
    _w(proj / "docs" / "decisions.md",
       "Question: which database for vectors?\n\nAnswer: DuckDB now, SQLite outgrew the index.\n\n"
       "Question: how do we batch?\n\nAnswer: one text per call.\n")
    r = eng.ingest(f, "file")
    assert r["added"] == 1 and r["retracted"] == 1 and r["kept"] >= 1
    assert any(e["type"] == "revises" for e in eng.arc.edges)
    texts = [h["text"] for h in eng.search("database for vectors", k=10)]
    assert not any("SQLite with a flat index" in t for t in texts), "retracted text must not be found"

    # same content again: nothing to do
    r = eng.ingest(f, "file")
    assert r["added"] == 0 and r["retracted"] == 0

    # delete a file: its atoms are retracted, not erased
    os.remove(proj / "README.md")
    todo, gone = eng.plan()
    assert gone == [str(proj / "README.md")]
    r = eng.retract_file(gone[0])
    assert r["retracted"] >= 1
    before = eng.stats()

    eng.save()
    eng2, _ = _engine(tmp_path, proj)
    after = eng2.stats()
    for k in ("atoms", "live", "retracted", "branches", "edges", "files"):
        assert before[k] == after[k], k


def test_recall_marks_and_budget(env):
    tmp_path, proj = env
    eng, _ = _engine(tmp_path, proj)
    for i in range(6):
        _w(proj / ("n%d.md" % i), "Note %d about the cache policy: we keep embeddings for %d days "
                                   "because access is bursty.\n" % (i, 10 + i))
    for path, kind in eng.plan()[0]:
        eng.ingest(path, kind)
    r = eng.recall("how long do we keep embeddings in the cache", budget_chars=700, max_items=3,
                   min_score=0.1)
    assert r["context"].startswith("<ptg-memory>") and r["context"].endswith("</ptg-memory>")
    assert len(r["context"]) <= 700 + 200
    assert 1 <= len(r["items"]) <= 3
    assert eng.recall("hi", min_score=0.1)["context"] == ""          # too short to recall for


def test_remember_and_secrets(env):
    tmp_path, proj = env
    eng, _ = _engine(tmp_path, proj)
    card = eng.remember("Decision: rotate the key sk-abcdefghijklmnopqrstuvwxyz0123 and never log it.")
    assert "sk-abcdefghijklmnop" not in card["text"]
    assert "<secret-removed" in card["text"]
    hits = eng.search("rotate the key and never log it", k=3)
    assert hits and hits[0]["id"] == card["id"]


def test_claude_transcript_parse(tmp_path):
    from ptg_mem import transcripts
    p = tmp_path / "abc12345-0000.jsonl"
    rows = [
        {"type": "user", "timestamp": "2026-09-24T10:00:00Z", "message": {"content": "Why is the build slow?"}},
        {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "secret thoughts"},
                                                       {"type": "text", "text": "Because ccache misses."},
                                                       {"type": "tool_use", "name": "Bash",
                                                        "input": {"command": "ccache -s"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "stats"}]}},
        {"type": "user", "message": {"content": "<task-notification>done</task-notification>"}},
        {"type": "system", "subtype": "compact_boundary", "timestamp": "2026-09-24T11:00:00Z"},
        {"type": "user", "isCompactSummary": True, "timestamp": "2026-09-24T11:00:01Z",
         "message": {"content": "Summary: build fixed by warming ccache."}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    atoms = transcripts.parse_claude(str(p))
    texts = " ".join(a["text"] for a in atoms)
    assert "Because ccache misses." in texts and "[Bash] ccache -s" in texts
    assert "secret thoughts" not in texts                      # reasoning never enters memory
    assert "task-notification" not in texts                    # harness messages are not the user
    assert any(a["question"].startswith("Session summary") for a in atoms)
    assert transcripts.CompactTracker().last_compact(str(p)) is not None
