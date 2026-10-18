# -*- coding: utf-8 -*-
"""PTG-MEM 0.3 engine (memcore + memshell + LanceDB): the 0.2 behaviour of tests/test_engine.py, plus what 0.3 adds.

Deterministic hash embedder, nothing downloaded. The 0.2 internals the old test looked at (``eng.arc.edges``, the
``<secret-removed`` mark) are read the 0.3 way: an edited paragraph supersedes its old version with source "edit";
a secret becomes ``<secret>`` (memcore.scrub).

    PTG-MEM\\rev-shell\\.venv\\Scripts\\python -m pytest tests/test_engine3.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("lancedb")
pytest.importorskip("memcore")
pytest.importorskip("memshell")


ENGINE = {"kind": "v3"}


@pytest.fixture(params=["v3", "full"])
def env(request, tmp_path, monkeypatch):
    """Each behaviour on both engines: engine3 and the full PTG (ptgcore, PTG-MEM/rev-core on the path)."""
    if request.param == "full":
        pytest.importorskip("ptgcore")
    monkeypatch.setitem(ENGINE, "kind", request.param)
    monkeypatch.setenv("PTG_MEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import importlib
    import ptg_mem.paths
    importlib.reload(ptg_mem.paths)
    proj = tmp_path / "proj"
    (proj / "docs").mkdir(parents=True)
    return tmp_path, proj


def _engine(tmp_path, proj, embedder=None):
    from ptg_mem import paths
    from ptg_mem.embedders import make_embedder
    from ptg_mem.engine3 import MemoryEngine3
    p = paths.new_project(str(proj), store=str(tmp_path / "store"), embedder=embedder or {"backend": "hash"})
    p["sources"] = {"folder": True, "claude": False, "codex": False}
    if ENGINE["kind"] == "full":
        from ptg_mem.engine_full import MemoryEngineFull
        return MemoryEngineFull(p, make_embedder(p["embedder"])).open(), p
    return MemoryEngine3(p, make_embedder(p["embedder"])).open(), p


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

    # edit one answer: exactly one atom retracted, one added, the new one supersedes the old (source "edit")
    _w(proj / "docs" / "decisions.md",
       "Question: which database for vectors?\n\nAnswer: DuckDB now, SQLite outgrew the index.\n\n"
       "Question: how do we batch?\n\nAnswer: one text per call.\n")
    r = eng.ingest(f, "file")
    assert r["added"] == 1 and r["retracted"] == 1 and r["kept"] >= 1
    assert any(e.kind == "supersedes" and e.source == "edit" for e in eng.edges)
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
    r = eng.recall("how long do we keep embeddings in the cache", budget_chars=700, max_items=3, min_score=0.1)
    assert r["context"].startswith("<ptg-memory>") and r["context"].endswith("</ptg-memory>")
    assert len(r["context"]) <= 700                                   # 0.3: strictly inside the budget
    assert 1 <= len(r["items"]) <= 3
    assert eng.recall("hi", min_score=0.1)["context"] == ""          # too short to recall for


def test_remember_and_secrets(env):
    tmp_path, proj = env
    eng, _ = _engine(tmp_path, proj)
    card = eng.remember("Decision: rotate the key sk-abcdefghijklmnopqrstuvwxyz0123 and never log it.")
    assert "sk-abcdefghijklmnop" not in card["text"]
    assert "<secret>" in card["text"]
    hits = eng.search("rotate the key and never log it", k=3)
    assert hits and hits[0]["id"] == card["id"]


def test_edited_note_is_marked_in_recall_and_decisions(env):
    """0.3: the old version of an edited paragraph, when recalled, says what replaced it; the decision feed shows the
    pair; the edge survives a reload."""
    tmp_path, proj = env
    eng, _ = _engine(tmp_path, proj)
    f = _w(proj / "docs" / "plan.md", "Storage decision: we keep vectors in SQLite with a flat index.\n\nBatching: one text per call.\n")
    eng.ingest(f, "file")
    _w(proj / "docs" / "plan.md", "Storage decision: we keep vectors in LanceDB, SQLite outgrew the index.\n\nBatching: one text per call.\n")
    eng.ingest(f, "file")
    d = eng.decisions()
    assert len(d) == 1 and d[0]["type"] == "supersedes" and "LanceDB" in d[0]["new"]["text"] and "SQLite with" in d[0]["old"]["text"]
    old_id = d[0]["old"]["id"]
    assert eng.card(old_id)["current_id"] == d[0]["new"]["id"] and eng.stats()["superseded"] == 0  # the old one is retracted
    eng2, _ = _engine(tmp_path, proj)
    assert [x["new"]["id"] for x in eng2.decisions()] == [d[0]["new"]["id"]]


def test_agent_claimed_relation_and_brief(env):
    tmp_path, proj = env
    eng, _ = _engine(tmp_path, proj)
    a = eng.remember("Decision: the recall window is 0.12 below the best hit.")
    b = eng.remember("Decision: the recall window is 0.10 below the best hit, 0.12 let noise in.")
    eng.relate(b["id"], a["id"], "supersedes", source="agent")
    assert eng.card(a["id"])["current_id"] == b["id"] and eng.card(a["id"])["status"] == "superseded"
    r = eng.recall("what is the recall window below the best hit", min_score=0.0, max_items=5)
    assert "replaced later" in r["context"] or "replaces an earlier version" in r["context"]
    brief = eng.brief()
    assert brief.startswith("<ptg-memory>") and brief.endswith("</ptg-memory>") and "2 memory atoms" in brief and "0.10" in brief


def test_search_falls_back_to_words_without_the_embedder(env):
    """P1 of memcore: the embedder is down (GPU busy) — memory still answers, by words."""
    tmp_path, proj = env
    eng, _ = _engine(tmp_path, proj)
    _w(proj / "docs" / "a.md", "LanceDB keeps the atom vectors.\n\nChonkie cuts long paragraphs only.\n")
    eng.ingest(str(proj / "docs" / "a.md"), "file")

    def broken(texts):
        raise RuntimeError("CUDA out of memory")

    eng.embedder.embed = broken
    hits = eng.search("Chonkie paragraphs", k=3)
    assert hits and "Chonkie" in hits[0]["text"] and eng.embedder_failed


def test_store_refuses_another_model(env):
    tmp_path, proj = env
    eng, _ = _engine(tmp_path, proj)
    eng.remember("Decision: one store, one embedding model.")
    with pytest.raises(RuntimeError, match="not comparable"):
        _engine(tmp_path, proj, embedder={"backend": "hash", "model": "hash-other", "dim": 128})


def test_binary_file_is_harmless(env):
    tmp_path, proj = env
    eng, _ = _engine(tmp_path, proj)
    (proj / "docs" / "blob.md").write_bytes(bytes(range(256)) * 8)
    r = eng.ingest(str(proj / "docs" / "blob.md"), "file")
    assert r["retracted"] == 0
