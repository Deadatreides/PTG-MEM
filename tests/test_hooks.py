# -*- coding: utf-8 -*-
"""Hooks must be silent and harmless outside registered projects, and must read
their JSON input even with a BOM in front of it (PowerShell pipes add one)."""
import io
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(REPO, "bin", "ptg-hook.py")


def _run(cmd, payload: bytes, home):
    env = dict(os.environ, PTG_MEM_HOME=str(home))
    r = subprocess.run([sys.executable, HOOK, cmd], input=payload, capture_output=True, env=env, timeout=30)
    return r.returncode, r.stdout.decode("utf-8", "replace")


def test_unregistered_folder_is_silent_and_registers_nothing(tmp_path):
    home = tmp_path / "home"
    body = json.dumps({"cwd": str(tmp_path), "prompt": "anything at all", "session_id": "x"}).encode()
    for cmd in ("session-start", "prompt", "stop"):
        code, out = _run(cmd, b"\xef\xbb\xbf" + body, home)
        assert code == 0 and out == ""
    assert not (home / "projects.json").exists()


def test_garbage_input_is_harmless(tmp_path):
    code, out = _run("prompt", b"\x00\xffnot json", tmp_path / "home")
    assert code == 0 and out == ""


def test_bom_input_is_parsed(monkeypatch):
    sys.path.insert(0, REPO)
    from ptg_mem import hooks
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b'\xef\xbb\xbf{"cwd": "X"}')))
    assert hooks._read_input() == {"cwd": "X"}
