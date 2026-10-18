"""Пишет rptg kit. Каждый пример и тест идёт в чистой папке с копией tests/fixtures: базы и файлы, которые пишут
примеры функций с effects: io, не остаются в проекте, и повторный `pytest` даёт тот же итог, что проверки kit на
свежей копии программы."""
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _rptg_clean_dir(tmp_path, monkeypatch):
    fixtures = ROOT / "tests" / "fixtures"
    if fixtures.is_dir():
        shutil.copytree(fixtures, tmp_path / "tests" / "fixtures")
    monkeypatch.chdir(tmp_path)
