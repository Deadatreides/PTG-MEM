# -*- coding: utf-8 -*-
r"""Ревизия PTG-MEM, замер 2: чтение документов готовым MarkItDown против своего извлекателя ptg_core.

Выборка — настоящие документы с диска пользователя (.docx, .pdf, .xlsx), без вендорных каталогов
(_internal, site-packages). Для .docx сравниваются оба; .pdf и .xlsx свой извлекатель не читает вовсе — у них
меряется только MarkItDown. Считается: прочитан ли файл, сколько символов текста, сколько кириллицы, время.

    .venv-rev\Scripts\python bench\parse_compare.py [на_тип]
"""
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ptg_mem.core import ptg_core  # noqa: E402

ROOT = Path(os.environ.get("PARSE_ROOT", r"H:\работа"))
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SKIP = re.compile(r"[\\/](_internal|site-packages|node_modules|\.git|dist|build|maker_built|maker_distr)[\\/]", re.I)
CYR = re.compile(r"[А-Яа-яЁё]")


def sample(ext):
    files = [p for p in ROOT.rglob(f"*{ext}") if not SKIP.search(str(p)) and 0 < p.stat().st_size < 20_000_000]
    random.Random(20261010).shuffle(files)
    return files[:PER]


def measure(fn, path):
    t0 = time.time()
    try:
        text = fn(path) or ""
        err = None
    except Exception as e:  # noqa: BLE001 — замер: любой сбой чтения — строка отчёта
        text, err = "", f"{type(e).__name__}: {str(e)[:120]}"
    return {"chars": len(text.strip()), "cyr": len(CYR.findall(text)), "s": round(time.time() - t0, 2), "err": err}


def main():
    from markitdown import MarkItDown
    md = MarkItDown(enable_plugins=False)
    ours = {".docx": ptg_core._extract_docx_text}
    rows = []
    for ext in (".docx", ".pdf", ".xlsx"):
        for p in sample(ext):
            row = {"file": str(p), "ext": ext, "markitdown": measure(lambda x: md.convert(str(x)).text_content, p)}
            if ext in ours:
                row["ours"] = measure(lambda x, f=ours[ext]: f(str(x)), p)
            rows.append(row)
    for ext in (".docx", ".pdf", ".xlsx"):
        part = [r for r in rows if r["ext"] == ext]
        if not part:
            continue
        def stat(key):
            got = [r[key] for r in part if key in r]
            ok = [g for g in got if g["chars"] > 0 and not g["err"]]
            return f"прочитано {len(ok)}/{len(got)}, символов {sum(g['chars'] for g in got)}, кириллицы {sum(g['cyr'] for g in got)}, " \
                   f"время {sum(g['s'] for g in got):.1f} с, сбоев {sum(1 for g in got if g['err'])}"
        print(f"{ext}: файлов {len(part)}")
        print(f"  MarkItDown: {stat('markitdown')}")
        if any("ours" in r for r in part):
            print(f"  свой:       {stat('ours')}")
            worse = [r for r in part if r["markitdown"]["chars"] < 0.8 * r["ours"]["chars"]]
            better = [r for r in part if r["markitdown"]["chars"] > 1.25 * max(1, r["ours"]["chars"])]
            print(f"  MarkItDown меньше на 20 %+: {len(worse)}, больше на 25 %+: {len(better)}")
            for r in worse[:3]:
                print(f"    меньше: {Path(r['file']).name}: {r['markitdown']['chars']} против {r['ours']['chars']}")
        errs = [r for r in part if r["markitdown"]["err"]]
        for r in errs[:3]:
            print(f"    сбой MarkItDown: {Path(r['file']).name}: {r['markitdown']['err']}")
    with io.open(HERE / "out" / "parse_compare.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("записано: bench/out/parse_compare.json")


if __name__ == "__main__":
    main()
