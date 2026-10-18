# -*- coding: utf-8 -*-
r"""Ревизия PTG-MEM, замер 7: устойчивость нарезки к правке.

Память обновляется по правкам: файл перечитан → атомы с прежним отпечатком остаются, остальные снимаются и
заново кодируются (memcore.plan_sync). Хорошая нарезка при правке одного абзаца меняет один атом. Нарезка с
упаковкой (Chonkie RecursiveChunker набирает куски до chunk_size) может сдвинуть границы всех следующих кусков.

Корпус — заметки памяти Claude Code (md). В каждом файле в абзац посередине дописывается одно предложение.
Считается, сколько атомов файла сменилось (снято + добавлено), и средняя длина атома.

Способы: абзацы ("\n\n", как прежний разбор); Chonkie 512 и 1200 символов (правила по умолчанию, без сети);
абзацы, а длинные (> 1200) — Chonkie внутри абзаца.

    PTG-MEM\rev-shell\.venv\Scripts\python bench\chunk_locality.py
"""
import collections
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from chonkie import RecursiveChunker  # noqa: E402

CORPUS = Path(r"C:\Users\user\.claude\projects\H---------------------------for-codex--1-1---------\memory")
ADD = " Это предложение дописано при правке, чтобы проверить, сколько атомов сменится."
C512 = RecursiveChunker(tokenizer="character", chunk_size=512)
C1200 = RecursiveChunker(tokenizer="character", chunk_size=1200)


def paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chonkie_with(ch):
    return lambda text: [c.text.strip() for c in ch.chunk(text) if c.text.strip()]


def para_then_chonkie(text):
    out = []
    for p in paragraphs(text):
        out += [p] if len(p) <= 1200 else [c.text.strip() for c in C1200.chunk(p) if c.text.strip()]
    return out


WAYS = {"абзацы": paragraphs, "Chonkie 512": chonkie_with(C512), "Chonkie 1200": chonkie_with(C1200),
        "абзацы + Chonkie для длинных": para_then_chonkie}


def edit(text):
    ps = text.split("\n\n")
    idx = [i for i, p in enumerate(ps) if len(p.strip()) > 40]
    if not idx:
        return None
    i = idx[len(idx) // 2]
    ps[i] = ps[i].rstrip() + ADD
    return "\n\n".join(ps)


def main():
    files = sorted(p for p in CORPUS.glob("*.md") if p.name != "MEMORY.md")
    churn, sizes, total = collections.defaultdict(list), collections.defaultdict(list), collections.Counter()
    for f in files:
        text = f.read_text(encoding="utf-8")
        new = edit(text)
        if new is None:
            continue
        for name, cut in WAYS.items():
            a, b = cut(text), cut(new)
            before, after = collections.Counter(a), collections.Counter(b)
            changed = sum((before - after).values()) + sum((after - before).values())
            churn[name].append(changed)
            sizes[name] += [len(x) for x in a]
            total[name] += len(a)
    n = len(churn["абзацы"])
    print(f"файлов: {n}; правка — одно предложение в один абзац\n")
    print(f"{'способ':<30} {'атомов на файл':>15} {'средняя длина':>14} {'сменилось атомов: среднее':>27} {'медиана':>8} {'макс':>6}")
    for name in WAYS:
        c = churn[name]
        print(f"{name:<30} {total[name] / n:>15.1f} {statistics.mean(sizes[name]):>14.0f} {statistics.mean(c):>27.2f} "
              f"{statistics.median(c):>8.0f} {max(c):>6}")


if __name__ == "__main__":
    main()
