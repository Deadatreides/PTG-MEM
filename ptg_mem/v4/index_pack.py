# -*- coding: utf-8 -*-
r"""PTG_V4/index_pack.py — индекс ветвей фиксированной записью вместо JSON.

ПРИНЦИП взят у архиваторов и методов доступа мейнфреймов: **индекс отделён от данных,
состоит из записей фиксированной длины и мал**. У ZIP это центральный каталог, у 7-Zip —
отдельный заголовок, у ISAM — индексная область. Его читают целиком, данные — выборочно.

ЧТО НЕ ТАК СЕЙЧАС. Центроиды ветвей лежат в `branch_bodies.json` текстом, по одному числу
на строку с отступами — около 30 байт на число вместо 4. Индекс раздут, читается разбором
всего файла, и держит ОЗУ, которая рядом с моделью стоит ~3 с на токен за гигабайт.

ЧТО ДЕЛАЕТ. Достаёт центроиды в матрицу `centroids.npy` (float32, запись фиксированной
длины) и порядок ветвей в `branch_ids.json`. Центроид ветви `i` после этого лежит по
смещению `i·dim·4` — адресная арифметика вместо поиска, и файл отображается в память
вместо разбора.

Архив не трогается: пишется в свой каталог.

Запуск: python index_pack.py <путь к .ptg> [<выходной каталог>]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


def pack(ptg_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(ptg_dir, "branch_bodies.json")
    size_json = os.path.getsize(src)

    t0 = time.time()
    with open(src, encoding="utf-8") as f:
        bodies = json.load(f)
    t_json = time.time() - t0

    ids, rows = [], []
    dim = None
    skipped = 0
    for bid, body in bodies.items():
        c = body.get("centroid")
        if not c:
            skipped += 1
            continue
        if dim is None:
            dim = len(c)
        if len(c) != dim:
            skipped += 1
            continue
        ids.append(bid)
        rows.append(c)
    mat = np.asarray(rows, dtype=np.float32)

    npy_path = os.path.join(out_dir, "centroids.npy")
    np.save(npy_path, mat)
    with open(os.path.join(out_dir, "branch_ids.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)
    size_npy = os.path.getsize(npy_path)

    t0 = time.time()
    m = np.load(npy_path, mmap_mode="r")
    _ = float(m[0, 0])                    # тронуть первую запись, чтобы отображение состоялось
    t_npy = time.time() - t0

    # остальное тело ветви (atom_ids и прочее) — без центроидов оно невелико
    rest = {bid: {k: v for k, v in body.items() if k != "centroid"}
            for bid, body in bodies.items()}
    rest_path = os.path.join(out_dir, "branch_meta.json")
    with open(rest_path, "w", encoding="utf-8") as f:
        json.dump(rest, f, ensure_ascii=False)
    size_rest = os.path.getsize(rest_path)

    print("ветвей           %d (пропущено без центроида %d), размерность %s" % (len(ids), skipped, dim))
    print()
    print("  %-34s %12s %12s" % ("", "было", "стало"))
    print("  %-34s %10.1f МБ %10.1f МБ" % ("индекс центроидов",
                                           size_json / 1e6, size_npy / 1e6))
    print("  %-34s %12s %10.1f МБ" % ("остальное тело ветви (отдельно)", "—", size_rest / 1e6))
    print("  %-34s %12s %11.1fx" % ("сжатие индекса", "—", size_json / size_npy if size_npy else 0))
    print("  %-34s %10.2f с %11.3f с" % ("время загрузки", t_json, t_npy))
    print("  %-34s %12s %11.0fx" % ("ускорение загрузки", "—", t_json / t_npy if t_npy else 0))
    return {"branches": len(ids), "dim": dim, "size_json": size_json,
            "size_npy": size_npy, "size_rest": size_rest,
            "t_json": t_json, "t_npy": t_npy}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out")
    r = pack(sys.argv[1], out)
    with open(os.path.join(out, "index_pack_report.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
