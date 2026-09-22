# -*- coding: utf-8 -*-
r"""PTG_V4/incremental.py — обновление архива без перепрогона хранилища.

ЗАДАЧА. Пересборка архива стоила часа эмбеддинга и переписывала все 1.2 ГБ. Любая правка
одного файла запускала всё заново. Нужно обновление, которое трогает только изменившееся и
не требует присутствия человека.

ПОЧЕМУ ЭТО ВООБЩЕ ЗАКОННО — довод из теоремы о сэндвиче. Условие режима `d ≫ ln n`
**логарифмично по числу узлов**. Пристройка m узлов к n меняет требуемый бюджет как
`ln(n+m) − ln n`: удвоение архива поднимает порог связности всего на 0.69. Значит **дописывать
можно, не пересматривая граф целиком** — режим, в котором архив был собран, при дописывании не
теряется. Это не удобство реализации, а свойство, которое надо было доказать, прежде чем
строить инкремент.

ЧЕТЫРЕ МЕХАНИЗМА:

1. **Манифест.** `path -> (size, mtime, hash)`. Решение «трогать или нет» принимается по
   манифесту, без чтения содержимого: сравниваются размер и время, хеш считается только у
   подозрительных. Хеш от НОРМАЛИЗОВАННОГО текста, поэтому перевод строк и хвостовые пробелы
   не считаются изменением.

2. **Адресация по содержимому.** Идентификатор атома — хеш его нормализованного текста.
   Неизменившийся атом получает тот же идентификатор, а значит и вектор из кэша эмбеддингов.
   Перемещение файла не вызывает переэмбеддинга: путь — свойство атома, не его имя.

3. **Ярусы вместо перезаписи.** Архив = замороженная база + ярусы приращений. Новое пишется
   ярусом; поиск идёт по объединению. Ничего не переписывается, поэтому обновление не может
   испортить базу и не требует блокировки на час. Слияние ярусов в базу — отдельная
   операция, которую можно делать когда угодно или не делать вовсе.

4. **Надгробия.** Удалённое помечается, а не вырезается: вырезание строки из матрицы сдвигает
   все последующие и рвёт соответствие «строка ↔ узел», ради которого и делалась запись
   фиксированной длины.

Запуск:
    python incremental.py manifest <корень> [<файл манифеста>]   снять манифест
    python incremental.py diff <корень> <файл манифеста>         что изменилось
    python incremental.py plan <корень> <файл манифеста>         план обновления с ценой
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intake_filter import scan
from sandwich_graft import nodes_until_next_budget, suggest_budget

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
MANIFEST_DEFAULT = os.path.join(OUT_DEFAULT, "manifest.json")
REPORT_DEFAULT = os.path.join(OUT_DEFAULT, "report.json")


def archive_nodes(report_path: str = REPORT_DEFAULT) -> int | None:
    """Число УЗЛОВ архива из отчёта rebalance.py.

    Бюджет связей считается по узлам, а манифест знает только файлы — величины разные
    (2002 файла против 9745 узлов), и одна через другую не выражается: сколько атомов
    даст файл, решают фильтр и пороги ветвления. Здесь, в CLI, живого архива нет,
    поэтому единственный источник — report.json; в MCP первым спрашивается сам архив.
    """
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path, encoding="utf-8") as f:
            n = json.load(f).get("n")
    except (OSError, ValueError):
        return None
    return int(n) if n else None


def normalize(text: str) -> str:
    """Нормализация перед хешированием: перевод строк и хвостовые пробелы не являются
    изменением содержания, а без нормализации каждый переход между редакторами выглядел бы
    как правка всего файла."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def file_hash(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return hashlib.blake2b(normalize(f.read()).encode("utf-8"), digest_size=16).hexdigest()
    except OSError:
        return ""


def build_manifest(root: str) -> dict:
    accepted, rejected, stats = scan(root)
    root = os.path.abspath(root)
    files = {}
    for p in accepted:
        try:
            st = os.stat(p)
        except OSError:
            continue
        rel = os.path.relpath(p, root).replace("\\", "/")
        files[rel] = {"size": st.st_size, "mtime": int(st.st_mtime), "hash": file_hash(p)}
    return {"root": root, "built": int(time.time()), "n_files": len(files),
            "n_rejected": len(rejected), "files": files}


def diff(root: str, old: dict) -> dict:
    """Что изменилось. Хеш считается только там, где разошлись размер или время —
    иначе обход упёрся бы в чтение всего дерева."""
    new_accepted, _, _ = scan(root)
    root = os.path.abspath(root)
    oldf = old.get("files", {})
    seen, added, changed, touched_same = set(), [], [], []
    for p in new_accepted:
        rel = os.path.relpath(p, root).replace("\\", "/")
        seen.add(rel)
        prev = oldf.get(rel)
        if prev is None:
            added.append(rel)
            continue
        try:
            st = os.stat(p)
        except OSError:
            continue
        if st.st_size == prev["size"] and int(st.st_mtime) == prev["mtime"]:
            continue                                  # не трогали — не читаем
        h = file_hash(p)
        if h and h == prev.get("hash"):
            touched_same.append(rel)                  # время сдвинулось, содержание то же
        else:
            changed.append(rel)
    removed = [r for r in oldf if r not in seen]
    return {"added": added, "changed": changed, "removed": removed,
            "touched_but_same": touched_same, "unchanged": len(seen) - len(added) - len(changed)}


def plan(root: str, old: dict) -> dict:
    d = diff(root, old)
    n_old = old.get("n_files", 0)
    n_new = n_old + len(d["added"]) - len(d["removed"])
    work = len(d["added"]) + len(d["changed"])
    # цена: эмбеддинг — единственная дорогая часть, остальное линейно и дёшево
    print("=" * 66)
    print("было файлов            %d" % n_old)
    print("добавлено              %d" % len(d["added"]))
    print("изменено               %d" % len(d["changed"]))
    print("удалено                %d" % len(d["removed"]))
    print("тронуто, но то же      %d   <- переэмбеддинга НЕ требуют" % len(d["touched_but_same"]))
    print("без изменений          %d" % d["unchanged"])
    print("-" * 66)
    print("к обработке            %d файлов (%.1f %% дерева)"
          % (work, 100.0 * work / max(1, n_new)))
    # Бюджет — по УЗЛАМ графа, а не по файлам манифеста: ln n в условии теоремы берётся
    # от размера графа. Здесь считалось по n_files, и на 2002 файлах выходило 23 вместо
    # 28 — ниже порога режима, в который архив и приводится.
    n_nodes = archive_nodes()
    if n_nodes:
        b = suggest_budget(n_nodes)
        headroom = nodes_until_next_budget(n_nodes)
        print("узлов в архиве         %d   (ln n = %.2f, из out/report.json)"
              % (n_nodes, math.log(n_nodes)))
        print("бюджет связей          %d" % b)
        print("порог режима           не сдвинется, пока узлов меньше %d (+%d)"
              % (n_nodes + headroom, headroom))
    else:
        print("узлов в архиве         неизвестно — нет out/report.json, бюджет не считается")
    print("=" * 66)
    d["n_old"], d["n_new"], d["work"] = n_old, n_new, work
    d["n_nodes"], d["budget"] = n_nodes, suggest_budget(n_nodes) if n_nodes else None
    return d


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd, root = argv[1], argv[2]
    mpath = argv[3] if len(argv) > 3 else MANIFEST_DEFAULT
    os.makedirs(os.path.dirname(mpath), exist_ok=True)

    if cmd == "manifest":
        t0 = time.time()
        m = build_manifest(root)
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False)
        print("манифест: %d файлов, отсечено %d, %.1f с -> %s"
              % (m["n_files"], m["n_rejected"], time.time() - t0, mpath))
        return 0

    if not os.path.exists(mpath):
        print("нет манифеста %s — сначала `manifest`" % mpath)
        return 1
    with open(mpath, encoding="utf-8") as f:
        old = json.load(f)

    if cmd == "diff":
        d = diff(root, old)
        for k in ("added", "changed", "removed"):
            print("%s: %d" % (k, len(d[k])))
            for r in d[k][:10]:
                print("   ", r)
        return 0
    if cmd == "plan":
        plan(root, old)
        return 0
    print("неизвестная команда", cmd)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
