# -*- coding: utf-8 -*-
r"""PTG_V4/intake_filter.py — что вообще имеет право попасть в архив.

ЗАЧЕМ ЭТО НЕ ГИГИЕНА, А ПАРАМЕТР РЕЖИМА. Условие переноса из теоремы о сэндвиче —
`d ≫ ln n`, где n есть число узлов. Мусор на входе растит n, а значит поднимает и требуемый
бюджет связей: при n = 10 тыс. порог связности `ln n` = 9.2, при n = 100 тыс. — 11.5, и
бюджет приходится держать выше. Каждый отсечённый дистрибутив — это не «чище стало», а
**сдвиг архива в режим, где его структура вообще что-то значит**.

Отсюда правило отбора: в архив идёт только то, что несёт **рассуждение о проекте**. Чужой
исходник, распакованный пакет, венв и дистрибутив рассуждения не несут — они несут чужое
рассуждение, одинаковое у миллиона проектов, и в графе дают плотные клики ни о чём.

ЧТО ЛОВИТ СВЕРХ ПРЕЖНЕГО `snapshot_filters.py`:
  * деревья пакетов и венвов: site-packages, dist-packages, Lib/, Scripts/, .cargo, .npm;
  * дистрибутивы и архивы: .whl, .tar.gz, .zip, .7z, .rar, .exe, .msi, .deb, .rpm, .iso;
  * вендоренное и стороннее: vendor/, third_party/, external/, deps/, _deps/, extern/;
  * деревья сборки: build*/, dist/, target/, obj/, x64/, Release/, Debug/, cmake-build-*;
  * бинарь по содержимому, а не по имени: нулевой байт в первых 8 КиБ;
  * чужой исходник по содержимому: файл с лицензионной шапкой стороннего правообладателя
    и без единого упоминания проекта;
  * зеркала и копии: пути, где тот же файл уже принят под другим корнем (по хешу).

Запуск:
    python intake_filter.py <корень> [--report]    посчитать и показать, что отсекается
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter

# --- каталоги, отсекаемые целиком ------------------------------------------
EXCLUDE_DIRS = {
    # служебное
    ".ptg", ".code_ptg", ".git", ".hg", ".svn", ".idea", ".vscode", ".vs",
    "__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    # венвы и пакетные деревья
    ".venv", "venv", "env", "virtualenv", "site-packages", "dist-packages",
    "node_modules", ".npm", ".yarn", ".pnpm-store", ".cargo", ".rustup", ".gradle",
    "lib", "lib64", "scripts", "bin", "include",        # характерная тройка венва
    # вендоренное и стороннее
    "vendor", "vendored", "third_party", "thirdparty", "external", "externals",
    "extern", "deps", "_deps", "subprojects", "vendor_ptg",
    # сборка
    "build", "dist", "target", "obj", "x64", "win32", "release", "debug",
    "cmake-build-debug", "cmake-build-release", "cmakefiles",
    # данные прогонов и веса
    "models", "traces", "runs", "raw", "out", "outputs", "logs", "log",
    "checkpoints", "ckpt", "wandb", "artifacts", "cache", "tmp", "temp",
    "huggingface_cache", "pip_cache", "embed_cache", "_snapshot_run",
}
# каталоги из списка выше, которые отсекаются ТОЛЬКО рядом с признаком венва
VENV_ONLY_DIRS = {"lib", "lib64", "scripts", "bin", "include"}
VENV_MARKERS = {"pyvenv.cfg", "activate", "activate.bat", "Activate.ps1"}

# --- расширения: что берём ---------------------------------------------------
DOC_EXTS    = {".md", ".markdown", ".rst", ".txt", ".org"}
CODE_EXTS   = {".py"}
NATIVE_EXTS = {".cpp", ".hpp", ".cc", ".c", ".h", ".cu", ".cuh"}
CFG_EXTS    = {".yaml", ".yml", ".toml", ".ini", ".cfg"}
SCRIPT_EXTS = {".sh", ".bat", ".ps1"}
INCLUDE_EXTS = DOC_EXTS | CODE_EXTS | NATIVE_EXTS | CFG_EXTS | SCRIPT_EXTS

# --- дистрибутивы, архивы, бинарь -------------------------------------------
DIST_SUFFIX = (
    ".whl", ".egg", ".tar", ".tar.gz", ".tgz", ".zip", ".7z", ".rar", ".gz", ".bz2", ".xz",
    ".exe", ".msi", ".dll", ".so", ".dylib", ".lib", ".a", ".o", ".obj", ".pdb", ".pyd",
    ".deb", ".rpm", ".iso", ".img", ".cab", ".jar", ".class",
    ".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".npz", ".npy", ".ckpt",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg", ".pdf",
    ".mp3", ".mp4", ".wav", ".avi", ".mkv", ".ttf", ".otf", ".woff", ".woff2",
    ".db", ".sqlite", ".sqlite-wal", ".sqlite-shm", ".idx", ".faiss",
    ".min.js", ".lock", ".orig", ".incomplete", ".crdownload", ".part",
)
EXCLUDE_NAMES = {
    "code_dump.txt", ".env", "poetry.lock", "package-lock.json", "yarn.lock",
    "pnpm-lock.yaml", "cargo.lock", "requirements.lock", "pyvenv.cfg",
    "license", "license.txt", "license.md", "copying", "notice", "authors",
}

# --- правила поддеревьев -----------------------------------------------------
# Внутри дерева форка 5 921 файл апстрима llama.cpp и несколько десятков наших.
# Лицензионных шапок в апстриме нет, поэтому детектор чужого его не ловит: нужно
# явное правило. Берём только то, что несёт мысль проекта, — организм и его пробы.
SUBTREE_RULES = {
    "fork-tree": re.compile(r"(myc|mycelium|substrate)", re.I),
    "llama.cpp": re.compile(r"$^"),          # апстрим-зеркало: не берём ничего
}

MAX_FILE_BYTES = 1_200_000
MIN_FILE_BYTES = 24

# чужая лицензионная шапка: правообладатель, которого в проекте быть не должно
FOREIGN_COPYRIGHT = re.compile(
    r"copyright\s*(\(c\)|©)?\s*\d{4}[^\n]{0,80}?"
    r"(inc\.|llc|ltd|gmbh|corporation|foundation|university|institute)",
    re.I)
LICENSE_HEADER = re.compile(
    r"(licensed under the apache license|mit license|bsd \d-clause|"
    r"gnu general public license|mozilla public license|spdx-license-identifier)", re.I)


def _is_venv_root(dirpath: str, names: set) -> bool:
    return bool(VENV_MARKERS & names)


def looks_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(8192)
    except OSError:
        return True
    return b"\x00" in head


def looks_foreign(path: str, project_marks: tuple) -> tuple[bool, str]:
    """Чужой исходник: лицензионная шапка стороннего правообладателя и ни одного
    упоминания проекта. Оба условия обязательны — иначе отсечётся собственный код,
    к которому кто-то приписал лицензию."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(4000)
    except OSError:
        return True, "нечитаем"
    lic = bool(LICENSE_HEADER.search(head)) or bool(FOREIGN_COPYRIGHT.search(head))
    if not lic:
        return False, ""
    low = head.lower()
    if any(m in low for m in project_marks):
        return False, ""
    return True, "чужая лицензионная шапка без упоминания проекта"


def scan(root: str, project_marks=("mycelium", "ptg", "форк", "организм", "полос"),
         dedup=True):
    """Обойти дерево. Возвращает (принятые, отвергнутые[(путь, причина)], статистика)."""
    accepted, rejected = [], []
    stats = Counter()
    seen_hashes = {}
    root = os.path.abspath(root)

    for dirpath, dirnames, filenames in os.walk(root):
        names = set(filenames)
        venv_here = _is_venv_root(dirpath, names)
        keep = []
        for d in dirnames:
            dl = d.lower()
            if dl in VENV_ONLY_DIRS:
                # Lib/Scripts/bin отсекаются только если рядом маркер венва: иначе
                # выбросили бы собственный каталог bin с осмысленными скриптами.
                if venv_here:
                    stats["каталог: венв"] += 1
                    continue
                keep.append(d)
                continue
            if dl in EXCLUDE_DIRS or (dl.startswith(".") and dl != ".claude"):
                stats["каталог: исключён"] += 1
                continue
            # build, build-fast, build-cuda, build-nomyc, cmake-build-* — всё это деревья
            # сборки. Прежнее правило по точному имени "build" их не ловило.
            if dl.startswith("build") or dl.startswith("cmake-build") \
               or dl.endswith(".egg-info") or dl.endswith(".dist-info"):
                stats["каталог: сборка/пакет"] += 1
                continue
            keep.append(d)
        dirnames[:] = keep

        for fn in filenames:
            path = os.path.join(dirpath, fn)
            low = fn.lower()
            ext = os.path.splitext(low)[1]
            rel = os.path.relpath(path, root).replace("\\", "/")
            top = rel.split("/", 1)[0]
            rule = SUBTREE_RULES.get(top)
            if rule is not None and not rule.search(rel):
                rejected.append((path, "поддерево %s: не своё" % top))
                stats["поддерево апстрима"] += 1
                continue
            if low in EXCLUDE_NAMES:
                rejected.append((path, "имя в стоп-листе")); stats["имя"] += 1; continue
            if low.endswith(DIST_SUFFIX):
                rejected.append((path, "дистрибутив/бинарь по расширению")); stats["дистрибутив"] += 1; continue
            if ext not in INCLUDE_EXTS:
                rejected.append((path, "расширение вне списка")); stats["расширение"] += 1; continue
            try:
                sz = os.path.getsize(path)
            except OSError:
                rejected.append((path, "нет доступа")); stats["доступ"] += 1; continue
            if sz < MIN_FILE_BYTES:
                rejected.append((path, "пустышка")); stats["пустышка"] += 1; continue
            if sz > MAX_FILE_BYTES:
                rejected.append((path, "больше предела %d" % MAX_FILE_BYTES)); stats["крупный"] += 1; continue
            if looks_binary(path):
                rejected.append((path, "бинарь по содержимому")); stats["бинарь"] += 1; continue
            foreign, why = looks_foreign(path, project_marks)
            if foreign:
                rejected.append((path, why)); stats["чужой исходник"] += 1; continue
            if dedup:
                try:
                    with open(path, "rb") as f:
                        h = hashlib.blake2b(f.read(), digest_size=16).hexdigest()
                except OSError:
                    h = None
                if h is not None:
                    if h in seen_hashes:
                        rejected.append((path, "копия " + os.path.relpath(seen_hashes[h], root)))
                        stats["копия"] += 1
                        continue
                    seen_hashes[h] = path
            accepted.append(path)
            stats["принято"] += 1
    return accepted, rejected, stats


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = argv[1]
    acc, rej, st = scan(root)
    total = len(acc) + len(rej)
    print("корень: %s" % root)
    print("файлов пройдено: %d, принято %d (%.1f %%), отсечено %d"
          % (total, len(acc), 100.0 * len(acc) / total if total else 0, len(rej)))
    print()
    print("по причинам:")
    for k, v in st.most_common():
        print("  %-24s %6d" % (k, v))
    if "--report" in argv:
        print("\nпримеры отсечённого:")
        shown = {}
        for p, why in rej:
            key = why.split()[0]
            if shown.get(key, 0) >= 3:
                continue
            shown[key] = shown.get(key, 0) + 1
            print("  [%s] %s" % (why[:40], os.path.relpath(p, root)[:90]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
