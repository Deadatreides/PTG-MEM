# -*- coding: utf-8 -*-
r"""PTG_V5/persist_v5.py — переигровка архива ядром V5 С СОХРАНЕНИЕМ.

ЧЕМ ОТЛИЧАЕТСЯ ОТ run_full.py. Тот собирает граф в памяти, прогоняет ворота и пишет
отчёт. Обратной записи у него нет вовсе, и поэтому «мы перешли на V5» до сих пор было
правдой про замеры и неправдой про данные: на диске лежал граф, собранный старым
графтом. Здесь результат переигровки сохраняется как ПОЛНОЦЕННОЕ хранилище.

ПОЧЕМУ ПЕРЕИГРОВКА, А НЕ ДОПИСКА ХВОСТА. Если поставить V5 только на новые атомы,
в одном архиве окажется два правила присоединения: старое у первых 27 тысяч и новое
у хвоста. Это шов внутри одной базы — ровно то, что в METATREE-ENGINES.md признано
непочинимым локальными средствами. Правило должно лечь на всё разом.

ЭМБЕДДЕР НЕ НУЖЕН. Атомы и векторы берутся с диска и переигрываются в том же порядке;
меняется только правило присоединения. Это и делает операцию посильной: дорогая часть
(эмбеддинг) уже оплачена, платим только за графтинг.

ЧТО ПЕРЕНОСИТСЯ ИЗ ИСХОДНОГО АРХИВА, А НЕ ПЕРЕСЧИТЫВАЕТСЯ. Переигровка вызывает
_add_atom и не вызывает build(), поэтому сама по себе НЕ создаёт:

  * root_project / root_embedding — иначе root_similarity у всех веток станет пустым;
  * file_edges — файловый граф ПАТЧА 2 строится обходом, а не присоединением;
  * processed_files / files / last_structure_hash — БЕЗ НИХ СЛЕДУЮЩАЯ ИНКРЕМЕНТАЛЬНАЯ
    СБОРКА СОЧТЁТ ВСЕ ФАЙЛЫ НОВЫМИ и переэмбеддит архив целиком.

Последнее — самая дорогая ошибка из возможных здесь, поэтому перенос проверяется
явной сверкой после сохранения.

ВОРОТА ДО ЗАПИСИ. Сохранение происходит, только если граф прошёл ворота (связность,
несхлопнутость, сходство рёбер, путь). Не прошёл — пишется отчёт, хранилище не
создаётся. Исходный архив не трогается ни при каком исходе: запись идёт в НОВЫЙ
каталог.

Запуск:
    python persist_v5.py [<исходное хранилище>] [<новое хранилище>] [margin]
"""
from __future__ import annotations

import collections
import gc
import io
import json
import os
import shutil
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
V4 = os.path.join(ROOT, "PTG_V4")
SNAP = os.path.join(ROOT, "trace-probe", "ptg", "_snapshot_run")
PTG_TEXT = os.path.join(ROOT, "trace-probe", "ptg", "PTG+MCP1.3")
for p in (HERE, V4, SNAP, PTG_TEXT):
    sys.path.insert(0, p)

import ptg_core                                        # noqa: E402
import graft_fast                                      # noqa: E402
import budget_graft                                    # noqa: E402
import gates                                           # noqa: E402

SRC_NAME = sys.argv[1] if len(sys.argv) > 1 else "store_text_v3"
DST_NAME = sys.argv[2] if len(sys.argv) > 2 else "store_text_v5"
MARGIN = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
# N>0 — смоук: пройти весь путь (переигровка → ворота → запись → сверка) на горстке
# атомов, прежде чем тратить часы. Ворота на обрезке ничего не доказывают про архив,
# доказывают они только то, что код доходит до конца и сохраняет полное хранилище.
N = int(sys.argv[4]) if len(sys.argv) > 4 else 0

SRC = os.path.join(SNAP, SRC_NAME)
DST = os.path.join(SNAP, DST_NAME)
SRC_PTG = os.path.join(SRC, ptg_core.PTG_DIRNAME)

# те же калиброванные пороги, которыми собраны store_text_cal и v3: переигровка другой
# шкалой дала бы другой граф не из-за ядра, а из-за порогов, и сравнение было бы ложью
ptg_core.BRANCH_THRESH = 0.47
ptg_core.CONTINUE_THRESH = 0.57
ptg_core.RETURN_THRESH = 0.65
ptg_core.CONTRADICT_SIM_THRESH = 0.62
ptg_core.SUPERSEDE_SIM_THRESH = 0.57
# индекс 287 МБ; при пороге 50 МБ np.load отобразил бы его в память, и процесс,
# держащий файл, сломал бы сохранение (ERROR_USER_MAPPED_FILE / OSError 22)
ptg_core.MMAP_THRESHOLD_BYTES = 1 << 62

# --- отсев битого разбора переписки ------------------------------------------
# 22.09.2026: 7181 из 8815 атомов переписки в архиве (81 %) — это СТАРЫЙ, сломанный
# разбор. До правки `_flatten` в prepare_intake.py метка «Claude:» стояла только на
# первом абзаце хода; доля размеченных абзацев обрушивалась, парсер ptg_core уходил в
# последовательный фолбэк и вместо ~1 атома на ход давал тысячи атомов «соседний абзац
# с соседним», без ролей. Числа на одном и том же сеансе 827b7940:
#
#     старый разбор  5706 атомов, медиана длины   235 симв.
#     новый разбор    388 атомов, медиана длины  3811 симв.
#
# Файлы переименовались (имя зависело от mtime журнала), старые копии удалились с
# диска, но надгробий у ptg_core нет — атомы остались в архиве навсегда. Все пять
# затронутых сеансов присутствуют в архиве И в правильном разборе, поэтому отсев
# ничего не теряет: он убирает обрубки, а не содержание.
#
# Критерий узкий сознательно: только `свод-работ-для-ptg/e-переписка` и только те
# имена, которых сейчас нет на диске. Отсеивать всё, чей файл исчез, — другая и
# куда более широкая политика, и здесь она не применяется.
KEEP_GHOSTS = "--keep-ghosts" in sys.argv
CHAT_DIR = os.path.join(ROOT, "свод-работ-для-ptg", "e-переписка")
try:
    CHAT_ON_DISK = set(os.listdir(CHAT_DIR))
except OSError:
    CHAT_ON_DISK = set()


def is_ghost_chat(node: dict) -> bool:
    f = node.get("file") or ""
    if "e-переписка" not in f:
        return False
    return os.path.basename(f) not in CHAT_ON_DISK


# поля, которые пересчитывает сам движок при присоединении
DROP = ("vec", "index", "branch", "semantic_epoch", "placement_info",
        "path_coherence", "short_path_coherence", "long_path_coherence",
        "reinforcement_count", "reinforcement_links", "repetition_epochs")


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def read_json(name: str):
    with io.open(os.path.join(SRC_PTG, name), encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    if not os.path.isdir(SRC_PTG):
        raise SystemExit("нет исходного хранилища: %s" % SRC_PTG)
    if os.path.exists(DST):
        raise SystemExit("каталог %s уже существует — снести или взять другое имя" % DST)

    t0 = time.time()
    # meta_tree.json — 214 МБ, и «branches» в нём нам не нужны: ветви пересоберёт
    # переигровка. Берём четыре поля и отпускаем остальное, иначе к пиковой памяти
    # прогона (векторы 281 МБ ×2 плюс узлы) прибавилась бы копия, которой никто
    # не пользуется.
    meta = read_json(ptg_core.META_FILE)
    order = meta["id_order"]
    carry = {"processed_files": meta.get("processed_files", {}),
             "files": meta.get("files", {}),
             "last_structure_hash": meta.get("last_structure_hash")}
    del meta
    gc.collect()
    src_nodes = read_json(ptg_core.NODES_FILE)
    root_project = read_json(ptg_core.ROOT_PROJECT_FILE)
    file_edges = read_json(ptg_core.FILE_EDGES_FILE)
    vecs = np.load(os.path.join(SRC_PTG, ptg_core.INDEX_FILE + ".npy"))
    rpath = os.path.join(SRC_PTG, "root_embedding.npy")
    root_embedding = np.load(rpath) if os.path.exists(rpath) else None
    log("исходник %s: узлов %d, векторы %s, чтение %.0f с"
        % (SRC_NAME, len(order), vecs.shape, time.time() - t0))
    if len(order) != len(vecs):
        raise SystemExit("рассинхрон: id_order %d, векторов %d — архив собирался "
                         "в этот момент?" % (len(order), len(vecs)))

    os.makedirs(DST, exist_ok=True)
    arc = ptg_core.Archive(folder=ROOT, output_dir=DST, progress_cb=lambda m: None,
                           extract_py_comments=True)
    arc.root_embedding = root_embedding
    arc.embedder.dim = int(vecs.shape[1])

    fg = graft_fast.install(arc, ptg_core, verify=0)
    bg = budget_graft.install(arc, ptg_core, fg, margin=MARGIN)

    t0 = time.time()
    done = skipped = ghosts = 0
    ghost_files = collections.Counter()
    limit = N or len(order)
    if N:
        log("СМОКЕ: только %d атомов — ворота на обрезке ничего не доказывают" % N)
    for i, nid in enumerate(order):
        if done >= limit:
            break
        nd = src_nodes.pop(nid, None)      # отпускаем узел сразу: держать
                                           # весь nodes.json до конца незачем
        if nd is None:
            skipped += 1
            continue
        if not KEEP_GHOSTS and is_ghost_chat(nd):
            ghosts += 1
            ghost_files[os.path.basename(nd.get("file") or "")] += 1
            continue
        v = np.asarray(vecs[i], dtype="float32")
        nrm = float(np.linalg.norm(v))
        if nrm == 0:
            skipped += 1
            continue
        atom = {k: nd[k] for k in nd if k not in DROP}
        atom["vec"] = v / nrm
        arc._add_atom(atom)
        done += 1
        if done % 2500 == 0:
            log("  %d/%d, %.0f с, рёбер %d, ветвей %d"
                % (done, len(order), time.time() - t0, len(arc.edges), len(arc.branches)))
    dt = time.time() - t0
    log("переигровка: %d атомов за %.0f мин (%.1f атом/с), пропущено %d, "
        "битой переписки отсеяно %d, узлов %d, ветвей %d, рёбер %d"
        % (done, dt / 60, done / max(dt, 1e-9), skipped, ghosts,
           len(arc.nodes), len(arc.branches), len(arc.edges)))
    for name, c in sorted(ghost_files.items()):
        log("    отсеяно %5d из %s" % (c, name))

    # --- ворота ДО записи ---------------------------------------------------
    snap = {"nodes": arc.nodes, "edges": arc.edges, "branches": arc.branches,
            "path_memory": arc.path_memory}
    rep = {
        "исходник": SRC_NAME, "новое": DST_NAME, "атомов": done, "margin": MARGIN,
        "секунд": round(dt, 1),
        "битой_переписки_отсеяно": ghosts,
        "отсеяно_по_файлам": dict(ghost_files),
        "связность": gates.connectivity(snap),
        "несхлопнутость": gates.no_coagulation(snap),
        "сходство_рёбер": gates.edge_similarity(snap),
        "путь": gates.path(snap),
        "счётчики": bg.stats(),
    }
    out_dir = os.path.join(HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    with io.open(os.path.join(out_dir, "persist_v5.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    for k in ("связность", "несхлопнутость", "сходство_рёбер", "путь"):
        log("ворота %-16s %s" % (k, json.dumps(rep[k], ensure_ascii=False)[:300]))

    # Ворота НЕ возвращают готового «прошло» — решение принимается здесь, по тем самым
    # величинам, ради которых ворота и писались. Проверять надо именно так: ключ
    # «прошло» напрашивался, но его нет, и условие по нему молча пропускало бы всё.
    con, coa = rep["связность"], rep["несхлопнутость"]
    bad, warn = [], []
    if con.get("изолированных", 0) != 0:
        bad.append("сироты: %d" % con["изолированных"])
    if coa.get("НАРУШЕНИЕ_далёкое_продолжение"):
        bad.append("continues через границу времени: %d" % coa.get("из_них_continues", 0))
    # РЕЖИМ — ПРЕДУПРЕЖДЕНИЕ, А НЕ ПРИГОВОР. Первая версия блокировала запись при
    # `в_режиме == false`, и 22.09 это снесло совершенно исправный граф: степень 20.89
    # при цели 3·ln n = 29.94, сирот ноль, глубина обхода 4. Разбор счётчиков показал,
    # что дело не в дефекте, а в арифметике эмиссии: e = ceil(d/2) рассчитан на то, что
    # КАЖДОЕ ребро находит место, а заполняется эмиссия на 74.9 % (226 268 из 301 937).
    # Отказов по бюджету 3.25 млн против 864 тыс. по порогу — упор в остаток бюджета
    # цели, а не в нехватку похожих. Значит это параметр управления (margin), а не
    # корректность. Множитель 3 — выбранный запас, а не закон: связность требует ln n,
    # и 2.09·ln n при нуле изолированных — рабочий граф. Уничтожать его нельзя.
    if not N and not con.get("в_режиме"):
        warn.append("не в режиме: степень %.2f при цели 3·ln n = %.2f (эмиссия "
                    "заполнена %s %%) — поднять margin"
                    % (con.get("средняя_степень", 0), 3 * con.get("ln_n", 0),
                       rep["счётчики"].get("эмиссия_заполнена_%")))
    if rep["счётчики"].get("рёбер_отношений", 0) <= 0:
        bad.append("V5 не добавил ни одного отношения — ядро не установлено?")
    if rep["счётчики"].get("отказов_по_порогу", 0) <= 0:
        bad.append("ни одного отказа по порогу — пол отношений не работает")

    # ПОЧЕМУ ЗДЕСЬ НЕТ АБСОЛЮТНОГО ПОЛА ПО КОСИНУСУ РЕБРА. Напрашивалось «косинус
    # ребра типа T не ниже порога T», и первая версия так и проверяла. Смоук 22.09
    # показал, что это неверно: движок решает не по сырому косинусу, а по составной
    # оценке графта (W_SEM 0.40, W_PATH 0.25, W_MOM 0.15, W_ROOT 0.10, W_ACT 0.10),
    # поэтому у законного `continues` косинус бывает 0.364 при CONTINUE_THRESH 0.57.
    # Вдобавок V5 выпускает `reinforces` и `returns_to` — ТЕМИ ЖЕ именами, что и
    # движок, так что по типу ребра их источник не различить. Такая проверка забракует
    # правильный архив. Пол V5 (`sim < BRANCH_THRESH` → отказ) живёт в самом ядре и
    # виден счётчиком «отказов_по_порогу», он и проверяется выше. Распределение
    # косинусов по типам остаётся в отчёте как диагностика для глаз.
    for w in warn:
        log("ПРЕДУПРЕЖДЕНИЕ: %s" % w)
    rep["предупреждения"] = warn
    if bad:
        # ignore_errors=True прятало настоящий отказ: 22.09 каталог остался на диске,
        # потому что ptg_core держит открытым ptg.log внутри него. Полумёртвое
        # хранилище выглядит как готовое — сообщаем явно.
        try:
            shutil.rmtree(DST)
        except OSError as e:
            log("каталог %s снести не удалось (%s) — убрать вручную" % (DST, e))
        raise SystemExit("ВОРОТА НЕ ПРОЙДЕНЫ: %s — хранилище не создано, "
                         "отчёт в PTG_V5/out/persist_v5.json" % ", ".join(bad))

    # --- перенос того, что переигровка не строит ----------------------------
    arc.root_project = root_project
    arc.file_edges = file_edges
    arc.processed_files = carry["processed_files"]
    arc.files = carry["files"]
    arc.last_structure_hash = carry["last_structure_hash"]
    log("перенесено: файлов в processed_files %d, files %d, рёбер файлового графа %d"
        % (len(arc.processed_files), len(arc.files), len(file_edges)))

    t0 = time.time()
    arc.save()
    log("сохранено в %s за %.0f с" % (DST, time.time() - t0))

    # --- сверка после записи -------------------------------------------------
    chk = json.load(io.open(os.path.join(DST, ptg_core.PTG_DIRNAME, ptg_core.META_FILE),
                            encoding="utf-8"))
    log("сверка: id_order %d, processed_files %d, files %d"
        % (len(chk["id_order"]), len(chk.get("processed_files", {})),
           len(chk.get("files", {}))))
    if not chk.get("processed_files"):
        log("ВНИМАНИЕ: processed_files пуст — следующая инкрементальная сборка "
            "переэмбеддит архив целиком")
    log("RUN-COMPLETE")


if __name__ == "__main__":
    main()
