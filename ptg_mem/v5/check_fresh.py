# -*- coding: utf-8 -*-
r"""PTG_V5/check_fresh.py — приёмка обновления: вправду ли свежая работа в архиве.

ЗАЧЕМ ОТДЕЛЬНАЯ ПРОВЕРКА. «Сборка закончилась, узлов стало больше» — это не то же
самое, что «новое знание в архиве и его видно». Счётчик узлов вырастет и если
проиндексировались логи, и если приёмка легла дублями. Здесь проверяется предметно:
лежат ли в архиве те самые вещи, которых 22.09.2026 в нём не было, и в каких файлах.

ДВЕ РАЗНЫЕ ВЕЛИЧИНЫ, КОТОРЫЕ НЕЛЬЗЯ ПУТАТЬ:

  * НАЛИЧИЕ — опора встречается в тексте атома буквально. Не требует эмбеддера,
    отвечает на вопрос «попало ли».
  * НАХОДИМОСТЬ — атом с опорой поднимается человеческим запросом БЕЗ опоры.
    Требует эмбеддера и отвечает на вопрос «вернётся ли оно мне». Замерено, что
    это плоский косинус, а не обход графа (см. память graph-does-not-improve-retrieval).

Наличие без находимости — обычное дело и не дефект сам по себе. Дефект — когда нет
наличия: тогда никакой поиск не поможет.

Запуск:
    python check_fresh.py [<хранилище>] [--search]
"""
from __future__ import annotations

import collections
import io
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SNAP = os.path.join(ROOT, "trace-probe", "ptg", "_snapshot_run")

STORE_NAME = "store_text_v5"
DO_SEARCH = False
for a in sys.argv[1:]:
    if a == "--search":
        DO_SEARCH = True
    else:
        STORE_NAME = a
PTG = os.path.join(SNAP, STORE_NAME, ".ptg")
GGUF = os.path.join(ROOT, r"trace-probe\models\Qwen3-Embedding-4B-GGUF"
                          r"\Qwen3-Embedding-4B-Q4_K_M.gguf")

# Опоры — редкие строки, которых в августовском архиве быть не могло. Числа с
# десятичной частью сознательно НЕ берём: память уже зафиксировала, что «0.60» и
# «3.53» встречаются в сотнях атомов по другим поводам и дают ложные попадания.
ANCHORS = [
    ("бюджет при вставке (ядро V5)",        "budget_graft"),
    ("вектор-графт V4",                     "graft_fast"),
    ("правила слияния движков",             "METATREE-ENGINES"),
    ("математика бюджета до кода",          "BUDGETED-GRAFT-MATH"),
    ("сжатие токенов для текста",           "Jasper-Token-Compression"),
    ("публичный бенчмарк памяти",           "LongMemEval"),
    ("разбор обсуждения с meta.ai",         "metaai"),
    ("рыночная записка",                    "MARKET-2026-09"),
    ("продуктовая стратегия",               "PRODUCT-STRATEGY"),
    ("тип спасения сироты в V5",            "RESCUE_TYPE"),
    ("ловушка отображения индекса",         "ERROR_USER_MAPPED_FILE"),
    ("обратный PTG — граф архитектуры",     "обратн"),
]

QUERIES = [
    "как устроено новое ядро, где бюджет степеней стоит в момент присоединения атома",
    "чем мерили полноту памяти на публичном наборе вопросов и что вышло против плоского поиска",
    "что решено по продукту: издания, цена, на ком зарабатывать",
]


def main() -> None:
    if not os.path.isdir(PTG):
        raise SystemExit("нет хранилища: %s" % PTG)
    t0 = time.time()
    with io.open(os.path.join(PTG, "meta_tree.json"), encoding="utf-8") as f:
        meta = json.load(f)
    order = meta["id_order"]
    with io.open(os.path.join(PTG, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    print("%s: атомов %d, файлов %d, чтение %.0f с"
          % (STORE_NAME, len(order), len(meta.get("files", {})), time.time() - t0))

    texts, files = [], []
    for nid in order:
        nd = nodes.get(nid) or {}
        texts.append((nd.get("text") or "").lower())
        files.append(nd.get("file") or "")

    print("\n=== НАЛИЧИЕ ===")
    print("  %-34s %7s  %s" % ("что ищем", "атомов", "где чаще всего"))
    missing = []
    for label, anchor in ANCHORS:
        a = anchor.lower()
        hits = [i for i, t in enumerate(texts) if a in t]
        if not hits:
            missing.append(label)
            print("  %-34s %7s  —" % (label[:34], "НЕТ"))
            continue
        top = collections.Counter(os.path.basename(files[i]) for i in hits).most_common(2)
        print("  %-34s %7d  %s" % (label[:34], len(hits),
                                   ", ".join("%s (%d)" % (n, c) for n, c in top)))

    print("\n  итог: опор найдено %d из %d" % (len(ANCHORS) - len(missing), len(ANCHORS)))
    if missing:
        print("  НЕ НАЙДЕНО:", "; ".join(missing))

    # переписка сеансов — отдельно: это то, ради чего заводилась приёмка
    chat = [i for i, f in enumerate(files) if "e-переписка" in f]
    by_file = collections.Counter(os.path.basename(files[i]) for i in chat)
    print("\n=== ПЕРЕПИСКА СЕАНСОВ В АРХИВЕ ===")
    print("  атомов %d из %d файлов" % (len(chat), len(by_file)))
    for n, c in sorted(by_file.items()):
        print("    %-44s %5d атом(ов)" % (n, c))

    if not DO_SEARCH:
        print("\n(находимость не мерилась: запустить с --search)")
        return

    import numpy as np
    V = np.load(os.path.join(PTG, "faiss.index.npy")).astype("float32")
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    from llama_cpp import Llama
    llm = Llama(model_path=GGUF, embedding=True, n_ctx=512, n_batch=512,
                n_ubatch=512, n_gpu_layers=0, verbose=False)
    print("\n=== НАХОДИМОСТЬ (плоский косинус, top-5) ===")
    for q in QUERIES:
        r = llm.create_embedding(q[:1500])
        qv = np.asarray(r["data"][0]["embedding"], dtype="float32")
        qv /= max(float(np.linalg.norm(qv)), 1e-9)
        idx = np.argsort(-(V @ qv))[:5]
        print("\n  запрос: %s" % q)
        for rank, i in enumerate(idx, 1):
            print("    %d. %-38s %s" % (rank, os.path.basename(files[i])[:38],
                                        texts[i][:90].replace("\n", " ")))


if __name__ == "__main__":
    main()
