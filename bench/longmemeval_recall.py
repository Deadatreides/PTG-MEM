# -*- coding: utf-8 -*-
r"""PTG на LongMemEval: полнота памяти против плоского векторного поиска.

ЧТО ИМЕННО МЕРИТСЯ, И ЧТО НЕТ.

Меряется **полнота памяти**: попадает ли в выдачу реплика-улика (`has_answer`)
и сессия-улика (`answer_session_ids`). Это НЕ та цифра, которой меряются
Zep (63.8 %) и Mem0 (49.0 %) — у них точность ОТВЕТА, посчитанная моделью-судьёй.
Сравнивать напрямую нельзя, и здесь этого не делается.

Почему так. Точность ответа требует модели-читателя на 9B и судьи gpt-класса.
Полнота памяти не требует ничего, кроме эмбеддера, и при этом изолирует ровно
вклад памяти: если улика не поднята, никакой читатель её не спасёт.

ДВА СПОСОБА НА ОДНИХ ДАННЫХ:

  * **плоский косинус** — то, что делает обычный RAG над репликами;
  * **PTG** — тот же косинус как затравка плюс расширение по типизированным
    рёбрам графа, собранного ядром V5 с бюджетом.

Разрез по типам вопросов обязателен: `knowledge-update` проверяет замещение,
`multi-session` — сборку из нескольких сессий, `temporal-reasoning` — время.
Если граф полезен, он полезен именно там, а не в среднем.

ПОРОГИ КАЛИБРУЮТСЯ НА КАЖДОМ СТОГЕ. Калиброванные 0.47/0.57/0.65 сняты с
русского корпуса на Qwen3-4B. У Jasper на английском чате распределение
косинусов другое, и перенос порогов дал бы граф, где всё связано со всем.
Поэтому пороги берутся процентилями фактического распределения сходств
этого самого стога — то же правило, другая шкала.

Запуск:  python longmemeval_recall.py [сколько_вопросов] [top_k]
"""
from __future__ import annotations

import collections
import io
import json
import os
import random
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
V4 = os.path.join(ROOT, "PTG_V4")
PTG_TEXT = os.path.join(ROOT, "trace-probe", "ptg", "PTG+MCP1.3")
for p in (HERE, V4, PTG_TEXT):
    sys.path.insert(0, p)

DATA = os.path.join(HERE, "data", "longmemeval_s.json")
MODEL = os.environ.get("PTG_EMBED_MODEL", "")  # путь к модели sentence-transformers
N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
# При k=10 плоский косинус упирается в потолок (замер на 12 вопросах: 100 %),
# и различать нечего. Полнота меряется сразу на нескольких k — запас есть
# только на малых.
KS = (1, 3, 5, 10)
SEED = 20260922

import ptg_core                                        # noqa: E402
import graft_fast                                      # noqa: E402
import budget_graft                                    # noqa: E402

TYPED = ("supersedes", "contradicts", "fixes", "refines", "returns_to",
         "reinforces", "continues")


def build_atoms(item):
    """Атом = реплика. Сессия играет роль файла, её дата — метку времени."""
    atoms = []
    dates = item.get("haystack_dates") or []
    for si, (sid, sess) in enumerate(zip(item["haystack_session_ids"],
                                         item["haystack_sessions"])):
        ts = 0.0
        if si < len(dates):
            try:
                ts = time.mktime(time.strptime(str(dates[si])[:10], "%Y/%m/%d"))
            except Exception:
                try:
                    ts = time.mktime(time.strptime(str(dates[si])[:10], "%Y-%m-%d"))
                except Exception:
                    ts = 0.0
        for ti, turn in enumerate(sess):
            txt = (turn.get("content") or "").strip()
            if not txt:
                continue
            atoms.append({
                "text": txt, "role": turn.get("role"),
                "session": sid, "turn": ti,
                "evidence": bool(turn.get("has_answer")),
                "timestamp": ts,
            })
    return atoms


def calibrate(V, sample=4000, rng=None):
    """Пороги — процентили фактического распределения сходств этого стога."""
    n = len(V)
    rng = rng or np.random.default_rng(SEED)
    a = rng.integers(0, n, size=min(sample, n * 4))
    b = rng.integers(0, n, size=len(a))
    m = a != b
    s = np.einsum("ij,ij->i", V[a[m]], V[b[m]])
    return (float(np.quantile(s, 0.90)), float(np.quantile(s, 0.97)),
            float(np.quantile(s, 0.99)))


def ptg_graph(atoms, V, margin=3.0):
    """Собрать граф ядром V5 на готовых векторах. Возвращает рёбра-соседство."""
    import tempfile
    work = os.path.join(tempfile.gettempdir(), "lme_ptg")
    os.makedirs(work, exist_ok=True)
    arc = ptg_core.Archive(folder=work, output_dir=work,
                           progress_cb=lambda m: None, extract_py_comments=False)
    arc.root_embedding = None
    arc.embedder.dim = int(V.shape[1])
    fg = graft_fast.install(arc, ptg_core, verify=0)
    budget_graft.install(arc, ptg_core, fg, margin=margin)
    ids = []
    for i, a in enumerate(atoms):
        node = arc._add_atom({
            "id": "a%05d" % i, "text": a["text"], "question": "", "answer": a["text"],
            "file": a["session"], "folder_path": a["session"],
            "session_id": a["session"], "order": i,
            "confidence": "structural", "timestamp": a["timestamp"] or time.time(),
            "vec": V[i],
        })
        ids.append(node["id"])
    pos = {nid: i for i, nid in enumerate(ids)}
    adj = collections.defaultdict(set)
    for e in arc.edges:
        u, v = pos.get(e["from"]), pos.get(e["to"])
        if u is not None and v is not None and e["type"] in TYPED:
            adj[u].add(v)
            adj[v].add(u)
    return adj, len(arc.edges)


def main():
    print("читаю стенд ...", flush=True)
    data = json.load(io.open(DATA, encoding="utf-8"))
    rng = random.Random(SEED)
    by_type = collections.defaultdict(list)
    for it in data:
        by_type[it["question_type"]].append(it)
    picked = []
    per = max(1, N_Q // len(by_type))
    for t, items in by_type.items():
        rng.shuffle(items)
        picked.extend(items[:per])
    picked = picked[:N_Q]
    print("вопросов взято: %d, по типам: %s"
          % (len(picked), dict(collections.Counter(p["question_type"] for p in picked))),
          flush=True)

    import torch
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    enc = SentenceTransformer(MODEL, trust_remote_code=True, device="cuda",
                              model_kwargs={"dtype": torch.float32})
    enc.max_seq_length = 512
    print("эмбеддер поднят за %.0f с" % (time.time() - t0), flush=True)

    res = collections.defaultdict(lambda: collections.Counter())
    tot = collections.Counter()
    edge_stat = []
    t0 = time.time()
    for qi, item in enumerate(picked, 1):
        atoms = build_atoms(item)
        if not atoms:
            continue
        texts = [a["text"][:1500] for a in atoms]
        V = enc.encode(texts, batch_size=32, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        qv = enc.encode([item["question"]], convert_to_numpy=True,
                        normalize_embeddings=True).astype("float32")[0]

        ev_turns = {i for i, a in enumerate(atoms) if a["evidence"]}
        ev_sess = set(item.get("answer_session_ids") or [])
        if not ev_turns and not ev_sess:
            continue
        qt = item["question_type"]
        tot[qt] += 1

        sims = V @ qv
        flat = list(np.argsort(-sims)[:TOP_K])

        b, c, r = calibrate(V)
        ptg_core.BRANCH_THRESH, ptg_core.CONTINUE_THRESH, ptg_core.RETURN_THRESH = b, c, r
        adj, n_edges = ptg_graph(atoms, V)
        edge_stat.append(n_edges / max(1, len(atoms)))
        seeds = list(np.argsort(-sims)[:3])
        pool = set(seeds)
        for s in seeds:
            pool |= adj.get(s, set())
        pool = np.array(sorted(pool))
        ptg = list(pool[np.argsort(-(V[pool] @ qv))][:TOP_K]) if len(pool) else flat

        for name, idxs in (("плоский", flat), ("PTG", ptg)):
            for kk in KS:
                cut = idxs[:kk]
                ht = any(i in ev_turns for i in cut)
                hs = any(atoms[i]["session"] in ev_sess for i in cut)
                res[name]["turn%d_%s" % (kk, qt)] += int(ht)
                res[name]["sess%d_%s" % (kk, qt)] += int(hs)
                res[name]["turn%d_all" % kk] += int(ht)
                res[name]["sess%d_all" % kk] += int(hs)
        if qi % 10 == 0:
            print("  %d/%d, %.0f с" % (qi, len(picked), time.time() - t0), flush=True)

    n = sum(tot.values())
    print("\nвопросов засчитано: %d, рёбер на атом: %.1f"
          % (n, float(np.mean(edge_stat)) if edge_stat else 0))
    print("\n%-5s %16s %12s   %16s %12s"
          % ("k", "реплика плоский", "реплика PTG", "сессия плоский", "сессия PTG"))
    for kk in KS:
        print("%-5d %15s%% %11s%%   %15s%% %11s%%" % (
            kk,
            "%.1f" % (100.0 * res["плоский"]["turn%d_all" % kk] / max(1, n)),
            "%.1f" % (100.0 * res["PTG"]["turn%d_all" % kk] / max(1, n)),
            "%.1f" % (100.0 * res["плоский"]["sess%d_all" % kk] / max(1, n)),
            "%.1f" % (100.0 * res["PTG"]["sess%d_all" % kk] / max(1, n))))
    print("\nпо типам, сессия-улика при k=3:")
    print("  %-28s %5s %10s %10s" % ("тип", "n", "плоский", "PTG"))
    for qt in sorted(tot):
        k3 = tot[qt]
        print("  %-28s %5d %9.1f%% %9.1f%%"
              % (qt, k3, 100.0 * res["плоский"]["sess3_" + qt] / k3,
                 100.0 * res["PTG"]["sess3_" + qt] / k3))

    out = {"вопросов": n,
           "recall": {"k%d" % kk: {
               "реплика": {m: round(100.0 * res[m]["turn%d_all" % kk] / max(1, n), 1)
                           for m in ("плоский", "PTG")},
               "сессия": {m: round(100.0 * res[m]["sess%d_all" % kk] / max(1, n), 1)
                          for m in ("плоский", "PTG")}} for kk in KS},
           "по_типам_k3": {qt: {m: round(100.0 * res[m]["sess3_" + qt] / tot[qt], 1)
                                for m in ("плоский", "PTG")} for qt in tot}}
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with io.open(os.path.join(HERE, "out", "longmemeval_recall.json"), "w",
                 encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nзаписано: PTG_V5/out/longmemeval_recall.json")


if __name__ == "__main__":
    main()
