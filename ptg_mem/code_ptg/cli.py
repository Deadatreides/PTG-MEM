#!/usr/bin/env python3
"""CLI для Code PTG (v2, на настоящем графовом движке PTG)."""

import argparse
from collections import Counter
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from code_ptg import CodeArchive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "index", "reindex", "card", "deps", "risks", "lineage",
        "list", "branches", "root",
    ])
    # Несколько целей за один запуск — не удобство, а требование.
    # Запуск на каждую цель означает загрузку архива заново, а `load_if_exists()`
    # при индексе больше 50 МБ отображает его в память, после чего сохранение
    # падает с ERROR_USER_MAPPED_FILE. Одна загрузка и одно сохранение эту
    # ловушку обходят по построению и заодно экономят минуты на перезагрузке.
    parser.add_argument("target", nargs="*", default=None)
    parser.add_argument("--project", default=".")
    parser.add_argument("--store", default=None)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--model", default=None,
                        help="путь к модели sentence-transformers для эмбеддинга "
                             "внутри процесса. Задан — LM Studio не нужен вовсе. "
                             "То же можно передать переменной CODE_PTG_MODEL")
    parser.add_argument("--device", default=None, help="cuda | cpu (по умолчанию — как есть)")
    parser.add_argument("--skip", default="", help="каталоги через запятую, которые не "
                                                   "индексировать (сверх умолчательных: "
                                                   ".venv, node_modules, __pycache__ и т.п.)")
    args = parser.parse_args()

    if args.model:
        os.environ["CODE_PTG_MODEL"] = args.model
    if args.device:
        os.environ["CODE_PTG_DEVICE"] = args.device

    archive = CodeArchive(args.project, store_dir=args.store, use_embeddings=not args.no_embeddings)
    archive.load()
    # команды кроме index работают с одной целью
    one = (args.target[0] if args.target else None)

    if args.command == "index":
        targets = list(args.target) or [args.project]
        skip = {d.strip() for d in args.skip.split(",") if d.strip()}
        if all(os.path.isdir(t) for t in targets):
            total_ok = total_bad = 0
            for t in targets:
                rep = archive.index_directory(t, skip_dirs=skip)
                total_ok += rep["проиндексировано"]
                total_bad += rep["пропущено"]
                print("%s: проиндексировано %d, пропущено %d"
                      % (t, rep["проиндексировано"], rep["пропущено"]))
                for p, why in rep["причины"][:5]:
                    print("    пропущен %s — %s" % (os.path.basename(p), why[:90]))
            print("ИТОГО: проиндексировано %d, пропущено %d" % (total_ok, total_bad))
        elif len(targets) == 1 and not os.path.isdir(targets[0]):
            target = targets[0]
            archive.index_file(target)
            archive._resolve_refs()
        archive.save()
        nodes = [n for n in archive.engine.archive.nodes.values() if n.get("domain") == "code"]
        print(f"Узлов в графе: {len(nodes)} (эмбеддинги: "
              f"{'доступны' if archive.engine.embeddings_available else 'недоступны — узлы изолированные'})")
        # Перечислять все узлы поимённо нельзя: на восьми тысячах это сотни
        # килобайт вывода, в которых теряется то, ради чего смотрят.
        by_kind = Counter((n.get("kind"), n.get("subkind")) for n in nodes)
        by_src = Counter((n.get("layer_1_provenance") or {}).get("source") for n in nodes)
        for (k, sk), c in by_kind.most_common():
            print(f"  {k}:{sk:12s} {c:6d}")
        print("  источник фактов:", dict(by_src))

    elif args.command == "reindex":
        result = archive.reindex_file(one)
        archive.save()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "card":
        node = archive.get_active_card_by_qname(one)
        print(json.dumps(node, ensure_ascii=False, indent=2, default=str) if node else "Карточка не найдена.")

    elif args.command == "deps":
        node = archive.get_active_card_by_qname(one)
        if not node:
            print("Карточка не найдена.")
            return
        for e in archive.get_dependencies(node["id"]):
            dst = archive.engine.archive.nodes.get(e["to"], {})
            print(f"  --{e['type']}--> {dst.get('qualified_name', e['to'])}")

    elif args.command == "risks":
        node = archive.get_active_card_by_qname(one)
        if not node:
            print("Карточка не найдена.")
            return
        for r in archive.get_risk_impact(node["id"]):
            print(f"  hop={r['hops']} via={r['via_edge']}  {r['qualified_name']}")

    elif args.command == "lineage":
        node = archive.get_active_card_by_qname(one)
        if not node:
            print("Карточка не найдена.")
            return
        for v in archive.get_lineage(node["id"]):
            print(f"  [{v['status']:10s}] {v['card_id'][:10]}  content={v['content_hash']} sig={v['signature_hash']}")

    elif args.command == "list":
        for n in archive.engine.archive.nodes.values():
            if n.get("domain") == "code":
                print(f"[{n['kind']:10s}] {n['qualified_name']:35s} status={n['status']:10s} branch={n['branch'][:8]}")

    elif args.command == "branches":
        print(json.dumps(archive.engine.archive.list_branches(), ensure_ascii=False, indent=2, default=str))

    elif args.command == "root":
        print(json.dumps(archive.engine.archive.root_overview(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
