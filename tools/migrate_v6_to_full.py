"""Move a 0.2 store (format 6) into the full engine without re-embedding, then build the code graph and the metatree.

Stop the daemon first (``ptg stop``). Run with the daemon's Python, the one that has the full engine installed.

    python tools/migrate_v6_to_full.py <store dir> <space> [--repo <path> ...] [--binary <codebase-memory-mcp>]

<store dir>  the project's "store" from ~/.ptg-mem/projects.json, with data/nodes.json and data/vectors.npy inside;
             the full engine is written to <store dir>/full next to it, the old files are not touched.
<space>      «name=provider:model@dim», the embedder that made the vectors, e.g. q4b=gguf:Qwen3-Embedding-4B-Q4_K_M@2560.
             Set the same name as "space" of the project in projects.json.
--repo       repositories for the code graph (also put them in the project's "code_repos").

Only the decision kinds of links are moved (supersedes, contradicts, fixes; revises becomes supersedes): the 0.2
guesses such as "reinforces" are not decisions, and there may be hundreds of thousands of them.
"""
import json
import os
import sys
import time

import ptgcore


def step(name, fn):
    t0 = time.time()
    out = fn()
    print(f"{name}: {out} ({time.time() - t0:.1f} s)", flush=True)
    return out


def main(argv: list[str]) -> int:
    repos, binary, plain = [], "codebase-memory-mcp", []
    it = iter(argv)
    for a in it:
        if a == "--repo":
            repos.append(next(it))
        elif a == "--binary":
            binary = next(it)
        else:
            plain.append(a)
    if len(plain) != 2:
        print(__doc__)
        return 2
    store, spec = plain
    home = os.path.join(store, "full")
    step("move the format 6 store", lambda: ptgcore.migrate_v6(home, store, spec))
    if repos:
        step("code graph", lambda: ptgcore.index_code(home, tuple(repos), binary))
    stats = step("metatree (save)", lambda: ptgcore.save(home))
    print(json.dumps(stats.__dict__, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
