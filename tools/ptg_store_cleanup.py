"""Drop old versions of the full engine's LanceDB tables, keep only the current one (the data does not change).

Needed for stores written before 0.3.0: they kept every version for 7 days, and each save rewrote the vector table,
so a 0.8 GB store could take ~90 GB on disk. Since 0.3.0 the engine keeps versions for 10 minutes only.

Stop the daemon first (``ptg stop``). Without --apply it only shows the sizes.

    python tools/ptg_store_cleanup.py <store dir> [--apply]

<store dir> is the project's "store" from ~/.ptg-mem/projects.json; the tables live in its ``full`` subfolder.
Run it with the daemon's Python (the one that has lancedb).
"""
import os
import sys
from datetime import timedelta

import lancedb


def disk(path: str) -> int:
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    root = os.path.join(args[0], "full") if os.path.isdir(os.path.join(args[0], "full")) else args[0]
    apply = "--apply" in argv
    db = lancedb.connect(root)
    for name in sorted(os.listdir(root)):
        if not name.endswith(".lance"):
            continue
        path = os.path.join(root, name)
        table = db.open_table(name[:-6])
        rows, before = table.count_rows(), disk(path)
        if apply:
            table.cleanup_old_versions(timedelta(0), delete_unverified=True)
            table = db.open_table(name[:-6])
            assert table.count_rows() == rows, f"{name}: {rows} rows before, {table.count_rows()} after"
        print(f"{name:20s} rows {rows:7d}  {before / 1e9:6.2f} GB -> {disk(path) / 1e9:6.2f} GB")
    print("done" if apply else "dry run; add --apply to delete old versions")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
