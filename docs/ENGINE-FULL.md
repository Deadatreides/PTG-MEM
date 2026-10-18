# The full engine (0.3): setup, layout, numbers

The daemon, hooks, MCP tools and the window are the same as in 0.2. What changes is the engine behind them. You
choose it per project in `~/.ptg-mem/projects.json`:

| `"engine"` | what it is |
|---|---|
| absent | the 0.2 engine (`ptg_mem/engine.py`): own graph store, format 6. Still the default |
| `"full"` | the full PTG (`ptg_mem/engine_full.py` + the four packages below): LanceDB, RRF, links with validity dates, code graph, metatree, any number of embedding models |
| `"v3"` | the intermediate LanceDB adapter (`ptg_mem/engine3.py`), single vector column; kept for its tests |

Why it was rewritten, with the measurements: [HISTORY.md](HISTORY.md).

## Layout

| folder | package | role | made with |
|---|---|---|---|
| `rev/` | `memcore` | what to keep, retract or replace when a source changes; links with validity dates (Graphiti-style); context budget; file selection; secret scrubbing (detect-secrets + vendor formats) | rptg kit |
| `rev-shell/` | `memshell` | files to text (MarkItDown), atoms (a paragraph, Chonkie for long ones), selection window, session rule, the `<ptg-memory>` block, the decision feed | rptg kit |
| `rev-full/` | `ptgfull` | LanceDB store, embedding spaces and RRF, Drift-Adapter between models, code graph through codebase-memory-mcp, the metatree and its seams | rptg kit |
| `rev-core/` | `ptgcore` | the 19 operations the daemon calls: ingest, search, recall, decisions, node, graph, remember, save, migration from format 6 | rptg kit |
| `ptg_mem/engine_full.py` | — | the glue: the daemon's embedder, one writer at a time, the answer shapes of 0.2 | by hand |

Each package has `spec.md` (what it must do), cards with conditions and examples in the code, and its own tests. The
rptg kit accepted the code only when ready checks passed: mypy, tach, ruff, deal, Hypothesis, CrossHair and radon.

## Install (Python 3.11+)

The daemon of the full engine runs in its own environment. Hooks keep using the Python that runs `ptg`. They are
HTTP clients only.

```bash
git clone https://github.com/Deadatreides/PTG-MEM.git && cd PTG-MEM
python -m venv .venv-full
.venv-full/Scripts/python -m pip install -e . -e rev -e rev-shell -e rev-full -e rev-core   # Linux/macOS: .venv-full/bin/python
.venv-full/Scripts/python -m pip install lancedb pyarrow markitdown chonkie detect-secrets deal networkx scipy pathspec llama-cpp-python
```

Use `sentence-transformers` instead of `llama-cpp-python` if your embedder is not GGUF.

For the code graph, download the `codebase-memory-mcp` binary from its releases and put the path in `code_binary`.
Do not run its `install` command: it edits your agents' settings.

`~/.ptg-mem/config.json`: the daemon's interpreter.

```json
{ "python": "C:/path/to/PTG-MEM/.venv-full/Scripts/python.exe" }
```

`~/.ptg-mem/projects.json`, inside your project's entry:

```json
"engine": "full",
"space": "main",
"code_repos": ["C:/path/to/repo-you-want-mapped"],
"code_binary": "C:/path/to/codebase-memory-mcp.exe"
```

Then `ptg stop`. The next hook starts the daemon with the new engine. A new project starts with an empty store at
`<store>/full`.

## Moving a 0.2 store without re-embedding

```bash
ptg stop
.venv-full/Scripts/python tools/migrate_v6_to_full.py <store> "main=gguf:Qwen3-Embedding-4B-Q4_K_M@2560" --repo <repo> --binary <codebase-memory-mcp>
```

The space string is `name=provider:model@dim` of the model that made the old vectors. The old files stay where they
are, and the full engine is written to `<store>/full`.

Only decision links are moved: supersedes, contradicts, fixes, and revises (which becomes supersedes). The 0.2
threshold guesses are dropped. On the author's store there were 417 442 `reinforces` links, and copying them one
batch at a time took hours.

## Disk

Stores written before 0.3.0 kept every LanceDB version for 7 days, and each save rewrote the vector table. One store
held 0.84 GB of data and took 89.8 GB on disk. Now versions are kept for 10 minutes.

To clean an older store, stop the daemon and run:

```bash
.venv-full/Scripts/python tools/ptg_store_cleanup.py <store> --apply
```

Without `--apply` it only shows the sizes.

## Numbers

Measured on the author's live store: 27 816 atoms, 16 702 links, 879 code symbols, 25 835 metatree edges. Embedder
Qwen3-Embedding-4B GGUF, GTX 1660 SUPER 6 GB, Windows 10. 47 own queries and 24 unrelated ones, old daemon and new
one in turn:

| | 0.2 engine | full engine |
|---|---|---|
| the right atom in the first ten | 29 of 47 | 40 of 47 |
| prompt context with the right atom | 18 of 47 | 31 of 47 |
| silent on unrelated prompts | 21 of 24 | 20 of 24 |
| prompt-context latency, median | 0.15 s | 1.3–1.5 s |
| ingest of one file | — | 3–5 s |

Search ranks by RRF of words and every embedding space. Neither single method wins everywhere:

| corpus | words | vectors | RRF |
|---|---|---|---|
| own archive (Qwen3-Embedding-4B) | 61.7 % | 38.3 % | 59.6 % |
| LongMemEval (Qwen3-Embedding-0.6B) | 88.3 % | 98.3 % | 91.7 % |

Graph neighbours do not raise recall on either corpus. The graph marks what replaced what and shows code and
neighbouring turns after a hit.

## Known problems

* **Latency.**
  * Prompt context takes 1.3–1.5 s, up to 11 s while files are being ingested. The hook waits 6 s and then sends
    the prompt without memory.
  * Causes: links and metatree edges are read on every recall, LanceDB does a flat scan, the full-text index is
    rebuilt on each write, and recall waits for the embedder during ingestion.
* **Real-model runs on the author's machine only:** Windows 10, one GPU. The four packages' test suites need no
  model and no GPU, and CI runs them on Linux and Windows, Python 3.11 and 3.12. macOS is not in CI yet.
* **The code graph** covers only the repositories listed in `code_repos`, not the whole folder.
* **Contracts in production.** The engine disables deal contracts at run time: with them, recall took 316 s.
  `PTG_MEM_CONTRACTS=1` turns them back on.
