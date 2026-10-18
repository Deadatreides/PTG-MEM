# Changelog

## 0.3.0 — 2026-10-18

### Added
* **The full engine**, opt-in per project with `"engine": "full"` in `~/.ptg-mem/projects.json`. See
  [docs/ENGINE-FULL.md](docs/ENGINE-FULL.md).
  * LanceDB store: vectors, full text and SQL in one table.
  * Search ranks by RRF of words and every embedding space.
  * Links with validity dates, Graphiti-style; the agent can declare their kind.
  * Code graph from codebase-memory-mcp for the repositories in `code_repos`.
  * The metatree: code tree, text sources, and `edits` / `reads` / `mentions` seams between session atoms and code.
  * Any number of embedding models, one space each. A Drift-Adapter between models proposes candidates.
  * Documents through MarkItDown; secrets through detect-secrets plus vendor formats.
* Four packages written through the rptg kit and accepted by ready checks: `memcore` (`rev/`), `memshell`
  (`rev-shell/`), `ptgfull` (`rev-full/`), `ptgcore` (`rev-core/`). Each has `spec.md` and its own tests.
* `ptg_mem/engine_full.py`: the daemon's adapter to `ptgcore`. `ptg_mem/engine3.py`: the intermediate LanceDB
  adapter (`"engine": "v3"`), kept for its tests.
* The daemon can run on its own interpreter. In `~/.ptg-mem/config.json`, `"python"` sets the interpreter and
  `"python_path"` adds package folders. Hooks stay on the Python that runs `ptg`.
* `tools/migrate_v6_to_full.py` moves a 0.2 store into the full engine without re-embedding. Only decision links
  are moved.
* `tools/ptg_store_cleanup.py` removes old LanceDB versions from stores written before 0.3.0.
* `docs/HISTORY.md` and `docs/HISTORY.ru.md`: why each part exists and what replaced what, with the measurements.
* Benchmarks in `bench/`:
  * hybrid recall on LongMemEval;
  * Drift-Adapter probe;
  * LanceDB exactness;
  * paragraph locality;
  * document parsing;
  * secret detection.

### Fixed
* Full engine stores grew without bound. LanceDB kept every table version for 7 days, and each save rewrote the
  vector table: 0.84 GB of data took 89.8 GB of disk. Versions are now kept for 10 minutes.

### Measured (author's live store, 47 own queries, 24 unrelated)
* The right atom in the first ten: 29 → 40.
* Prompt context with the right atom: 18 → 31.
* Silent on unrelated prompts: 21 → 20.
* Prompt-context latency, median: 0.15 s → 1.3–1.5 s. This is the open problem, and the reason the 0.2 engine
  stays the default.

## 0.2.0 — 2026-10-05

* Background daemon, prompt hooks and MCP for Claude Code and Codex.
* Browser window, decision log, Pro key.
* Store format 6.
* Code map (Code PTG) as a separate CLI and MCP server.
