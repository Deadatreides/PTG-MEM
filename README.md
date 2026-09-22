# PTG-MEM

**Local memory for AI agents that accepts a folder — not a repository, not an
instrumented app, not a vector database. A folder.**

Point it at the mess on your disk: chat exports, half-broken scripts, archives,
someone else's code mixed with your own. It builds a graph of what you were
thinking, and serves it to your agent over MCP.

> *"It's not that you're dumb — the model kept forgetting."*

---

## Why this exists

Every tool in this space needs you to have prepared something first:

| tool | what you must already have |
|---|---|
| Mem0, Zep, Letta | an instrumented app emitting episodes through an API |
| CodeGraph, Sourcegraph | a repository |
| Claude Context | a vector database and an embedding provider |
| Spec Kit | a project and process discipline |

**None of them takes a folder.** Most people who need memory the most have
exactly a folder — and nothing else.

## What it actually does

It builds a *path-dependent trajectory graph*: atoms of thought connected by
**typed** edges — `continues`, `refines`, `fixes`, `supersedes`, `contradicts`,
`returns_to`. Flat search gives you a thought. The graph gives you its fate:
what refined it, what fixed it, what overrode it, what it contradicts.

Everything runs on your machine. The embedder is in-process. No server, no API
key, no network after install, nothing leaves the disk.

## Measured

Numbers below are from real archives in this repository's history, not
projections.

| | |
|---|---|
| traversal of a whole codebase | **3 hops** at 21 678 nodes |
| cost of a question | 2 430 tokens at top-12, against ~118 000 for a repo map |
| edges built with zero LLM calls | 346 621 |
| typed history recovered | 1 103 contradictions, 4 600 unresolved lines |
| intake from an unprepared folder | 6 184 files scanned, 1 999 accepted by rule |
| graph connectivity | mean degree 30.15 against a threshold of ln n = 9.98, **zero orphans** |
| storage after packing | 180 MB for 8 436 nodes |

## What this project does NOT claim

This section is deliberate, and it stays.

* **Retrieval quality is not better than flat vector search.** Measured, twice,
  against ourselves. On LongMemEval V1 at k=10, flat cosine recalls 98.3 % of
  evidence sessions; graph expansion recalls 91.7 %. At a fixed result budget,
  graph expansion can only *displace* a true hit. We publish the harness that
  shows this: [`bench/`](bench/).
* **Answer accuracy is unmeasured.** The published numbers of Zep (63.8) and
  Mem0 (49.0) are answer accuracy judged by a model. Ours is recall. They are
  not comparable and we do not compare them.
* **It does not scale to millions of nodes yet.** Build cost is quadratic;
  verified to ~28 000 nodes.
* It does not "understand" code.

If you find a claim in this README that isn't backed by a number in this
repository, it's a bug — open an issue.

## Status

Early. Packaging (`pip install`) is not finished; right now this is source you
run directly. See [docs/MARKET.md](docs/MARKET.md) for where this sits against
the rest of the field, including the parts where it loses.

## Licence

AGPL-3.0. If AGPL doesn't work for your company, see
[COMMERCIAL.md](COMMERCIAL.md) — a commercial licence is a file and an invoice,
with no support obligation attached.

## Support

There isn't any, and that's a decision rather than neglect. Issues may go
unanswered. What replaces support is release rhythm: each release announces the
next one.
