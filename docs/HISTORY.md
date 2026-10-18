# How PTG-MEM came to be, and why 0.3 was rewritten

[Русская версия](HISTORY.ru.md)

Every step below was a decision made for a measured reason. The numbers come from the author's own archive and from
the reports in [LLM-MEASUREMENTS](https://github.com/Deadatreides/LLM-MEASUREMENTS) and [`bench/`](../bench/).

## 1. PTG, spring–summer 2026: one person's chaos

The author is a design engineer, not a programmer.
* For months, ideas, architectures, maths and logic were discussed in free browser chats: ChatGPT, Grok, DeepSeek,
  Gemini, Meta.
* The answers were copied into text files in a project folder.
* After a couple of months the folder was chaos. Archived project dumps, project files, notes, measurements and
  logs were mixed together in no order.
* After a month of illness nobody, the author included, remembered what was where, why it was there and what came
  next.
* A fresh chat could not be given that context either.

**PTG (Personal Thought Graph)** was the fix. It is a desktop app with an MCP layer. It turns raw chat exports into a
path-dependent graph:
* an atom is one question with its answer;
* links are typed: continues, branches, returns, supersedes, contradicts, fixes, refines;
* placement goes along several axes: meaning, time, file and path.

The agent pulled in only the context that mattered at the moment, and it worked.

## 2. Code PTG: legacy that nobody here wrote

The code of the author's experiments, and of his llama.cpp fork, was written by AI agents. Nobody had read it. Code
PTG turns a codebase into cards:
* a card for every function and class;
* links: calls, imports, implements, tests;
* risks: what an edit can reach;
* a version history for each card;
* provenance: no confidence without evidence.

The steps:
* **26–27.08.2026.** 4 723 cards from the AST, served over MCP.
* **20–21.09.2026.** Made into a product:
  * a card shrank 7.2×, a hop 23.6×;
  * three hops cost about 410 tokens instead of about 10 600;
  * C/C++/CUDA through tree-sitter;
  * the embedder runs in-process, LM Studio is no longer needed.

## 3. Metatree: one memory across text, code and models

The author defined it on 20.09.2026:
* prose is the root: notes, chats, documentation;
* file paths and names hang on the prose;
* comments come next, and function vectors attach to them.

The metatree stitches the chat graph and the code graph together by provenance (file, symbol). It does not compare
vectors of different spaces.

A second job was to survive a change of embedding model. The plan was to keep model-independent layers and to move
queries between models with a Procrustes map. Measured 21.09, the map reached 0.49–0.65 agreement on the ten nearest
neighbours. The gate was 0.90. A translated query can only propose candidates, never take a model's place.

In 0.2 the metatree stayed a blueprint: nothing imported it.

## 4. PTG-MEM 0.2, 24.09–05.10.2026: a product instead of a desktop tool

The author asked for:
* no rebuild chains between store versions;
* no LM Studio;
* background embedding;
* context attached without being asked;
* a window to see what was attached;
* a Pro edition.

So PTG became a daemon:
* One process owns the memory. Hooks, MCP, CLI and the browser window are thin clients.
* A changed file is diffed atom by atom: unchanged atoms stay, vanished ones are retracted and kept in history, new
  ones get a `revises` link.
* Store format 6: 646 MB instead of 1.7 GB, load 4 s instead of 40 s.
* Graph growth under a degree budget: the earlier graph had a hub of 386 links and 470 orphans, after the fix zero
  orphans.

Two measurements shaped it:
* **The graph does not find more than plain vector search.** On the author's archive plain cosine found 70.3 %
  against 64.9 % for graph-first search. On LongMemEval at k = 10 it was 98.3 % against 91.7 %. So search went
  plain, and the graph only says what is current.
* **The chat parser was broken.** 81 % of the chat atoms were fragments. It was fixed before the release.

0.2.0 was published on 05.10.2026.

## 5. 08.10.2026: the market check that should have come first

A check against ready-made tools showed that both shelves are full:
* agent memory: claude-mem, mem0, Zep, Letta, basic-memory;
* code graphs: Graphify, codegraph, codebase-memory-mcp, Understand-Anything.

The rule since then: take the best ready parts, and write only the glue.

## 6. 0.3, 10–14.10.2026: the full PTG on ready parts

The revision (10.10) kept only what nobody else has:
* the parser for five vendors' chat exports;
* the question-and-answer atom;
* the hooks, the daemon and MCP.

Everything else was replaced by measured ready parts:

| part | 0.2 | 0.3 | why |
|---|---|---|---|
| store and search | own graph store, format 6 | LanceDB: vectors, full text and SQL in one table | LanceDB without an index equals numpy cosine, 200 of 200 |
| ranking | plain cosine | RRF of words and every embedding space | own corpus: words 61.7 %, vectors 38.3 %, RRF 59.6 %; LongMemEval: vectors 98.3 %, RRF 91.7 %. RRF never collapses |
| links | guessed by cosine thresholds | Graphiti-style links with validity dates; the agent declares the kind | 68 % of guessed "A replaces B" were false |
| code map | own Code PTG | codebase-memory-mcp (static, no LLM) | it found 9 of 9 callers of `client.call` on this repository in 12 s |
| documents | own extractors | MarkItDown | on the test set: PDF 17 of 25 where ours read 0, XLSX 25 of 25, DOCX the same Cyrillic; legacy .doc stays ours |
| secrets | own regexes | detect-secrets plus vendor formats | 13 of 14 caught, 0 false of 11 |
| other models | one model per store | an embedding space per model, Drift-Adapter between them | 0.7475 agreement on the 10 nearest neighbours; candidates only |
| metatree | blueprint | real: code tree, text sources, `edits` / `reads` / `mentions` seams between session atoms and code | — |

How it was built:
* The engine was written through the rptg kit, card by card, accepted by ready checks (mypy, tach, ruff, deal,
  Hypothesis, CrossHair) rather than by a model.
* It is four packages: `memcore`, `memshell`, `ptgfull`, `ptgcore`.
* The daemon talks to it through `ptg_mem/engine_full.py`.

The author's live store was migrated on 14.10:
* 27 816 atoms;
* 16 702 links — only the decision kinds; 417 442 guessed `reinforces` links were dropped;
* 879 code symbols;
* 25 835 metatree edges.

Paired measurement of the live daemon, old engine → new, 47 own queries and 24 unrelated ones:

| | 0.2 | 0.3 |
|---|---|---|
| the right atom in the first ten | 29 of 47 | **40 of 47** |
| prompt context with the right atom | 18 of 47 | **31 of 47** |
| silent on unrelated prompts | 21 of 24 | 20 of 24 |
| prompt-context latency, median | 0.15 s | **1.3–1.5 s** |

The full engine finds more and is slower. Latency is the open problem. Its sources: link and metatree reads on every
recall, a flat LanceDB scan, full-text rebuilds on writes, and recall waiting for the embedder during ingestion.

## 7. 18.10.2026: the store that ate the disk

LanceDB keeps old table versions for 7 days. Every save rewrote the vector table, 0.63 GB each time. The author's
store held 0.84 GB of data and took 89.8 GB on disk, growing by 5 GB in half an hour.

Since then:
* the engine keeps versions for 10 minutes (`KEEP_VERSIONS` in `ptgcore/ledger/file_table.py`);
* `tools/ptg_store_cleanup.py` removes old versions from stores written earlier.
