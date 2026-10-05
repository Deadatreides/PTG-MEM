---
description: Start PTG-MEM memory for this project (shows what will be indexed first)
allowed-tools: Bash(ptg:*), Bash(python:*), Bash(pip:*)
---
Set up PTG-MEM memory for the current project.

1. Check that the `ptg` command exists (`ptg --help`). If it does not, tell the user to install it with
   `pip install "ptg-mem[st]"` (or `uv tool install "ptg-mem[st]"`) and stop.
2. Run `ptg init "$CLAUDE_PROJECT_DIR" --dry-run` and show the user the output verbatim: how many files
   will be indexed and which embedding model will be downloaded (and its size).
3. Ask the user to confirm. Only after an explicit yes run `ptg init "$CLAUDE_PROJECT_DIR" --yes`,
   then `ptg install claude --path "$CLAUDE_PROJECT_DIR"`.
4. Tell the user that indexing runs in the background (`ptg status`, or `ptg gui` for the live view),
   and that memory is attached to prompts automatically from the next session on.
