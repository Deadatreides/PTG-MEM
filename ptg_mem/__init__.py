# -*- coding: utf-8 -*-
"""PTG-MEM — local memory for coding agents.

The product layer lives in this package's top-level modules (engine, daemon,
hooks, mcp_server, cli, gui). The graph engine itself — ``core/ptg_core.py``,
the vectorised graft ``v4/graft_fast.py`` and the budgeted graft
``v5/budget_graft.py`` — is research code with flat imports, so its folders are
put on ``sys.path`` here once, before anything imports it.
"""
import os
import sys

__version__ = "0.2.0"

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("core", "v4", "v5"):
    _p = os.path.join(_HERE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
