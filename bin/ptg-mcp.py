# -*- coding: utf-8 -*-
"""MCP server entry point that works from a checkout or a plugin folder."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptg_mem.mcp_server import main  # noqa: E402

main()
