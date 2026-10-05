# -*- coding: utf-8 -*-
"""Claude Code hook entry point that works from a checkout or a plugin folder
(no pip install needed): puts the repository on sys.path and runs ptg_mem.hooks."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptg_mem.hooks import main  # noqa: E402

sys.exit(main())
