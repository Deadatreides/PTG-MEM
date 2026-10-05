# -*- coding: utf-8 -*-
"""`ptg` command-line entry point that works from a checkout."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ptg_mem.cli import main  # noqa: E402

sys.exit(main())
