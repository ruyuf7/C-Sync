"""Stub package for the AI Trend Agent's future agent workflow.

The package adds 02_src to the import path so its modules can use the existing
``from schemas import ...`` imports when the package is loaded from the repo
root.
"""

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
