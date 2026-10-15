"""Make ``src/`` and ``scripts/`` importable when the package is not installed."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    path = PROJECT_ROOT / sub
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
