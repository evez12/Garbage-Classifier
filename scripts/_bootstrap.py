"""Put ``src/`` on ``sys.path`` so the scripts run from a clone without installing.

``pip install -e .`` makes this unnecessary, but not everyone wants to install a
package just to run a training script.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
