"""Import shim so the notebook runs straight from a clone, with no install step.

``model_training_notebook.ipynb`` does ``from garbage_dataset import
GarbageDataset``. Jupyter's working directory is this ``notebooks/`` folder, so
Python looks for ``garbage_dataset`` here. This file puts ``src/`` on
``sys.path`` and re-exports the real implementation from
:mod:`garbage_classifier.data`, which keeps a single copy of the class.

``sys.path`` is inherited by DataLoader worker processes (multiprocessing passes
it in the spawn preparation data), so ``num_workers > 0`` works on Windows too.

If the package is installed instead (``pip install -e .``), that installation
takes precedence and this shim is harmless.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from garbage_classifier.data import GarbageDataset  # noqa: E402

__all__ = ["GarbageDataset"]
