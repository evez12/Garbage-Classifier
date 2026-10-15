"""Top-level ``garbage_dataset`` module kept for the notebook's import.

``notebooks/model_training_notebook.ipynb`` contains
``from garbage_dataset import GarbageDataset`` and must not be modified, so
this module re-exports the class from its real home in
:mod:`garbage_classifier.data`. Keeping the dataset in a module of its own
(rather than defining it in the notebook) is what lets DataLoader worker
processes pickle it on Windows, where multiprocessing uses ``spawn``.
"""

from __future__ import annotations

from garbage_classifier.data import GarbageDataset

__all__ = ["GarbageDataset"]
