"""Garbage Classifier — RegNetY-1.6GF transfer learning for 10-class waste images.

The package is a faithful extraction of the pipeline in
``notebooks/model_training_notebook.ipynb``, reorganised into importable
modules so the same pipeline can be driven from the command line.

Module map
----------
``config``       Typed configuration loaded from ``config.yaml``.
``data``         Dataset, data audit, near-duplicate grouping, group-aware splits, transforms.
``model``        Backbone construction, parameter freezing, optimiser groups, LR schedules.
``metrics``      Metric and calibration primitives (macro-F1, ECE, temperature, bootstrap).
``training``     Training and evaluation loops.
``experiments``  Run configuration, artifact layout, caching, registry.
``selection``    Model selection, inference recipe, deployable bundle export.
``inference``    Bundle loading and single-image prediction.
``viz``          Figures used by the reports and the demo site.
"""

from garbage_classifier.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
__version__ = "1.0.0"
