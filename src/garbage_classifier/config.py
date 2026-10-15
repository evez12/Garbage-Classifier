"""Typed configuration for the pipeline, loaded from ``config.yaml``.

The notebook keeps its configuration in module-level constants. Here the same
values live in frozen dataclasses so a caller can point the pipeline at a
different dataset or change hyperparameters without editing any source file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAME = "config.yaml"


def project_root() -> Path:
    """Repository root, resolved from this file's location (``src/garbage_classifier/``)."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    """Where the images live and how duplicates and splits are handled."""

    root: Path
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    phash_max_distance: int = 4
    test_n_splits: int = 7
    val_n_splits: int = 6


@dataclass(frozen=True)
class ImageConfig:
    """Input geometry and the normalisation the pretrained backbone expects."""

    width: int = 224
    height: int = 224
    channels: int = 3
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def size(self) -> tuple[int, int]:
        """``(height, width)``, the order torchvision transforms expect."""
        return self.height, self.width


@dataclass(frozen=True)
class LoaderConfig:
    """DataLoader settings shared by every split."""

    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True


@dataclass(frozen=True)
class LinearProbeConfig:
    """Stage 1 — a frozen backbone with a freshly initialised linear head."""

    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    eta_min: float = 1e-5
    patience: int = 5
    use_class_weights: bool = False
    class_weight_alpha: float = 1.0


@dataclass(frozen=True)
class FineTuneSharedConfig:
    """Settings every stage-2 run shares."""

    epochs: int = 20
    warmup_epochs: int = 1
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    patience: int = 5


@dataclass(frozen=True)
class SelectionConfig:
    """How the winning run, the inference recipe and the review threshold are chosen."""

    tta_significance: float = 0.05
    bootstrap_resamples: int = 2000
    calibration_bins: int = 15
    selective_targets: tuple[float, ...] = (0.97, 0.98, 0.99, 0.995)
    review_target: float = 0.99


@dataclass(frozen=True)
class Config:
    """The whole pipeline configuration, plus the derived output paths."""

    seed: int
    data: DataConfig
    image: ImageConfig
    loader: LoaderConfig
    linear_probe: LinearProbeConfig
    finetune_shared: FineTuneSharedConfig
    selection: SelectionConfig
    finetune_experiments: dict[str, dict[str, Any]] = field(default_factory=dict)
    root: Path = field(default_factory=project_root)

    # --- Derived paths. These match the notebook's layout exactly, so artifacts
    # --- produced by the scripts are picked up by the notebook's own caches.
    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def tensorboard_dir(self) -> Path:
        return self.root / "tensorboard"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def linear_probe_checkpoint(self) -> Path:
        """Stage-1 weights. The notebook calls this ``CKPT_PATH``."""
        return self.checkpoints_dir / "regnet_y_1_6gf_head.pt"

    @property
    def linear_probe_history(self) -> Path:
        """Stage-1 per-epoch history; its presence is what lets a re-run skip training."""
        return self.reports_dir / "regnet_y_1_6gf_head" / "history.csv"

    @property
    def bundle_path(self) -> Path:
        """The single deployable artifact: weights + preprocessing + recipe."""
        return self.checkpoints_dir / "best_model_bundle.pt"

    @property
    def registry_path(self) -> Path:
        return self.experiments_dir / "registry.csv"

    @property
    def eval_dir(self) -> Path:
        return self.reports_dir / "best_model"

    def with_overrides(self, **kwargs: Any) -> Config:
        """Return a copy with top-level fields replaced (used by the CLI flags)."""
        return replace(self, **kwargs)


def _as_path(value: str | Path, root: Path) -> Path:
    """Resolve a possibly relative path against the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path)


def load_config(path: str | Path | None = None, root: Path | None = None) -> Config:
    """Read ``config.yaml`` (or *path*) and build a :class:`Config`.

    Parameters
    ----------
    path
        Configuration file. Defaults to ``config.yaml`` at the repository root.
    root
        Repository root that relative paths resolve against. Defaults to the
        directory two levels above this module.
    """
    root = Path(root).resolve() if root is not None else project_root()
    config_path = Path(path) if path is not None else root / DEFAULT_CONFIG_NAME
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_raw = dict(raw.get("data", {}))
    data = DataConfig(
        root=_as_path(data_raw.pop("root", "data/garbage"), root),
        extensions=tuple(data_raw.pop("extensions", DataConfig.extensions)),
        **data_raw,
    )

    image_raw = dict(raw.get("image", {}))
    image = ImageConfig(
        mean=tuple(image_raw.pop("mean", ImageConfig.mean)),
        std=tuple(image_raw.pop("std", ImageConfig.std)),
        **image_raw,
    )

    selection_raw = dict(raw.get("selection", {}))
    selection = SelectionConfig(
        selective_targets=tuple(selection_raw.pop("selective_targets", SelectionConfig.selective_targets)),
        **selection_raw,
    )

    finetune_raw = dict(raw.get("finetune", {}))
    return Config(
        seed=int(raw.get("seed", 11)),
        data=data,
        image=image,
        loader=LoaderConfig(**raw.get("loader", {})),
        linear_probe=LinearProbeConfig(**raw.get("linear_probe", {})),
        finetune_shared=FineTuneSharedConfig(**finetune_raw.get("shared", {})),
        selection=selection,
        finetune_experiments=dict(finetune_raw.get("experiments", {})),
        root=root,
    )
