"""Dataset, data audit, near-duplicate grouping, group-aware splits and transforms.

This is the extraction of notebook sections 1, 3, 4, 5, 6 and 7. The important
property it preserves is the **leakage-free split**: images are grouped by
perceptual hash first, and the train / validation / test folds are stratified by
class *and* disjoint at the group level, so no near-duplicate of a test image
can appear in training.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import RegNet_Y_1_6GF_Weights
from tqdm.auto import tqdm

from garbage_classifier.config import Config

__all__ = [
    "GarbageDataset",
    "audit_dataset",
    "build_dataloaders",
    "build_splits",
    "compute_class_weights",
    "drop_conflicting_groups",
    "eval_transform",
    "finetune_train_transform",
    "linear_probe_train_transform",
    "load_splits",
    "phash_groups",
    "seed_everything",
    "to_absolute",
    "to_relative",
]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs.

    Seeding alone does not make CUDA runs bit-exact: cuDNN may pick
    non-deterministic kernels. For strict reproducibility also set
    ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False``, at some speed cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GarbageDataset(Dataset):
    """Image-classification dataset backed by a ``(path, y)`` DataFrame.

    Parameters
    ----------
    df
        Frame with a ``path`` column of absolute image paths and an integer
        ``y`` column of class indices.
    transform
        Callable applied to the decoded ``PIL.Image`` in RGB.
    """

    def __init__(self, df: pd.DataFrame, transform: Callable) -> None:
        self.paths: list[str] = df["path"].tolist()
        self.targets: list[int] = df["y"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[idx]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.targets[idx]


# ---------------------------------------------------------------------------
# Portable paths
# ---------------------------------------------------------------------------
def to_relative(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Store paths relative to *root*, so split files are portable across machines."""
    return frame.assign(path=frame["path"].map(lambda p: Path(p).relative_to(root).as_posix()))


def to_absolute(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Resolve relative paths back to absolute ones for loading."""
    return frame.assign(path=frame["path"].map(lambda p: str(root / p)))


# ---------------------------------------------------------------------------
# Data audit
# ---------------------------------------------------------------------------
def audit_file(path: Path) -> dict[str, Any]:
    """Open one image and record integrity, format, geometry and both hashes.

    ``md5`` catches byte-identical duplicates; ``phash`` is a 64-bit perceptual
    hash that also catches re-encoded, resized and lightly edited copies.
    A corrupt file is reported with ``ok=False`` rather than raising, because a
    single unreadable image would otherwise crash a DataLoader worker mid-epoch.
    """
    row: dict[str, Any] = {"path": str(path), "label": path.parent.name, "ext": path.suffix.lower(), "ok": True}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            row.update(mode=im.mode, fmt=im.format, width=im.width, height=im.height)
            rgb = im.convert("RGB")  # grayscale / RGBA / CMYK / P -> RGB
            row["phash"] = str(imagehash.phash(rgb))
        row["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        row.update(ok=False, error=repr(exc))
    return row


def audit_dataset(data_root: Path, extensions: Sequence[str], progress: bool = True) -> pd.DataFrame:
    """Audit every image under *data_root*, one row per file.

    Returns a frame with the :func:`audit_file` fields plus ``aspect`` and
    ``pixels``. The class label is the name of the containing directory.
    """
    wanted = {e.lower() for e in extensions}
    paths = sorted(p for p in data_root.rglob("*") if p.suffix.lower() in wanted)
    if not paths:
        raise FileNotFoundError(
            f"No images with extensions {sorted(wanted)} under {data_root}. "
            "See README.md -> Dataset for the download instructions."
        )
    frame = pd.DataFrame([audit_file(p) for p in tqdm(paths, desc="Auditing images", disable=not progress)])
    frame["aspect"] = frame["width"] / frame["height"]
    frame["pixels"] = frame["width"] * frame["height"]
    return frame


# ---------------------------------------------------------------------------
# Near-duplicate grouping
# ---------------------------------------------------------------------------
_POPCOUNT_U8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Bit count for a ``uint64`` array (NumPy >= 2.0 fast path, lookup-table fallback)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POPCOUNT_U8[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def phash_groups(hashes: pd.Series, max_dist: int = 4, chunk: int = 512) -> np.ndarray:
    """Group images so that near-duplicates share a group id.

    Two images are linked when the Hamming distance between their perceptual
    hashes is at most *max_dist* (out of 64 bits); links are merged
    transitively via connected components on a sparse adjacency matrix.
    Distances are computed in chunks with a vectorised popcount, so the O(N^2)
    comparison stays fast and memory-bounded.
    """
    h = np.array([int(s, 16) for s in hashes], dtype=np.uint64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for i in range(0, len(h), chunk):
        dist = popcount64(h[i : i + chunk, None] ^ h[None, :])
        r, c = np.nonzero(dist <= max_dist)
        r += i
        keep = r < c  # upper triangle only, no self-pairs
        rows.append(r[keep])
        cols.append(c[keep])
    row_idx, col_idx = np.concatenate(rows), np.concatenate(cols)
    n = len(h)
    adjacency = coo_matrix((np.ones(len(row_idx), dtype=np.int8), (row_idx, col_idx)), shape=(n, n))
    return connected_components(adjacency, directed=False)[1]


def drop_conflicting_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove near-duplicate groups whose members carry different labels.

    A group of visually identical images annotated with two different classes is
    label noise, not signal; keeping it would both train on contradictions and
    make the metrics unachievable. Returns the cleaned frame and the number of
    conflicting groups removed.
    """
    labels_per_group = frame.groupby("group")["label"].nunique()
    conflicting = labels_per_group[labels_per_group > 1].index
    cleaned = frame[~frame["group"].isin(conflicting)].reset_index(drop=True)
    return cleaned, len(conflicting)


# ---------------------------------------------------------------------------
# Group-aware stratified split
# ---------------------------------------------------------------------------
def holdout_group_split(frame: pd.DataFrame, n_splits: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out one stratified, group-disjoint fold (~``1 / n_splits`` of the data)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, hold_idx = next(sgkf.split(frame, frame["y"], groups=frame["group"]))
    return frame.iloc[rest_idx], frame.iloc[hold_idx]


def build_splits(config: Config, progress: bool = True) -> tuple[dict[str, pd.DataFrame], list[str], pd.DataFrame]:
    """Audit the dataset, drop label noise and carve out leakage-free splits.

    Returns ``({"train": ..., "val": ..., "test": ...}, classes, audit_frame)``.
    The splits are also written to ``splits/*.csv`` with paths relative to the
    data root, so they are portable and can be reloaded by :func:`load_splits`.
    """
    audit = audit_dataset(config.data.root, config.data.extensions, progress=progress)
    n_corrupt = int((~audit["ok"]).sum())
    if n_corrupt:
        print(f"Skipping {n_corrupt} corrupt image(s).")

    ok = audit[audit["ok"]].reset_index(drop=True).copy()
    ok["group"] = phash_groups(ok["phash"], max_dist=config.data.phash_max_distance)
    ok, n_conflicting = drop_conflicting_groups(ok)
    if n_conflicting:
        print(f"Removed {n_conflicting} near-duplicate group(s) with conflicting labels (label noise).")

    frame = ok[["path", "label", "group"]].reset_index(drop=True)
    classes: list[str] = sorted(frame["label"].unique())
    frame["y"] = frame["label"].map({c: i for i, c in enumerate(classes)})

    trainval_df, test_df = holdout_group_split(frame, config.data.test_n_splits, config.seed)
    train_df, val_df = holdout_group_split(trainval_df, config.data.val_n_splits, config.seed)

    # Leakage guard: no near-duplicate group may span two splits.
    assert not set(train_df["group"]) & set(val_df["group"])
    assert not set(train_df["group"]) & set(test_df["group"])
    assert not set(val_df["group"]) & set(test_df["group"])

    splits = {"train": train_df, "val": val_df, "test": test_df}
    config.splits_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        to_relative(part, config.data.root).to_csv(config.splits_dir / f"{name}.csv", index=False)
    return splits, classes, audit


def load_splits(config: Config) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Reload the splits written by :func:`build_splits`, resolving paths to absolute."""
    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        path = config.splits_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing — run the pipeline once to generate the splits.")
        splits[name] = to_absolute(pd.read_csv(path), config.data.root).reset_index(drop=True)
    classes = sorted(pd.concat([s["label"] for s in splits.values()]).unique())
    return splits, classes


# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------
def compute_class_weights(labels: pd.Series, num_classes: int, alpha: float = 1.0) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1.

    ``w_c = (N / (K * n_c)) ** alpha``, so ``alpha=1.0`` is full inverse
    frequency and ``alpha=0.5`` a softer square-root variant. Normalising to
    mean 1 keeps the loss scale — and therefore the effective learning rate —
    comparable to the unweighted loss. Always computed on the **training**
    split only, so no validation or test statistic leaks into training.
    """
    y = torch.from_numpy(labels.to_numpy(dtype=np.int64, copy=True))
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = (counts.sum() / (num_classes * counts)) ** alpha
    return weights / weights.mean()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def linear_probe_train_transform(config: Config) -> T.Compose:
    """Stage-1 augmentation: mild crop, flip and colour jitter.

    The conservative ``scale`` lower bound avoids crops that lose the object
    entirely, and waste items have no canonical left/right orientation.
    """
    return T.Compose(
        [
            T.RandomResizedCrop(size=config.image.size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(mean=config.image.mean, std=config.image.std),
        ]
    )


def finetune_train_transform(config: Config) -> T.Compose:
    """Stage-2 augmentation: stronger, because far more capacity is trainable."""
    return T.Compose(
        [
            T.RandomResizedCrop(size=config.image.size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.Normalize(mean=config.image.mean, std=config.image.std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )


def eval_transform() -> Any:
    """The deterministic preprocessing published with the pretrained weights.

    Resize the shorter side to 232 (bilinear), centre-crop 224, normalise with
    the ImageNet statistics. Taken straight from the weights enum rather than
    hand-copied, so it cannot drift from what the backbone was trained on.
    """
    return RegNet_Y_1_6GF_Weights.DEFAULT.transforms()


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders(
    config: Config,
    splits: dict[str, pd.DataFrame],
    train_transform: Callable,
    batch_size: int | None = None,
) -> dict[str, DataLoader]:
    """Build the three loaders.

    Training shuffles with a seeded generator and drops the last partial batch
    so no tiny, noisy batch skews an epoch. Validation and test keep
    ``shuffle=False``, which the evaluation code relies on to map predictions
    back to file paths — asserted rather than assumed.
    """
    batch_size = batch_size or config.loader.batch_size
    eval_tf = eval_transform()
    common = {"num_workers": config.loader.num_workers, "pin_memory": config.loader.pin_memory}
    train_kwargs = {**common, "persistent_workers": config.loader.num_workers > 0}
    eval_kwargs = {**common, "persistent_workers": False}

    return {
        "train": DataLoader(
            GarbageDataset(splits["train"], train_transform),
            batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(config.seed),
            **train_kwargs,
        ),
        "val": DataLoader(GarbageDataset(splits["val"], eval_tf), batch_size, shuffle=False, **eval_kwargs),
        "test": DataLoader(GarbageDataset(splits["test"], eval_tf), batch_size, shuffle=False, **eval_kwargs),
    }
