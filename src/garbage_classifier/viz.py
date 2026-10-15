"""Figures for the reports and the demo site.

Matplotlib's non-interactive Agg backend is selected on import, so these work
unchanged in a headless script run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from garbage_classifier.metrics import calibration_table, expected_calibration_error  # noqa: E402

#: Fixed colour roles, used consistently across every figure.
TRAIN_COLOR = "#2a78d6"  # blue   -> train / correct / selected model
VAL_COLOR = "#eb6834"  # orange -> validation / incorrect
NEUTRAL_COLOR = "#52514e"  # gray   -> reference lines
SEQ_CMAP = "Blues"  # single-hue sequential map for heatmaps

__all__ = [
    "NEUTRAL_COLOR",
    "SEQ_CMAP",
    "TRAIN_COLOR",
    "VAL_COLOR",
    "plot_calibration",
    "plot_confusion_matrices",
    "plot_per_class_f1",
    "plot_training_history",
    "save_figure",
    "set_theme",
    "top_confusions",
]


def set_theme() -> None:
    """Apply the seaborn theme the figures were designed with."""
    sns.set_theme(style="whitegrid", context="notebook")


def save_figure(fig: plt.Figure, directory: Path, name: str, dpi: int = 150) -> Path:
    """Write *fig* to ``<directory>/<name>.png`` and close it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_history(history: pd.DataFrame, monitor: str = "val_macro_f1") -> plt.Figure:
    """Loss, accuracy, the monitored metric and the LR per epoch, with the best epoch marked.

    Train metrics are measured on augmented batches in ``train()`` mode and
    validation on clean centre crops in ``eval()`` mode, so a validation loss
    *below* the training loss is expected here and not a bug. A widening gap
    signals overfitting; flat, close curves signal a capacity limit.
    """
    epochs = history["epoch"].to_numpy() if "epoch" in history else np.arange(1, len(history) + 1)
    best_epoch = int(epochs[int(np.argmax(history[monitor].to_numpy()))])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    panels = [
        (axes[0, 0], "Cross-entropy loss", [("train_loss", "Train", TRAIN_COLOR), ("val_loss", "Validation", VAL_COLOR)], 1.0),
        (axes[0, 1], "Accuracy (%)", [("train_acc", "Train", TRAIN_COLOR), ("val_acc", "Validation", VAL_COLOR)], 100.0),
        (axes[1, 0], "Validation macro-F1", [("val_macro_f1", "Validation", VAL_COLOR)], 1.0),
        (axes[1, 1], "Learning rate", [("lr", "Learning rate", NEUTRAL_COLOR)], 1.0),
    ]

    for ax, title, series, scale in panels:
        for key, label, color in series:
            if key not in history:
                continue
            ax.plot(epochs, history[key].to_numpy() * scale, color=color, lw=2, marker="o", ms=4, label=label)
        ax.axvline(best_epoch, color=NEUTRAL_COLOR, ls="--", lw=1, label=f"Best epoch ({best_epoch})")
        ax.set_title(title)
        ax.legend(frameon=False)

    axes[1, 1].set_yscale("log")
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    fig.suptitle("Training history", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(cm: np.ndarray, classes: list[str], title: str) -> plt.Figure:
    """Side-by-side annotated heatmaps: raw counts and row-normalised percentages.

    The row-normalised panel's diagonal is per-class recall.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0) * 100
    pct_annot = np.where(cm > 0, np.char.mod("%.1f", cm_pct), "")

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    heatmap_kw = dict(
        cmap=SEQ_CMAP,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"shrink": 0.8},
    )
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[0], **heatmap_kw)
    sns.heatmap(cm_pct, annot=pct_annot, fmt="", vmin=0, vmax=100, ax=axes[1], **heatmap_kw)

    for ax, subtitle in zip(axes, ["Counts", "Row-normalised (% of true class) — diagonal = recall"]):
        ax.set_title(subtitle)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def top_confusions(cm: np.ndarray, classes: list[str], k: int = 10) -> pd.DataFrame:
    """The *k* largest off-diagonal cells of the confusion matrix."""
    rows = [
        {
            "true": classes[i],
            "predicted": classes[j],
            "count": int(cm[i, j]),
            "% of true class": 100 * cm[i, j] / cm[i].sum(),
        }
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]
    if not rows:
        return pd.DataFrame(columns=["true", "predicted", "count", "% of true class"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(k).reset_index(drop=True)


def plot_calibration(y_true: np.ndarray, probs: np.ndarray, title: str, n_bins: int = 15) -> plt.Figure:
    """Reliability diagram plus the confidence histogram split by correctness."""
    table = calibration_table(y_true, probs, n_bins)
    ece = expected_calibration_error(y_true, probs, n_bins)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    width = 1.0 / n_bins

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    filled = table.dropna()
    axes[0].bar(
        filled["bin_lower"], filled["accuracy"], width=width * 0.95, align="edge",
        color=TRAIN_COLOR, label="Empirical accuracy",
    )
    axes[0].plot([0, 1], [0, 1], ls="--", color=NEUTRAL_COLOR, lw=1.5, label="Perfect calibration")
    axes[0].set(
        xlim=(0, 1), ylim=(0, 1), xlabel="Predicted confidence", ylabel="Accuracy",
        title=f"Reliability diagram (ECE = {ece:.3f})",
    )
    axes[0].legend(frameon=False, loc="upper left")

    bins = np.linspace(0, 1, 26)
    axes[1].hist(confidence[correct], bins=bins, color=TRAIN_COLOR, alpha=0.75, label=f"Correct (n={correct.sum()})")
    axes[1].hist(
        confidence[~correct], bins=bins, color=VAL_COLOR, alpha=0.85, label=f"Incorrect (n={(~correct).sum()})"
    )
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Predicted confidence (max softmax)", ylabel="Images (log scale)", title="Confidence distribution"
    )
    axes[1].legend(frameon=False, loc="upper left")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_per_class_f1(per_class: pd.DataFrame, title: str) -> plt.Figure:
    """Horizontal bars of per-class precision / recall / F1, weakest class first."""
    frame = per_class.sort_values("f1")
    y = np.arange(len(frame))
    height = 0.26

    fig, ax = plt.subplots(figsize=(10, 0.62 * len(frame) + 2))
    for offset, column, color, label in [
        (-height, "precision", TRAIN_COLOR, "Precision"),
        (0.0, "recall", VAL_COLOR, "Recall"),
        (height, "f1", NEUTRAL_COLOR, "F1"),
    ]:
        ax.barh(y + offset, frame[column], height=height, color=color, label=label)

    ax.set_yticks(y, frame.index)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
