"""Metric, calibration and statistical-testing primitives.

Extraction of the shared evaluation helpers of notebook sections 10, 12 and 13.
Everything here works on stored NumPy logits, so the whole post-training
analysis runs without a GPU and without rebuilding the network.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

MetricFn = Callable[[np.ndarray, np.ndarray], float]

__all__ = [
    "apply_temperature",
    "apply_threshold",
    "bootstrap_ci",
    "brier_score",
    "calibration_table",
    "compute_metrics",
    "expected_calibration_error",
    "fast_macro_f1",
    "fit_temperature",
    "holm_adjust",
    "mcnemar_exact",
    "paired_bootstrap_delta",
    "risk_coverage",
    "softmax_np",
    "threshold_for_target",
]


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted by the row max for numerical stability."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Macro-F1 over all *k* classes via a bincount confusion matrix.

    Equivalent to ``sklearn.metrics.f1_score(average="macro", zero_division=0)``
    but roughly two orders of magnitude faster, which matters inside a
    2,000-resample bootstrap.
    """
    cm = np.bincount(k * y_true + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0).mean())


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_table(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Bin predictions by confidence (max softmax) and compare to empirical accuracy."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        rows.append(
            {
                "bin_lower": edges[b],
                "bin_upper": edges[b + 1],
                "count": int(mask.sum()),
                "confidence": confidence[mask].mean() if mask.any() else np.nan,
                "accuracy": correct[mask].mean() if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: bin-count-weighted |accuracy - confidence|."""
    table = calibration_table(y_true, probs, n_bins).dropna()
    weights = table["count"] / table["count"].sum()
    return float((weights * (table["accuracy"] - table["confidence"]).abs()).sum())


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """The single scalar ``T`` minimising the NLL of ``softmax(logits / T)``.

    Temperature scaling rescales confidences without touching the argmax, so
    accuracy and macro-F1 are unchanged while the probabilities become usable
    as confidences. Fitted on validation only.
    """
    rows = np.arange(len(y_true))

    def nll(t: float) -> float:
        return float(-_log_softmax(logits / t)[rows, y_true].mean())

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    """Temperature-scaled probabilities."""
    return softmax_np(logits / t)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score — mean squared error of the probability vector (lower is better)."""
    onehot = np.eye(probs.shape[1])[y_true]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Global metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Global classification metrics for one split."""
    labels = np.arange(probs.shape[1])
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        metrics.update({f"{avg}_precision": p, f"{avg}_recall": r, f"{avg}_f1": f})

    metrics["top3_accuracy"] = top_k_accuracy_score(y_true, probs, k=3, labels=labels)
    metrics["log_loss"] = log_loss(y_true, probs, labels=labels)
    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)
    except ValueError:  # a class is missing from y_true -> AUC undefined
        metrics["roc_auc_ovr_macro"] = float("nan")
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["brier"] = brier_score(y_true, probs)
    metrics["ece"] = expected_calibration_error(y_true, probs)
    return {k: float(v) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Uncertainty and significance
# ---------------------------------------------------------------------------
def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: MetricFn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 11,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI for ``metric_fn(y_true, y_pred)``."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(metric_fn(y_true, y_pred)), float(lo), float(hi)


def paired_bootstrap_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: MetricFn,
    n_boot: int = 2000,
    seed: int = 11,
) -> tuple[float, float, float, float]:
    """``delta = metric(b) - metric(a)``: estimate, paired 95% CI and ``P(delta <= 0)``.

    Both models are scored on the *same* resample, so per-image difficulty noise
    cancels and the interval around the difference is far tighter than two
    independent confidence intervals would suggest.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = metric_fn(y_true[idx], pred_b[idx]) - metric_fn(y_true[idx], pred_a[idx])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    point = float(metric_fn(y_true, pred_b) - metric_fn(y_true, pred_a))
    return point, float(lo), float(hi), float((deltas <= 0).mean())


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[int, int, float]:
    """``(#fixed, #broken, two-sided exact p)`` for model *b* against model *a*.

    Only the images the two models disagree on carry information, so the test is
    an exact binomial test on those discordant pairs.
    """
    ok_a, ok_b = pred_a == y_true, pred_b == y_true
    fixed, broken = int((~ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())
    p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken > 0 else 1.0
    return fixed, broken, float(p)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values for a family of tests."""
    m = len(p_values)
    adjusted, running = np.empty(m), 0.0
    for rank, i in enumerate(np.argsort(p_values)):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------
def risk_coverage(y_true: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coverage, accuracy on the covered part and the threshold, by decreasing confidence."""
    conf = probs.max(axis=1)
    order = np.argsort(-conf, kind="stable")
    correct = (probs.argmax(axis=1) == y_true)[order]
    n_cov = np.arange(1, len(y_true) + 1)
    return n_cov / len(y_true), np.cumsum(correct) / n_cov, conf[order]


def threshold_for_target(y_true: np.ndarray, probs: np.ndarray, target: float) -> float:
    """Lowest confidence threshold whose covered subset still reaches *target* accuracy.

    Lowest, therefore maximising coverage — the fewest images sent to manual
    review for a given accuracy guarantee. ``nan`` when the target is
    unreachable at any threshold.
    """
    _, acc, conf = risk_coverage(y_true, probs)
    ok = np.nonzero(acc >= target)[0]
    return float(conf[ok.max()]) if len(ok) else float("nan")


def apply_threshold(y_true: np.ndarray, probs: np.ndarray, t: float) -> tuple[float, float, int]:
    """``(coverage, accuracy on covered images, number sent to review)`` at threshold *t*."""
    keep = probs.max(axis=1) >= t
    acc = float((probs.argmax(axis=1)[keep] == y_true[keep]).mean()) if keep.any() else float("nan")
    return float(keep.mean()), acc, int((~keep).sum())
