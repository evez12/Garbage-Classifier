"""Model selection, the final inference recipe and the deployable bundle.

Extraction of notebook sections 12, 13.1-13.7 and 13.14. Everything here reads
the stored logits, so it needs no GPU and never retrains.

The protocol has one rule: **every decision is made on validation only.** Which
experiment wins, whether to use test-time augmentation, the temperature and the
manual-review threshold are all chosen on the validation split. Test numbers are
recorded for transparency but never drive a choice, which is what keeps the final
test estimate unbiased.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support

from garbage_classifier.config import Config
from garbage_classifier.experiments import BASELINE_NAME
from garbage_classifier.metrics import (
    apply_temperature,
    apply_threshold,
    bootstrap_ci,
    expected_calibration_error,
    fast_macro_f1,
    fit_temperature,
    paired_bootstrap_delta,
    softmax_np,
    threshold_for_target,
)
from garbage_classifier.model import ARCH_NAME, file_md5

__all__ = [
    "Recipe",
    "SelectionResult",
    "build_comparison",
    "decide_recipe",
    "export_bundle",
    "per_class_table",
    "select_best",
]


class _MacroF1:
    """Macro-F1 bound to a fixed class count, usable as a metric callable."""

    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return fast_macro_f1(y_true, y_pred, self.k)


def build_comparison(records: dict[str, Any], y_val: np.ndarray, y_test: np.ndarray, n_classes: int) -> pd.DataFrame:
    """One row per run: cost and the headline validation / test metrics."""
    macro_f1 = _MacroF1(n_classes)
    rows = []
    for name, rec in records.items():
        rows.append(
            {
                "experiment": name,
                "description": rec["description"],
                "trainable params (M)": rec["n_trainable"] / 1e6,
                "train time (min)": rec["train_time_s"] / 60,
                "best epoch": rec["meta"]["best_epoch"],
                "val acc": float((rec["val"]["y_pred"] == y_val).mean()),
                "val macro-F1": macro_f1(y_val, rec["val"]["y_pred"]),
                "test acc": float((rec["test"]["y_pred"] == y_test).mean()),
                "test macro-F1": macro_f1(y_test, rec["test"]["y_pred"]),
            }
        )
    return pd.DataFrame(rows).set_index("experiment")


class SelectionResult:
    """Outcome of the validation-only selection step."""

    def __init__(
        self,
        best_name: str,
        runner_up_name: str,
        comparison: pd.DataFrame,
        calibration: pd.DataFrame,
        selection: pd.DataFrame,
    ) -> None:
        self.best_name = best_name
        self.runner_up_name = runner_up_name
        self.comparison = comparison
        self.calibration = calibration
        self.selection = selection

    @property
    def tied(self) -> list[str]:
        return list(self.selection.index[self.selection["tied with top"]])


def select_best(
    records: dict[str, Any],
    y_val: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    config: Config,
) -> SelectionResult:
    """Pick the winning run on validation macro-F1, with a tie check and a proper tie-break.

    Ranking by a point estimate alone would pick winners by noise, so every run
    is compared to the top one with a *paired* bootstrap. Runs whose confidence
    interval for the difference includes zero are statistically tied with the
    top, and among those the one with the lowest temperature-scaled validation
    NLL wins — a proper scoring rule, so the tie-break rewards honest
    probabilities rather than a luckier argmax.
    """
    macro_f1 = _MacroF1(n_classes)
    labels = np.arange(n_classes)

    # Temperature per run, fitted on validation.
    calib_rows = []
    for name, rec in records.items():
        t = fit_temperature(rec["val"]["logits"], y_val)
        rec["temperature"] = t
        for split, y in (("val", y_val), ("test", y_test)):
            rec[split]["probs_ts"] = apply_temperature(rec[split]["logits"], t)
        calib_rows.append(
            {
                "experiment": name,
                "T (fit on val)": t,
                "val NLL raw": log_loss(y_val, rec["val"]["probs"], labels=labels),
                "val NLL scaled": log_loss(y_val, rec["val"]["probs_ts"], labels=labels),
                "test ECE raw": expected_calibration_error(y_test, rec["test"]["probs"], config.selection.calibration_bins),
                "test ECE scaled": expected_calibration_error(y_test, rec["test"]["probs_ts"], config.selection.calibration_bins),
                "test log-loss raw": log_loss(y_test, rec["test"]["probs"], labels=labels),
                "test log-loss scaled": log_loss(y_test, rec["test"]["probs_ts"], labels=labels),
            }
        )
    calibration = pd.DataFrame(calib_rows).set_index("experiment")

    comparison = build_comparison(records, y_val, y_test, n_classes)
    ranked = comparison["val macro-F1"].sort_values(ascending=False)
    top_name = ranked.index[0]

    sel_rows = []
    for name in ranked.index:
        if name == top_name:
            delta, lo, hi, p_le0, tied = 0.0, 0.0, 0.0, float("nan"), True
        else:  # delta = F1(top) - F1(name)
            delta, lo, hi, p_le0 = paired_bootstrap_delta(
                y_val,
                records[name]["val"]["y_pred"],
                records[top_name]["val"]["y_pred"],
                macro_f1,
                n_boot=config.selection.bootstrap_resamples,
                seed=config.seed,
            )
            tied = lo <= 0.0
        sel_rows.append(
            {
                "experiment": name,
                "val macro-F1": ranked[name],
                "delta to top (pp)": 100 * delta,
                "delta CI low (pp)": 100 * lo,
                "delta CI high (pp)": 100 * hi,
                "P(delta<=0)": p_le0,
                "tied with top": tied,
                "val NLL (T-scaled)": calibration.loc[name, "val NLL scaled"],
                "trainable params (M)": records[name]["n_trainable"] / 1e6,
            }
        )
    selection = pd.DataFrame(sel_rows).set_index("experiment")

    candidates = selection[selection["tied with top"]]
    best_name = str(candidates["val NLL (T-scaled)"].idxmin())
    runner_up_name = str(comparison["val macro-F1"].drop(best_name).idxmax())

    if len(candidates) == 1:
        print(f"Selected: {best_name} — its val macro-F1 lead over every other run is statistically supported.")
    else:
        print(f"Statistically tied on val macro-F1: {list(candidates.index)}")
        print(
            f"Tie broken by temperature-scaled val NLL -> selected: {best_name} "
            f"(NLL {candidates.loc[best_name, 'val NLL (T-scaled)']:.4f})"
        )
    print(f"Runner-up (next best val macro-F1): {runner_up_name}")
    return SelectionResult(best_name, runner_up_name, comparison, calibration, selection)


class Recipe:
    """The deployable inference recipe, all of it decided on validation."""

    def __init__(self, use_tta: bool, temperature: float, review_threshold: float, review_target: float) -> None:
        self.use_tta = use_tta
        self.temperature = temperature
        self.review_threshold = review_threshold
        self.review_target = review_target

    def as_dict(self) -> dict[str, Any]:
        return {
            "tta_hflip": bool(self.use_tta),
            "temperature": float(self.temperature),
            "review_threshold": float(self.review_threshold) if np.isfinite(self.review_threshold) else None,
            "review_target": self.review_target,
        }


def final_logits(record: dict[str, Any], split: str, tta: bool) -> np.ndarray:
    """Logits for *split*, averaged with the mirrored pass when TTA is on."""
    s = record[split]
    return 0.5 * (s["logits"] + s["logits_hflip"]) if tta else s["logits"]


def decide_recipe(
    best_rec: dict[str, Any],
    y_val: np.ndarray,
    n_classes: int,
    config: Config,
) -> tuple[Recipe, dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """Decide TTA, fit the temperature and choose the manual-review threshold.

    Horizontal-flip TTA doubles inference cost, so it is adopted only when its
    validation macro-F1 gain is statistically supported — a gain of one or two
    images is noise and does not justify the cost. The temperature is then
    refitted on the final (TTA or plain) validation logits.
    """
    macro_f1 = _MacroF1(n_classes)
    tta_delta, tta_lo, tta_hi, tta_p = paired_bootstrap_delta(
        y_val,
        best_rec["val"]["y_pred"],
        final_logits(best_rec, "val", True).argmax(axis=1),
        macro_f1,
        n_boot=config.selection.bootstrap_resamples,
        seed=config.seed,
    )
    use_tta = bool(tta_p < config.selection.tta_significance)
    print(
        f"hflip TTA on val: delta macro-F1 {100 * tta_delta:+.2f} pp, "
        f"95% CI [{100 * tta_lo:+.2f}, {100 * tta_hi:+.2f}], P(delta <= 0) = {tta_p:.3f}  ->  use TTA: {use_tta}"
    )

    temperature = fit_temperature(final_logits(best_rec, "val", use_tta), y_val)
    print(
        f"Temperature (fit on val{' + TTA' if use_tta else ''}): T = {temperature:.4f} "
        f"({'under' if temperature < 1 else 'over'}-confident raw model)"
    )

    final: dict[str, dict[str, np.ndarray]] = {}
    for split in ("val", "test"):
        logits = final_logits(best_rec, split, use_tta)
        final[split] = {
            "y_true": best_rec[split]["y_true"],
            "logits": logits,
            "y_pred": logits.argmax(axis=1),
            "probs_raw": softmax_np(logits),
            "probs": apply_temperature(logits, temperature),
        }

    threshold = threshold_for_target(y_val, final["val"]["probs"], config.selection.review_target)
    recipe = Recipe(use_tta, temperature, threshold, config.selection.review_target)
    tta_stats = {"delta": tta_delta, "ci_low": tta_lo, "ci_high": tta_hi, "p_le0": tta_p}
    return recipe, final, tta_stats


def selective_table(
    final: dict[str, dict[str, np.ndarray]],
    y_val: np.ndarray,
    y_test: np.ndarray,
    targets: tuple[float, ...],
    model_name: str,
) -> pd.DataFrame:
    """Thresholds chosen on validation for each target accuracy, evaluated on test."""
    rows = []
    for target in targets:
        t = threshold_for_target(y_val, final["val"]["probs"], target)
        cov, acc, n_review = (
            apply_threshold(y_test, final["test"]["probs"], t) if np.isfinite(t) else (np.nan, np.nan, -1)
        )
        rows.append(
            {
                "model": model_name,
                "target accuracy": target,
                "threshold (val)": t,
                "test coverage": cov,
                "test accuracy (covered)": acc,
                "test images to review": n_review,
            }
        )
    return pd.DataFrame(rows)


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Per-class precision / recall / F1 with support, weakest class first."""
    labels = np.arange(len(classes))
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": support},
        index=pd.Index(classes, name="class"),
    ).sort_values("f1")


def export_bundle(
    config: Config,
    best_rec: dict[str, Any],
    recipe: Recipe,
    classes: list[str],
    test_metrics: dict[str, float],
) -> Path:
    """Write ``checkpoints/best_model_bundle.pt`` — everything inference needs, nothing more.

    Architecture name, weights, class order, the exact preprocessing parameters,
    the fitted temperature, the TTA decision and the review threshold. A
    deployment loads only this file: no training code, no config, no dataset.
    """
    import torch
    from garbage_classifier.data import eval_transform

    eval_tf = eval_transform()
    state = torch.load(best_rec["ckpt"], map_location="cpu", weights_only=False)

    config.bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": ARCH_NAME,
            "model": state["model"],
            "classes": classes,
            "preprocess": {
                "resize": list(eval_tf.resize_size),
                "crop": list(eval_tf.crop_size),
                "interpolation": eval_tf.interpolation.value,
                "mean": list(eval_tf.mean),
                "std": list(eval_tf.std),
            },
            "temperature": float(recipe.temperature),
            "tta_hflip": bool(recipe.use_tta),
            "review_threshold": recipe.as_dict()["review_threshold"],
            "source_run": best_rec["name"],
            "source_checkpoint_md5": file_md5(best_rec["ckpt"]),
            "test_metrics": test_metrics,
        },
        config.bundle_path,
    )
    return config.bundle_path


def bootstrap_metric_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    config: Config,
) -> dict[str, dict[str, float]]:
    """Accuracy and macro-F1 with percentile bootstrap confidence intervals."""
    macro_f1 = _MacroF1(n_classes)
    out: dict[str, dict[str, float]] = {}
    for key, fn in (
        ("accuracy", lambda t, p: float((t == p).mean())),
        ("macro_f1", macro_f1),
    ):
        point, lo, hi = bootstrap_ci(
            y_true, y_pred, fn, n_boot=config.selection.bootstrap_resamples, seed=config.seed
        )
        out[key] = {"value": point, "ci_low": lo, "ci_high": hi}
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion matrix scoring every class, even one absent from *y_true*."""
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))


def _json_safe(obj: Any) -> Any:
    """Replace NaN / infinity with null so the result is strict, parseable JSON.

    Python's ``json`` happily writes the bare tokens ``NaN`` and ``Infinity``,
    which every browser's ``JSON.parse`` rejects — and this file is read by the
    demo page. Unreachable accuracy targets legitimately produce NaN, so they
    are converted rather than treated as a bug.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write *payload* as indented, strictly valid JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), indent=2, allow_nan=False, default=str)
    path.write_text(text, encoding="utf-8")
    return path
