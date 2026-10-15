#!/usr/bin/env python
"""Select the best run, fix the inference recipe, export the bundle and the reports.

Runs entirely from the artifacts in ``experiments/`` — no GPU, no retraining,
a few seconds. Mirrors notebook sections 12, 13.1-13.7 and 13.14.

Every decision is taken on **validation**: which experiment wins (paired
bootstrap tie check, temperature-scaled NLL tie-break), whether to use
horizontal-flip TTA, the temperature, and the confidence threshold for manual
review. The test split is scored once, for reporting only.

Writes
------
``checkpoints/best_model_bundle.pt``   the single deployable artifact
``reports/comparison/``                comparison, selection and calibration tables
``reports/best_model/``                metrics, per-class and per-image tables, figures
``docs/metrics.json`` + ``docs/figures/``  what the GitHub Pages demo reads

Usage::

    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import _bootstrap  # noqa: F401

from garbage_classifier.config import Config, load_config
from garbage_classifier.data import load_splits, to_relative
from garbage_classifier.experiments import BASELINE_NAME, build_configs, is_complete, load_record
from garbage_classifier.metrics import compute_metrics, fast_macro_f1
from garbage_classifier.selection import (
    bootstrap_metric_table,
    confusion,
    decide_recipe,
    export_bundle,
    per_class_table,
    select_best,
    selective_table,
    write_json,
)
from garbage_classifier.viz import (
    plot_calibration,
    plot_confusion_matrices,
    plot_per_class_f1,
    plot_training_history,
    save_figure,
    set_theme,
    top_confusions,
)

PROJECT_ROOT = _bootstrap.PROJECT_ROOT

#: Figures the demo site embeds, mapped from their name in reports/best_model/
#: to the name docs/figures/ serves them under. Copied because reports/ is
#: gitignored while docs/ is published.
DEMO_FIGURES = {
    "confusion_matrix_test": "confusion_matrix",
    "per_class_metrics": "per_class_metrics",
    "calibration": "calibration",
    "training_history": "training_history",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (default: repo root).")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset directory (for resolving image paths).")
    parser.add_argument("--device", default=None, help="Unused here; accepted so train.py can forward its flags.")
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Do not refresh docs/metrics.json and docs/figures/ (the demo site's inputs).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config: Config = load_config(args.config, root=PROJECT_ROOT)
    if args.data_root is not None:
        root = args.data_root.expanduser()
        root = root if root.is_absolute() else (config.root / root)
        config = config.with_overrides(data=dataclasses.replace(config.data, root=root))

    set_theme()
    splits, classes = load_splits(config)
    n_classes = len(classes)
    y_val = splits["val"]["y"].to_numpy()
    y_test = splits["test"]["y"].to_numpy()

    configs = build_configs(config)
    available = [name for name, cfg in configs.items() if is_complete(config, cfg)]
    if BASELINE_NAME not in available:
        raise SystemExit(
            "The linear-probe baseline has no completed run. Train it first:\n"
            "    python scripts/train.py --model all"
        )
    missing = [n for n in configs if n not in available]
    if missing:
        print(f"Note: comparing {len(available)} of {len(configs)} runs — not yet trained: {missing}")

    records = {name: load_record(config, name, classes) for name in available}
    for rec in records.values():  # identical, identically ordered eval sets -> paired tests are valid
        assert np.array_equal(rec["val"]["y_true"], y_val) and np.array_equal(rec["test"]["y_true"], y_test), (
            f"[{rec['name']}] stored labels do not match splits/*.csv — rebuild the splits and retrain."
        )
    print(f"Loaded {len(records)} run(s): {list(records)}\n")

    # --- Selection ---------------------------------------------------------
    result = select_best(records, y_val, y_test, n_classes, config)
    best_rec = records[result.best_name]

    compare_dir = config.reports_dir / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(compare_dir / "comparison.csv")
    result.selection.to_csv(compare_dir / "selection.csv")
    result.calibration.to_csv(compare_dir / "calibration.csv")

    # --- Final inference recipe -------------------------------------------
    print()
    recipe, final, tta_stats = decide_recipe(best_rec, y_val, n_classes, config)

    # --- Test-set reporting (scored once) ---------------------------------
    eval_dir = config.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_pred = final["test"]["y_pred"]
    test_probs = final["test"]["probs"]
    headline = bootstrap_metric_table(y_test, test_pred, n_classes, config)
    full_metrics = compute_metrics(y_test, test_pred, test_probs)
    per_class = per_class_table(y_test, test_pred, classes)
    cm = confusion(y_test, test_pred, n_classes)
    selective = selective_table(final, y_val, y_test, config.selection.selective_targets, result.best_name)

    print(
        f"\nTest — accuracy {headline['accuracy']['value']:.4f} "
        f"[{headline['accuracy']['ci_low']:.4f}, {headline['accuracy']['ci_high']:.4f}], "
        f"macro-F1 {headline['macro_f1']['value']:.4f} "
        f"[{headline['macro_f1']['ci_low']:.4f}, {headline['macro_f1']['ci_high']:.4f}]"
    )
    print(f"Weakest class: '{per_class.index[0]}' (F1 {per_class['f1'].iloc[0]:.4f})")
    print(f"Misclassified: {int((test_pred != y_test).sum())} / {len(y_test)}")

    # Per-image predictions with the full calibrated probability vector.
    predictions = splits["test"][["path", "label"]].reset_index(drop=True).rename(columns={"label": "true"})
    predictions["pred"] = [classes[i] for i in test_pred]
    predictions["confidence"] = test_probs[np.arange(len(test_probs)), test_pred]
    predictions["p_true"] = test_probs[np.arange(len(test_probs)), y_test]
    predictions["correct"] = predictions["true"] == predictions["pred"]
    predictions = pd.concat(
        [predictions, pd.DataFrame(test_probs, columns=[f"p_{c}" for c in classes])], axis=1
    )
    out = to_relative(predictions, config.data.root)
    out.to_csv(eval_dir / "predictions_test.csv", index=False)
    out.loc[~out["correct"]].sort_values("confidence", ascending=False).to_csv(
        eval_dir / "misclassified_test.csv", index=False
    )
    per_class.to_csv(eval_dir / "per_class_test.csv")
    selective.to_csv(eval_dir / "selective_prediction.csv", index=False)
    top_confusions(cm, classes, k=10).to_csv(eval_dir / "top_confusions.csv", index=False)

    # --- Figures -----------------------------------------------------------
    save_figure(
        plot_confusion_matrices(cm, classes, f"Confusion matrix — test set — {result.best_name}"),
        eval_dir, "confusion_matrix_test",
    )
    save_figure(
        plot_per_class_f1(per_class, f"Per-class test metrics — {result.best_name}"),
        eval_dir, "per_class_metrics",
    )
    save_figure(
        plot_calibration(y_test, test_probs, f"Calibration — test set — {result.best_name} (T = {recipe.temperature:.3f})"),
        eval_dir, "calibration",
    )
    save_figure(plot_training_history(best_rec["history"]), eval_dir, "training_history")

    # --- Deployable bundle -------------------------------------------------
    bundle_path = export_bundle(
        config, best_rec, recipe, classes,
        {k: v["value"] for k, v in headline.items()},
    )
    print(f"\nBundle: {bundle_path.relative_to(config.root)}")

    # --- metrics.json ------------------------------------------------------
    review = recipe.as_dict()["review_threshold"]
    review_row = selective[selective["target accuracy"] == config.selection.review_target]
    summary = {
        "selected_model": result.best_name,
        "selected_description": best_rec["description"],
        "runner_up": result.runner_up_name,
        "statistically_tied": result.tied,
        "classes": classes,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "recipe": recipe.as_dict(),
        "tta_decision": tta_stats,
        "headline_test": headline,
        "full_test_metrics": full_metrics,
        "per_class_test": per_class.reset_index().to_dict(orient="records"),
        "confusion_matrix_test": cm.tolist(),
        "n_misclassified_test": int((test_pred != y_test).sum()),
        "comparison": result.comparison.reset_index().to_dict(orient="records"),
        "selection": result.selection.reset_index().to_dict(orient="records"),
        "calibration": result.calibration.reset_index().to_dict(orient="records"),
        "selective_prediction": selective.to_dict(orient="records"),
        "review_threshold": review,
        "review_coverage": (
            float(review_row["test coverage"].iloc[0]) if len(review_row) else None
        ),
        "training": {
            name: {
                "trainable_params": rec["n_trainable"],
                "train_time_min": rec["train_time_s"] / 60,
                "best_epoch": rec["meta"]["best_epoch"],
                "epochs_run": rec["meta"]["epochs_run"],
                "device": rec["meta"].get("device"),
                "torch": rec["meta"].get("torch"),
            }
            for name, rec in records.items()
        },
    }
    write_json(eval_dir / "metrics.json", summary)
    print(f"Reports: {eval_dir.relative_to(config.root)} and {compare_dir.relative_to(config.root)}")

    # --- Demo site inputs --------------------------------------------------
    if not args.no_docs:
        docs = config.root / "docs"
        page_payload = build_page_payload(
            config, result, best_rec, records, recipe, classes, splits,
            headline, per_class, cm, selective, review,
        )
        merge_page_metrics(docs / "data" / "metrics.json", page_payload)

        figures_dir = docs / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        for source_name, published_name in DEMO_FIGURES.items():
            src = eval_dir / f"{source_name}.png"
            if src.exists():
                shutil.copy2(src, figures_dir / f"{published_name}.png")
        print("Demo site inputs refreshed: docs/data/metrics.json, docs/figures/")
        print("Next: python scripts/export_onnx.py   (builds the model the browser demo runs)")

    return 0


def build_page_payload(
    config: Config,
    result,
    best_rec: dict,
    records: dict,
    recipe,
    classes: list[str],
    splits: dict[str, pd.DataFrame],
    headline: dict,
    per_class: pd.DataFrame,
    cm: np.ndarray,
    selective: pd.DataFrame,
    review: float | None,
) -> dict:
    """Shape the results the way docs/assets/app.js reads them.

    Deliberately a different shape from ``reports/best_model/metrics.json``:
    that file is the complete record, this one is only what the page renders.
    """
    review_row = selective[selective["target accuracy"] == config.selection.review_target]
    baseline = records.get(BASELINE_NAME)

    def with_baseline(key: str, baseline_value: float) -> dict:
        m = headline[key]
        return {"value": m["value"], "ci": [m["ci_low"], m["ci_high"]], "baseline": baseline_value}

    y_test = splits["test"]["y"].to_numpy()
    base_pred = baseline["test"]["y_pred"] if baseline else None
    base_acc = float((base_pred == y_test).mean()) if base_pred is not None else float("nan")
    base_f1 = fast_macro_f1(y_test, base_pred, len(classes)) if base_pred is not None else float("nan")

    base_class_f1: dict[str, float] = {}
    if base_pred is not None:
        _, _, f_base, _ = precision_recall_fscore_support(
            y_test, base_pred, labels=np.arange(len(classes)), average=None, zero_division=0
        )
        base_class_f1 = dict(zip(classes, (float(v) for v in f_base)))

    return {
        "selected_model": result.best_name,
        "selected_description": best_rec["description"],
        "runner_up": result.runner_up_name,
        "classes": classes,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "recipe": recipe.as_dict(),
        "headline": {
            "accuracy": with_baseline("accuracy", base_acc),
            "macro_f1": with_baseline("macro_f1", base_f1),
        },
        "per_class": [
            {
                "class": name,
                "support": int(row["support"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "missed": int(row["support"] - cm[classes.index(name), classes.index(name)]),
                "false_alarms": int(cm[:, classes.index(name)].sum() - cm[classes.index(name), classes.index(name)]),
                "delta_f1_pp": (
                    100 * (float(row["f1"]) - base_class_f1[name]) if name in base_class_f1 else None
                ),
            }
            for name, row in per_class.iterrows()
        ],
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_test)),
        "n_misclassified": int(cm.sum() - np.trace(cm)),
        "experiments": [
            {
                "name": name,
                "description": rec["description"],
                "trainable_m": rec["n_trainable"] / 1e6,
                "epochs_run": int(rec["meta"]["epochs_run"]),
                "best_epoch": int(rec["meta"]["best_epoch"]),
                "train_time_min": (
                    None if not np.isfinite(rec["train_time_s"]) else rec["train_time_s"] / 60
                ),
                "val_acc": float(rec["meta"]["metrics"]["val_acc"]),
                "val_macro_f1": float(rec["meta"]["metrics"]["val_macro_f1"]),
                "test_acc": float(rec["meta"]["metrics"]["test_acc"]),
                "test_macro_f1": float(rec["meta"]["metrics"]["test_macro_f1"]),
            }
            for name, rec in records.items()
        ],
        "selective_prediction": selective.to_dict(orient="records"),
        "review_threshold": review,
        "review_coverage": float(review_row["test coverage"].iloc[0]) if len(review_row) else None,
    }


def merge_page_metrics(path: Path, payload: dict) -> Path:
    """Write the page payload, keeping sections only the notebook can produce.

    Robustness, latency, Grad-CAM-era efficiency figures and the embedding
    analysis come from notebook sections the scripts do not reimplement. A
    retrain refreshes everything it computed and leaves those untouched rather
    than deleting them from the page.
    """
    existing: dict = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    merged = {**existing, **payload}
    # headline carries extra metrics on the published file; keep any we did not
    # recompute rather than dropping them.
    if isinstance(existing.get("headline"), dict):
        merged["headline"] = {**existing["headline"], **payload["headline"]}
    return write_json(path, merged)


if __name__ == "__main__":
    raise SystemExit(main())
