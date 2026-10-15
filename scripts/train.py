#!/usr/bin/env python
"""Train the garbage classifier — the notebook's pipeline, from the command line.

The two stages are the same as in ``notebooks/model_training_notebook.ipynb``:

1. **Linear probe** — freeze the ImageNet backbone, fit a new 10-class head.
   Every fine-tuning run starts from these weights (the LP-FT recipe), so this
   stage runs first whether or not you asked for it.
2. **Fine-tuning** — four experiments that unfreeze progressively more of the
   backbone with discriminative learning rates.

Examples
--------
Train everything, then select, calibrate and export the deployable bundle::

    python scripts/train.py --model all

Train a single experiment (the linear probe is reused from its checkpoint)::

    python scripts/train.py --model ft_block34

Point it at your own copy of the dataset, on a smaller GPU::

    python scripts/train.py --model all --data-root /data/garbage --batch-size 32

Completed runs are cached: re-running is a no-op unless a config changed or you
pass ``--force-retrain``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (side effect: makes `garbage_classifier` importable)

from garbage_classifier.config import Config, load_config
from garbage_classifier.data import build_splits, load_splits, seed_everything
from garbage_classifier.experiments import (
    BASELINE_NAME,
    build_configs,
    is_complete,
    make_run_context,
    register_linear_probe,
    run_experiment,
    train_linear_probe,
)
from garbage_classifier.model import resolve_device

PROJECT_ROOT = _bootstrap.PROJECT_ROOT


def build_parser(experiment_names: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        nargs="+",
        default=["all"],
        metavar="NAME",
        help=f"Which run(s) to train: 'all' or any of {experiment_names}. Several names may be given.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (default: repo root).")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset directory, one subfolder per class.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the batch size (lower on GPU OOM).")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (use 0 to debug).")
    parser.add_argument("--seed", type=int, default=None, help="Override the global seed.")
    parser.add_argument("--device", default=None, help="Force a device, e.g. 'cuda', 'cuda:1' or 'cpu'.")
    parser.add_argument(
        "--amp",
        choices=["bf16", "fp16", "off"],
        default="bf16",
        help="Mixed-precision mode for training (default: bf16, which needs no loss scaling).",
    )
    parser.add_argument(
        "--rebuild-splits",
        action="store_true",
        help="Re-audit the dataset and rebuild splits/*.csv instead of reusing them.",
    )
    parser.add_argument(
        "--retrain-linear-probe",
        action="store_true",
        help="Retrain stage 1 even if its checkpoint exists. This invalidates every fine-tuning run.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain the selected run(s) even when their cached artifacts are up to date.",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip the selection / calibration / export step that normally follows training.",
    )
    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Fold the CLI flags into the loaded configuration."""
    updates: dict[str, object] = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.data_root is not None:
        root = args.data_root.expanduser()
        root = root if root.is_absolute() else (config.root / root)
        updates["data"] = dataclasses.replace(config.data, root=root)
    loader_updates = {}
    if args.batch_size is not None:
        loader_updates["batch_size"] = args.batch_size
    if args.num_workers is not None:
        loader_updates["num_workers"] = args.num_workers
    if loader_updates:
        updates["loader"] = dataclasses.replace(config.loader, **loader_updates)
    return config.with_overrides(**updates) if updates else config


def resolve_requested(names: list[str], all_names: list[str]) -> list[str]:
    """Expand ``all`` and validate the requested run names, preserving the canonical order."""
    if any(n == "all" for n in names):
        return list(all_names)
    unknown = [n for n in names if n not in all_names]
    if unknown:
        raise SystemExit(f"Unknown model name(s) {unknown}. Choose from: {['all', *all_names]}")
    return [n for n in all_names if n in set(names)]


def main(argv: list[str] | None = None) -> int:
    base_config = load_config(root=PROJECT_ROOT)
    all_names = list(build_configs(base_config))

    args = build_parser(all_names).parse_args(argv)
    config = apply_overrides(base_config, args)
    requested = resolve_requested(args.model, all_names)

    if not config.data.root.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {config.data.root}\n"
            "Download Garbage Classification v2 and lay it out as described in README.md -> Dataset, "
            "or point --data-root at your copy."
        )

    device = resolve_device(args.device)
    print(f"Device: {device}  |  dataset: {config.data.root}  |  seed: {config.seed}")
    seed_everything(config.seed)

    # --- Splits ------------------------------------------------------------
    if args.rebuild_splits or not (config.splits_dir / "train.csv").exists():
        print("Auditing the dataset and building leakage-free splits ...")
        splits, classes, _ = build_splits(config)
    else:
        splits, classes = load_splits(config)
        print(f"Reusing splits from {config.splits_dir.relative_to(config.root)}")
    print(
        f"Classes ({len(classes)}): {', '.join(classes)}\n"
        f"Images — train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}"
    )

    configs = build_configs(config)
    force = set(requested) if args.force_retrain else set()
    ctx = make_run_context(
        config, classes, splits, device,
        amp=args.amp, batch_size_ft=args.batch_size, force_retrain=force,
    )

    # --- Stage 1 -----------------------------------------------------------
    # Needed either way: it is both the baseline run and the initialisation of
    # every fine-tuning run.
    print("\n=== Stage 1 — linear probe ===")
    lp_history = train_linear_probe(ctx, retrain=args.retrain_linear_probe)
    register_linear_probe(ctx, lp_history, all_names)

    # --- Stage 2 -----------------------------------------------------------
    finetune_requested = [n for n in requested if n != BASELINE_NAME]
    if finetune_requested:
        print("\n=== Stage 2 — fine-tuning ===")
    for name in finetune_requested:
        print()
        run_experiment(ctx, configs[name], all_names)

    # --- Selection, calibration, export ------------------------------------
    if args.no_evaluate:
        print("\nSkipping evaluation (--no-evaluate).")
        return 0

    incomplete = [n for n, cfg in configs.items() if not is_complete(config, cfg)]
    if incomplete:
        print(
            f"\nSkipping evaluation: {incomplete} not trained yet. "
            f"Run 'python scripts/train.py --model all' to complete them, "
            f"or 'python scripts/evaluate.py' to compare only what exists."
        )
        return 0

    print("\n=== Selection, calibration and export ===")
    import evaluate as evaluate_script  # noqa: PLC0415  (deferred: only needed on this path)

    return evaluate_script.main(
        [
            *(["--config", str(args.config)] if args.config else []),
            *(["--data-root", str(args.data_root)] if args.data_root else []),
            *(["--device", args.device] if args.device else []),
        ]
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
