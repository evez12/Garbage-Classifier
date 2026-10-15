#!/usr/bin/env python
"""Classify images with the exported bundle.

Loads only ``checkpoints/best_model_bundle.pt`` — weights, class order,
preprocessing, temperature, TTA flag and review threshold all come from that one
file, so this script is a realistic stand-in for a deployment.

Examples
--------
::

    python scripts/predict.py photo.jpg
    python scripts/predict.py a.jpg b.png --top-k 5
    python scripts/predict.py "images/*.jpg" --json predictions.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from garbage_classifier.config import load_config
from garbage_classifier.inference import load_bundle, predict_image
from garbage_classifier.model import resolve_device

PROJECT_ROOT = _bootstrap.PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="Image files or glob patterns.")
    parser.add_argument("--bundle", type=Path, default=None, help="Path to best_model_bundle.pt.")
    parser.add_argument("--top-k", type=int, default=3, help="How many classes to show per image (default: 3).")
    parser.add_argument("--device", default=None, help="Force a device, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--json", type=Path, default=None, help="Also write the predictions to this JSON file.")
    return parser


def expand(patterns: list[str]) -> list[Path]:
    """Expand glob patterns, keeping literal paths that exist as given."""
    paths: list[Path] = []
    for pattern in patterns:
        if Path(pattern).exists():
            paths.append(Path(pattern))
            continue
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise SystemExit(f"No such file or pattern: {pattern}")
        paths.extend(Path(m) for m in matched)
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(root=PROJECT_ROOT)
    bundle_path = args.bundle or config.bundle_path
    if not bundle_path.exists():
        raise SystemExit(
            f"Bundle not found: {bundle_path}\n"
            "Train the model first (python scripts/train.py --model all), or download the release asset "
            "listed in README.md -> Pretrained weights."
        )

    device = resolve_device(args.device)
    net, bundle = load_bundle(bundle_path, device)
    threshold = bundle.get("review_threshold")
    print(
        f"Model: {bundle['source_run']} on {device} | TTA {'on' if bundle['tta_hflip'] else 'off'} | "
        f"T = {bundle['temperature']:.3f} | review below {threshold if threshold is None else round(threshold, 3)}"
    )

    results = []
    for path in expand(args.images):
        pred = predict_image(path, net, bundle, top_k=args.top_k, device=device)
        flag = "  -> manual review" if pred.needs_review else ""
        ranked = ", ".join(f"{c} {p:.3f}" for c, p in pred.top_k)
        print(f"{path.name}: {pred.label} ({pred.confidence:.3f}){flag}\n    {ranked}")
        results.append({"image": str(path), **pred.as_dict()})

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
