#!/usr/bin/env python
"""Export the trained bundle to ONNX for in-browser inference.

The GitHub Pages demo runs the model client-side with onnxruntime-web, so the
network is exported as plain fp32 ONNX taking an already-normalised
``(N, 3, 224, 224)`` tensor and returning raw logits. Preprocessing (resize,
centre crop, normalise), the temperature division and the review threshold all
live in ``docs/model/model_meta.json`` and are applied by the page's JavaScript,
which keeps the graph itself simple and portable.

fp32 is deliberate: it reproduces the PyTorch numbers exactly, so the accuracy
quoted on the page is the accuracy the page delivers.

The weights are written **inside** the ``.onnx`` file rather than to a sidecar
``.onnx.data``, so the browser fetches exactly one artifact and no
external-data wiring is needed. The batch axis stays dynamic, which lets the
same file be used for batched inference from Python.

Outputs
-------
``docs/model/garbage_regnety16gf.onnx``   the model the browser downloads
``docs/model/model_meta.json``            classes, preprocessing and recipe

Usage::

    pip install -r requirements-onnx.txt
    python scripts/export_onnx.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from garbage_classifier.config import load_config
from garbage_classifier.inference import load_bundle

PROJECT_ROOT = _bootstrap.PROJECT_ROOT
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "model" / "garbage_regnety16gf.onnx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=None, help="Path to best_model_bundle.pt.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write the .onnx file.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (default: 18).")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2e-4,
        help="Maximum allowed absolute difference between PyTorch and onnxruntime logits.",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip the onnxruntime parity check.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(root=PROJECT_ROOT)
    bundle_path = args.bundle or config.bundle_path
    if not bundle_path.exists():
        raise SystemExit(f"Bundle not found: {bundle_path}\nTrain the model first: python scripts/train.py --model all")

    device = torch.device("cpu")  # export on CPU so the graph has no device-specific nodes
    net, bundle = load_bundle(bundle_path, device)
    net.eval()

    crop = bundle["preprocess"]["crop"]
    height, width = (crop * 2)[:2] if len(crop) == 1 else crop[:2]
    example = torch.randn(1, 3, int(height), int(width))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch = torch.export.Dim("batch", min=1, max=64)
    torch.onnx.export(
        net,
        (example,),
        str(args.output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: batch}},
        opset_version=args.opset,
        external_data=False,  # one self-contained file for the browser to fetch
        optimize=True,
        dynamo=True,
    )

    sidecar = args.output.with_suffix(args.output.suffix + ".data")
    if sidecar.exists():  # older torch versions ignore external_data=False
        import onnx

        model = onnx.load(str(args.output))  # pulls the sidecar in
        onnx.save_model(model, str(args.output), save_as_external_data=False)
        sidecar.unlink()
        print(f"Inlined {sidecar.name} into the .onnx file")

    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB, opset {args.opset})")

    # --- Metadata the page needs to reproduce the exact inference recipe ----
    meta = {
        "model_file": args.output.name,
        "arch": bundle["arch"],
        "source_run": bundle["source_run"],
        "classes": bundle["classes"],
        "input": {"name": "input", "layout": "NCHW", "height": int(height), "width": int(width)},
        "preprocess": bundle["preprocess"],
        "temperature": float(bundle["temperature"]),
        "tta_hflip": bool(bundle["tta_hflip"]),
        "review_threshold": bundle.get("review_threshold"),
        "test_metrics": bundle.get("test_metrics", {}),
        "opset": args.opset,
        "size_bytes": args.output.stat().st_size,
    }
    meta_path = args.output.parent / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path.relative_to(PROJECT_ROOT)}")

    # --- Parity check -------------------------------------------------------
    if args.no_verify:
        return 0
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed — skipping the parity check (pip install -r requirements-onnx.txt).")
        return 0

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(config.seed)
    worst = 0.0
    for _ in range(5):
        batch = rng.standard_normal((2, 3, int(height), int(width))).astype(np.float32)
        with torch.inference_mode():
            expected = net(torch.from_numpy(batch)).numpy()
        actual = session.run(["logits"], {"input": batch})[0]
        worst = max(worst, float(np.abs(expected - actual).max()))

    print(f"Parity check: max |PyTorch - onnxruntime| = {worst:.2e} (tolerance {args.tolerance:.0e})")
    if worst > args.tolerance:
        raise SystemExit("ONNX output does not match PyTorch within tolerance — not safe to ship.")
    print("ONNX export verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
