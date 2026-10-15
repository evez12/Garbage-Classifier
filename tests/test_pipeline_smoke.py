"""End-to-end smoke test of the training pipeline.

Runs the real scripts — audit, near-duplicate grouping, group-aware split,
linear probe, one fine-tuning run, selection, calibration and bundle export —
against a tiny synthetic dataset, on CPU, with randomly initialised weights.

It is a wiring test, not an accuracy test: the numbers it produces are
meaningless, but every artifact the rest of the project depends on has to come
out of it with the right name, shape and contents. It needs no dataset and no
network access.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import PROJECT_ROOT

CLASSES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]
IMAGES_PER_CLASS = 12  # must exceed the larger n_splits (7) in config.yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A throwaway copy of the project with a tiny dataset and small budgets."""
    root = tmp_path_factory.mktemp("garbage")
    for item in ("src", "scripts", "config.yaml"):
        source = PROJECT_ROOT / item
        destination = root / item
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    data_root = root / "data" / "garbage"
    rng = np.random.default_rng(0)
    for index, name in enumerate(CLASSES):
        folder = data_root / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            # A per-class colour bias plus noise: distinct perceptual hashes, so
            # the duplicate grouping has something real to do.
            canvas = np.zeros((64, 64, 3), dtype=np.float64)
            canvas[:, :, index % 3] = 30 + 20 * index
            pixels = np.clip(canvas + rng.normal(0, 55, (64, 64, 3)), 0, 255)
            Image.fromarray(pixels.astype(np.uint8)).save(folder / f"{name}_{i:02d}.jpg", quality=92)

    config = (root / "config.yaml").read_text(encoding="utf-8")
    config = (
        config.replace("  epochs: 15", "  epochs: 1")
        .replace("    epochs: 20", "    epochs: 1")
        .replace("  batch_size: 64", "  batch_size: 4")
        .replace("  num_workers: 4", "  num_workers: 0")
        .replace("  bootstrap_resamples: 2000", "  bootstrap_resamples: 25")
    )
    (root / "config.yaml").write_text(config, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def pipeline(workspace: Path):
    """Run train + evaluate once for the whole module."""
    import garbage_classifier.inference as inference
    import garbage_classifier.model as model_module
    from torchvision.models import regnet_y_1_6gf as real_regnet

    def offline_regnet(weights=None, **kwargs):
        """Skip the ImageNet download: the test only checks plumbing."""
        return real_regnet(weights=None, **kwargs)

    original_model, original_inference = model_module.regnet_y_1_6gf, inference.regnet_y_1_6gf
    model_module.regnet_y_1_6gf = offline_regnet
    inference.regnet_y_1_6gf = offline_regnet

    import evaluate as evaluate_script
    import train as train_script

    original_roots = train_script.PROJECT_ROOT, evaluate_script.PROJECT_ROOT
    train_script.PROJECT_ROOT = workspace
    evaluate_script.PROJECT_ROOT = workspace
    try:
        assert train_script.main(["--model", "ft_block4", "--device", "cpu", "--amp", "off"]) == 0
        # train.py only runs the evaluation step once every run is complete, so
        # with a partial set it is invoked directly. That also exercises the
        # "compare only what exists" path.
        assert evaluate_script.main([]) == 0
        yield workspace
    finally:
        model_module.regnet_y_1_6gf = original_model
        inference.regnet_y_1_6gf = original_inference
        train_script.PROJECT_ROOT, evaluate_script.PROJECT_ROOT = original_roots


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_splits_are_written_and_disjoint(pipeline: Path) -> None:
    """Three split files, covering every image exactly once, sharing no group."""
    import pandas as pd

    frames = {
        name: pd.read_csv(pipeline / "splits" / f"{name}.csv")
        for name in ("train", "val", "test")
    }
    total = sum(len(f) for f in frames.values())
    assert total == len(CLASSES) * IMAGES_PER_CLASS

    paths = pd.concat([f["path"] for f in frames.values()])
    assert paths.duplicated().sum() == 0, "an image appears in more than one split"

    groups = {name: set(f["group"]) for name, f in frames.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_run_artifacts_exist(pipeline: Path) -> None:
    """Both runs produce the full artifact set, plus a registry row each."""
    import pandas as pd

    for run in ("linear_probe", "ft_block4"):
        folder = pipeline / "experiments" / run
        for artifact in ("run.json", "history.csv", "predictions.npz"):
            assert (folder / artifact).is_file(), f"{run}/{artifact} missing"

        meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        assert meta["name"] == run
        assert (pipeline / meta["checkpoint"]).is_file()
        assert set(meta["metrics"]) >= {"val_macro_f1", "test_macro_f1", "val_acc", "test_acc"}

        with np.load(folder / "predictions.npz") as npz:
            assert [str(c) for c in npz["classes"]] == CLASSES
            for split in ("val", "test"):
                logits = npz[f"{split}_logits"]
                assert logits.shape[1] == len(CLASSES)
                assert logits.shape == npz[f"{split}_logits_hflip"].shape
                assert len(npz[f"{split}_y_true"]) == logits.shape[0]

    registry = pd.read_csv(pipeline / "experiments" / "registry.csv")
    assert set(registry["name"]) == {"linear_probe", "ft_block4"}


def test_caching_skips_a_completed_run(pipeline: Path) -> None:
    """A finished, unchanged run is recognised as complete and not retrained."""
    from garbage_classifier.config import load_config
    from garbage_classifier.experiments import build_configs, is_complete

    config = load_config(root=pipeline)
    configs = build_configs(config)
    assert is_complete(config, configs["linear_probe"])
    assert is_complete(config, configs["ft_block4"])
    # Never trained, so it must report incomplete rather than silently pass.
    assert not is_complete(config, configs["ft_full_llrd"])


def test_bundle_is_self_contained_and_predicts(pipeline: Path) -> None:
    """The exported bundle carries the whole recipe and classifies an image."""
    import torch

    from garbage_classifier.inference import load_bundle, predict_image

    bundle_path = pipeline / "checkpoints" / "best_model_bundle.pt"
    assert bundle_path.is_file()

    raw = torch.load(bundle_path, map_location="cpu", weights_only=False)
    assert raw["arch"] == "regnet_y_1_6gf"
    assert raw["classes"] == CLASSES
    assert set(raw["preprocess"]) == {"resize", "crop", "interpolation", "mean", "std"}
    assert isinstance(raw["temperature"], float) and raw["temperature"] > 0
    assert isinstance(raw["tta_hflip"], bool)

    net, bundle = load_bundle(bundle_path, torch.device("cpu"))
    image = next((pipeline / "data" / "garbage" / "glass").glob("*.jpg"))
    prediction = predict_image(image, net, bundle, top_k=3, device=torch.device("cpu"))

    assert prediction.label in CLASSES
    assert 0.0 <= prediction.confidence <= 1.0
    assert len(prediction.top_k) == 3
    assert prediction.top_k == sorted(prediction.top_k, key=lambda pair: -pair[1])
    assert isinstance(prediction.needs_review, bool)


def test_reports_and_demo_inputs_are_written(pipeline: Path) -> None:
    """Evaluation writes its tables, its figures and the page's metrics.json."""
    eval_dir = pipeline / "reports" / "best_model"
    for name in (
        "metrics.json",
        "per_class_test.csv",
        "predictions_test.csv",
        "selective_prediction.csv",
        "confusion_matrix_test.png",
        "per_class_metrics.png",
    ):
        assert (eval_dir / name).is_file(), f"reports/best_model/{name} missing"

    for name in ("comparison.csv", "selection.csv", "calibration.csv"):
        assert (pipeline / "reports" / "comparison" / name).is_file()

    assert (pipeline / "docs" / "data" / "metrics.json").is_file()
    assert (pipeline / "docs" / "figures" / "confusion_matrix.png").is_file()


def test_metrics_json_is_strict_json(pipeline: Path) -> None:
    """The demo page parses this file, and JSON.parse rejects NaN and Infinity."""
    for path in (
        pipeline / "reports" / "best_model" / "metrics.json",
        pipeline / "docs" / "data" / "metrics.json",
    ):
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)  # strict by default: NaN/Infinity would raise
        assert payload["selected_model"] in {"linear_probe", "ft_block4"}
        assert payload["classes"] == CLASSES
        json.dumps(payload, allow_nan=False)

    # The page payload has to carry exactly the keys docs/assets/app.js reads.
    page = json.loads((pipeline / "docs" / "data" / "metrics.json").read_text(encoding="utf-8"))
    assert len(page["confusion_matrix"]) == len(CLASSES)
    assert all(len(row) == len(CLASSES) for row in page["confusion_matrix"])
    assert sum(sum(row) for row in page["confusion_matrix"]) == page["n_test"]
    assert {"accuracy", "macro_f1"} <= set(page["headline"])
    for metric in page["headline"].values():
        assert len(metric["ci"]) == 2 and metric["ci"][0] <= metric["value"] <= metric["ci"][1]
    assert len(page["per_class"]) == len(CLASSES)
    for entry in page["per_class"]:
        assert {"class", "support", "precision", "recall", "f1", "missed", "false_alarms"} <= set(entry)
    assert {"tta_hflip", "temperature", "review_threshold"} <= set(page["recipe"])
    assert page["experiments"] and {"name", "val_macro_f1", "test_macro_f1"} <= set(page["experiments"][0])


def test_notebook_is_unmodified() -> None:
    """The notebook is the record of the reported run and must not drift."""
    import hashlib

    notebook = PROJECT_ROOT / "notebooks" / "model_training_notebook.ipynb"
    if not notebook.is_file():
        pytest.skip("notebook not present in this checkout")
    digest = hashlib.sha256(notebook.read_bytes()).hexdigest()
    assert digest == "e19487130933f0fcd51846390883fb9258fd6f41d97b662fddb34cbed83a7968"
