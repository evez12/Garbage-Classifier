"""Load the exported bundle and predict on single images.

Extraction of notebook section 14. The predictor is built **only** from
``checkpoints/best_model_bundle.pt``: it rebuilds the network and the evaluation
transform from the stored parameters, applies TTA if that was the selected
recipe, divides the logits by the fitted temperature, and flags low-confidence
images for manual review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import regnet_y_1_6gf

from garbage_classifier.model import ARCH_NAME

__all__ = ["Prediction", "load_bundle", "predict_image", "preprocess_image"]


class Prediction:
    """One image's prediction: ranked ``(class, probability)`` pairs and a review flag."""

    def __init__(self, top_k: list[tuple[str, float]], needs_review: bool) -> None:
        self.top_k = top_k
        self.needs_review = needs_review

    @property
    def label(self) -> str:
        return self.top_k[0][0]

    @property
    def confidence(self) -> float:
        return self.top_k[0][1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "top_k": [{"class": c, "probability": p} for c, p in self.top_k],
            "needs_review": self.needs_review,
        }


def _build_transform(preprocess: dict[str, Any]) -> T.Compose:
    """Rebuild the evaluation transform from the parameters stored in the bundle."""
    return T.Compose(
        [
            T.Resize(preprocess["resize"], interpolation=T.InterpolationMode(preprocess["interpolation"]), antialias=True),
            T.CenterCrop(preprocess["crop"]),
            T.ToTensor(),
            T.Normalize(preprocess["mean"], preprocess["std"]),
        ]
    )


def load_bundle(path: str | Path, device: torch.device | None = None) -> tuple[nn.Module, dict[str, Any]]:
    """Load the deployable bundle and return ``(model in eval mode, bundle dict)``.

    The returned bundle carries a ready-to-use ``transform`` under that key.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
    if bundle["arch"] != ARCH_NAME:
        raise ValueError(f"Unsupported architecture in bundle: {bundle['arch']!r} (expected {ARCH_NAME!r})")

    net = regnet_y_1_6gf(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(bundle["classes"]))
    net.load_state_dict(bundle["model"])
    bundle["transform"] = _build_transform(bundle["preprocess"])
    return net.to(device).eval(), bundle


def preprocess_image(image_path: str | Path, bundle: dict[str, Any]) -> torch.Tensor:
    """Decode one image file and apply the bundle's evaluation transform."""
    with Image.open(Path(image_path)) as im:
        return bundle["transform"](im.convert("RGB")).unsqueeze(0)


@torch.inference_mode()
def predict_image(
    image_path: str | Path,
    net: nn.Module,
    bundle: dict[str, Any],
    top_k: int = 3,
    device: torch.device | None = None,
) -> Prediction:
    """Calibrated top-*k* prediction and a manual-review flag for one image file."""
    device = device or next(net.parameters()).device
    x = preprocess_image(image_path, bundle).to(device)

    logits = net(x).float()
    if bundle["tta_hflip"]:
        logits = 0.5 * (logits + net(torch.flip(x, dims=[3])).float())

    probs = (logits / bundle["temperature"]).softmax(dim=1).squeeze(0).cpu()
    top = probs.topk(min(top_k, len(bundle["classes"])))
    ranked = [(bundle["classes"][i], float(p)) for p, i in zip(top.values, top.indices)]

    threshold = bundle.get("review_threshold")
    return Prediction(ranked, needs_review=threshold is not None and ranked[0][1] < threshold)
