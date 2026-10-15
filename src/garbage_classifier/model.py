"""RegNetY-1.6GF construction, freezing strategy, optimiser groups and LR schedules.

Extraction of notebook sections 8, 9 and 11.2.

RegNet (Radosavovic et al., *Designing Network Design Spaces*, 2020) is a family
of CNNs whose stage widths and depths follow a quantised linear rule found by
design-space search; the **Y** variant adds Squeeze-and-Excitation blocks.
RegNetY-1.6GF has ~11.2M parameters and ~1.6 GFLOPs at 224x224 — a strong
accuracy/compute trade-off for a ~9k-image dataset on a single laptop GPU.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import RegNet_Y_1_6GF_Weights, regnet_y_1_6gf

ARCH_NAME = "regnet_y_1_6gf"

#: Parameter-name prefixes, outermost stage first. Used for layer-wise LR decay
#: and as the fixed hyperparameter columns in the TensorBoard HPARAMS table.
ALL_GROUPS = ["fc", "block4", "block3", "block2", "block1", "stem"]

__all__ = [
    "ALL_GROUPS",
    "ARCH_NAME",
    "build_model",
    "build_pretrained_model",
    "file_md5",
    "free_gpu",
    "freeze_bn_of_frozen_modules",
    "load_checkpoint",
    "make_param_groups",
    "make_scheduler",
    "resolve_device",
    "set_trainable",
]


def resolve_device(preference: str | None = None) -> torch.device:
    """Pick the compute device: *preference* if given, else CUDA when available."""
    if preference:
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pretrained_model(num_classes: int, device: torch.device) -> nn.Module:
    """ImageNet-pretrained backbone, frozen, with a fresh ``Linear(888, num_classes)`` head.

    Stage 1 trains only the head: ~8.9k trainable parameters out of 10.3M. Linear
    probing is the right first step — fast, nearly impossible to overfit, and it
    gives a clean baseline before any backbone weight is touched.
    """
    model = regnet_y_1_6gf(weights=RegNet_Y_1_6GF_Weights.DEFAULT).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes).to(device)
    return model


def build_model(classes: list[str], checkpoint: Path, device: torch.device) -> nn.Module:
    """Backbone with a ``len(classes)``-class head, initialised from a full-model checkpoint.

    ``weights=None`` on purpose: the checkpoint already holds the (fine-tuned or
    probed) backbone, so nothing is downloaded and nothing is overwritten.
    """
    model = regnet_y_1_6gf(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    load_checkpoint(model, checkpoint, torch.device("cpu"), expected_classes=classes)
    return model.to(device)


def load_checkpoint(
    model: nn.Module,
    path: Path,
    device: torch.device,
    expected_classes: list[str] | None = None,
) -> dict:
    """Load weights from a training checkpoint, validating its class order.

    A silently permuted class list is the kind of bug that produces plausible
    but wrong predictions, so the order is asserted rather than trusted.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    if expected_classes is not None and state.get("classes") is not None:
        assert list(state["classes"]) == list(expected_classes), (
            f"Class order mismatch: checkpoint={state['classes']} vs current={expected_classes}"
        )
    model.load_state_dict(state["model"])
    model.eval()
    return state


def freeze_bn_of_frozen_modules(model: nn.Module) -> None:
    """Put BatchNorm layers whose affine parameters are frozen into ``eval()`` mode.

    Without this, ``model.train()`` keeps updating BN running statistics from
    small, augmented batches even though the BN weights are frozen, silently
    shifting the features the head is being fitted on.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in module.parameters()
        ):
            module.eval()


def set_trainable(model: nn.Module, prefixes: tuple[str, ...]) -> int:
    """Train only parameters whose name starts with one of *prefixes*.

    Returns the number of trainable parameters.
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_param_groups(model: nn.Module, lrs: dict[str, float], weight_decay: float) -> list[dict]:
    """One optimiser group per prefix, with weight decay on >= 2-D tensors only.

    Decaying 1-D parameters (BatchNorm weights and biases) only hurts
    normalisation layers, so they get ``weight_decay=0.0``. Every trainable
    parameter must land in exactly one group — asserted, because a parameter
    without a group would silently never be updated.
    """
    groups: list[dict] = []
    assigned: set[str] = set()
    for prefix, lr in lrs.items():
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith(prefix)]
        assigned.update(n for n, _ in named)
        decay = [p for _, p in named if p.ndim > 1]
        no_decay = [p for _, p in named if p.ndim <= 1]
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay, "name": f"{prefix}/decay"})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": f"{prefix}/no_decay"})

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert assigned == trainable, f"Params without an LR group: {sorted(trainable - assigned)[:5]}"
    return groups


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warm-up (x0.1 -> x1) followed by cosine annealing; stepped once per epoch.

    The warm-up protects pretrained weights from the first, largest updates.
    """
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=eta_min
    )
    if warmup_epochs == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def free_gpu() -> None:
    """Release cached GPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_MD5_CACHE: dict[tuple[str, float], str] = {}


def file_md5(path: Path) -> str:
    """MD5 of a file, memoised on ``(path, mtime)``.

    Fingerprints the checkpoint a run was initialised from, so retraining the
    linear probe invalidates exactly the runs that depend on it and nothing else.
    """
    key = (str(path), path.stat().st_mtime)
    if key not in _MD5_CACHE:
        _MD5_CACHE[key] = hashlib.md5(path.read_bytes()).hexdigest()
    return _MD5_CACHE[key]
