"""Training and evaluation loops.

Extraction of notebook sections 9 and 11.3. Model selection is identical in both
stages: the best **validation macro-F1** epoch wins, with early stopping, and
the best weights are restored at the end so the in-memory model is the selected
one.

Macro-F1 rather than accuracy, because the dataset is imbalanced (max/min ~ 4.2)
and accuracy can look good while a minority class is neglected.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from garbage_classifier.model import freeze_bn_of_frozen_modules, load_checkpoint

__all__ = [
    "check_alignment",
    "collect_predictions",
    "evaluate",
    "fit",
    "predict_logits",
    "resolve_amp_dtype",
    "train_one_epoch",
]


def resolve_amp_dtype(device: torch.device, requested: str = "bf16") -> torch.dtype | None:
    """Pick the autocast dtype, degrading gracefully when unsupported.

    bf16 needs no loss scaling and is ~2x faster on RTX 30-series and newer. On
    a GPU without bf16 support this falls back to fp16 (which needs a
    :class:`~torch.amp.GradScaler`); on CPU, autocast is disabled entirely.
    """
    if device.type != "cuda" or requested == "off":
        return None
    if requested == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------------
# Inference passes
# ---------------------------------------------------------------------------
@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    tta_hflip: bool = False,
) -> dict[str, Any]:
    """One inference pass gathering everything downstream analysis needs.

    Returns ``y_true``, ``y_pred``, softmax ``probs`` and, when *loss_fn* is
    given, the mean ``loss``. With ``tta_hflip=True`` the logits of the image
    and of its horizontal mirror are averaged.
    """
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    total_loss, n_seen = 0.0, 0

    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X).float()
        if tta_hflip:
            logits = 0.5 * (logits + model(torch.flip(X, dims=[3])).float())

        if loss_fn is not None:
            total_loss += loss_fn(logits, y).item() * y.size(0)
        n_seen += y.size(0)

        logits_chunks.append(logits.cpu())
        target_chunks.append(y.cpu())

    probs = torch.cat(logits_chunks).softmax(dim=1)
    return {
        "y_true": torch.cat(target_chunks).numpy(),
        "y_pred": probs.argmax(dim=1).numpy(),
        "probs": probs.numpy(),
        "loss": total_loss / n_seen if loss_fn is not None else float("nan"),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    tta_hflip: bool = False,
) -> dict[str, Any]:
    """Loss, accuracy and macro-F1 on a loader — the metrics model selection uses.

    The raw predictions come back too, so post-training analysis reuses exactly
    the code path that drove selection.
    """
    out = collect_predictions(model, dataloader, device, loss_fn, tta_hflip)
    labels = np.arange(out["probs"].shape[1])  # score every class even if absent in a split
    out["acc"] = float((out["y_true"] == out["y_pred"]).mean())
    out["macro_f1"] = float(f1_score(out["y_true"], out["y_pred"], labels=labels, average="macro", zero_division=0))
    return out


@torch.inference_mode()
def predict_logits(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    """fp32 logits of every image *and* of its horizontal mirror, plus labels, in loader order.

    Caching both lets temperature scaling, the TTA decision and every metric in
    the comparison and evaluation sections be computed later without the model.
    """
    model.eval()
    out: dict[str, list[torch.Tensor]] = {"logits": [], "logits_hflip": [], "y_true": []}
    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        out["logits"].append(model(X).float().cpu())
        out["logits_hflip"].append(model(torch.flip(X, dims=[3])).float().cpu())
        out["y_true"].append(y)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def check_alignment(frame: pd.DataFrame, res: dict[str, np.ndarray], split: str) -> None:
    """Fail loudly if loader order differs from DataFrame order.

    Per-image analysis maps row *i* of the predictions to row *i* of the split
    frame. If an evaluation loader were ever created with ``shuffle=True`` that
    mapping would be silently wrong, so it is asserted.
    """
    assert len(res["y_true"]) == len(frame), f"{split}: {len(res['y_true'])} predictions vs {len(frame)} rows"
    assert np.array_equal(res["y_true"], frame["y"].to_numpy()), (
        f"{split}: loader order differs from DataFrame order -- use shuffle=False"
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    scaler: GradScaler | None = None,
    writer: Any | None = None,
    global_step: int = 0,
    log_every_n_steps: int = 10,
) -> tuple[float, float, int]:
    """One training epoch. Returns ``(mean loss, accuracy, next global step)``.

    Forward and loss run under autocast; the backward pass stays outside it.
    Accuracy uses ``argmax`` on the logits — softmax is monotonic, so it would
    be wasted work. The global gradient norm is measured with
    ``clip_grad_norm_(max_norm=inf)``, which computes the norm without clipping,
    because instabilities (loss spikes, exploding gradients) are invisible in
    epoch averages.
    """
    model.train()
    freeze_bn_of_frozen_modules(model)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    total_loss, correct, total = 0.0, 0, 0

    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(X)
            loss = loss_fn(logits, y)

        if scaler is not None:  # fp16 path
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
            scaler.step(optimizer)
            scaler.update()
        else:  # bf16 / fp32 path
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
            optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

        if writer is not None and global_step % log_every_n_steps == 0:
            writer.add_scalar("train_step/loss", batch_loss, global_step)
            writer.add_scalar("train_step/grad_norm", grad_norm.item(), global_step)
        global_step += 1

    return total_loss / total, correct / total, global_step


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn_train: nn.Module,
    loss_fn_val: nn.Module,
    device: torch.device,
    epochs: int,
    classes: list[str],
    ckpt_path: Path,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    patience: int | None = None,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    scaler: GradScaler | None = None,
    extra_checkpoint_fields: dict[str, Any] | None = None,
    run_label: str = "run",
    writer: Any | None = None,
    log_every_n_steps: int = 10,
    on_epoch_end: Any | None = None,
) -> pd.DataFrame:
    """Train with best-checkpoint selection on validation macro-F1.

    A checkpoint is written whenever validation macro-F1 improves, and the best
    weights are reloaded before returning, so the returned model is the selected
    one rather than the last-epoch one.

    The validation loss is always plain, unweighted, unsmoothed cross-entropy
    (*loss_fn_val*) so it stays comparable across runs — the training loss is
    not, because label smoothing and class weights change its scale.

    Returns the per-epoch history as a DataFrame.
    """
    history: list[dict[str, Any]] = []
    best_f1, best_epoch, no_improve, step = -1.0, 0, 0, 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        lrs_now = {
            g.get("name", f"group{i}").split("/")[0].split(".")[-1]: g["lr"]
            for i, g in enumerate(optimizer.param_groups)
        }
        t0 = time.perf_counter()
        train_loss, train_acc, step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn_train,
            device,
            amp_dtype=amp_dtype,
            scaler=scaler,
            writer=writer,
            global_step=step,
            log_every_n_steps=log_every_n_steps,
        )
        val_res = evaluate(model, val_loader, loss_fn_val, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_res["macro_f1"])
            else:
                scheduler.step()

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_res["loss"],
            "val_acc": val_res["acc"],
            "val_macro_f1": val_res["macro_f1"],
            "lr": next(iter(lrs_now.values())),
            **{f"lr_{g}": lr for g, lr in lrs_now.items()},
            "epoch_time_s": time.perf_counter() - t0,
        }
        history.append(row)
        if on_epoch_end is not None:
            on_epoch_end(epoch, row)

        improved = row["val_macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, no_improve = row["val_macro_f1"], epoch, 0
            payload: dict[str, Any] = {
                "model": model.state_dict(),
                "classes": classes,
                "best_epoch": epoch,
                "epoch": epoch - 1,  # 0-based, kept for compatibility with the notebook's checkpoint
                "val_macro_f1": best_f1,
            }
            if extra_checkpoint_fields:
                payload.update(extra_checkpoint_fields)
            torch.save(payload, ckpt_path)
        else:
            no_improve += 1

        print(
            f"[{run_label}] epoch {epoch:02d}/{epochs} | lr {row['lr']:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
            f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.2%} F1 {row['val_macro_f1']:.4f} | "
            f"{row['epoch_time_s']:.0f}s{'  *best*' if improved else ''}",
            flush=True,
        )

        if patience is not None and no_improve >= patience:
            print(f"[{run_label}] early stopping: no val macro-F1 improvement for {patience} epochs")
            break

    if ckpt_path.exists():
        load_checkpoint(model, ckpt_path, device, expected_classes=classes)
        print(f"[{run_label}] restored best epoch {best_epoch} (val macro-F1 = {best_f1:.4f})")
    return pd.DataFrame(history)
