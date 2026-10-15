"""Run configuration, artifact layout, caching and the experiment registry.

Extraction of notebook section 11. Every run — the linear-probe baseline
included — writes the same set of artifacts, so model comparison and the
in-depth evaluation run **from disk alone**, without retraining anything::

    experiments/
    |- registry.csv           one row per run: cost + key val/test metrics
    |- linear_probe/          run.json, history.csv, predictions.npz
    |- ft_block4/             best.pt, run.json, history.csv, predictions.npz
    |- ft_block34/            ...
    |- ft_full_llrd/          ...
    `- ft_block34_cw/         ...

Cache rule
----------
A run is skipped when all its artifacts exist, its stored config equals the
current config, **and** the linear-probe checkpoint it was initialised from is
unchanged (MD5 fingerprint). Editing an experiment's hyperparameters or
retraining the linear probe therefore invalidates exactly the runs that depend
on it, and nothing else. ``run.json`` is written last and deleted when a run
starts, so an interrupted run is never mistaken for a finished one.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, log_loss
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from garbage_classifier.config import Config
from garbage_classifier.data import (
    GarbageDataset,
    build_dataloaders,
    compute_class_weights,
    finetune_train_transform,
    linear_probe_train_transform,
)
from garbage_classifier.metrics import fast_macro_f1, softmax_np
from garbage_classifier.model import (
    ALL_GROUPS,
    build_model,
    build_pretrained_model,
    file_md5,
    free_gpu,
    make_param_groups,
    make_scheduler,
    set_trainable,
)
from garbage_classifier.training import check_alignment, fit, predict_logits, resolve_amp_dtype
from garbage_classifier.viz import plot_confusion_matrices

BASELINE_NAME = "linear_probe"
LOG_EVERY_N_STEPS = 10

__all__ = [
    "BASELINE_NAME",
    "FTConfig",
    "RunContext",
    "build_configs",
    "is_complete",
    "load_record",
    "run_experiment",
    "short_group",
    "train_linear_probe",
]


@dataclass(frozen=True)
class FTConfig:
    """One training run.

    ``lrs`` maps parameter-name prefixes to learning rates: listed prefixes are
    trainable, everything else stays frozen. Lower learning rates for earlier
    stages, because early stages hold generic features that need little change.

    The field names and defaults match the notebook's dataclass exactly, so a
    ``run.json`` written here is accepted by the notebook's own cache check.
    """

    name: str
    lrs: dict[str, float]
    description: str = ""
    epochs: int = 20
    warmup_epochs: int = 1
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    class_weight_alpha: float | None = None  # None -> unweighted loss
    patience: int = 5


@dataclass
class RunContext:
    """Everything a run needs that is shared between runs: data, loaders, device."""

    config: Config
    classes: list[str]
    splits: dict[str, pd.DataFrame]
    loaders: dict[str, DataLoader]
    device: torch.device
    amp_dtype: torch.dtype | None
    batch_size_ft: int
    force_retrain: set[str] = field(default_factory=set)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def macro_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return fast_macro_f1(y_true, y_pred, self.num_classes)

    def make_train_loader(self) -> DataLoader:
        """Fresh, seeded stage-2 train loader, so every experiment sees the same batch order."""
        cfg = self.config
        return DataLoader(
            GarbageDataset(self.splits["train"], finetune_train_transform(cfg)),
            self.batch_size_ft,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(cfg.seed),
            num_workers=cfg.loader.num_workers,
            pin_memory=cfg.loader.pin_memory,
            persistent_workers=cfg.loader.num_workers > 0,
        )


def short_group(prefix: str) -> str:
    """``'trunk_output.block4' -> 'block4'`` — used in logs and column names."""
    return prefix.split(".")[-1]


def build_configs(config: Config) -> dict[str, FTConfig]:
    """All run configs in order, baseline first, built from ``config.yaml``."""
    lp = config.linear_probe
    baseline = FTConfig(
        BASELINE_NAME,
        {"fc": lp.learning_rate},
        "Frozen backbone, linear head (Section 9)",
        epochs=lp.epochs,
        warmup_epochs=0,
        weight_decay=lp.weight_decay,
        label_smoothing=0.0,
        patience=lp.patience,
    )

    shared = config.finetune_shared
    configs: dict[str, FTConfig] = {BASELINE_NAME: baseline}
    for name, raw in config.finetune_experiments.items():
        configs[name] = FTConfig(
            name=name,
            lrs={k: float(v) for k, v in raw["learning_rates"].items()},
            description=raw.get("description", ""),
            epochs=int(raw.get("epochs", shared.epochs)),
            warmup_epochs=int(raw.get("warmup_epochs", shared.warmup_epochs)),
            weight_decay=float(raw.get("weight_decay", shared.weight_decay)),
            label_smoothing=float(raw.get("label_smoothing", shared.label_smoothing)),
            class_weight_alpha=(
                float(raw["class_weight_alpha"]) if raw.get("class_weight_alpha") is not None else None
            ),
            patience=int(raw.get("patience", shared.patience)),
        )
    return configs


# ---------------------------------------------------------------------------
# Artifact layout and caching
# ---------------------------------------------------------------------------
def run_dir(config: Config, name: str) -> Path:
    return config.experiments_dir / name


def config_as_json(cfg: FTConfig) -> dict[str, Any]:
    """JSON round-trip of the config, so dict comparison is exact."""
    return json.loads(json.dumps(asdict(cfg)))


def is_complete(config: Config, cfg: FTConfig) -> bool:
    """True when the run's artifacts exist and match the current config and init checkpoint."""
    d = run_dir(config, cfg.name)
    if not all((d / f).exists() for f in ("run.json", "history.csv", "predictions.npz")):
        return False
    if not config.linear_probe_checkpoint.exists():
        return False
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    return (
        (config.root / meta["checkpoint"]).exists()
        and meta["config"] == config_as_json(cfg)
        and meta["init_checkpoint_md5"] == file_md5(config.linear_probe_checkpoint)
    )


def update_registry(config: Config, row: dict[str, Any], order: list[str]) -> pd.DataFrame:
    """Insert or replace one run's row in ``experiments/registry.csv``."""
    new = pd.DataFrame([row])
    if config.registry_path.exists():
        old = pd.read_csv(config.registry_path)
        new = pd.concat([old[old["name"] != row["name"]], new], ignore_index=True)
    rank = {n: i for i, n in enumerate(order)}
    new = new.sort_values("name", key=lambda s: s.map(rank).fillna(len(rank))).reset_index(drop=True)
    config.registry_path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(config.registry_path, index=False)
    return new


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------
def open_writer(config: Config, name: str, fresh: bool = True) -> SummaryWriter:
    """TensorBoard writer for one run; *fresh* removes old event files so curves never duplicate."""
    log_dir = config.tensorboard_dir / name
    if fresh and log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def tb_log_epoch(writer: SummaryWriter, epoch: int, row: dict[str, Any]) -> None:
    """Per-epoch scalars: convergence, overfitting, the LR of every group, cost."""
    writer.add_scalar("loss/train", row["train_loss"], epoch)
    writer.add_scalar("loss/val", row["val_loss"], epoch)
    writer.add_scalar("accuracy/train", row["train_acc"], epoch)
    writer.add_scalar("accuracy/val", row["val_acc"], epoch)
    writer.add_scalar("macro_f1/val", row["val_macro_f1"], epoch)
    for key, value in row.items():
        if key.startswith("lr_"):
            writer.add_scalar(f"lr/{key[3:]}", value, epoch)
    if np.isfinite(row.get("epoch_time_s", np.nan)):
        writer.add_scalar("time/epoch_s", row["epoch_time_s"], epoch)


def tb_log_final(
    writer: SummaryWriter,
    cfg: FTConfig,
    classes: list[str],
    best_epoch: int,
    val_out: dict[str, np.ndarray],
    test_out: dict[str, np.ndarray],
) -> dict[str, float]:
    """Best-checkpoint diagnostics on validation, plus this run's HPARAMS entry."""
    labels = np.arange(len(classes))
    metrics: dict[str, float] = {}
    for split, out in (("val", val_out), ("test", test_out)):
        probs = softmax_np(out["logits"])
        pred = probs.argmax(axis=1)
        metrics[f"hparam/{split}_macro_f1"] = fast_macro_f1(out["y_true"], pred, len(classes))
        metrics[f"hparam/{split}_acc"] = float((pred == out["y_true"]).mean())
        metrics[f"hparam/{split}_log_loss"] = float(log_loss(out["y_true"], probs, labels=labels))

    probs_val = softmax_np(val_out["logits"])
    for k, name in enumerate(classes):
        writer.add_pr_curve(f"pr_curve_val/{name}", val_out["y_true"] == k, probs_val[:, k], global_step=best_epoch)
    cm = confusion_matrix(val_out["y_true"], probs_val.argmax(axis=1), labels=labels)
    writer.add_figure(
        "figures/confusion_val",
        plot_confusion_matrices(cm, classes, f"{cfg.name} — validation, best epoch {best_epoch}"),
        global_step=best_epoch,
    )

    lrs = {short_group(k): v for k, v in cfg.lrs.items()}
    hparams = {
        "name": cfg.name,
        **{f"lr_{g}": float(lrs.get(g, 0.0)) for g in ALL_GROUPS},
        "weight_decay": cfg.weight_decay,
        "label_smoothing": cfg.label_smoothing,
        "class_weight_alpha": float(cfg.class_weight_alpha or 0.0),
        "max_epochs": cfg.epochs,
        "best_epoch": best_epoch,
    }
    writer.add_hparams(hparams, metrics, run_name=".")
    return {k.removeprefix("hparam/"): v for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Run tail shared by every run
# ---------------------------------------------------------------------------
def finalize_run(
    ctx: RunContext,
    cfg: FTConfig,
    model: nn.Module,
    history: pd.DataFrame,
    ckpt_path: Path,
    writer: SummaryWriter,
    n_trainable: int,
    batch_size: int,
    order: list[str],
) -> dict[str, Any]:
    """Cache predictions, run the integrity checks, write every artifact.

    The integrity checks matter: loader order must equal DataFrame order, and the
    *saved* weights must reproduce the best validation macro-F1 the history
    claims. Either failing means the stored logits do not describe the stored
    checkpoint, which would quietly invalidate every later analysis.
    """
    config = ctx.config
    d = run_dir(config, cfg.name)
    d.mkdir(parents=True, exist_ok=True)

    val_out = predict_logits(model, ctx.loaders["val"], ctx.device)
    test_out = predict_logits(model, ctx.loaders["test"], ctx.device)
    check_alignment(ctx.splits["val"], val_out, "val")
    check_alignment(ctx.splits["test"], test_out, "test")

    val_f1 = ctx.macro_f1(val_out["y_true"], val_out["logits"].argmax(axis=1))
    assert abs(val_f1 - history["val_macro_f1"].max()) < 1e-3, (
        f"[{cfg.name}] saved weights give val macro-F1 {val_f1:.4f}, "
        f"history says {history['val_macro_f1'].max():.4f}"
    )

    history.to_csv(d / "history.csv", index=False)
    np.savez_compressed(
        d / "predictions.npz",
        classes=np.array(ctx.classes),
        **{
            f"{split}_{k}": v
            for split, out in (("val", val_out), ("test", test_out))
            for k, v in out.items()
        },
    )
    best_epoch = int(history.loc[history["val_macro_f1"].idxmax(), "epoch"])
    metrics = tb_log_final(writer, cfg, ctx.classes, best_epoch, val_out, test_out)
    writer.close()

    meta: dict[str, Any] = {
        "name": cfg.name,
        "description": cfg.description,
        "config": config_as_json(cfg),
        "checkpoint": ckpt_path.relative_to(config.root).as_posix(),
        "init_checkpoint_md5": file_md5(config.linear_probe_checkpoint),
        "n_trainable": int(n_trainable),
        "n_total": int(sum(p.numel() for p in model.parameters())),
        "epochs_run": int(len(history)),
        "best_epoch": best_epoch,
        "train_time_s": float(history["epoch_time_s"].sum(skipna=False)),
        "metrics": metrics,
        "batch_size": batch_size,
        "seed": config.seed,
        "amp_dtype": str(ctx.amp_dtype),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (d / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    update_registry(
        config,
        {
            "name": cfg.name,
            "description": cfg.description,
            "trainable params (M)": n_trainable / 1e6,
            "epochs run": len(history),
            "best epoch": best_epoch,
            "train time (min)": meta["train_time_s"] / 60,
            "val acc": metrics["val_acc"],
            "val macro-F1": metrics["val_macro_f1"],
            "test acc": metrics["test_acc"],
            "test macro-F1": metrics["test_macro_f1"],
            "finished_at": meta["finished_at"],
        },
        order,
    )
    return meta


def _print_cached(meta: dict[str, Any]) -> None:
    m = meta["metrics"]
    print(
        f"[{meta['name']}] cached run ({meta['finished_at']}) — best epoch "
        f"{meta['best_epoch']}/{meta['epochs_run']}, val macro-F1 {m['val_macro_f1']:.4f}, "
        f"test macro-F1 {m['test_macro_f1']:.4f}. Not retrained."
    )


# ---------------------------------------------------------------------------
# Stage 1 — linear probe
# ---------------------------------------------------------------------------
def train_linear_probe(ctx: RunContext, retrain: bool = False) -> pd.DataFrame:
    """Train (or restore) the frozen-backbone linear probe and return its history.

    Every fine-tuning run starts from these weights — the linear-probe-then-
    fine-tune (LP-FT) recipe. Because the head is already fitted when the first
    gradients reach the backbone, the early updates are not driven by a random
    classifier, which protects the pretrained features.
    """
    config = ctx.config
    lp = config.linear_probe
    ckpt, history_path = config.linear_probe_checkpoint, config.linear_probe_history

    if ckpt.exists() and history_path.exists() and not retrain:
        state = torch.load(ckpt, map_location=ctx.device, weights_only=False)
        print(
            f"Loaded cached linear probe from {ckpt.name}: best epoch {state.get('best_epoch', '?')}, "
            f"val macro-F1 = {state['val_macro_f1']:.4f}. Pass --retrain-linear-probe to train again."
        )
        return pd.read_csv(history_path)

    model = build_pretrained_model(ctx.num_classes, ctx.device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{BASELINE_NAME}] frozen backbone, new head | trainable params: {n_trainable:,}")

    class_weights = (
        compute_class_weights(ctx.splits["train"]["y"], ctx.num_classes, alpha=lp.class_weight_alpha).to(ctx.device)
        if lp.use_class_weights
        else None
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lp.learning_rate, weight_decay=lp.weight_decay
    )
    optimizer.param_groups[0]["name"] = "fc"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lp.epochs, eta_min=lp.eta_min)

    loaders = build_dataloaders(config, ctx.splits, linear_probe_train_transform(config))
    history = fit(
        model=model,
        train_loader=loaders["train"],
        val_loader=ctx.loaders["val"],
        optimizer=optimizer,
        loss_fn_train=loss_fn,
        loss_fn_val=nn.CrossEntropyLoss(),
        device=ctx.device,
        epochs=lp.epochs,
        classes=ctx.classes,
        ckpt_path=ckpt,
        scheduler=scheduler,
        patience=lp.patience,
        amp_dtype=ctx.amp_dtype,
        extra_checkpoint_fields={"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        run_label=BASELINE_NAME,
    )

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.set_index("epoch").to_csv(history_path)
    print(f"Linear-probe history saved: {history_path.relative_to(config.root)}")
    del model
    free_gpu()
    return history


def register_linear_probe(ctx: RunContext, history: pd.DataFrame, order: list[str]) -> dict[str, Any]:
    """Give the baseline the same artifacts as a fine-tuning run, without retraining it."""
    config, cfg = ctx.config, build_configs(ctx.config)[BASELINE_NAME]
    d = run_dir(config, cfg.name)
    d.mkdir(parents=True, exist_ok=True)

    if cfg.name not in ctx.force_retrain and is_complete(config, cfg):
        meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
        _print_cached(meta)
        return meta

    (d / "run.json").unlink(missing_ok=True)
    model = build_model(ctx.classes, config.linear_probe_checkpoint, ctx.device)

    history = history.copy()
    if "epoch" not in history:
        history.insert(0, "epoch", np.arange(1, len(history) + 1))
    if "lr_fc" not in history:
        history["lr_fc"] = history["lr"]
    if "epoch_time_s" not in history:
        history["epoch_time_s"] = np.nan

    writer = open_writer(config, cfg.name, fresh=True)
    for row in history.to_dict(orient="records"):
        tb_log_epoch(writer, int(row["epoch"]), row)
    try:
        meta = finalize_run(
            ctx, cfg, model, history, config.linear_probe_checkpoint, writer,
            sum(p.numel() for p in model.fc.parameters()), config.loader.batch_size, order,
        )
    finally:
        writer.close()
        del model
        free_gpu()
    print(f"[{cfg.name}] registered -> {d.relative_to(config.root)}")
    return meta


# ---------------------------------------------------------------------------
# Stage 2 — one fine-tuning run
# ---------------------------------------------------------------------------
def run_experiment(ctx: RunContext, cfg: FTConfig, order: list[str]) -> dict[str, Any]:
    """Train one fine-tuning config, or return its cached record if it is up to date."""
    config = ctx.config
    d = run_dir(config, cfg.name)
    d.mkdir(parents=True, exist_ok=True)

    if cfg.name not in ctx.force_retrain and is_complete(config, cfg):
        meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
        _print_cached(meta)
        return meta

    (d / "run.json").unlink(missing_ok=True)  # incomplete until finalize_run succeeds
    model = build_model(ctx.classes, config.linear_probe_checkpoint, ctx.device)
    n_trainable = set_trainable(model, tuple(cfg.lrs))
    print(f"[{cfg.name}] {cfg.description} | trainable params: {n_trainable / 1e6:.2f} M")

    class_weights = (
        compute_class_weights(ctx.splits["train"]["y"], ctx.num_classes, alpha=cfg.class_weight_alpha).to(ctx.device)
        if cfg.class_weight_alpha is not None
        else None
    )
    optimizer = torch.optim.AdamW(make_param_groups(model, cfg.lrs, cfg.weight_decay))
    scheduler = make_scheduler(optimizer, cfg.epochs, cfg.warmup_epochs)
    scaler = torch.amp.GradScaler(ctx.device.type) if ctx.amp_dtype is torch.float16 else None

    writer = open_writer(config, cfg.name, fresh=True)
    train_loader = ctx.make_train_loader()
    try:
        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=ctx.loaders["val"],
            optimizer=optimizer,
            loss_fn_train=nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing),
            loss_fn_val=nn.CrossEntropyLoss(),
            device=ctx.device,
            epochs=cfg.epochs,
            classes=ctx.classes,
            ckpt_path=d / "best.pt",
            scheduler=scheduler,
            patience=cfg.patience,
            amp_dtype=ctx.amp_dtype,
            scaler=scaler,
            extra_checkpoint_fields={"config": asdict(cfg)},
            run_label=cfg.name,
            writer=writer,
            log_every_n_steps=LOG_EVERY_N_STEPS,
            on_epoch_end=lambda epoch, row: tb_log_epoch(writer, epoch, row),
        )
        meta = finalize_run(ctx, cfg, model, history, d / "best.pt", writer, n_trainable, ctx.batch_size_ft, order)
    finally:
        writer.close()
        del train_loader, model
        free_gpu()

    print(f"[{cfg.name}] done in {meta['train_time_s'] / 60:.1f} min -> {d.relative_to(config.root)}")
    return meta


# ---------------------------------------------------------------------------
# Reading runs back from disk
# ---------------------------------------------------------------------------
def load_record(config: Config, name: str, classes: list[str]) -> dict[str, Any]:
    """Everything the comparison needs for one run, read from disk — no model, no GPU."""
    d = run_dir(config, name)
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    with np.load(d / "predictions.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    assert [str(c) for c in arrays["classes"]] == classes, f"[{name}] class order mismatch"

    record: dict[str, Any] = {
        "name": name,
        "description": meta["description"],
        "meta": meta,
        "history": pd.read_csv(d / "history.csv"),
        "ckpt": config.root / meta["checkpoint"],
        "n_trainable": meta["n_trainable"],
        "train_time_s": meta["train_time_s"],
    }
    for split in ("val", "test"):
        logits = arrays[f"{split}_logits"]
        probs = softmax_np(logits)
        record[split] = {
            "y_true": arrays[f"{split}_y_true"],
            "logits": logits,
            "logits_hflip": arrays[f"{split}_logits_hflip"],
            "probs": probs,
            "y_pred": probs.argmax(axis=1),
        }
    return record


def make_run_context(
    config: Config,
    classes: list[str],
    splits: dict[str, pd.DataFrame],
    device: torch.device,
    amp: str = "bf16",
    batch_size_ft: int | None = None,
    force_retrain: set[str] | None = None,
) -> RunContext:
    """Assemble the shared :class:`RunContext` (loaders use the stage-2 augmentation)."""
    loaders = build_dataloaders(config, splits, finetune_train_transform(config), batch_size=batch_size_ft)
    return RunContext(
        config=config,
        classes=classes,
        splits=splits,
        loaders=loaders,
        device=device,
        amp_dtype=resolve_amp_dtype(device, amp),
        batch_size_ft=batch_size_ft or config.loader.batch_size,
        force_retrain=force_retrain or set(),
    )
