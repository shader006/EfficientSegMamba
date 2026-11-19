#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained training script for official SegMamba on BraTS 2023 inside `all/`.

Connects:
- `all/settings.py` (hyperparameters, optimizer, scheduler, reproducibility)
- `all/losses.py` (DiceCELoss supporting `multiclass` and `regions` modes)
- `all/model/segmamba.py` (official ge-xing/SegMamba architecture)
- `all/script/dataset.py` (patient-disjoint `.npy` RAM/mmap DataLoader)
- `all/script/metrics.py` (BraTS-standard WT/TC/ET Dice & HD95)

Examples:
    # 1. Dry-run verification on CPU login node (DataLoader + Model + Optimizer):
    python train.py --dry-run --cache-mode mmap --num-workers 2

    # 2. Quick overfit check on GPU node (4 cases, 20 epochs):
    python train.py --overfit-cases 4 --epochs 20 --val-every 5 --cache-mode mmap

    # 3. Full training on GPU node (reads defaults from settings.py):
    python train.py --run-name segmamba_brats23_baseline

    # 4. Resume from checkpoint:
    python train.py --resume /home/cuc.buithi/BRATS/all/runs/segmamba_brats23_baseline/checkpoints/last_model.pt
"""

from __future__ import annotations

import os
os.environ["TMPDIR"] = "/tmp"
import tempfile
tempfile.tempdir = "/tmp"

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from torch.utils.data import DataLoader
from tqdm import tqdm


ALL_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ALL_DIR / "script"
for p in (str(SCRIPT_DIR), str(ALL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import settings  # noqa: E402
from losses import build_loss  # noqa: E402
from model import SegMamba  # noqa: E402
from dataset import (  # noqa: E402
    BraTS23PreprocessedDataset,
    _worker_init_fn,
    build_dataloaders,
    load_split_case_ids,
)
from metrics import aggregate_metrics, compute_batch_metrics  # noqa: E402


def build_model(label_mode: str, device: torch.device) -> torch.nn.Module:
    """Instantiate official SegMamba from `settings.py`."""
    out_chans = 4 if label_mode == "multiclass" else 3
    model_name = str(settings.MODEL_NAME).lower()
    if model_name == "segmamba":
        model = SegMamba(
            in_chans=settings.IN_CHANS,
            out_chans=out_chans,
            depths=list(settings.SEGMAMBA_DEPTHS),
            feat_size=list(settings.SEGMAMBA_FEAT_SIZE),
            drop_path_rate=settings.SEGMAMBA_DROP_PATH_RATE,
            hidden_size=settings.SEGMAMBA_HIDDEN_SIZE,
            norm_name=settings.SEGMAMBA_NORM_NAME,
        )
        return model.to(device)
    raise ValueError(f"Unsupported MODEL_NAME in settings.py: {settings.MODEL_NAME}")


def resolve_device(requested: str) -> torch.device:
    """Select CUDA device when available, raising cleanly if GPU is missing for real train."""
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def make_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders (supports `--overfit-cases` subset mode)."""
    patch_size = tuple(int(x) for x in settings.INPUT_SIZE)
    if args.overfit_cases > 0:
        ids = load_split_case_ids(Path(args.split_json), "train", fold=args.fold)[: args.overfit_cases]
        train_ds = BraTS23PreprocessedDataset(
            data_dir=args.data_dir,
            case_ids=ids,
            is_train=True,
            patch_size=patch_size,
            fg_prob=args.fg_prob,
            label_mode=args.label_mode,
            cache_mode=args.cache_mode,
            aug_mode=args.aug_mode,
            modality_dropout_prob=settings.MODALITY_DROPOUT_PROB,
        )
        val_ds = BraTS23PreprocessedDataset(
            data_dir=args.data_dir,
            case_ids=ids,
            is_train=False,
            label_mode=args.label_mode,
            cache_mode=args.cache_mode,
        )
        kw = {
            "num_workers": args.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": args.num_workers > 0,
            "worker_init_fn": _worker_init_fn if args.num_workers > 0 else None,
        }
        train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(ids)), shuffle=True, **kw)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **kw)
        return train_loader, val_loader

    train_loader, val_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        split_json=args.split_json,
        patch_size=patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_mode=args.cache_mode,
        label_mode=args.label_mode,
        aug_mode=args.aug_mode,
        fg_prob=args.fg_prob,
        modality_dropout_prob=settings.MODALITY_DROPOUT_PROB,
        fold=args.fold,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    grad_clip_norm: float,
    epoch: int,
    total_epochs: int = 0,
    logger: Optional["RunLogger"] = None,
    overall_eta: Optional[str] = None,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    running_loss = 0.0
    n_steps = 0
    total_steps = len(loader)
    use_amp = amp_dtype is not None and device.type == "cuda"
    is_tty = sys.stdout.isatty()
    t_start = time.perf_counter()
    log_interval = max(1, total_steps // 4) if total_steps > 0 else 1

    desc = f"Epoch {epoch:03d}/{total_epochs:03d} [Train]" if total_epochs > 0 else f"Epoch {epoch:03d} [Train]"
    pbar = tqdm(loader, desc=desc, leave=False, disable=not is_tty)
    for step_idx, batch in enumerate(pbar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}: {loss.item()}")

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.item())
        running_loss += loss_val
        n_steps += 1
        cur_lr = optimizer.param_groups[0]["lr"]
        postfix = {"loss": f"{loss_val:.4f}", "lr": f"{cur_lr:.2e}"}
        if overall_eta:
            postfix["eta_all"] = overall_eta
        pbar.set_postfix(postfix)

        if logger is not None and total_steps > 0:
            if step_idx % log_interval == 0 or step_idx == total_steps:
                elapsed = time.perf_counter() - t_start
                ep_eta = format_duration((elapsed / step_idx) * (total_steps - step_idx))
                pct = (step_idx / total_steps) * 100.0
                logger.log(
                    f"  [Train Ep {epoch:03d}] {pct:5.1f}% ({step_idx}/{total_steps}) | "
                    f"avg_loss={running_loss / n_steps:.4f} | lr={cur_lr:.2e} | ep_ETA={ep_eta}",
                    echo=not is_tty,
                )

    return running_loss / max(1, n_steps)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    inferer: SlidingWindowInferer,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    label_mode: str,
    compute_hd95: bool,
    epoch: int,
    logger: Optional["RunLogger"] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate validation set with SlidingWindowInferer and BraTS metrics."""
    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    records: List[Dict[str, float]] = []
    total_steps = len(loader)
    is_tty = sys.stdout.isatty()
    t_start = time.perf_counter()
    log_interval = max(1, total_steps // 4) if total_steps > 0 else 1

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", leave=False, disable=not is_tty)
    for step_idx, batch in enumerate(pbar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = inferer(images, model)

        batch_metrics = compute_batch_metrics(
            logits.float().cpu(),
            labels,
            label_mode=label_mode,
            from_logits=True,
            empty_hd95_penalty=settings.EMPTY_HD95_PENALTY,
            compute_hd95=compute_hd95,
        )
        records.extend(batch_metrics)
        if batch_metrics:
            pbar.set_postfix(dice=f"{batch_metrics[-1]['Mean_Dice']:.4f}")

        if logger is not None and total_steps > 0:
            if step_idx % log_interval == 0 or step_idx == total_steps:
                elapsed = time.perf_counter() - t_start
                val_eta = format_duration((elapsed / step_idx) * (total_steps - step_idx))
                pct = (step_idx / total_steps) * 100.0
                logger.log(
                    f"  [Val Ep {epoch:03d}] {pct:5.1f}% ({step_idx}/{total_steps}) | val_ETA={val_eta}",
                    echo=not is_tty,
                )

    return aggregate_metrics(records)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    best_val_dice: float,
    args: argparse.Namespace,
) -> None:
    """Save full training state checkpoint (model, optimizer, scheduler, scaler, metadata)."""
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_dice": float(best_val_dice),
        "label_mode": args.label_mode,
        "model_name": settings.MODEL_NAME,
        "settings": settings.export_config_dict(),
    }
    torch.save(payload, path)


class TopKCheckpointManager:
    """Maintain top-K model-weight checkpoints ranked by validation Mean Dice."""

    def __init__(self, ckpt_dir: Path, k: int = 3) -> None:
        self.k = max(0, int(k))
        self.topk_dir = ckpt_dir / "topk"
        self.manifest_path = ckpt_dir / "topk_manifest.json"
        self.entries: List[Dict[str, Any]] = []
        if self.k > 0:
            self.topk_dir.mkdir(parents=True, exist_ok=True)
            if self.manifest_path.is_file():
                try:
                    loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                    self.entries = [e for e in loaded if Path(e["path"]).is_file()]
                except Exception:
                    self.entries = []

    def maybe_save(
        self,
        epoch: int,
        val_mean: float,
        val_wt: float,
        val_tc: float,
        val_et: float,
        model: torch.nn.Module,
        args: argparse.Namespace,
    ) -> bool:
        """Save lightweight weights if `val_mean` qualifies for top-K; return True if saved."""
        if self.k <= 0 or np.isnan(val_mean):
            return False

        qualifies = len(self.entries) < self.k or val_mean > self.entries[-1]["val_mean_dice"]
        if not qualifies:
            return False

        ckpt_path = self.topk_dir / f"ep{epoch:03d}_dice{val_mean:.4f}.pt"
        payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "val_mean_dice": float(val_mean),
            "val_wt_dice": float(val_wt),
            "val_tc_dice": float(val_tc),
            "val_et_dice": float(val_et),
            "label_mode": args.label_mode,
            "model_name": settings.MODEL_NAME,
            "settings": settings.export_config_dict(),
        }
        torch.save(payload, ckpt_path)

        # Replace any prior entry at the same epoch/path, then sort descending by Dice.
        self.entries = [e for e in self.entries if int(e["epoch"]) != int(epoch)]
        self.entries.append(
            {
                "epoch": int(epoch),
                "val_mean_dice": float(val_mean),
                "val_wt_dice": float(val_wt),
                "val_tc_dice": float(val_tc),
                "val_et_dice": float(val_et),
                "path": str(ckpt_path),
            }
        )
        self.entries.sort(key=lambda e: (-e["val_mean_dice"], -e["epoch"]))

        while len(self.entries) > self.k:
            evicted = self.entries.pop()
            evicted_path = Path(evicted["path"])
            if evicted_path.is_file() and evicted_path != ckpt_path:
                evicted_path.unlink(missing_ok=True)

        self.manifest_path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")
        return True


class RunLogger:
    """Mirror training messages to stdout and an append-only `train.log` file."""

    def __init__(self, log_path: Optional[Path]) -> None:
        self.log_path = log_path
        self._file = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = log_path.open("a", encoding="utf-8")

    def log(self, msg: str = "", echo: bool = True) -> None:
        if echo:
            print(msg, flush=True)
        if self._file is not None:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = msg.splitlines() if msg else [""]
            for line in lines:
                self._file.write(f"[{ts}] {line}\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


def format_duration(seconds: float) -> str:
    """Format duration in seconds into compact human-readable form."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def append_history_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append one epoch summary row to `history.csv`."""
    write_header = not csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegMamba on BraTS 2023 inside `all/`.")
    parser.add_argument("--data-dir", type=Path, default=settings.DATA_DIR)
    parser.add_argument("--split-json", type=Path, default=settings.SPLIT_JSON)
    parser.add_argument("--runs-dir", type=Path, default=settings.RUNS_DIR)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--fold", type=int, default=settings.FOLD)

    parser.add_argument("--epochs", type=int, default=settings.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--val-every", type=int, default=settings.VAL_EVERY)
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--cache-mode", choices=("ram", "mmap"), default=settings.CACHE_MODE)
    parser.add_argument("--aug-mode", choices=("strong", "light", "none"), default=settings.AUG_MODE)
    parser.add_argument("--label-mode", choices=("multiclass", "regions"), default=settings.LABEL_MODE)
    parser.add_argument("--fg-prob", type=float, default=settings.FG_PROB)
    parser.add_argument("--amp-precision", choices=("bf16", "fp16", "fp32"), default=settings.AMP_PRECISION)
    parser.add_argument("--device", type=str, default=settings.DEVICE)

    parser.add_argument("--val-hd95", action="store_true", help="Compute HD95 during validation epochs.")
    parser.add_argument("--top-k", type=int, default=settings.TOP_K_CHECKPOINTS, help="Keep top-K checkpoints by val Dice.")
    parser.add_argument("--overfit-cases", type=int, default=0, help="Train/eval on N cases for quick overfit test.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint (.pt) to resume training.")
    parser.add_argument("--dry-run", action="store_true", help="Verify setup, batch, model, and loss without GPU loop.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings.set_global_reproducibility(settings.REPRO_SEED)

    device = resolve_device(args.device)
    if device.type != "cuda" and not args.dry_run:
        raise RuntimeError(
            "No CUDA GPU detected on this node. Official SegMamba (`mamba_ssm`) requires a CUDA GPU "
            "for forward/backward passes. Submit via SLURM (`sbatch`/`srun`) or run `--dry-run` on CPU."
        )

    run_name = args.run_name.strip() or f"{settings.MODEL_NAME}_{args.label_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.runs_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    if not args.dry_run:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(None if args.dry_run else (run_dir / "train.log"))
    try:
        train_loader, val_loader = make_dataloaders(args)
        model = build_model(args.label_mode, device)
        loss_fn = build_loss(label_mode=args.label_mode)
        optimizer = settings.build_optimizer(model, lr=args.lr)

        steps_per_epoch = max(1, len(train_loader))
        total_steps = args.epochs * steps_per_epoch
        scheduler = settings.build_scheduler(optimizer, total_steps=total_steps)

        amp_dtype = settings.get_amp_dtype(args.amp_precision) if settings.AMP_ENABLED else None
        scaler = (
            torch.amp.GradScaler("cuda")
            if (amp_dtype == torch.float16 and device.type == "cuda")
            else None
        )

        n_params = sum(p.numel() for p in model.parameters())
        logger.log(
            f"[INIT] run_name={run_name} | model={settings.MODEL_NAME} ({n_params / 1e6:.2f}M params) | "
            f"device={device} | amp={args.amp_precision if amp_dtype else 'fp32'} | "
            f"label_mode={args.label_mode} | cache={args.cache_mode} | aug={args.aug_mode}"
        )
        logger.log(
            f"[DATA] train_cases={len(train_loader.dataset)} | val_cases={len(val_loader.dataset)} | "
            f"steps/epoch={steps_per_epoch} | epochs={args.epochs} | val_every={args.val_every}"
        )

        if args.dry_run:
            batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            logger.log(f"[DRY-RUN] train batch image={tuple(batch['image'].shape)}, label={tuple(batch['label'].shape)}")
            logger.log(f"[DRY-RUN] val sample image={tuple(val_batch['image'].shape)}, label={tuple(val_batch['label'].shape)}")
            logger.log("[DRY-RUN PASSED] Pipeline ready to submit to a GPU node.")
            return 0

        start_epoch = 1
        best_val_dice = -1.0
        if args.resume is not None:
            if not args.resume.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if scaler is not None and ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_dice = float(ckpt.get("best_val_dice", -1.0))
            logger.log(f"[RESUME] Loaded {args.resume} (resuming at epoch {start_epoch}, best_val_dice={best_val_dice:.4f})")

        config_snapshot = {
            "run_name": run_name,
            "cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "model_params": n_params,
            "settings": settings.export_config_dict(),
        }
        (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

        inferer = SlidingWindowInferer(
            roi_size=tuple(int(x) for x in settings.INPUT_SIZE),
            sw_batch_size=settings.SW_BATCH_SIZE,
            overlap=settings.SW_OVERLAP,
            mode=settings.SW_MODE,
        )
        history_csv = run_dir / "history.csv"
        topk_manager = TopKCheckpointManager(ckpt_dir=ckpt_dir, k=args.top_k)
        epoch_times: List[float] = []

        for epoch in range(start_epoch, args.epochs + 1):
            t_ep = time.perf_counter()
            prev_eta = (
                format_duration((sum(epoch_times) / len(epoch_times)) * (args.epochs - epoch + 1))
                if epoch_times
                else None
            )

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                amp_dtype=amp_dtype,
                grad_clip_norm=settings.GRAD_CLIP_NORM,
                epoch=epoch,
                total_epochs=args.epochs,
                logger=logger,
                overall_eta=prev_eta,
            )

            val_summary: Dict[str, Dict[str, float]] = {}
            should_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
            if should_val:
                val_summary = validate(
                    model=model,
                    loader=val_loader,
                    inferer=inferer,
                    device=device,
                    amp_dtype=amp_dtype,
                    label_mode=args.label_mode,
                    compute_hd95=args.val_hd95,
                    epoch=epoch,
                    logger=logger,
                )

            val_mean = val_summary.get("Mean_Dice", {}).get("mean", float("nan"))
            val_wt = val_summary.get("WT_Dice", {}).get("mean", float("nan"))
            val_tc = val_summary.get("TC_Dice", {}).get("mean", float("nan"))
            val_et = val_summary.get("ET_Dice", {}).get("mean", float("nan"))
            val_hd95 = val_summary.get("Mean_HD95", {}).get("mean", float("nan"))
            val_wt_hd95 = val_summary.get("WT_HD95", {}).get("mean", float("nan"))
            val_tc_hd95 = val_summary.get("TC_HD95", {}).get("mean", float("nan"))
            val_et_hd95 = val_summary.get("ET_HD95", {}).get("mean", float("nan"))

            saved_best = False
            saved_topk = False
            if should_val and not np.isnan(val_mean):
                if val_mean > best_val_dice:
                    best_val_dice = val_mean
                    save_checkpoint(
                        ckpt_dir / "best_model.pt",
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        best_val_dice,
                        args,
                    )
                    saved_best = True
                saved_topk = topk_manager.maybe_save(
                    epoch=epoch,
                    val_mean=val_mean,
                    val_wt=val_wt,
                    val_tc=val_tc,
                    val_et=val_et,
                    model=model,
                    args=args,
                )

            save_checkpoint(
                ckpt_dir / "last_model.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val_dice,
                args,
            )

            ep_sec = time.perf_counter() - t_ep
            epoch_times.append(ep_sec)
            recent_window = epoch_times[-max(10, args.val_every * 3):]
            avg_ep_sec = sum(recent_window) / len(recent_window)
            eta_str = format_duration(avg_ep_sec * (args.epochs - epoch))

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "lr": optimizer.param_groups[0]["lr"],
                "val_mean_dice": "" if np.isnan(val_mean) else round(val_mean, 6),
                "val_wt_dice": "" if np.isnan(val_wt) else round(val_wt, 6),
                "val_tc_dice": "" if np.isnan(val_tc) else round(val_tc, 6),
                "val_et_dice": "" if np.isnan(val_et) else round(val_et, 6),
                "val_mean_hd95": "" if np.isnan(val_hd95) else round(val_hd95, 4),
                "val_wt_hd95": "" if np.isnan(val_wt_hd95) else round(val_wt_hd95, 4),
                "val_tc_hd95": "" if np.isnan(val_tc_hd95) else round(val_tc_hd95, 4),
                "val_et_hd95": "" if np.isnan(val_et_hd95) else round(val_et_hd95, 4),
                "best_val_dice": round(best_val_dice, 6) if best_val_dice >= 0 else "",
                "epoch_sec": round(ep_sec, 2),
            }
            append_history_csv(history_csv, row)

            val_msg = (
                f" | val_dice={val_mean:.4f} (WT={val_wt:.4f}, TC={val_tc:.4f}, ET={val_et:.4f})"
                if should_val
                else ""
            )
            hd95_msg = (
                f" | hd95={val_hd95:.2f} (WT={val_wt_hd95:.2f}, TC={val_tc_hd95:.2f}, ET={val_et_hd95:.2f})"
                if (should_val and not np.isnan(val_hd95))
                else ""
            )
            logger.log(
                f"[Epoch {epoch:03d}/{args.epochs:03d}] loss={train_loss:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}{val_msg}{hd95_msg} | "
                f"best={best_val_dice:.4f} | {ep_sec:.1f}s | ETA={eta_str}"
            )
            if saved_best:
                logger.log(f"[BEST] Saved {ckpt_dir / 'best_model.pt'} (epoch={epoch}, val_mean_dice={best_val_dice:.4f})")
            if saved_topk:
                logger.log(f"[TOP-K] Updated {topk_manager.manifest_path} (kept {len(topk_manager.entries)}/{args.top_k})")

        logger.log(f"\n[DONE] Training complete. Best val Mean Dice: {best_val_dice:.4f}")
        logger.log(f"  - Best checkpoint : {ckpt_dir / 'best_model.pt'}")
        if topk_manager.entries:
            logger.log(f"  - Top-{len(topk_manager.entries)} manifest  : {topk_manager.manifest_path}")
        logger.log(f"  - History CSV     : {history_csv}")
        logger.log(f"  - Train log       : {run_dir / 'train.log'}")
        return 0
    except Exception:
        logger.log("[ERROR] Unhandled exception during training:")
        logger.log(traceback.format_exc().rstrip())
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
