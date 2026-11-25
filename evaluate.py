#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test/Validation evaluation script for trained SegMamba checkpoints inside `all/`.

Evaluates a selected checkpoint (e.g. `best_model_selected.pt` produced by
`select_best_checkpoint.py`, or any `.pt` checkpoint) on the configured split
(`test` by default) using `SlidingWindowInferer` and BraTS-standard metrics
(WT/TC/ET/Mean Dice and HD95, with Mean/Std/Median/P25/P75/IQR summary).

Optionally supports:
- 8-view 3D flip test-time augmentation (`--tta`)
- Uncropping and exporting discrete predictions to `.nii.gz` (`--save-nii`)

Outputs inside `--output-dir` (default: `<run_dir>/eval_<split>_<ckpt_stem>`):
- `per_case_metrics.csv`
- `summary.json`
- `predictions/<case_id>.nii.gz` (when `--save-nii` is set)
"""

from __future__ import annotations

import os
os.environ["TMPDIR"] = "/tmp"
import tempfile
tempfile.tempdir = "/tmp"

import argparse
import csv
import itertools
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
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
from dataset import BraTS23PreprocessedDataset, _worker_init_fn, load_split_case_ids  # noqa: E402
from metrics import (  # noqa: E402
    BRATS_REGIONS,
    aggregate_metrics,
    compute_case_metrics,
    multiclass_to_brats_regions,
    predictions_to_region_masks,
    uncrop_to_original,
)
from train import build_model, format_duration, resolve_device  # noqa: E402


def resolve_checkpoint_path(run_dir: Optional[Path], checkpoint: Optional[Path]) -> Path:
    """Resolve explicit `--checkpoint` or auto-discover best checkpoint under `--run-dir`."""
    if checkpoint is not None:
        ckpt_path = checkpoint.resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    if run_dir is None:
        raise ValueError("Provide either `--run-dir` or `--checkpoint`.")

    ckpt_dir = run_dir.resolve() / "checkpoints"
    for name in ("best_model_selected.pt", "best_model.pt", "last_model.pt"):
        candidate = ckpt_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir} (checked best_model_selected.pt, best_model.pt, last_model.pt)."
    )


def infer_run_dir(run_dir: Optional[Path], ckpt_path: Path) -> Optional[Path]:
    """Infer `<run_dir>` from `.../checkpoints/<file>.pt` when `--run-dir` is omitted."""
    if run_dir is not None:
        return run_dir.resolve()
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent
    if ckpt_path.parent.name == "topk" and ckpt_path.parent.parent.name == "checkpoints":
        return ckpt_path.parent.parent.parent
    return None


def apply_saved_settings(ckpt: Dict[str, Any], run_dir: Optional[Path]) -> Dict[str, Any]:
    """Restore model/input settings from checkpoint or `config.json` to match training."""
    saved: Dict[str, Any] = {}
    if run_dir is not None:
        cfg_path = run_dir / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                saved.update(cfg.get("settings", {}))
                cli_args = cfg.get("cli_args", {})
                if "label_mode" in cli_args:
                    saved.setdefault("LABEL_MODE", cli_args["label_mode"])
            except Exception:
                pass

    if isinstance(ckpt.get("settings"), dict):
        saved.update(ckpt["settings"])
    if "label_mode" in ckpt:
        saved["LABEL_MODE"] = ckpt["label_mode"]

    model_keys = (
        "MODEL_NAME",
        "IN_CHANS",
        "OUT_CHANS",
        "INPUT_SIZE",
        "LABEL_MODE",
        "SEGMAMBA_DEPTHS",
        "SEGMAMBA_FEAT_SIZE",
        "SEGMAMBA_HIDDEN_SIZE",
        "SEGMAMBA_NORM_NAME",
        "SEGMAMBA_DROP_PATH_RATE",
        "EMPTY_HD95_PENALTY",
    )
    for key in model_keys:
        if key in saved and hasattr(settings, key):
            setattr(settings, key, saved[key])
    return saved


def predict_with_tta(
    inferer: SlidingWindowInferer,
    images: torch.Tensor,
    model: torch.nn.Module,
    use_tta: bool,
) -> torch.Tensor:
    """Run SlidingWindowInferer with optional 8-view 3D flip test-time augmentation."""
    if not use_tta:
        return inferer(images, model)

    flip_combos: List[Tuple[int, ...]] = []
    for r in range(4):
        flip_combos.extend(itertools.combinations((2, 3, 4), r))

    logits_sum: Optional[torch.Tensor] = None
    for axes in flip_combos:
        inp = torch.flip(images, dims=axes) if axes else images
        out = inferer(inp, model)
        if axes:
            out = torch.flip(out, dims=axes)
        out_f = out.float()
        logits_sum = out_f if logits_sum is None else (logits_sum + out_f)
    return logits_sum / float(len(flip_combos))


def logits_to_discrete_seg(logits: np.ndarray, label_mode: str) -> np.ndarray:
    """Convert `(C, X, Y, Z)` logits to discrete uint8 segmentation `(X, Y, Z)`."""
    if label_mode == "multiclass":
        return np.argmax(logits, axis=0).astype(np.uint8)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -60.0, 60.0)))
    wt, tc, et = probs[0] >= 0.5, probs[1] >= 0.5, probs[2] >= 0.5
    seg = np.zeros(logits.shape[1:], dtype=np.uint8)
    seg[wt] = 2
    seg[tc] = 1
    seg[et] = 3
    return seg


def extract_voxel_spacing(meta: Optional[Dict[str, Any]], pkl_path: Path) -> Tuple[float, float, float]:
    """Extract `(sx, sy, sz)` voxel spacing from collated batch metadata or `.pkl`."""
    if meta is not None and "spacing" in meta:
        sp = meta["spacing"]
        if isinstance(sp, (list, tuple)) and len(sp) == 3:
            return tuple(float(v[0] if isinstance(v, (torch.Tensor, np.ndarray, list, tuple)) else v) for v in sp)
    with pkl_path.open("rb") as f:
        raw_meta = pickle.load(f)
    sp = raw_meta.get("spacing", (1.0, 1.0, 1.0))
    return tuple(float(v) for v in sp)


def save_nifti_prediction(
    discrete_seg: np.ndarray,
    pkl_path: Path,
    out_path: Path,
) -> None:
    """Uncrop discrete segmentation back to original volume shape and save `.nii.gz`."""
    with pkl_path.open("rb") as f:
        meta = pickle.load(f)
    restored = uncrop_to_original(
        discrete_seg.astype(np.uint8, copy=False),
        crop_bbox=meta["crop_bbox"],
        original_shape=meta["original_shape"],
        fill_value=0,
    )
    affine = np.asarray(meta.get("affine", np.eye(4)), dtype=np.float64)
    nii = nib.Nifti1Image(restored, affine)
    nib.save(nii, str(out_path))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SegMamba checkpoint on BraTS test/val split inside `all/`."
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory under `all/runs/`.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit `.pt` checkpoint path.")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test", help="Dataset split to evaluate.")
    parser.add_argument("--data-dir", type=Path, default=settings.DATA_DIR)
    parser.add_argument("--split-json", type=Path, default=settings.SPLIT_JSON)
    parser.add_argument("--fold", type=int, default=settings.FOLD)
    parser.add_argument("--cache-mode", choices=("ram", "mmap"), default="mmap")
    parser.add_argument("--num-workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--sw-batch-size", type=int, default=settings.SW_BATCH_SIZE)
    parser.add_argument("--sw-overlap", type=float, default=settings.SW_OVERLAP)
    parser.add_argument("--sw-mode", choices=("gaussian", "constant"), default=settings.SW_MODE)
    parser.add_argument("--tta", action="store_true", help="Enable 8-view 3D flip test-time augmentation.")
    parser.add_argument("--amp-precision", choices=("bf16", "fp16", "fp32"), default=None)
    parser.add_argument("--device", type=str, default=settings.DEVICE)
    parser.add_argument("--no-hd95", action="store_true", help="Skip HD95 computation for fast Dice-only check.")
    parser.add_argument("--save-nii", action="store_true", help="Save uncropped `.nii.gz` predictions.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write CSV/JSON reports.")
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    settings.set_global_reproducibility(settings.REPRO_SEED)

    ckpt_path = resolve_checkpoint_path(args.run_dir, args.checkpoint)
    run_dir = infer_run_dir(args.run_dir, ckpt_path)

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for SegMamba SlidingWindowInferer evaluation.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_settings = apply_saved_settings(ckpt, run_dir)
    label_mode = str(saved_settings.get("LABEL_MODE", settings.LABEL_MODE))
    ckpt_epoch = int(ckpt.get("epoch", -1))

    if args.amp_precision is not None:
        amp_precision = args.amp_precision
    elif saved_settings.get("AMP_ENABLED", settings.AMP_ENABLED):
        amp_precision = str(saved_settings.get("AMP_PRECISION", settings.AMP_PRECISION))
    else:
        amp_precision = "fp32"
    amp_dtype = settings.get_amp_dtype(amp_precision)
    use_amp = amp_dtype is not None and device.type == "cuda"

    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    elif run_dir is not None:
        out_dir = run_dir / f"eval_{args.split}_{ckpt_path.stem}"
    else:
        out_dir = ALL_DIR / f"eval_{args.split}_{ckpt_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    if args.save_nii:
        pred_dir.mkdir(parents=True, exist_ok=True)

    case_ids = load_split_case_ids(Path(args.split_json), args.split, fold=args.fold)
    eval_ds = BraTS23PreprocessedDataset(
        data_dir=args.data_dir,
        case_ids=case_ids,
        is_train=False,
        label_mode=label_mode,
        cache_mode=args.cache_mode,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )

    model = build_model(label_mode, device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    inferer = SlidingWindowInferer(
        roi_size=tuple(int(x) for x in settings.INPUT_SIZE),
        sw_batch_size=args.sw_batch_size,
        overlap=args.sw_overlap,
        mode=args.sw_mode,
    )

    compute_hd95 = not args.no_hd95
    print(
        f"[EVAL] split={args.split} ({len(eval_ds)} cases) | ckpt={ckpt_path.name} (ep={ckpt_epoch}) | "
        f"label_mode={label_mode} | precision={amp_precision} | overlap={args.sw_overlap} | "
        f"mode={args.sw_mode} | tta={args.tta} | hd95={compute_hd95}"
    )

    records: List[Dict[str, Any]] = []
    case_times: List[float] = []
    is_tty = sys.stdout.isatty()
    pbar = tqdm(eval_loader, desc=f"Eval [{args.split}]", disable=not is_tty)
    log_interval = max(1, len(eval_loader) // 10) if len(eval_loader) > 0 else 1

    for idx, batch in enumerate(pbar, start=1):
        t_case = time.perf_counter()
        case_id = str(batch["case_id"][0])
        images = batch["image"].to(device, non_blocking=True)
        pkl_path = Path(args.data_dir) / f"{case_id}.pkl"
        spacing = extract_voxel_spacing(batch.get("meta"), pkl_path)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = predict_with_tta(inferer, images, model, use_tta=args.tta)

        logits_np = logits[0].float().cpu().numpy()
        pred_regions = predictions_to_region_masks(logits_np, label_mode=label_mode, from_logits=True)

        if label_mode == "regions":
            gt_regions = batch["label"][0].numpy() > 0.5
        else:
            gt_regions = multiclass_to_brats_regions(batch["label"][0].numpy())

        case_metrics = compute_case_metrics(
            pred_regions,
            gt_regions,
            voxel_spacing=spacing,
            empty_hd95_penalty=settings.EMPTY_HD95_PENALTY,
            compute_hd95=compute_hd95,
        )

        if args.save_nii:
            discrete_seg = logits_to_discrete_seg(logits_np, label_mode=label_mode)
            save_nifti_prediction(discrete_seg, pkl_path, pred_dir / f"{case_id}.nii.gz")

        case_sec = time.perf_counter() - t_case
        case_times.append(case_sec)

        row: Dict[str, Any] = {"case_id": case_id}
        for key, val in case_metrics.items():
            row[key] = round(float(val), 6) if "Dice" in key else (round(float(val), 4) if "HD95" in key else int(val))
        row["infer_sec"] = round(case_sec, 3)
        records.append(row)

        if is_tty:
            pbar.set_postfix(dice=f"{row['Mean_Dice']:.4f}")
        elif idx % log_interval == 0 or idx == len(eval_loader):
            eta = format_duration((sum(case_times) / len(case_times)) * (len(eval_loader) - idx))
            hd_txt = f" | HD95={row['Mean_HD95']:.2f}" if compute_hd95 else ""
            print(f"  [{idx}/{len(eval_loader)}] {case_id} | Mean_Dice={row['Mean_Dice']:.4f}{hd_txt} | ETA={eta}", flush=True)

    summary = aggregate_metrics(records)

    csv_path = out_dir / "per_case_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "num_cases": len(records),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt_epoch,
        "label_mode": label_mode,
        "amp_precision": amp_precision,
        "sw_batch_size": args.sw_batch_size,
        "sw_overlap": args.sw_overlap,
        "sw_mode": args.sw_mode,
        "tta": args.tta,
        "compute_hd95": compute_hd95,
        "total_sec": round(sum(case_times), 2),
        "mean_sec_per_case": round(sum(case_times) / max(1, len(case_times)), 3),
        "summary": summary,
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n================ EVALUATION SUMMARY ================")
    for metric_name in (*[f"{r}_Dice" for r in BRATS_REGIONS], "Mean_Dice"):
        st = summary.get(metric_name)
        if st:
            print(
                f"  {metric_name:10s}: mean={st['mean']:.4f} +/- {st['std']:.4f} | "
                f"median={st['median']:.4f} (IQR={st['iqr']:.4f})"
            )
    if compute_hd95:
        for metric_name in (*[f"{r}_HD95" for r in BRATS_REGIONS], "Mean_HD95"):
            st = summary.get(metric_name)
            if st:
                print(
                    f"  {metric_name:10s}: mean={st['mean']:.2f} +/- {st['std']:.2f} mm | "
                    f"median={st['median']:.2f} (IQR={st['iqr']:.2f})"
                )
    print(f"  Per-case CSV : {csv_path}")
    print(f"  Summary JSON : {json_path}")
    if args.save_nii:
        print(f"  Predictions  : {pred_dir}")
    print("====================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
