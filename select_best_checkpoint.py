#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-training validation selector for Top-K checkpoints (Dice + HD95 tie-break).

Evaluates all saved top-K checkpoints (and `best_model.pt` if not in top-K) on the
validation split with full BraTS Dice and HD95 metrics, then selects the checkpoint
with the lowest validation Mean HD95 among candidates within `HD95_TIEBREAK_TOLERANCE`
of the highest validation Mean Dice.

Saves:
  - `<run_dir>/checkpoints/best_model_selected.pt`
  - `<run_dir>/val_topk_selection.json`
  - `<run_dir>/val_topk_selection.csv`
"""

from __future__ import annotations

import os
os.environ["TMPDIR"] = "/tmp"
import tempfile
tempfile.tempdir = "/tmp"

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from torch.utils.data import DataLoader

ALL_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ALL_DIR / "script"
for p in (str(SCRIPT_DIR), str(ALL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import settings  # noqa: E402
from dataset import BraTS23PreprocessedDataset, _worker_init_fn, load_split_case_ids  # noqa: E402
from train import build_model, format_duration, resolve_device, validate  # noqa: E402


def discover_candidate_checkpoints(ckpt_dir: Path) -> List[Path]:
    """Collect unique candidate checkpoints from `topk/` and `best_model.pt`."""
    candidates: List[Path] = []
    topk_dir = ckpt_dir / "topk"
    if topk_dir.is_dir():
        candidates.extend(sorted(topk_dir.glob("*.pt")))

    best_path = ckpt_dir / "best_model.pt"
    if best_path.is_file():
        best_meta = torch.load(best_path, map_location="cpu", weights_only=False)
        best_ep = int(best_meta.get("epoch", -1))
        existing_eps = {
            int(torch.load(p, map_location="cpu", weights_only=False).get("epoch", -2))
            for p in candidates
        }
        if best_ep not in existing_eps:
            candidates.append(best_path)

    return candidates


def select_best_by_dice_hd95(
    records: List[Dict[str, Any]],
    tolerance: float,
) -> Dict[str, Any]:
    """Among checkpoints within `tolerance` of max Mean_Dice, pick lowest Mean_HD95."""
    if not records:
        raise ValueError("No candidate evaluation records to select from.")

    max_dice = max(r["Mean_Dice"] for r in records)
    eligible = [r for r in records if r["Mean_Dice"] >= (max_dice - tolerance)]
    eligible.sort(
        key=lambda r: (
            float("inf") if np.isnan(r["Mean_HD95"]) else r["Mean_HD95"],
            -r["Mean_Dice"],
        )
    )
    return eligible[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Top-K checkpoints on val set with HD95 and select best balanced model."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory under `all/runs/`.")
    parser.add_argument("--data-dir", type=Path, default=settings.DATA_DIR)
    parser.add_argument("--split-json", type=Path, default=settings.SPLIT_JSON)
    parser.add_argument("--fold", type=int, default=settings.FOLD)
    parser.add_argument("--cache-mode", choices=("ram", "mmap"), default="mmap")
    parser.add_argument("--num-workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--amp-precision", choices=("bf16", "fp16", "fp32"), default=settings.AMP_PRECISION)
    parser.add_argument("--device", type=str, default=settings.DEVICE)
    parser.add_argument("--tolerance", type=float, default=settings.HD95_TIEBREAK_TOLERANCE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_dir}")

    candidates = discover_candidate_checkpoints(ckpt_dir)
    if not candidates:
        raise RuntimeError(f"No candidate checkpoints found in {ckpt_dir}")

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for SegMamba SlidingWindowInferer evaluation.")

    config_path = run_dir / "config.json"
    label_mode = settings.LABEL_MODE
    if config_path.is_file():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        label_mode = cfg.get("cli_args", {}).get("label_mode", label_mode)

    val_ids = load_split_case_ids(Path(args.split_json), "val", fold=args.fold)
    val_ds = BraTS23PreprocessedDataset(
        data_dir=args.data_dir,
        case_ids=val_ids,
        is_train=False,
        label_mode=label_mode,
        cache_mode=args.cache_mode,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )

    model = build_model(label_mode, device)
    inferer = SlidingWindowInferer(
        roi_size=tuple(int(x) for x in settings.INPUT_SIZE),
        sw_batch_size=settings.SW_BATCH_SIZE,
        overlap=settings.SW_OVERLAP,
        mode=settings.SW_MODE,
    )
    amp_dtype = settings.get_amp_dtype(args.amp_precision) if settings.AMP_ENABLED else None

    print(f"[INFO] Evaluating {len(candidates)} candidate checkpoints on {len(val_ds)} val cases (tolerance={args.tolerance:.4f})...")
    results: List[Dict[str, Any]] = []
    cand_times: List[float] = []
    for idx, ckpt_path in enumerate(candidates, start=1):
        t_cand = time.perf_counter()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = int(ckpt.get("epoch", -1))

        summary = validate(
            model=model,
            loader=val_loader,
            inferer=inferer,
            device=device,
            amp_dtype=amp_dtype,
            label_mode=label_mode,
            compute_hd95=True,
            epoch=epoch,
        )
        cand_sec = time.perf_counter() - t_cand
        cand_times.append(cand_sec)
        eta_str = format_duration((sum(cand_times) / len(cand_times)) * (len(candidates) - idx))
        row = {
            "checkpoint": ckpt_path.name,
            "path": str(ckpt_path),
            "epoch": epoch,
            "Mean_Dice": round(summary["Mean_Dice"]["mean"], 6),
            "WT_Dice": round(summary["WT_Dice"]["mean"], 6),
            "TC_Dice": round(summary["TC_Dice"]["mean"], 6),
            "ET_Dice": round(summary["ET_Dice"]["mean"], 6),
            "Mean_HD95": round(summary["Mean_HD95"]["mean"], 4),
            "WT_HD95": round(summary["WT_HD95"]["mean"], 4),
            "TC_HD95": round(summary["TC_HD95"]["mean"], 4),
            "ET_HD95": round(summary["ET_HD95"]["mean"], 4),
        }
        results.append(row)
        print(
            f"  [{idx}/{len(candidates)}] {row['checkpoint']:28s} | ep={epoch:03d} | "
            f"Dice={row['Mean_Dice']:.4f} (WT={row['WT_Dice']:.4f}, TC={row['TC_Dice']:.4f}, ET={row['ET_Dice']:.4f}) | "
            f"HD95={row['Mean_HD95']:.2f} mm | {cand_sec:.1f}s | ETA={eta_str}"
        )

    winner = select_best_by_dice_hd95(results, tolerance=args.tolerance)
    selected_path = ckpt_dir / "best_model_selected.pt"
    shutil.copy2(winner["path"], selected_path)

    out_json = run_dir / "val_topk_selection.json"
    out_csv = run_dir / "val_topk_selection.csv"
    payload = {
        "tolerance": args.tolerance,
        "selected_checkpoint": str(selected_path),
        "winner": winner,
        "candidates": results,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(
        f"\n[SELECTED] {winner['checkpoint']} (ep={winner['epoch']}, "
        f"Mean_Dice={winner['Mean_Dice']:.4f}, Mean_HD95={winner['Mean_HD95']:.2f} mm)"
    )
    print(f"  - Saved selected checkpoint: {selected_path}")
    print(f"  - Selection report         : {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
