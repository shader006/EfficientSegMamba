#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast, paper-ready preprocessing pipeline for BraTS 2023 GLI (.nii.gz -> .npy + .pkl).

Design choices derived directly from our 1,251-case BraTS 2023 GLI dataset analysis:
1. Uncompressed `.npy` storage (`float16` or `float32` image, `uint8` seg) for maximum
   training throughput with RAM / `/dev/shm` caching (avoids `.npz` gzip overhead
   and NFS LMDB lock/mmap bottlenecks).
2. Union brain-foreground cropping (`data > 0` or `seg > 0`) to remove ~75-83% empty
   background voxels while preserving 100% of brain tissue and tumor regions.
3. Artifact-safe normalization per case & per channel:
   - Replace non-finite values, clip negative MRI artifacts to 0.
   - Optional foreground percentile clipping (default: [0.5, 99.5]) to suppress
     extreme intensity outliers (e.g., >1e6 values seen in raw t1c/t2w).
   - Per-channel Z-score normalization computed strictly on nonzero brain voxels,
     keeping background strictly at 0.0.
4. Pre-computed `class_locations` (up to N sampled voxel coordinates per label 1, 2, 3
   and WT) stored in `<case_id>.pkl` so training DataLoaders can perform fast
   foreground-oversampled 3D patch cropping without running `np.argwhere` per epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ALL_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = ALL_DIR.parent / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
DEFAULT_OUTPUT_DIR = ALL_DIR.parent / "data"
DEFAULT_MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def parse_patient_timepoint(case_id: str) -> Tuple[str, str]:
    """Extract patient_id and timepoint from `BraTS-GLI-XXXXX-YYY`."""
    match = re.match(r"^BraTS-GLI-(\d+)-(\d+)$", case_id)
    if match:
        return match.group(1), match.group(2)
    parts = case_id.split("-")
    return (parts[-2], parts[-1]) if len(parts) >= 4 else (case_id, "000")


def compute_nonzero_bbox(mask: np.ndarray, margin: int = 0) -> Tuple[Tuple[int, int], ...]:
    """Compute tight 3D bounding box [(x0, x1), (y0, y1), (z0, z1)] (end-exclusive)."""
    if not np.any(mask):
        return tuple((0, int(s)) for s in mask.shape)

    bbox: List[Tuple[int, int]] = []
    for axis, size in enumerate(mask.shape):
        reduce_axes = tuple(i for i in range(mask.ndim) if i != axis)
        idx = np.flatnonzero(np.any(mask, axis=reduce_axes))
        lo = max(0, int(idx[0]) - margin)
        hi = min(int(size), int(idx[-1]) + 1 + margin)
        bbox.append((lo, hi))
    return tuple(bbox)


def normalize_channel(
    vol: np.ndarray,
    brain_mask: np.ndarray,
    clip_low: float,
    clip_high: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Clip outliers and apply Z-score normalization strictly inside brain_mask."""
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.maximum(vol, 0.0)

    fg_vals = vol[brain_mask]
    if fg_vals.size == 0:
        return np.zeros_like(vol, dtype=np.float32), {
            "clip_lo_val": 0.0,
            "clip_hi_val": 0.0,
            "fg_mean": 0.0,
            "fg_std": 1.0,
        }

    lo_val = 0.0
    hi_val = float(fg_vals.max())
    if clip_low > 0.0 or clip_high < 100.0:
        lo_val, hi_val = np.percentile(fg_vals, [clip_low, clip_high]).astype(np.float64)
        fg_vals = np.clip(fg_vals, lo_val, hi_val)
        vol = np.where(brain_mask, np.clip(vol, lo_val, hi_val), 0.0)

    mean = float(fg_vals.mean())
    std = float(fg_vals.std())
    safe_std = std if std > 1e-8 else 1.0

    out = np.zeros_like(vol, dtype=np.float32)
    out[brain_mask] = (fg_vals - mean) / safe_std

    return out, {
        "clip_lo_val": float(lo_val),
        "clip_hi_val": float(hi_val),
        "fg_mean": mean,
        "fg_std": safe_std,
    }


def sample_class_locations(
    seg: np.ndarray,
    labels: Sequence[int] = (1, 2, 3),
    max_samples_per_class: int = 10000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Sample voxel coordinates per tumor label (and WT) for fast foreground cropping."""
    rng = np.random.default_rng(seed)
    locations: Dict[str, np.ndarray] = {}

    for lbl in labels:
        coords = np.argwhere(seg == lbl)
        if len(coords) > max_samples_per_class:
            idx = rng.choice(len(coords), size=max_samples_per_class, replace=False)
            coords = coords[idx]
        locations[f"label_{lbl}"] = coords.astype(np.int16)

    wt_coords = np.argwhere(seg > 0)
    if len(wt_coords) > max_samples_per_class:
        idx = rng.choice(len(wt_coords), size=max_samples_per_class, replace=False)
        wt_coords = wt_coords[idx]
    locations["wt"] = wt_coords.astype(np.int16)
    return locations


def preprocess_single_case(task: Dict[str, Any]) -> Dict[str, Any]:
    """Load raw NIfTI files for one case, crop, normalize, and write `.npy` + `.pkl`."""
    case_dir = Path(task["case_dir"])
    case_id = case_dir.name
    out_dir = Path(task["output_dir"])
    modalities: Tuple[str, ...] = tuple(task["modalities"])
    image_dtype = np.float16 if task["image_dtype"] == "float16" else np.float32

    img_out = out_dir / f"{case_id}.npy"
    seg_out = out_dir / f"{case_id}_seg.npy"
    pkl_out = out_dir / f"{case_id}.pkl"

    if task["skip_existing"] and img_out.is_file() and pkl_out.is_file():
        if not task["has_seg"] or seg_out.is_file():
            with pkl_out.open("rb") as f:
                meta = pickle.load(f)
            return {
                "case_id": case_id,
                "status": "skipped",
                "original_voxels": int(np.prod(meta["original_shape"])),
                "cropped_voxels": int(np.prod(meta["cropped_shape"])),
                "img_mb": img_out.stat().st_size / (1024.0 * 1024.0),
                "seg_mb": seg_out.stat().st_size / (1024.0 * 1024.0) if seg_out.is_file() else 0.0,
            }

    try:
        mod_arrays: List[np.ndarray] = []
        ref_affine: Optional[np.ndarray] = None
        ref_spacing: Optional[Tuple[float, ...]] = None
        raw_shape: Optional[Tuple[int, ...]] = None

        for mod in modalities:
            mod_path = case_dir / f"{case_id}-{mod}.nii.gz"
            if not mod_path.is_file():
                matches = sorted(case_dir.glob(f"*{mod}.nii.gz"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Missing modality {mod} in {case_dir}")
                mod_path = matches[0]

            nii = nib.load(str(mod_path))
            arr = np.asarray(nii.dataobj, dtype=np.float32)
            if raw_shape is None:
                raw_shape = tuple(int(x) for x in arr.shape)
                ref_affine = nii.affine.astype(np.float32)
                ref_spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])
            elif tuple(int(x) for x in arr.shape) != raw_shape:
                raise ValueError(f"Shape mismatch in {case_id} ({mod}): {arr.shape} vs {raw_shape}")
            mod_arrays.append(arr)

        assert raw_shape is not None
        seg_arr: Optional[np.ndarray] = None
        if task["has_seg"]:
            seg_path = case_dir / f"{case_id}-seg.nii.gz"
            if not seg_path.is_file():
                matches = sorted(case_dir.glob("*seg.nii.gz"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Missing seg mask in {case_dir}")
                seg_path = matches[0]
            seg_raw = np.asarray(nib.load(str(seg_path)).dataobj)
            seg_arr = np.rint(seg_raw).astype(np.uint8)
            if seg_arr.shape != raw_shape:
                raise ValueError(f"Seg shape mismatch in {case_id}: {seg_arr.shape} vs {raw_shape}")

        data = np.stack(mod_arrays, axis=0)  # (C, X, Y, Z)
        union_fg = np.any(data > 0, axis=0)
        if seg_arr is not None:
            union_fg |= seg_arr > 0

        bbox = compute_nonzero_bbox(union_fg, margin=task["crop_margin"])
        xs, ys, zs = (slice(lo, hi) for lo, hi in bbox)

        cropped_data = data[:, xs, ys, zs]
        cropped_seg = seg_arr[xs, ys, zs] if seg_arr is not None else None
        cropped_fg = union_fg[xs, ys, zs]

        channel_stats: Dict[str, Dict[str, float]] = {}
        norm_channels: List[np.ndarray] = []
        for idx, mod in enumerate(modalities):
            c_vol = cropped_data[idx]
            c_mask = (c_vol > 0) if np.any(c_vol > 0) else cropped_fg
            norm_vol, stats_dict = normalize_channel(
                c_vol,
                c_mask,
                clip_low=task["clip_low"],
                clip_high=task["clip_high"],
            )
            norm_channels.append(norm_vol)
            channel_stats[mod] = stats_dict

        final_img = np.stack(norm_channels, axis=0).astype(image_dtype, copy=False)
        np.save(img_out, final_img)

        seg_mb = 0.0
        class_locations: Dict[str, np.ndarray] = {}
        if cropped_seg is not None:
            np.save(seg_out, cropped_seg)
            seg_mb = seg_out.stat().st_size / (1024.0 * 1024.0)
            class_locations = sample_class_locations(
                cropped_seg,
                max_samples_per_class=task["max_fg_samples"],
                seed=task["seed"],
            )

        patient_id, timepoint = parse_patient_timepoint(case_id)
        metadata = {
            "case_id": case_id,
            "patient_id": patient_id,
            "timepoint": timepoint,
            "modalities": list(modalities),
            "image_dtype": str(final_img.dtype),
            "original_shape": raw_shape,
            "cropped_shape": tuple(int(x) for x in final_img.shape[1:]),
            "crop_bbox": bbox,
            "spacing": ref_spacing,
            "affine": ref_affine,
            "clip_percentiles": [task["clip_low"], task["clip_high"]],
            "channel_stats": channel_stats,
            "class_locations": class_locations,
        }
        with pkl_out.open("wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "case_id": case_id,
            "status": "ok",
            "original_voxels": int(np.prod(raw_shape)),
            "cropped_voxels": int(np.prod(final_img.shape[1:])),
            "img_mb": img_out.stat().st_size / (1024.0 * 1024.0),
            "seg_mb": seg_mb,
        }
    except Exception as exc:
        return {"case_id": case_id, "status": "error", "error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS 2023 GLI NIfTI data into fast cropped/normalized .npy + .pkl files."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modalities",
        nargs=4,
        default=list(DEFAULT_MODALITIES),
        help="Channel order for the 4 MRI modalities (default: t1n t1c t2w t2f).",
    )
    parser.add_argument(
        "--image-dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Output image .npy dtype (`float32` for max precision, `float16` for 2x smaller size; default: float32).",
    )
    parser.add_argument(
        "--clip-percentiles",
        nargs=2,
        type=float,
        default=[0.5, 99.5],
        metavar=("LOW", "HIGH"),
        help="Foreground intensity clipping percentiles per channel (use 0 100 to disable).",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=2,
        help="Extra voxel margin around brain bounding box (default: 2).",
    )
    parser.add_argument(
        "--max-fg-samples",
        type=int,
        default=10000,
        help="Max sampled coordinates per class stored in .pkl for fast patch sampling.",
    )
    parser.add_argument("--no-seg", action="store_true", help="Set for unlabeled ValidationData.")
    parser.add_argument("--skip-existing", action="store_true", help="Resume without overwriting existing files.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases (0 = all).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not data_dir.is_dir():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    case_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if args.max_cases > 0:
        case_dirs = case_dirs[: args.max_cases]

    output_dir.mkdir(parents=True, exist_ok=True)
    clip_low, clip_high = float(args.clip_percentiles[0]), float(args.clip_percentiles[1])

    tasks = [
        {
            "case_dir": str(d),
            "output_dir": str(output_dir),
            "modalities": list(args.modalities),
            "image_dtype": args.image_dtype,
            "clip_low": clip_low,
            "clip_high": clip_high,
            "crop_margin": int(args.crop_margin),
            "max_fg_samples": int(args.max_fg_samples),
            "has_seg": not args.no_seg,
            "skip_existing": bool(args.skip_existing),
            "seed": int(args.seed),
        }
        for d in case_dirs
    ]

    print(f"[INFO] Preprocessing {len(tasks)} cases from: {data_dir}")
    print(
        f"[INFO] Output: {output_dir} | modalities={args.modalities} | dtype={args.image_dtype} | "
        f"clip=[{clip_low}, {clip_high}] | workers={args.workers}"
    )

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(preprocess_single_case, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            results.append(fut.result())

    errors = [r for r in results if r["status"] == "error"]
    valid = [r for r in results if r["status"] in ("ok", "skipped")]
    orig_vox = sum(r["original_voxels"] for r in valid)
    crop_vox = sum(r["cropped_voxels"] for r in valid)
    img_gb = sum(r["img_mb"] for r in valid) / 1024.0
    seg_gb = sum(r["seg_mb"] for r in valid) / 1024.0
    reduction_pct = (1.0 - (crop_vox / orig_vox)) * 100.0 if orig_vox else 0.0

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "modalities": list(args.modalities),
        "image_dtype": args.image_dtype,
        "clip_percentiles": [clip_low, clip_high],
        "crop_margin": int(args.crop_margin),
        "total_cases": len(tasks),
        "processed_ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "errors": errors,
        "voxel_reduction_pct": round(reduction_pct, 2),
        "total_image_npy_gb": round(img_gb, 3),
        "total_seg_npy_gb": round(seg_gb, 3),
        "total_npy_gb": round(img_gb + seg_gb, 3),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    summary_path = output_dir / "preprocessing_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"\n[DONE] {summary['processed_ok']} ok, {summary['skipped']} skipped, "
        f"{len(errors)} errors in {summary['elapsed_sec']}s"
    )
    print(f"  - Voxel reduction from brain cropping : {reduction_pct:.2f}%")
    print(f"  - Total .npy size (img + seg)         : {img_gb + seg_gb:.2f} GB ({img_gb:.2f} + {seg_gb:.2f} GB)")
    print(f"  - Summary saved to                    : {summary_path}")
    if errors:
        print(f"[WARNING] Failed cases: {errors[:5]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
