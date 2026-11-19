#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard BraTS 2023 evaluation metrics (Dice & HD95) without CCA post-processing.

Evaluates the three clinically meaningful nested BraTS tumor sub-regions in the
exact channel order used by `all/script/dataset.py`:
  - Channel 0: WT (Whole Tumor  = labels 1 + 2 + 3)
  - Channel 1: TC (Tumor Core   = labels 1 + 3)
  - Channel 2: ET (Enhancing    = label 3)

Features:
- Supports both `label_mode="regions"` (3-channel logits/probs/masks) and
  `label_mode="multiclass"` (4-class logits/probs or discrete `{0,1,2,3}` masks).
- Exact BraTS empty-region handling:
    * GT empty & Pred empty -> Dice = 1.0, HD95 = 0.0
    * One empty & one non-empty -> Dice = 0.0, HD95 = `empty_hd95_penalty`
      (default: 373.13 mm = diagonal of 240x240x155 volume; pass `None` for NaN).
- Fast exact HD95 via tight joint bounding-box cropping around `pred | gt` before
  calling `medpy.metric.binary.hd95` (mathematically identical to full-volume HD95).
- `uncrop_to_original` helper to restore cropped predictions back to `240x240x155`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from medpy import metric as medpy_metric


BRATS_REGIONS: Tuple[str, ...] = ("WT", "TC", "ET")
BRATS_DEFAULT_HD95_PENALTY: float = 373.13  # sqrt(240^2 + 240^2 + 155^2) mm

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def multiclass_to_brats_regions(seg: ArrayLike) -> np.ndarray:
    """Convert discrete `{0,1,2,3}` mask `(X,Y,Z)` or `(B,X,Y,Z)` to bool regions `(...,3,X,Y,Z)`."""
    arr = _to_numpy(seg)
    wt = arr > 0
    tc = (arr == 1) | (arr == 3)
    et = arr == 3
    if arr.ndim == 3:
        return np.stack([wt, tc, et], axis=0)
    if arr.ndim == 4:
        return np.stack([wt, tc, et], axis=1)
    raise ValueError(f"Expected 3D or 4D discrete mask, got shape {arr.shape}")


def predictions_to_region_masks(
    pred: ArrayLike,
    label_mode: str = "regions",
    from_logits: bool = True,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert model predictions into boolean `(3, X, Y, Z)` or `(B, 3, X, Y, Z)` region masks.

    Args:
        pred: Predictions (`(3, X, Y, Z)` / `(B, 3, X, Y, Z)` for `regions`, or
            `(4, X, Y, Z)` / `(B, 4, X, Y, Z)` logits/probs or `(X, Y, Z)` / `(B, X, Y, Z)`
            discrete masks for `multiclass`).
        label_mode: `"regions"` or `"multiclass"`.
        from_logits: Apply sigmoid (for `regions`) before thresholding.
        threshold: Probability threshold for `regions` mode.
    """
    arr = _to_numpy(pred)
    if label_mode == "regions":
        if arr.ndim not in (4, 5):
            raise ValueError(f"Expected (3,X,Y,Z) or (B,3,X,Y,Z) for regions mode, got {arr.shape}")
        if from_logits:
            arr = 1.0 / (1.0 + np.exp(-np.clip(arr.astype(np.float32), -60.0, 60.0)))
        return arr >= threshold

    if label_mode == "multiclass":
        if arr.ndim == 3:
            return multiclass_to_brats_regions(arr)
        if arr.ndim == 5 and arr.shape[1] == 4:
            return multiclass_to_brats_regions(np.argmax(arr, axis=1))
        if arr.ndim == 5 and arr.shape[1] == 1:
            return multiclass_to_brats_regions(arr[:, 0])
        if arr.ndim == 4:
            if arr.shape[0] == 4 and (from_logits or np.issubdtype(arr.dtype, np.floating)):
                return multiclass_to_brats_regions(np.argmax(arr, axis=0))
            return multiclass_to_brats_regions(arr)
        raise ValueError(f"Unsupported multiclass prediction shape: {arr.shape}")

    raise ValueError(f"Unsupported label_mode: {label_mode}")


def _tight_union_bbox(a: np.ndarray, b: np.ndarray) -> Tuple[slice, slice, slice]:
    """Compute minimal 3D bounding-box slices covering `a | b` with 1-voxel margin."""
    union = a | b
    slices: List[slice] = []
    for axis, size in enumerate(union.shape):
        reduce_axes = tuple(i for i in range(3) if i != axis)
        idx = np.flatnonzero(np.any(union, axis=reduce_axes))
        lo = max(0, int(idx[0]) - 1)
        hi = min(int(size), int(idx[-1]) + 2)
        slices.append(slice(lo, hi))
    return (slices[0], slices[1], slices[2])


def compute_binary_dice_hd95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    empty_hd95_penalty: Optional[float] = BRATS_DEFAULT_HD95_PENALTY,
) -> Tuple[float, float]:
    """
    Compute BraTS-standard binary Dice and HD95 (in mm) for one 3D region.

    Returns:
        (dice, hd95)
    """
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    pred_sum = int(np.count_nonzero(pred))
    gt_sum = int(np.count_nonzero(gt))

    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 0.0
    if pred_sum == 0 or gt_sum == 0:
        hd_val = float("nan") if empty_hd95_penalty is None else float(empty_hd95_penalty)
        return 0.0, hd_val

    inter = int(np.count_nonzero(pred & gt))
    dice = (2.0 * inter) / (pred_sum + gt_sum)

    # Cropping to tight union bbox gives identical HD95 much faster than full volume EDT.
    xs, ys, zs = _tight_union_bbox(pred, gt)
    hd95 = float(
        medpy_metric.binary.hd95(
            pred[xs, ys, zs],
            gt[xs, ys, zs],
            voxelspacing=tuple(float(v) for v in voxel_spacing),
        )
    )
    return float(dice), hd95


def compute_case_metrics(
    pred_regions: ArrayLike,
    gt_regions: ArrayLike,
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    empty_hd95_penalty: Optional[float] = BRATS_DEFAULT_HD95_PENALTY,
    compute_hd95: bool = True,
) -> Dict[str, float]:
    """
    Compute per-region and mean BraTS metrics for a single 3D case `(3, X, Y, Z)`.

    Channel order must be `(WT, TC, ET)`.
    """
    pred = _to_numpy(pred_regions).astype(bool, copy=False)
    gt = _to_numpy(gt_regions).astype(bool, copy=False)
    if pred.shape != gt.shape or pred.ndim != 4 or pred.shape[0] != 3:
        raise ValueError(f"Expected (3,X,Y,Z) shapes, got pred={pred.shape}, gt={gt.shape}")

    out: Dict[str, float] = {}
    dice_vals: List[float] = []
    hd_vals: List[float] = []

    for idx, region in enumerate(BRATS_REGIONS):
        p_reg = pred[idx]
        g_reg = gt[idx]
        out[f"Pred_{region}_vox"] = float(np.count_nonzero(p_reg))
        out[f"GT_{region}_vox"] = float(np.count_nonzero(g_reg))

        if compute_hd95:
            d, hd = compute_binary_dice_hd95(
                p_reg,
                g_reg,
                voxel_spacing=voxel_spacing,
                empty_hd95_penalty=empty_hd95_penalty,
            )
            out[f"{region}_Dice"] = d
            out[f"{region}_HD95"] = hd
            hd_vals.append(hd)
        else:
            p_sum = int(out[f"Pred_{region}_vox"])
            g_sum = int(out[f"GT_{region}_vox"])
            if p_sum == 0 and g_sum == 0:
                d = 1.0
            elif p_sum == 0 or g_sum == 0:
                d = 0.0
            else:
                inter = int(np.count_nonzero(p_reg & g_reg))
                d = (2.0 * inter) / (p_sum + g_sum)
            out[f"{region}_Dice"] = float(d)

        dice_vals.append(out[f"{region}_Dice"])

    out["Mean_Dice"] = float(np.mean(dice_vals))
    if compute_hd95:
        valid_hd = [h for h in hd_vals if not np.isnan(h)]
        out["Mean_HD95"] = float(np.mean(valid_hd)) if valid_hd else float("nan")
    return out


def compute_batch_metrics(
    pred: ArrayLike,
    target: ArrayLike,
    label_mode: str = "regions",
    from_logits: bool = True,
    threshold: float = 0.5,
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    empty_hd95_penalty: Optional[float] = BRATS_DEFAULT_HD95_PENALTY,
    compute_hd95: bool = True,
) -> List[Dict[str, float]]:
    """Compute `compute_case_metrics` for every item in a batch."""
    pred_regions = predictions_to_region_masks(
        pred, label_mode=label_mode, from_logits=from_logits, threshold=threshold
    )
    gt_arr = _to_numpy(target)
    if label_mode == "regions":
        gt_regions = gt_arr > 0.5
    else:
        if gt_arr.ndim == 5 and gt_arr.shape[1] == 1:
            gt_arr = gt_arr[:, 0]
        gt_regions = multiclass_to_brats_regions(gt_arr)

    if pred_regions.ndim == 4:
        pred_regions = pred_regions[None]
        gt_regions = gt_regions[None]

    return [
        compute_case_metrics(
            pred_regions[b],
            gt_regions[b],
            voxel_spacing=voxel_spacing,
            empty_hd95_penalty=empty_hd95_penalty,
            compute_hd95=compute_hd95,
        )
        for b in range(pred_regions.shape[0])
    ]


def aggregate_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-case metric dicts into Mean, Std, Median, P25, P75, and IQR."""
    if not records:
        return {}

    metric_keys = [
        f"{r}_{m}"
        for r in (*BRATS_REGIONS, "Mean")
        for m in ("Dice", "HD95")
        if f"{r}_{m}" in records[0]
    ]

    summary: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        vals = np.array(
            [float(rec[key]) for rec in records if key in rec and not np.isnan(float(rec[key]))],
            dtype=np.float64,
        )
        if vals.size == 0:
            continue
        p25, med, p75 = np.percentile(vals, [25.0, 50.0, 75.0])
        summary[key] = {
            "count": int(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "median": float(med),
            "p25": float(p25),
            "p75": float(p75),
            "iqr": float(p75 - p25),
        }
    return summary


def uncrop_to_original(
    cropped: ArrayLike,
    crop_bbox: Sequence[Sequence[int]],
    original_shape: Sequence[int],
    fill_value: Union[int, float] = 0,
) -> np.ndarray:
    """Restore cropped volume `(X',Y',Z')` or `(C,X',Y',Z')` back to `original_shape`."""
    arr = _to_numpy(cropped)
    orig = tuple(int(x) for x in original_shape)
    bbox = [(int(lo), int(hi)) for lo, hi in crop_bbox]

    if arr.ndim == 3:
        out = np.full(orig, fill_value, dtype=arr.dtype)
        out[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]] = arr
        return out
    if arr.ndim == 4:
        out = np.full((arr.shape[0], *orig), fill_value, dtype=arr.dtype)
        out[:, bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]] = arr
        return out
    raise ValueError(f"Expected 3D or 4D cropped array, got shape {arr.shape}")


if __name__ == "__main__":
    # Built-in self-test: verify exactness vs full-volume medpy, edge cases, and uncrop.
    rng = np.random.default_rng(42)
    gt_seg = np.zeros((140, 175, 150), dtype=np.uint8)
    gt_seg[40:90, 50:110, 40:90] = 2
    gt_seg[55:80, 65:95, 55:75] = 1
    gt_seg[60:75, 70:88, 58:72] = 3

    # Pred with slight shift and missing ET on a second sample
    pred_seg = np.roll(gt_seg, shift=(2, -1, 1), axis=(0, 1, 2))
    gt_reg = multiclass_to_brats_regions(gt_seg)
    pred_reg = multiclass_to_brats_regions(pred_seg)

    fast_res = compute_case_metrics(pred_reg, gt_reg)
    for idx, reg in enumerate(BRATS_REGIONS):
        full_dc = medpy_metric.binary.dc(pred_reg[idx], gt_reg[idx])
        full_hd = medpy_metric.binary.hd95(pred_reg[idx], gt_reg[idx], voxelspacing=(1.0, 1.0, 1.0))
        assert abs(fast_res[f"{reg}_Dice"] - full_dc) < 1e-12
        assert abs(fast_res[f"{reg}_HD95"] - full_hd) < 1e-12

    # Edge case: both empty -> Dice=1.0, HD95=0.0; one empty -> Dice=0.0, HD95=penalty
    assert compute_binary_dice_hd95(np.zeros((16, 16, 16)), np.zeros((16, 16, 16))) == (1.0, 0.0)
    nz = np.zeros((16, 16, 16), dtype=bool)
    nz[4:8, 4:8, 4:8] = True
    assert compute_binary_dice_hd95(nz, np.zeros((16, 16, 16))) == (0.0, BRATS_DEFAULT_HD95_PENALTY)

    # Uncrop check
    restored = uncrop_to_original(gt_seg, ((50, 190), (38, 213), (2, 152)), (240, 240, 155))
    assert restored.shape == (240, 240, 155) and int(restored.sum()) == int(gt_seg.sum())

    summary = aggregate_metrics([fast_res])
    print("[SELF-TEST PASSED]")
    print("  Metrics:", {k: round(v["mean"], 4) for k, v in summary.items()})
