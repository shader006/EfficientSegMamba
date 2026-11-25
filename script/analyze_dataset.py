#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phân tích toàn diện bộ dữ liệu BraTS 2023 GLI TrainingData.

Script thực hiện kiểm tra và thống kê chi tiết trên từng ca (case) cũng như
tổng hợp toàn bộ tập dữ liệu:
1. Tính toàn vẹn file, dung lượng đĩa, định dạng tên file.
2. Thông số không gian NIfTI (shape, voxel spacing, orientation, dtype, tính nhất quán).
3. Cường độ tín hiệu 4 kênh MRI (t1n, t1c, t2w, t2f): tỉ lệ nền, phân vị foreground,
   giá trị bất thường (NaN/Inf/âm), tương phản cường độ theo từng vùng khối u.
4. Phân tích nhãn phân đoạn (0: BG, 1: NCR, 2: ED, 3: ET) và vùng gộp (WT, TC, ET):
   mất cân bằng lớp, ca thiếu nhãn, thể tích (voxels, cm³), tỉ lệ so với thể tích não.
5. Hình thái & vị trí không gian khối u: bounding box, số lát cắt axial có u, tâm khối u,
   phân tích thành phần liên thông 3D (CCA) cho WT và ET.
6. Đối chiếu độ cân bằng với file chia tập train/val/test (nếu có).
7. Xuất báo cáo CSV, JSON, Markdown tiếng Việt và các biểu đồ trực quan PNG.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ALL_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = str(ALL_DIR.parent / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "output_brats23_analysis")
DEFAULT_SPLIT_JSON = str(SCRIPT_DIR / "output_brats23_analysis" / "brats23_patient_split_70_10_20.json")

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
RAW_LABELS = {0: "BG", 1: "NCR", 2: "ED", 3: "ET"}
PERCENTILES = (1, 5, 50, 95, 99, 99.5)


def parse_patient_timepoint(case_id: str) -> Tuple[str, str]:
    """Tách patient_id và timepoint từ tên ca chuẩn BraTS-GLI-XXXXX-YYY."""
    match = re.match(r"^BraTS-GLI-(\d+)-(\d+)$", str(case_id))
    if match:
        return match.group(1), match.group(2)
    parts = str(case_id).split("-")
    if len(parts) >= 4:
        return parts[-2], parts[-1]
    return str(case_id), "000"


def resolve_case_file(case_dir: Path, case_id: str, suffix: str) -> Optional[Path]:
    """Tìm file NIfTI theo chuẩn BraTS 2023 (`<case_id>-<suffix>.nii.gz`)."""
    exact = case_dir / f"{case_id}-{suffix}.nii.gz"
    if exact.is_file():
        return exact
    matches = sorted(case_dir.glob(f"*{suffix}.nii.gz"))
    return matches[0] if len(matches) == 1 else None


def compute_cca_stats(mask: np.ndarray, prefix: str) -> Dict[str, Any]:
    """Tính thống kê thành phần liên thông 3D (6-connectivity) cho mask nhị phân."""
    if not np.any(mask):
        return {
            f"{prefix}_n_cc": 0,
            f"{prefix}_max_cc_vox": 0,
            f"{prefix}_min_cc_vox": 0,
            f"{prefix}_cc_lt50": 0,
            f"{prefix}_cc_lt100": 0,
            f"{prefix}_cc_lt200": 0,
        }
    labeled, n_cc = ndimage.label(mask)
    sizes = np.bincount(labeled.ravel())[1:]
    return {
        f"{prefix}_n_cc": int(n_cc),
        f"{prefix}_max_cc_vox": int(sizes.max()),
        f"{prefix}_min_cc_vox": int(sizes.min()),
        f"{prefix}_cc_lt50": int(np.sum(sizes < 50)),
        f"{prefix}_cc_lt100": int(np.sum(sizes < 100)),
        f"{prefix}_cc_lt200": int(np.sum(sizes < 200)),
    }


def compute_bbox_and_centroid(mask: np.ndarray) -> Dict[str, Any]:
    """Tính bounding box và tọa độ tâm chuẩn hóa của vùng khối u."""
    if not np.any(mask):
        return {
            "wt_bbox_x": 0,
            "wt_bbox_y": 0,
            "wt_bbox_z": 0,
            "wt_bbox_max_dim": 0,
            "wt_axial_slices": 0,
            "wt_centroid_norm_x": np.nan,
            "wt_centroid_norm_y": np.nan,
            "wt_centroid_norm_z": np.nan,
        }
    x_any = np.any(mask, axis=(1, 2))
    y_any = np.any(mask, axis=(0, 2))
    z_any = np.any(mask, axis=(0, 1))
    x_idx = np.flatnonzero(x_any)
    y_idx = np.flatnonzero(y_any)
    z_idx = np.flatnonzero(z_any)

    dim_x = int(x_idx[-1] - x_idx[0] + 1)
    dim_y = int(y_idx[-1] - y_idx[0] + 1)
    dim_z = int(z_idx[-1] - z_idx[0] + 1)

    total_vox = float(np.count_nonzero(mask))
    cx = float(np.sum(np.arange(mask.shape[0]) * np.sum(mask, axis=(1, 2))) / total_vox)
    cy = float(np.sum(np.arange(mask.shape[1]) * np.sum(mask, axis=(0, 2))) / total_vox)
    cz = float(np.sum(np.arange(mask.shape[2]) * np.sum(mask, axis=(0, 1))) / total_vox)
    return {
        "wt_bbox_x": dim_x,
        "wt_bbox_y": dim_y,
        "wt_bbox_z": dim_z,
        "wt_bbox_max_dim": max(dim_x, dim_y, dim_z),
        "wt_axial_slices": int(z_idx.size),
        "wt_centroid_norm_x": cx / float(mask.shape[0]),
        "wt_centroid_norm_y": cy / float(mask.shape[1]),
        "wt_centroid_norm_z": cz / float(mask.shape[2]),
    }


def analyze_single_case(case_dir_str: str) -> Dict[str, Any]:
    """Phân tích toàn bộ file NIfTI trong một thư mục ca bệnh."""
    case_dir = Path(case_dir_str)
    case_id = case_dir.name
    patient_id, timepoint = parse_patient_timepoint(case_id)
    record: Dict[str, Any] = {
        "case_id": case_id,
        "patient_id": patient_id,
        "timepoint": timepoint,
        "status": "ok",
        "error": "",
    }

    files: Dict[str, Optional[Path]] = {
        key: resolve_case_file(case_dir, case_id, key)
        for key in (*MODALITIES, "seg")
    }
    missing = [k for k, p in files.items() if p is None]
    record["missing_files"] = ",".join(missing)
    record["total_files_in_dir"] = len(list(case_dir.iterdir()))
    if missing:
        record["status"] = "missing_files"
        record["error"] = f"Thiếu file: {record['missing_files']}"
        return record

    try:
        total_mb = 0.0
        for key, path in files.items():
            assert path is not None
            mb = path.stat().st_size / (1024.0 * 1024.0)
            record[f"size_mb_{key}"] = round(mb, 4)
            total_mb += mb
        record["size_mb_total"] = round(total_mb, 4)

        seg_img = nib.load(str(files["seg"]))
        seg_arr = np.asarray(seg_img.dataobj)
        seg_shape = tuple(int(x) for x in seg_arr.shape)
        seg_zooms = tuple(round(float(z), 4) for z in seg_img.header.get_zooms()[:3])
        seg_axcodes = "".join(nib.aff2axcodes(seg_img.affine))
        voxel_vol_mm3 = float(np.prod(seg_zooms))

        record["shape"] = "x".join(map(str, seg_shape))
        record["dim_x"], record["dim_y"], record["dim_z"] = seg_shape
        record["spacing"] = "x".join(map(str, seg_zooms))
        record["voxel_vol_mm3"] = voxel_vol_mm3
        record["axcodes"] = seg_axcodes
        record["seg_dtype"] = str(seg_arr.dtype)
        record["seg_has_nan_inf"] = bool(not np.all(np.isfinite(seg_arr)))

        seg_int = np.rint(seg_arr).astype(np.int32)
        uniq_vals, uniq_counts = np.unique(seg_int, return_counts=True)
        count_map = dict(zip(uniq_vals.tolist(), uniq_counts.tolist()))
        unexpected_labels = sorted(k for k in count_map if k not in RAW_LABELS)
        record["unique_labels"] = ",".join(map(str, sorted(count_map.keys())))
        record["unexpected_labels"] = ",".join(map(str, unexpected_labels))

        total_voxels = int(seg_int.size)
        record["total_voxels"] = total_voxels

        ncr_mask = seg_int == 1
        ed_mask = seg_int == 2
        et_mask = seg_int == 3
        wt_mask = seg_int > 0
        tc_mask = ncr_mask | et_mask

        region_masks = {
            "bg": seg_int == 0,
            "ncr": ncr_mask,
            "ed": ed_mask,
            "et": et_mask,
            "wt": wt_mask,
            "tc": tc_mask,
        }
        for name, mask in region_masks.items():
            vox = int(np.count_nonzero(mask))
            record[f"{name}_voxels"] = vox
            record[f"{name}_cm3"] = (vox * voxel_vol_mm3) / 1000.0
            record[f"{name}_pct_vol"] = (vox / total_voxels) * 100.0

        record["has_ncr"] = record["ncr_voxels"] > 0
        record["has_ed"] = record["ed_voxels"] > 0
        record["has_et"] = record["et_voxels"] > 0
        record["has_wt"] = record["wt_voxels"] > 0
        record["et_to_wt_ratio"] = (
            record["et_voxels"] / record["wt_voxels"] if record["wt_voxels"] > 0 else 0.0
        )
        record["tc_to_wt_ratio"] = (
            record["tc_voxels"] / record["wt_voxels"] if record["wt_voxels"] > 0 else 0.0
        )

        record.update(compute_bbox_and_centroid(wt_mask))
        record.update(compute_cca_stats(wt_mask, "wt"))
        record.update(compute_cca_stats(et_mask, "et"))

        # Phân tích 4 kênh MRI và tạo brain foreground union mask
        brain_union_mask = np.zeros(seg_shape, dtype=bool)
        shape_mismatch = False
        spacing_mismatch = False
        axcodes_mismatch = False

        for mod in MODALITIES:
            mod_img = nib.load(str(files[mod]))
            mod_arr = np.asarray(mod_img.dataobj)
            mod_shape = tuple(int(x) for x in mod_arr.shape)
            mod_zooms = tuple(round(float(z), 4) for z in mod_img.header.get_zooms()[:3])
            mod_ax = "".join(nib.aff2axcodes(mod_img.affine))

            if mod_shape != seg_shape:
                shape_mismatch = True
            if mod_zooms != seg_zooms:
                spacing_mismatch = True
            if mod_ax != seg_axcodes:
                axcodes_mismatch = True

            record[f"{mod}_dtype"] = str(mod_arr.dtype)
            is_float = np.issubdtype(mod_arr.dtype, np.floating)
            if is_float:
                is_finite = np.isfinite(mod_arr)
                record[f"{mod}_has_nan_inf"] = bool(not np.all(is_finite))
                fg_mask = is_finite & (mod_arr > 0)
            else:
                record[f"{mod}_has_nan_inf"] = False
                fg_mask = mod_arr > 0

            record[f"{mod}_neg_voxels"] = int(np.count_nonzero(mod_arr < 0))
            brain_union_mask |= fg_mask
            fg_vals = mod_arr[fg_mask].astype(np.float32, copy=False)
            fg_count = int(fg_vals.size)

            record[f"{mod}_fg_ratio_pct"] = (fg_count / total_voxels) * 100.0
            record[f"{mod}_bg_zero_pct"] = 100.0 - record[f"{mod}_fg_ratio_pct"]

            if fg_count > 0:
                record[f"{mod}_fg_min"] = float(fg_vals.min())
                record[f"{mod}_fg_max"] = float(fg_vals.max())
                record[f"{mod}_fg_mean"] = float(fg_vals.mean())
                record[f"{mod}_fg_std"] = float(fg_vals.std())
                pct_vals = np.percentile(fg_vals, PERCENTILES)
                for p, val in zip(PERCENTILES, pct_vals):
                    p_key = str(p).replace(".", "_")
                    record[f"{mod}_fg_p{p_key}"] = float(val)
            else:
                for stat in ("min", "max", "mean", "std"):
                    record[f"{mod}_fg_{stat}"] = np.nan
                for p in PERCENTILES:
                    p_key = str(p).replace(".", "_")
                    record[f"{mod}_fg_p{p_key}"] = np.nan

            # Trung bình cường độ bên trong từng vùng u (giúp đánh giá tương phản)
            fg_mean = record[f"{mod}_fg_mean"]
            for reg in ("ncr", "ed", "et", "wt"):
                reg_mask = region_masks[reg]
                if np.any(reg_mask):
                    reg_mean = float(mod_arr[reg_mask].mean())
                    record[f"{mod}_{reg}_mean"] = reg_mean
                    record[f"{mod}_{reg}_to_fg_ratio"] = (
                        reg_mean / fg_mean if fg_mean and fg_mean > 0 else np.nan
                    )
                else:
                    record[f"{mod}_{reg}_mean"] = np.nan
                    record[f"{mod}_{reg}_to_fg_ratio"] = np.nan

        record["shape_mismatch"] = shape_mismatch
        record["spacing_mismatch"] = spacing_mismatch
        record["axcodes_mismatch"] = axcodes_mismatch

        brain_vox = int(np.count_nonzero(brain_union_mask))
        record["brain_voxels"] = brain_vox
        record["brain_cm3"] = (brain_vox * voxel_vol_mm3) / 1000.0
        record["wt_pct_of_brain"] = (
            (record["wt_voxels"] / brain_vox) * 100.0 if brain_vox > 0 else np.nan
        )

    except Exception as exc:  # pragma: no cover - defensive error capture per case
        record["status"] = "error"
        record["error"] = str(exc)

    return record


def summarize_series(s: pd.Series) -> Dict[str, float]:
    """Tính thống kê mô tả chuẩn cho một cột số."""
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty:
        return {}
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.count() > 1 else 0.0,
        "min": float(clean.min()),
        "p05": float(clean.quantile(0.05)),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def aggregate_summary(
    df: pd.DataFrame, data_dir: Path, split_json_path: Optional[Path]
) -> Dict[str, Any]:
    """Tổng hợp các chỉ số toàn tập từ DataFrame từng ca."""
    ok_df = df[df["status"] == "ok"].copy()
    if "patient_id" not in ok_df.columns or "timepoint" not in ok_df.columns:
        parsed = ok_df["case_id"].map(parse_patient_timepoint)
        ok_df["patient_id"] = [p[0] for p in parsed]
        ok_df["timepoint"] = [p[1] for p in parsed]
    else:
        ok_df["patient_id"] = ok_df["patient_id"].astype(str).str.zfill(5)
        ok_df["timepoint"] = ok_df["timepoint"].astype(str).str.zfill(3)

    scans_per_patient = ok_df["patient_id"].value_counts()
    scans_dist = {
        f"{int(k)}_scan": int(v)
        for k, v in scans_per_patient.value_counts().sort_index().items()
    }
    timepoint_counts = {
        str(k): int(v) for k, v in ok_df["timepoint"].value_counts().sort_index().items()
    }

    # Mất cân bằng lớp trên toàn tập (tính trên tổng voxel cộng dồn)
    total_vox_all = int(ok_df["total_voxels"].sum())
    total_brain_vox = int(ok_df["brain_voxels"].sum())
    class_totals = {
        name: int(ok_df[f"{name}_voxels"].sum())
        for name in ("bg", "ncr", "ed", "et", "wt", "tc")
    }
    class_imbalance = {
        name: {
            "total_voxels": vox,
            "pct_of_total_volume": (vox / total_vox_all * 100.0) if total_vox_all else 0.0,
            "pct_of_brain_volume": (
                None if name == "bg" else ((vox / total_brain_vox * 100.0) if total_brain_vox else 0.0)
            ),
        }
        for name, vox in class_totals.items()
    }

    # Phân tích cặp chụp lặp theo thời gian (-000 vs -001)
    tp0 = ok_df[ok_df["timepoint"] == "000"].set_index("patient_id")
    tp1 = ok_df[ok_df["timepoint"] == "001"].set_index("patient_id")
    common_pids = tp0.index.intersection(tp1.index)
    longitudinal: Dict[str, Any] = {"paired_patients": int(len(common_pids))}
    if len(common_pids) > 0:
        for reg in ("wt_cm3", "tc_cm3", "et_cm3"):
            abs_diff = (tp1.loc[common_pids, reg] - tp0.loc[common_pids, reg]).abs()
            longitudinal[f"{reg}_abs_diff"] = summarize_series(abs_diff)

    # Phân nhóm độ khó theo kích thước và số ổ khối u
    q33, q66 = np.quantile(ok_df["wt_cm3"].values, [1.0 / 3.0, 2.0 / 3.0])
    subgroups = {
        "wt_tertile_cutoffs_cm3": {"q33": float(q33), "q66": float(q66)},
        "wt_small_le_q33": int((ok_df["wt_cm3"] <= q33).sum()),
        "wt_medium_q33_to_q66": int(((ok_df["wt_cm3"] > q33) & (ok_df["wt_cm3"] <= q66)).sum()),
        "wt_large_gt_q66": int((ok_df["wt_cm3"] > q66).sum()),
        "wt_lt_20cm3": int((ok_df["wt_cm3"] < 20.0).sum()),
        "et_lt_2cm3_nonzero": int(((ok_df["et_voxels"] > 0) & (ok_df["et_cm3"] < 2.0)).sum()),
        "wt_multifocal_cc_gt1": int((ok_df["wt_n_cc"] > 1).sum()),
    }

    summary: Dict[str, Any] = {
        "dataset_path": str(data_dir),
        "patients": {
            "total_cases": int(len(ok_df)),
            "unique_patients": int(ok_df["patient_id"].nunique()),
            "scans_per_patient_dist": scans_dist,
            "timepoint_counts": timepoint_counts,
            "longitudinal_pairs": longitudinal,
        },
        "subgroups": subgroups,
        "integrity": {
            "total_case_dirs": int(len(df)),
            "valid_cases": int(len(ok_df)),
            "missing_file_cases": int((df["status"] == "missing_files").sum()),
            "error_cases": int((df["status"] == "error").sum()),
            "total_disk_gb": float(ok_df["size_mb_total"].sum() / 1024.0),
            "avg_case_size_mb": float(ok_df["size_mb_total"].mean()),
            "shape_mismatches": int(ok_df["shape_mismatch"].sum()),
            "spacing_mismatches": int(ok_df["spacing_mismatch"].sum()),
            "axcodes_mismatches": int(ok_df["axcodes_mismatch"].sum()),
            "seg_nan_inf_cases": int(ok_df["seg_has_nan_inf"].sum()),
            "unexpected_label_cases": int((ok_df["unexpected_labels"] != "").sum()),
        },
        "geometry": {
            "shapes": ok_df["shape"].value_counts().to_dict(),
            "spacings": ok_df["spacing"].value_counts().to_dict(),
            "axcodes": ok_df["axcodes"].value_counts().to_dict(),
            "seg_dtypes": ok_df["seg_dtype"].value_counts().to_dict(),
            "unique_label_sets": ok_df["unique_labels"].value_counts().to_dict(),
        },
        "missing_tumor_regions": {
            "no_wt_cases": int((~ok_df["has_wt"]).sum()),
            "no_ncr_cases": int((~ok_df["has_ncr"]).sum()),
            "no_ed_cases": int((~ok_df["has_ed"]).sum()),
            "no_et_cases": int((~ok_df["has_et"]).sum()),
            "no_et_pct": float((~ok_df["has_et"]).mean() * 100.0),
            "no_ncr_pct": float((~ok_df["has_ncr"]).mean() * 100.0),
            "no_ed_pct": float((~ok_df["has_ed"]).mean() * 100.0),
        },
        "class_imbalance": class_imbalance,
        "volumes_cm3": {
            reg: summarize_series(ok_df[f"{reg}_cm3"])
            for reg in ("brain", "wt", "tc", "et", "ncr", "ed")
        },
        "ratios": {
            "wt_pct_of_brain": summarize_series(ok_df["wt_pct_of_brain"]),
            "et_to_wt_ratio": summarize_series(ok_df["et_to_wt_ratio"]),
            "tc_to_wt_ratio": summarize_series(ok_df["tc_to_wt_ratio"]),
        },
        "morphology_and_bbox": {
            col: summarize_series(ok_df[col])
            for col in (
                "wt_bbox_x",
                "wt_bbox_y",
                "wt_bbox_z",
                "wt_bbox_max_dim",
                "wt_axial_slices",
                "wt_n_cc",
                "et_n_cc",
                "et_max_cc_vox",
                "et_min_cc_vox",
            )
        },
        "cca_insights": {
            "cases_wt_multi_cc": int((ok_df["wt_n_cc"] > 1).sum()),
            "cases_et_has_cc_lt50": int((ok_df["et_cc_lt50"] > 0).sum()),
            "cases_et_has_cc_lt100": int((ok_df["et_cc_lt100"] > 0).sum()),
            "cases_et_has_cc_lt200": int((ok_df["et_cc_lt200"] > 0).sum()),
            "cases_et_max_cc_lt100_nonzero": int(
                ((ok_df["et_voxels"] > 0) & (ok_df["et_max_cc_vox"] < 100)).sum()
            ),
            "patch_coverage": {
                f"le_{p}": float((ok_df["wt_bbox_max_dim"] <= p).mean() * 100.0)
                for p in (64, 96, 128, 160)
            },
        },
        "modalities": {},
    }

    for mod in MODALITIES:
        summary["modalities"][mod] = {
            "dtypes": ok_df[f"{mod}_dtype"].value_counts().to_dict(),
            "nan_inf_cases": int(ok_df[f"{mod}_has_nan_inf"].sum()),
            "cases_with_negative_voxels": int((ok_df[f"{mod}_neg_voxels"] > 0).sum()),
            "fg_ratio_pct": summarize_series(ok_df[f"{mod}_fg_ratio_pct"]),
            "fg_mean": summarize_series(ok_df[f"{mod}_fg_mean"]),
            "fg_std": summarize_series(ok_df[f"{mod}_fg_std"]),
            "fg_p50": summarize_series(ok_df[f"{mod}_fg_p50"]),
            "fg_p99": summarize_series(ok_df[f"{mod}_fg_p99"]),
            "fg_p99_5": summarize_series(ok_df[f"{mod}_fg_p99_5"]),
            "fg_max": summarize_series(ok_df[f"{mod}_fg_max"]),
            "contrast_to_fg": {
                reg: summarize_series(ok_df[f"{mod}_{reg}_to_fg_ratio"])
                for reg in ("ncr", "ed", "et", "wt")
            },
        }

    # Đối chiếu file split 70/10/20 nếu tồn tại
    if split_json_path and split_json_path.is_file():
        with open(split_json_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        split_summary: Dict[str, Any] = {
            "split_file": str(split_json_path),
            "splits": {},
            "patient_overlap": {},
            "statistical_tests": {"continuous_kruskal_wallis": {}, "categorical_chi2": {}},
        }
        split_pats: Dict[str, set] = {}
        split_dfs: Dict[str, pd.DataFrame] = {}
        multi_pids = set(scans_per_patient[scans_per_patient > 1].index.tolist())
        ok_df["is_multi_scan"] = ok_df["patient_id"].isin(multi_pids)

        for split_name in ("train", "val", "test"):
            raw_items = split_data.get(split_name, [])
            ids = [
                item["id"] if isinstance(item, dict) and "id" in item else Path(str(item)).name
                for item in raw_items
            ]
            sub = ok_df[ok_df["case_id"].isin(ids)].copy()
            split_dfs[split_name] = sub
            pats = set(sub["patient_id"].tolist())
            split_pats[split_name] = pats
            split_summary["splits"][split_name] = {
                "expected_count": len(ids),
                "matched_count": int(len(sub)),
                "unique_patients": int(len(pats)),
                "multi_scan_cases": int(sub["is_multi_scan"].sum()),
                "multi_scan_pct": float(sub["is_multi_scan"].mean() * 100.0) if len(sub) else 0.0,
                "wt_cm3_mean": float(sub["wt_cm3"].mean()) if len(sub) else 0.0,
                "wt_cm3_median": float(sub["wt_cm3"].median()) if len(sub) else 0.0,
                "tc_cm3_mean": float(sub["tc_cm3"].mean()) if len(sub) else 0.0,
                "et_cm3_mean": float(sub["et_cm3"].mean()) if len(sub) else 0.0,
                "no_et_pct": float((~sub["has_et"]).mean() * 100.0) if len(sub) else 0.0,
                "no_ncr_pct": float((~sub["has_ncr"]).mean() * 100.0) if len(sub) else 0.0,
                "volumes_cm3": {
                    reg: summarize_series(sub[f"{reg}_cm3"])
                    for reg in ("brain", "wt", "tc", "et", "ncr", "ed")
                },
            }
        split_summary["patient_overlap"] = {
            "train_val": int(len(split_pats["train"] & split_pats["val"])),
            "train_test": int(len(split_pats["train"] & split_pats["test"])),
            "val_test": int(len(split_pats["val"] & split_pats["test"])),
        }
        groups = [split_dfs[s] for s in ("train", "val", "test") if len(split_dfs[s]) > 0]
        if len(groups) == 3:
            for col in ("brain_cm3", "wt_cm3", "tc_cm3", "et_cm3", "ncr_cm3", "ed_cm3"):
                stat, pval = stats.kruskal(*(g[col].dropna().values for g in groups))
                split_summary["statistical_tests"]["continuous_kruskal_wallis"][col] = {
                    "statistic": float(stat),
                    "p_value": float(pval),
                }
            combined = pd.concat(
                [split_dfs[s].assign(split=s) for s in ("train", "val", "test")],
                ignore_index=True,
            )
            for col in ("has_et", "has_ncr", "is_multi_scan"):
                ct = pd.crosstab(combined["split"], combined[col])
                chi2, pval, _, _ = stats.chi2_contingency(ct)
                split_summary["statistical_tests"]["categorical_chi2"][col] = {
                    "statistic": float(chi2),
                    "p_value": float(pval),
                }
        summary["split_comparison"] = split_summary

    return summary


def generate_markdown_report(summary: Dict[str, Any], out_path: Path) -> None:
    """Sinh báo cáo phân tích tổng hợp dạng Markdown tiếng Việt."""
    integ = summary["integrity"]
    pats = summary["patients"]
    geom = summary["geometry"]
    miss = summary["missing_tumor_regions"]
    imb = summary["class_imbalance"]
    vols = summary["volumes_cm3"]
    cca = summary["cca_insights"]
    bbox = summary["morphology_and_bbox"]

    lines: List[str] = [
        "# Báo Cáo Phân Tích Tập Dữ Liệu BraTS 2023 GLI TrainingData",
        "",
        f"- **Đường dẫn dữ liệu:** `{summary['dataset_path']}`",
        f"- **Tổng số ca hợp lệ:** **{integ['valid_cases']} / {integ['total_case_dirs']}** ca "
        f"(thiếu file: {integ['missing_file_cases']}, lỗi đọc: {integ['error_cases']})",
        f"- **Số lượng bệnh nhân riêng biệt:** **{pats['unique_patients']}** bệnh nhân "
        f"(phân bố số lần chụp: `{pats['scans_per_patient_dist']}`, timepoints: `{pats['timepoint_counts']}`)",
        f"- **Tổng dung lượng trên đĩa (5 file `.nii.gz`/ca):** **{integ['total_disk_gb']:.2f} GB** "
        f"(trung bình {integ['avg_case_size_mb']:.2f} MB/ca)",
        "",
        "## 1. Thông số Không gian & Tính nhất quán NIfTI",
        f"- **Kích thước ảnh (Shape):** `{geom['shapes']}`",
        f"- **Voxel spacing (mm):** `{geom['spacings']}`",
        f"- **Hệ trục (Orientation):** `{geom['axcodes']}`",
        f"- **Kiểu dữ liệu mask (`seg`):** `{geom['seg_dtypes']}`",
        f"- **Các tập giá trị nhãn xuất hiện:** `{geom['unique_label_sets']}`",
        f"- **Lỗi không nhất quán nội tại ca:** shape mismatch = {integ['shape_mismatches']}, "
        f"spacing mismatch = {integ['spacing_mismatches']}, axcodes mismatch = {integ['axcodes_mismatches']}, "
        f"nhãn lạ ngoài {{0,1,2,3}} = {integ['unexpected_label_cases']}",
        "",
        "## 2. Mất cân bằng lớp (Class Imbalance) & Ca thiếu nhãn",
        "| Thành phần | Tổng số voxels | % Toàn volume | % Thể tích não (foreground) |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("bg", "Background (0)"),
        ("ncr", "NCR (1)"),
        ("ed", "ED (2)"),
        ("et", "ET (3)"),
        ("wt", "WT (1+2+3)"),
        ("tc", "TC (1+3)"),
    ]:
        info = imb[key]
        brain_pct_str = (
            "-" if info["pct_of_brain_volume"] is None else f"{info['pct_of_brain_volume']:.3f}%"
        )
        lines.append(
            f"| {label} | {info['total_voxels']:,} | "
            f"{info['pct_of_total_volume']:.3f}% | {brain_pct_str} |"
        )

    lines.extend(
        [
            "",
            f"- **Số ca không có Whole Tumor (WT = 0):** {miss['no_wt_cases']}",
            f"- **Số ca thiếu Enhancing Tumor (ET = 0):** **{miss['no_et_cases']}** ({miss['no_et_pct']:.2f}%)",
            f"- **Số ca thiếu Necrotic Core (NCR = 0):** **{miss['no_ncr_cases']}** ({miss['no_ncr_pct']:.2f}%)",
            f"- **Số ca thiếu Edema (ED = 0):** **{miss['no_ed_cases']}** ({miss['no_ed_pct']:.2f}%)",
            "",
            "## 3. Phân bố Thể tích (cm³)",
            "| Vùng | Mean ± Std | Median (IQR: P25 - P75) | Min - Max |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for key, label in [
        ("brain", "Não (Union Foreground)"),
        ("wt", "Whole Tumor (WT)"),
        ("tc", "Tumor Core (TC)"),
        ("et", "Enhancing Tumor (ET)"),
        ("ncr", "NCR (Label 1)"),
        ("ed", "Edema (Label 2)"),
    ]:
        v = vols[key]
        lines.append(
            f"| {label} | {v['mean']:.2f} ± {v['std']:.2f} | "
            f"{v['median']:.2f} ({v['p25']:.2f} - {v['p75']:.2f}) | "
            f"{v['min']:.2f} - {v['max']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Hình thái khối u, Bounding Box & Thành phần liên thông (CCA)",
            f"- **Bounding box WT (X × Y × Z) trung vị:** "
            f"{bbox['wt_bbox_x']['median']:.0f} × {bbox['wt_bbox_y']['median']:.0f} × {bbox['wt_bbox_z']['median']:.0f} voxels "
            f"(P95: {bbox['wt_bbox_x']['p95']:.0f} × {bbox['wt_bbox_y']['p95']:.0f} × {bbox['wt_bbox_z']['p95']:.0f})",
            f"- **Số lát cắt axial chứa u (WT):** trung vị {bbox['wt_axial_slices']['median']:.0f} "
            f"(min {bbox['wt_axial_slices']['min']:.0f}, max {bbox['wt_axial_slices']['max']:.0f})",
            "- **Tỉ lệ ca có toàn bộ khối u nằm gọn trong kích thước patch vuông:** "
            + ", ".join(f"`{k}`: {v:.1f}%" for k, v in cca["patch_coverage"].items()),
            f"- **WT có nhiều hơn 1 thành phần liên thông:** {cca['cases_wt_multi_cc']} ca",
            f"- **Phân tích cụm nhỏ vùng ET (liên quan hậu xử lý CCA):** "
            f"có cụm < 50 vox: {cca['cases_et_has_cc_lt50']} ca; "
            f"< 100 vox: {cca['cases_et_has_cc_lt100']} ca; "
            f"< 200 vox: {cca['cases_et_has_cc_lt200']} ca; "
            f"ca mà cụm ET lớn nhất < 100 vox (rủi ro bị xóa sạch nếu CCA ngưỡng 100): **{cca['cases_et_max_cc_lt100_nonzero']} ca**",
            "",
            "## 5. Thống kê Cường độ tín hiệu 4 Kênh MRI (trên vùng Foreground > 0)",
            "| Kênh | Dtype | % Voxel nền (=0) | Mean ± Std | Median | P99 | P99.5 | Max | Tỉ lệ tương phản ET/FG | Tỉ lệ tương phản ED/FG |",
            "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mod in MODALITIES:
        m = summary["modalities"][mod]
        bg_pct = 100.0 - m["fg_ratio_pct"]["mean"]
        et_r = m["contrast_to_fg"]["et"]["median"]
        ed_r = m["contrast_to_fg"]["ed"]["median"]
        dtype_str = "/".join(m["dtypes"].keys())
        lines.append(
            f"| `{mod}` | {dtype_str} | {bg_pct:.2f}% | "
            f"{m['fg_mean']['mean']:.1f} ± {m['fg_std']['mean']:.1f} | "
            f"{m['fg_p50']['median']:.1f} | {m['fg_p99']['median']:.1f} | "
            f"{m['fg_p99_5']['median']:.1f} | {m['fg_max']['max']:.1f} | "
            f"{et_r:.2f}x | {ed_r:.2f}x |"
        )

    if "split_comparison" in summary:
        sc = summary["split_comparison"]
        ov = sc.get("patient_overlap", {})
        st = sc.get("statistical_tests", {})
        kw = st.get("continuous_kruskal_wallis", {})
        ch = st.get("categorical_chi2", {})
        lines.extend(
            [
                "",
                f"## 6. Kiểm tra độ cân bằng của Split (`{Path(sc['split_file']).name}`)",
                "| Tập | Số ca | Số bệnh nhân | WT Mean (cm³) | WT Median (cm³) | TC Mean (cm³) | ET Mean (cm³) | % Ca thiếu ET |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for sname in ("train", "val", "test"):
            sp = sc["splits"][sname]
            lines.append(
                f"| `{sname}` | {sp['matched_count']}/{sp['expected_count']} | "
                f"{sp.get('unique_patients', '-')} | "
                f"{sp['wt_cm3_mean']:.2f} | {sp['wt_cm3_median']:.2f} | "
                f"{sp['tc_cm3_mean']:.2f} | {sp['et_cm3_mean']:.2f} | {sp['no_et_pct']:.2f}% |"
            )
        if ov:
            lines.append(
                f"\n- **Kiểm tra trùng bệnh nhân giữa các tập:** "
                f"`train ∩ val` = **{ov['train_val']}**, "
                f"`train ∩ test` = **{ov['train_test']}**, "
                f"`val ∩ test` = **{ov['val_test']}**"
            )
        if kw or ch:
            kw_str = ", ".join(f"`{k}`: p={v['p_value']:.4f}" for k, v in kw.items())
            ch_str = ", ".join(f"`{k}`: p={v['p_value']:.4f}" for k, v in ch.items())
            lines.append(f"- **Kiểm định đồng nhất Kruskal-Wallis:** {kw_str}")
            lines.append(f"- **Kiểm định đồng nhất Chi-square:** {ch_str}")

    lp = summary["patients"].get("longitudinal_pairs", {})
    sg = summary.get("subgroups", {})
    lines.extend(
        [
            "",
            "## 7. Phân tích Cặp chụp lặp & Nhóm độ khó (Chuẩn bị cho Paper)",
            f"- **Số bệnh nhân có 2 lần chụp (`-000` & `-001`):** **{lp.get('paired_patients', 0)}** bệnh nhân",
        ]
    )
    if lp.get("paired_patients", 0) > 0:
        lines.append(
            f"  - Biến thiên tuyệt đối giữa `-000` và `-001` (Median [IQR]): "
            f"WT = {lp['wt_cm3_abs_diff']['median']:.2f} ({lp['wt_cm3_abs_diff']['p25']:.2f}–{lp['wt_cm3_abs_diff']['p75']:.2f}) cm³, "
            f"ET = {lp['et_cm3_abs_diff']['median']:.2f} ({lp['et_cm3_abs_diff']['p25']:.2f}–{lp['et_cm3_abs_diff']['p75']:.2f}) cm³"
        )
    if sg:
        cuts = sg["wt_tertile_cutoffs_cm3"]
        lines.extend(
            [
                f"- **Phân tầng kích thước WT theo Tertiles:** Small (≤ {cuts['q33']:.2f} cm³): {sg['wt_small_le_q33']} ca | "
                f"Medium ({cuts['q33']:.2f}–{cuts['q66']:.2f} cm³): {sg['wt_medium_q33_to_q66']} ca | "
                f"Large (> {cuts['q66']:.2f} cm³): {sg['wt_large_gt_q66']} ca",
                f"- **Nhóm ca khó:** WT < 20 cm³ = {sg['wt_lt_20cm3']} ca | "
                f"ET nhỏ (< 2 cm³, > 0) = {sg['et_lt_2cm3_nonzero']} ca | "
                f"WT đa ổ (> 1 cụm liên thông) = {sg['wt_multifocal_cc_gt1']} ca",
            ]
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_latex_table1(summary: Dict[str, Any], out_path: Path) -> None:
    """Xuất Table 1 chuẩn LaTeX (booktabs) cho phần Dataset của bài báo."""
    if "split_comparison" not in summary:
        return
    sc = summary["split_comparison"]
    st = sc.get("statistical_tests", {})
    kw = st.get("continuous_kruskal_wallis", {})
    ch = st.get("categorical_chi2", {})
    pats = summary["patients"]
    vols_all = summary["volumes_cm3"]
    miss_all = summary["missing_tumor_regions"]
    multi_all = int(sum(sp.get("multi_scan_cases", 0) for sp in sc["splits"].values()))
    total_cases = pats["total_cases"]
    multi_pct_all = (multi_all / total_cases * 100.0) if total_cases else 0.0

    def fmt_iqr(v: Dict[str, float]) -> str:
        return f"{v['median']:.2f} [{v['p25']:.2f}--{v['p75']:.2f}]"

    def fmt_p(p: Optional[float]) -> str:
        if p is None:
            return "--"
        return "$<0.001$" if p < 0.001 else f"{p:.3f}"

    rows = [
        (
            "Unique patients ($n$)",
            str(pats["unique_patients"]),
            *(str(sc["splits"][s]["unique_patients"]) for s in ("train", "val", "test")),
            "--",
        ),
        (
            "MRI cases / scans ($n$)",
            str(total_cases),
            *(str(sc["splits"][s]["matched_count"]) for s in ("train", "val", "test")),
            "--",
        ),
        (
            "Multi-timepoint cases (\\%)",
            f"{multi_all} ({multi_pct_all:.2f}\\%)",
            *(
                f"{sc['splits'][s]['multi_scan_cases']} ({sc['splits'][s]['multi_scan_pct']:.2f}\\%)"
                for s in ("train", "val", "test")
            ),
            fmt_p(ch.get("is_multi_scan", {}).get("p_value")),
        ),
    ]

    for reg, title in [
        ("brain", "Brain foreground (cm$^3$)"),
        ("wt", "Whole Tumor / WT (cm$^3$)"),
        ("tc", "Tumor Core / TC (cm$^3$)"),
        ("et", "Enhancing Tumor / ET (cm$^3$)"),
        ("ncr", "NCR / Label 1 (cm$^3$)"),
        ("ed", "Edema / Label 2 (cm$^3$)"),
    ]:
        rows.append(
            (
                title,
                fmt_iqr(vols_all[reg]),
                *(fmt_iqr(sc["splits"][s]["volumes_cm3"][reg]) for s in ("train", "val", "test")),
                fmt_p(kw.get(f"{reg}_cm3", {}).get("p_value")),
            )
        )

    rows.extend(
        [
            (
                "Missing ET cases (\\%)",
                f"{miss_all['no_et_cases']} ({miss_all['no_et_pct']:.2f}\\%)",
                *(f"{sc['splits'][s]['no_et_pct']:.2f}\\%" for s in ("train", "val", "test")),
                fmt_p(ch.get("has_et", {}).get("p_value")),
            ),
            (
                "Missing NCR cases (\\%)",
                f"{miss_all['no_ncr_cases']} ({miss_all['no_ncr_pct']:.2f}\\%)",
                *(f"{sc['splits'][s]['no_ncr_pct']:.2f}\\%" for s in ("train", "val", "test")),
                fmt_p(ch.get("has_ncr", {}).get("p_value")),
            ),
        ]
    )

    tex_lines = [
        "% Auto-generated by analyze_dataset.py (requires \\usepackage{booktabs})",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Baseline characteristics and patient-level 70/10/20 split homogeneity on the BraTS 2023 GLI TrainingData. Continuous volumes are reported as Median [IQR].}",
        "\\label{tab:brats23_dataset}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Characteristic & All & Train & Val & Test & $p$-value \\\\",
        "\\midrule",
    ]
    for r in rows:
        tex_lines.append(" & ".join(r) + " \\\\")
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    out_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def generate_plots(df: pd.DataFrame, summary: Dict[str, Any], fig_dir: Path) -> List[Path]:
    """Tạo bộ biểu đồ trực quan phân tích BraTS 2023 TrainingData."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    ok_df = df[df["status"] == "ok"]
    saved: List[Path] = []

    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})

    # 1. Phân bố thể tích các vùng khối u
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    regions = [("wt_cm3", "WT", "#1f77b4"), ("tc_cm3", "TC", "#ff7f0e"), ("et_cm3", "ET", "#2ca02c")]
    for col, lbl, color in regions:
        axes[0].hist(ok_df[col], bins=45, alpha=0.55, label=lbl, color=color, edgecolor="white")
    axes[0].set_title("Phân bố thể tích WT, TC, ET (cm³)")
    axes[0].set_xlabel("Thể tích (cm³)")
    axes[0].set_ylabel("Số ca")
    axes[0].legend()

    box_cols = ["wt_cm3", "tc_cm3", "et_cm3", "ncr_cm3", "ed_cm3"]
    box_labels = ["WT", "TC", "ET (3)", "NCR (1)", "ED (2)"]
    axes[1].boxplot([ok_df[c].values for c in box_cols], tick_labels=box_labels, showfliers=False)
    axes[1].set_title("Boxplot thể tích các vùng (ẩn outliers)")
    axes[1].set_ylabel("Thể tích (cm³)")
    fig.tight_layout()
    p1 = fig_dir / "01_tumor_volume_distributions.png"
    fig.savefig(p1)
    plt.close(fig)
    saved.append(p1)

    # 2. Mất cân bằng lớp & ca thiếu nhãn
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    imb = summary["class_imbalance"]
    cls_names = ["NCR (1)", "ED (2)", "ET (3)"]
    cls_keys = ["ncr", "ed", "et"]
    cls_pcts = [imb[k]["pct_of_brain_volume"] for k in cls_keys]
    bars = axes[0].bar(cls_names, cls_pcts, color=["#d62728", "#2ca02c", "#ffbf00"])
    axes[0].set_title("Tỉ lệ voxel các nhãn khối u trong thể tích não (%)")
    axes[0].set_ylabel("% Thể tích não")
    for b, val in zip(bars, cls_pcts):
        axes[0].text(b.get_x() + b.get_width() / 2, val + 0.05, f"{val:.2f}%", ha="center", fontsize=9)

    miss = summary["missing_tumor_regions"]
    miss_names = ["Thiếu WT", "Thiếu NCR (1)", "Thiếu ED (2)", "Thiếu ET (3)"]
    miss_vals = [miss["no_wt_cases"], miss["no_ncr_cases"], miss["no_ed_cases"], miss["no_et_cases"]]
    bars2 = axes[1].bar(miss_names, miss_vals, color="#5c7cfa")
    axes[1].set_title("Số ca thiếu từng nhãn khối u (trên 1.251 ca)")
    axes[1].set_ylabel("Số ca")
    for b, val in zip(bars2, miss_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, val + 0.5, f"{val}", ha="center", fontsize=9)
    fig.tight_layout()
    p2 = fig_dir / "02_class_imbalance_and_missing_labels.png"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(p2)

    # 3. Bounding box của khối u (tham chiếu patch size)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for col, lbl, c in [("wt_bbox_x", "BBox X", "#1f77b4"), ("wt_bbox_y", "BBox Y", "#ff7f0e"), ("wt_bbox_z", "BBox Z", "#2ca02c")]:
        axes[0].hist(ok_df[col], bins=35, alpha=0.5, label=lbl, color=c)
    for psz in (96, 128):
        axes[0].axvline(psz, color="red", linestyle="--", linewidth=1.2, label=f"Patch {psz}")
    axes[0].set_title("Phân bố kích thước Bounding Box của WT (voxels)")
    axes[0].set_xlabel("Kích thước (voxels)")
    axes[0].set_ylabel("Số ca")
    axes[0].legend(fontsize=8)

    cov = summary["cca_insights"]["patch_coverage"]
    cov_keys = list(cov.keys())
    cov_vals = list(cov.values())
    bars3 = axes[1].bar([k.replace("le_", "≤ ") for k in cov_keys], cov_vals, color="#20c997")
    axes[1].set_title("Tỉ lệ ca có toàn bộ WT lọt vào Patch khối lập phương")
    axes[1].set_ylabel("% Số ca")
    axes[1].set_ylim(0, 110)
    for b, val in zip(bars3, cov_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    p3 = fig_dir / "03_tumor_bbox_and_patch_coverage.png"
    fig.savefig(p3)
    plt.close(fig)
    saved.append(p3)

    # 4. Cường độ MRI foreground & độ tương phản khối u
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].boxplot([ok_df[f"{m}_fg_mean"].values for m in MODALITIES], tick_labels=[m.upper() for m in MODALITIES], showfliers=False)
    axes[0].set_title("Phân bố Mean cường độ vùng não (Foreground > 0)")
    axes[0].set_ylabel("Giá trị cường độ gốc")

    x = np.arange(len(MODALITIES))
    w = 0.25
    for idx, (reg, lbl, c) in enumerate([("ncr", "NCR (1)", "#d62728"), ("ed", "ED (2)", "#2ca02c"), ("et", "ET (3)", "#ffbf00")]):
        meds = [ok_df[f"{m}_{reg}_to_fg_ratio"].median() for m in MODALITIES]
        axes[1].bar(x + (idx - 1) * w, meds, width=w, label=lbl, color=c)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.upper() for m in MODALITIES])
    axes[1].set_title("Tỉ lệ cường độ vùng u / trung bình não (Median)")
    axes[1].set_ylabel("Ratio so với FG Mean")
    axes[1].legend()
    fig.tight_layout()
    p4 = fig_dir / "04_modality_intensity_and_contrast.png"
    fig.savefig(p4)
    plt.close(fig)
    saved.append(p4)

    # 5. Phân bố vị trí tâm khối u & số thành phần liên thông ET
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    hb = axes[0].hexbin(
        ok_df["wt_centroid_norm_x"],
        ok_df["wt_centroid_norm_y"],
        gridsize=28,
        cmap="inferno",
        mincnt=1,
    )
    fig.colorbar(hb, ax=axes[0], label="Số ca")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Mật độ tâm khối u WT (mặt phẳng X-Y chuẩn hóa)")
    axes[0].set_xlabel("Tọa độ X chuẩn hóa")
    axes[0].set_ylabel("Tọa độ Y chuẩn hóa")

    et_cc_counts = ok_df["et_n_cc"].clip(upper=10).value_counts().sort_index()
    axes[1].bar(et_cc_counts.index.astype(str), et_cc_counts.values, color="#845ef7")
    axes[1].set_title("Số thành phần liên thông 3D của ET mỗi ca (gộp ≥10)")
    axes[1].set_xlabel("Số cụm ET liên thông")
    axes[1].set_ylabel("Số ca")
    fig.tight_layout()
    p5 = fig_dir / "05_tumor_centroid_and_et_components.png"
    fig.savefig(p5)
    plt.close(fig)
    saved.append(p5)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phân tích toàn diện bộ dữ liệu BraTS 2023 GLI TrainingData."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split-json", type=Path, default=Path(DEFAULT_SPLIT_JSON))
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="Số tiến trình song song để đọc NIfTI.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Giới hạn số ca để chạy thử nhanh (0 = toàn bộ).",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Đọc trực tiếp từ per_case_analysis.csv có sẵn để cập nhật report nhanh không cần đọc lại NIfTI.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Không vẽ biểu đồ PNG.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    csv_path = output_dir / "per_case_analysis.csv"
    if args.from_csv is not None:
        if not args.from_csv.is_file():
            print(f"[ERROR] Không tìm thấy CSV: {args.from_csv}", file=sys.stderr)
            return 1
        print(f"[INFO] Đọc dữ liệu đã phân tích từ CSV: {args.from_csv}")
        df = pd.read_csv(args.from_csv, dtype={"patient_id": str, "timepoint": str})
        if "unexpected_labels" in df.columns:
            df["unexpected_labels"] = df["unexpected_labels"].fillna("")
        if "patient_id" not in df.columns or "timepoint" not in df.columns:
            parsed = df["case_id"].map(parse_patient_timepoint)
            df.insert(1, "patient_id", [p[0] for p in parsed])
            df.insert(2, "timepoint", [p[1] for p in parsed])
            df.to_csv(csv_path, index=False)
    else:
        if not data_dir.is_dir():
            print(f"[ERROR] Không tìm thấy thư mục dữ liệu: {data_dir}", file=sys.stderr)
            return 1

        case_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
        if args.max_cases > 0:
            case_dirs = case_dirs[: args.max_cases]

        print(f"[INFO] Bắt đầu phân tích {len(case_dirs)} ca từ: {data_dir}")
        print(f"[INFO] Số workers: {args.workers} | Output: {output_dir}")

        records: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(analyze_single_case, str(d)) for d in case_dirs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Đang phân tích"):
                records.append(fut.result())

        df = pd.DataFrame(records).sort_values("case_id").reset_index(drop=True)
        df.to_csv(csv_path, index=False)

    split_path = args.split_json if args.split_json.is_file() else None
    summary = aggregate_summary(df, data_dir, split_path)

    json_path = output_dir / "summary_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_path = output_dir / "summary_report.md"
    generate_markdown_report(summary, md_path)

    tex_path = output_dir / "table1_dataset_summary.tex"
    generate_latex_table1(summary, tex_path)

    plots: List[Path] = []
    if not args.no_plots and (df["status"] == "ok").any():
        plots = generate_plots(df, summary, output_dir / "figures")

    elapsed = time.time() - t0
    print(f"\n[DONE] Hoàn tất phân tích trong {elapsed:.1f}s")
    print(f"  - CSV chi tiết từng ca : {csv_path}")
    print(f"  - JSON tổng hợp         : {json_path}")
    print(f"  - Báo cáo Markdown      : {md_path}")
    if tex_path.is_file():
        print(f"  - Bảng LaTeX (Table 1)  : {tex_path}")
    if plots:
        print(f"  - Biểu đồ ({len(plots)} ảnh PNG): {output_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
