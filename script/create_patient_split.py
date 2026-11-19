#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a patient-level (leakage-free) Train / Val / Test split for BraTS 2023 GLI.

In BraTS 2023 GLI, case IDs follow `BraTS-GLI-{PatientID}-{Timepoint}` (e.g.,
`BraTS-GLI-00008-000` and `BraTS-GLI-00008-001` are two scans of the same patient).
This script groups all scans by `PatientID` before splitting so that no patient
appears in more than one split, while preserving the target 70/10/20 ratio and
stratifying by scan count (single-scan vs. multi-scan patients) and optional
tumor metadata (ET-missing status and WT volume tertiles from `per_case_analysis.csv`).

It also runs statistical homogeneity tests (Kruskal-Wallis and Chi-square) across
splits and can optionally generate patient-grouped K-fold splits (`--k-folds`).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_DATA_ROOT = Path("/home/cuc.buithi/BRATS/BraTS2023_TrainVal")
DEFAULT_OUTPUT_JSON = Path(
    "/home/cuc.buithi/BRATS/all/script/output_brats23_analysis/brats23_patient_split_70_10_20.json"
)
DEFAULT_ANALYSIS_CSV = Path(
    "/home/cuc.buithi/BRATS/all/script/output_brats23_analysis/per_case_analysis.csv"
)
MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def parse_patient_id(case_id: str) -> str:
    """Extract the 5-digit patient ID from a `BraTS-GLI-XXXXX-YYY` folder name."""
    match = re.match(r"^BraTS-GLI-(\d+)-(\d+)$", case_id)
    if match:
        return match.group(1)
    parts = case_id.split("-")
    return parts[-2] if len(parts) >= 4 else case_id


def find_cases(root: Path) -> List[Dict[str, Any]]:
    """Collect valid BraTS 2023 GLI cases containing all 4 modalities and seg mask."""
    records: List[Dict[str, Any]] = []
    case_dirs = sorted(p for p in root.rglob("BraTS-GLI-*") if p.is_dir())

    for case_dir in case_dirs:
        case_id = case_dir.name
        image_paths: List[str] = []
        for mod in MODALITIES:
            mod_file = case_dir / f"{case_id}-{mod}.nii.gz"
            if not mod_file.is_file():
                image_paths = []
                break
            image_paths.append(str(mod_file.resolve()))

        mask_file = case_dir / f"{case_id}-seg.nii.gz"
        if not image_paths or not mask_file.is_file():
            continue

        records.append(
            {
                "id": case_id,
                "patient_id": parse_patient_id(case_id),
                "image": image_paths,
                "mask": str(mask_file.resolve()),
            }
        )
    return records


def allocate_counts(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    """Allocate integer counts (n_train, n_val, n_test) using rounding with exact sum."""
    n_train = round(total * train_ratio)
    n_val = round(total * val_ratio)
    if n_train + n_val > total:
        n_val = max(0, total - n_train)
    n_test = total - n_train - n_val
    return n_train, n_val, n_test


def load_case_metadata(analysis_csv: Optional[Path]) -> Tuple[Dict[str, bool], Dict[str, float]]:
    """Load per-case `has_et` and `wt_cm3` from analysis CSV if available."""
    if analysis_csv is None or not analysis_csv.is_file():
        return {}, {}
    df = pd.read_csv(analysis_csv, usecols=["case_id", "status", "has_et", "wt_cm3"])
    ok_df = df[df["status"] == "ok"]
    et_flags = {str(r.case_id): bool(r.has_et) for r in ok_df.itertuples(index=False)}
    wt_vols = {str(r.case_id): float(r.wt_cm3) for r in ok_df.itertuples(index=False)}
    return et_flags, wt_vols


def build_patient_strata(
    by_patient: Dict[str, List[Dict[str, Any]]],
    et_flags: Dict[str, bool],
    wt_vols: Dict[str, float],
) -> Dict[Tuple[int, bool, int], List[str]]:
    """Group patients into strata by (scan_count, any_missing_et, wt_tertile)."""
    patient_wt: Dict[str, float] = {}
    if wt_vols:
        for pid, pid_records in by_patient.items():
            vals = [wt_vols[r["id"]] for r in pid_records if r["id"] in wt_vols]
            patient_wt[pid] = float(np.mean(vals)) if vals else 0.0
        q33, q66 = np.quantile(list(patient_wt.values()), [1.0 / 3.0, 2.0 / 3.0])
    else:
        q33, q66 = 0.0, 0.0

    strata: Dict[Tuple[int, bool, int], List[str]] = defaultdict(list)
    for pid, pid_records in sorted(by_patient.items()):
        n_scans = len(pid_records)
        patient_missing_et = (
            any(not et_flags.get(r["id"], True) for r in pid_records) if et_flags else False
        )
        if wt_vols:
            wt_val = patient_wt[pid]
            wt_bin = 0 if wt_val <= q33 else (1 if wt_val <= q66 else 2)
        else:
            wt_bin = 0
        strata[(n_scans, patient_missing_et, wt_bin)].append(pid)

    return strata


def patient_stratified_split(
    records: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    et_flags: Dict[str, bool],
    wt_vols: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group records by patient_id and split with stratification (no patient leakage)."""
    by_patient: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_patient[rec["patient_id"]].append(rec)

    strata = build_patient_strata(by_patient, et_flags, wt_vols)
    rng = random.Random(seed)
    split_pids: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    # Assign each stratum independently to preserve scan-count, missing-ET, and WT-size balance.
    for _, pids in sorted(strata.items()):
        shuffled = pids[:]
        rng.shuffle(shuffled)
        n_tr, n_va, _ = allocate_counts(len(shuffled), train_ratio, val_ratio)
        split_pids["train"].extend(shuffled[:n_tr])
        split_pids["val"].extend(shuffled[n_tr : n_tr + n_va])
        split_pids["test"].extend(shuffled[n_tr + n_va :])

    def collect(pids: List[str]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for pid in sorted(pids):
            for rec in sorted(by_patient[pid], key=lambda r: r["id"]):
                items.append({"id": rec["id"], "image": rec["image"], "mask": rec["mask"]})
        return items

    train = collect(split_pids["train"])
    val = collect(split_pids["val"])
    test = collect(split_pids["test"])

    tr_p, va_p, te_p = set(split_pids["train"]), set(split_pids["val"]), set(split_pids["test"])
    assert len(tr_p & va_p) == 0, "Patient overlap detected between train and val!"
    assert len(tr_p & te_p) == 0, "Patient overlap detected between train and test!"
    assert len(va_p & te_p) == 0, "Patient overlap detected between val and test!"
    assert len(train) + len(val) + len(test) == len(records), "Case count mismatch after split!"

    return train, val, test


def create_patient_kfolds(
    records: List[Dict[str, Any]],
    k_folds: int,
    seed: int,
    et_flags: Dict[str, bool],
    wt_vols: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Create patient-grouped stratified K-fold splits (zero patient overlap across folds)."""
    by_patient: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_patient[rec["patient_id"]].append(rec)

    strata = build_patient_strata(by_patient, et_flags, wt_vols)
    rng = random.Random(seed)
    fold_pids: List[List[str]] = [[] for _ in range(k_folds)]

    # Round-robin assignment per stratum ensures balanced folds at patient level.
    offset = 0
    for _, pids in sorted(strata.items()):
        shuffled = pids[:]
        rng.shuffle(shuffled)
        for idx, pid in enumerate(shuffled):
            fold_pids[(idx + offset) % k_folds].append(pid)
        offset = (offset + len(shuffled)) % k_folds

    folds: List[Dict[str, Any]] = []
    for val_idx in range(k_folds):
        val_p = set(fold_pids[val_idx])
        train_p = set().union(*(fold_pids[i] for i in range(k_folds) if i != val_idx))
        assert len(val_p & train_p) == 0
        train_items = [
            {"id": r["id"], "image": r["image"], "mask": r["mask"]}
            for pid in sorted(train_p)
            for r in sorted(by_patient[pid], key=lambda x: x["id"])
        ]
        val_items = [
            {"id": r["id"], "image": r["image"], "mask": r["mask"]}
            for pid in sorted(val_p)
            for r in sorted(by_patient[pid], key=lambda x: x["id"])
        ]
        folds.append(
            {
                "fold": val_idx,
                "train_cases": len(train_items),
                "val_cases": len(val_items),
                "train_patients": len(train_p),
                "val_patients": len(val_p),
                "train": train_items,
                "val": val_items,
            }
        )
    return folds


def compute_split_statistical_tests(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    analysis_csv: Optional[Path],
) -> Dict[str, Any]:
    """Run Kruskal-Wallis and Chi-square homogeneity tests across train/val/test."""
    if analysis_csv is None or not analysis_csv.is_file():
        return {}

    df = pd.read_csv(analysis_csv)
    ok_df = df[df["status"] == "ok"].copy()
    split_map = {}
    for name, items in (("train", train), ("val", val), ("test", test)):
        for item in items:
            split_map[item["id"]] = name
    ok_df["split"] = ok_df["case_id"].map(split_map)
    ok_df = ok_df[ok_df["split"].isin(["train", "val", "test"])].copy()

    if "patient_id" not in ok_df.columns:
        ok_df["patient_id"] = ok_df["case_id"].map(parse_patient_id)
    scan_counts = ok_df.groupby("patient_id")["case_id"].transform("count")
    ok_df["is_multi_scan"] = scan_counts > 1

    tests: Dict[str, Any] = {"continuous_kruskal_wallis": {}, "categorical_chi2": {}}
    groups = [ok_df[ok_df["split"] == s] for s in ("train", "val", "test")]

    for col in ("brain_cm3", "wt_cm3", "tc_cm3", "et_cm3", "ncr_cm3", "ed_cm3"):
        stat, pval = stats.kruskal(*(g[col].dropna().values for g in groups))
        tests["continuous_kruskal_wallis"][col] = {
            "statistic": float(stat),
            "p_value": float(pval),
        }

    for col in ("has_et", "has_ncr", "is_multi_scan"):
        ct = pd.crosstab(ok_df["split"], ok_df[col])
        chi2, pval, _, _ = stats.chi2_contingency(ct)
        tests["categorical_chi2"][col] = {
            "statistic": float(chi2),
            "p_value": float(pval),
        }

    return tests


def print_balance_summary(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    analysis_csv: Optional[Path],
    stat_tests: Dict[str, Any],
) -> None:
    """Print patient counts, case counts, tumor volume balance, and p-values."""
    splits = {"train": train, "val": val, "test": test}
    total_cases = sum(len(v) for v in splits.values())
    total_patients = sum(len({parse_patient_id(x["id"]) for x in v}) for v in splits.values())

    print("\n[SUMMARY] Patient-level split (0 patient overlap):")
    for name, items in splits.items():
        n_cases = len(items)
        n_pats = len({parse_patient_id(x["id"]) for x in items})
        print(
            f"  - {name:5s}: {n_cases:4d} cases ({n_cases / total_cases * 100:5.2f}%) | "
            f"{n_pats:4d} patients ({n_pats / total_patients * 100:5.2f}%)"
        )

    if analysis_csv is not None and analysis_csv.is_file():
        df = pd.read_csv(analysis_csv)
        ok_df = df[df["status"] == "ok"]
        print("\n[BALANCE] Tumor statistics from analysis CSV:")
        for name, items in splits.items():
            ids = {x["id"] for x in items}
            sub = ok_df[ok_df["case_id"].isin(ids)]
            print(
                f"  - {name:5s}: WT mean={sub['wt_cm3'].mean():.2f} (med={sub['wt_cm3'].median():.2f}), "
                f"TC mean={sub['tc_cm3'].mean():.2f}, ET mean={sub['et_cm3'].mean():.2f} cm³, "
                f"missing ET={(~sub['has_et']).mean() * 100:.2f}%"
            )

    if stat_tests:
        print("\n[HOMOGENEITY TESTS] Across train / val / test (target p > 0.05):")
        for col, res in stat_tests["continuous_kruskal_wallis"].items():
            print(f"  - Kruskal-Wallis {col:10s}: H={res['statistic']:.4f}, p={res['p_value']:.4f}")
        for col, res in stat_tests["categorical_chi2"].items():
            print(f"  - Chi-square     {col:10s}: chi2={res['statistic']:.4f}, p={res['p_value']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a patient-grouped 70/10/20 train/val/test split for BraTS 2023 GLI."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--analysis-csv", type=Path, default=DEFAULT_ANALYSIS_CSV)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--k-folds",
        type=int,
        default=0,
        help="Optional number of patient-grouped CV folds to export alongside 70/10/20 (e.g. 5).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    records = find_cases(data_root)
    if not records:
        raise RuntimeError(f"No valid BraTS-GLI cases found under: {data_root}")

    analysis_csv = args.analysis_csv if args.analysis_csv.is_file() else None
    et_flags, wt_vols = load_case_metadata(analysis_csv)

    train, val, test = patient_stratified_split(
        records=records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        et_flags=et_flags,
        wt_vols=wt_vols,
    )

    tr_p = {parse_patient_id(x["id"]) for x in train}
    va_p = {parse_patient_id(x["id"]) for x in val}
    te_p = {parse_patient_id(x["id"]) for x in test}
    stat_tests = compute_split_statistical_tests(train, val, test, analysis_csv)

    payload = {
        "meta": {
            "data_root": str(data_root),
            "seed": args.seed,
            "split_strategy": "patient_grouped_stratified",
            "stratified_by": ["scan_count", "et_missing", "wt_tertile"]
            if wt_vols
            else (["scan_count", "et_missing"] if et_flags else ["scan_count"]),
            "ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "total": len(records),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "unique_patients": len(tr_p | va_p | te_p),
            "patient_counts": {
                "train": len(tr_p),
                "val": len(va_p),
                "test": len(te_p),
            },
            "patient_overlap": {
                "train_val": len(tr_p & va_p),
                "train_test": len(tr_p & te_p),
                "val_test": len(va_p & te_p),
            },
            "statistical_tests": stat_tests,
        },
        "train": train,
        "val": val,
        "test": test,
    }

    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[DONE] Patient-level split saved to: {output_json}")
    print_balance_summary(train, val, test, analysis_csv, stat_tests)

    if args.k_folds and args.k_folds > 1:
        folds = create_patient_kfolds(records, args.k_folds, args.seed, et_flags, wt_vols)
        kfold_path = output_json.with_name(f"brats23_patient_{args.k_folds}fold.json")
        with kfold_path.open("w", encoding="utf-8") as f:
            json.dump({"meta": {"k_folds": args.k_folds, "seed": args.seed}, "folds": folds}, f, indent=2)
        print(f"[DONE] Patient-level {args.k_folds}-fold split saved to: {kfold_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
