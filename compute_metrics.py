"""
Evaluation script for BraTS 2023 segmentation results.
Computes:
- Dice Similarity Coefficient (DSC)
- 95% Hausdorff Distance (HD95)
Across the three BraTS clinical sub-regions:
1. Whole Tumor (WT): Label 1 + 2 + 3
2. Tumor Core (TC): Label 1 + 3
3. Enhancing Tumor (ET): Label 3
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm

def cal_metric(pred, gt, voxel_spacing):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if pred.sum() > 0 and gt.sum() > 0:
        dice_score = metric.binary.dc(pred, gt)
        hd95_score = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return dice_score, hd95_score
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0
    else:
        return 0.0, np.nan

def to_brats_regions(seg):
    """Convert multi-label BraTS mask to binary masks for TC, WT, and ET."""
    tc = (seg == 1) | (seg == 3)
    wt = (seg == 1) | (seg == 2) | (seg == 3)
    et = (seg == 3)
    return {"TC": tc, "WT": wt, "ET": et}

def evaluate_predictions(pred_dir, gt_dir, output_csv="evaluation_results.csv", voxel_spacing=(1.0, 1.0, 1.0)):
    if not os.path.exists(pred_dir):
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")

    pred_files = [f for f in sorted(os.listdir(pred_dir)) if f.endswith(".npy") or f.endswith(".nii.gz")]
    if not pred_files:
        raise ValueError(f"No prediction files found in {pred_dir}")

    records = []
    print(f"Evaluating {len(pred_files)} prediction cases against ground truth in {gt_dir}...")

    for pf in tqdm(pred_files):
        case_id = pf.replace(".npy", "").replace(".nii.gz", "")
        pred_path = os.path.join(pred_dir, pf)

        if pf.endswith(".npy"):
            pred = np.load(pred_path)
            if pred.ndim == 4:
                pred = np.argmax(pred, axis=0)
        else:
            pred_itk = sitk.ReadImage(pred_path)
            pred = sitk.GetArrayFromImage(pred_itk)
            voxel_spacing = pred_itk.GetSpacing()

        # Find corresponding GT file
        gt_candidates = [
            os.path.join(gt_dir, f"{case_id}.npy"),
            os.path.join(gt_dir, f"{case_id}-seg.nii.gz"),
            os.path.join(gt_dir, case_id, f"{case_id}-seg.nii.gz"),
            os.path.join(gt_dir, case_id, "seg.nii.gz"),
        ]
        gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)
        if gt_path is None:
            print(f"[!] Warning: Ground truth not found for {case_id}, skipping.")
            continue

        if gt_path.endswith(".npy"):
            gt_data = np.load(gt_path)
            gt = gt_data["seg"] if isinstance(gt_data, dict) or hasattr(gt_data, "files") else gt_data
            if gt.ndim == 4:
                gt = gt[0]
        else:
            gt_itk = sitk.ReadImage(gt_path)
            gt = sitk.GetArrayFromImage(gt_itk)
            voxel_spacing = gt_itk.GetSpacing()

        pred_regs = to_brats_regions(pred)
        gt_regs = to_brats_regions(gt)

        case_record = {"case": case_id}
        for reg_name in ["WT", "TC", "ET"]:
            d, h = cal_metric(pred_regs[reg_name], gt_regs[reg_name], voxel_spacing)
            case_record[f"{reg_name}_dice"] = d
            case_record[f"{reg_name}_hd95"] = h

        records.append(case_record)

    df = pd.DataFrame(records)
    if len(df) > 0:
        mean_row = {"case": "Mean"}
        std_row = {"case": "Std"}
        for col in ["WT_dice", "TC_dice", "ET_dice", "WT_hd95", "TC_hd95", "ET_hd95"]:
            mean_row[col] = df[col].mean()
            std_row[col] = df[col].std()
        df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        df.to_csv(output_csv, index=False)
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY:")
        print(f"WT Dice: {mean_row['WT_dice']*100:.2f}% ± {std_row['WT_dice']*100:.2f}%  |  HD95: {mean_row['WT_hd95']:.2f}")
        print(f"TC Dice: {mean_row['TC_dice']*100:.2f}% ± {std_row['TC_dice']*100:.2f}%  |  HD95: {mean_row['TC_hd95']:.2f}")
        print(f"ET Dice: {mean_row['ET_dice']*100:.2f}% ± {std_row['ET_dice']*100:.2f}%  |  HD95: {mean_row['ET_hd95']:.2f}")
        mean_dice = (mean_row['WT_dice'] + mean_row['TC_dice'] + mean_row['ET_dice']) / 3.0 * 100
        print(f"Overall Mean Dice: {mean_dice:.2f}%")
        print("=" * 60)
        print(f"Results saved to: {output_csv}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Compute BraTS Metrics (Dice and HD95)")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing prediction files")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth files")
    parser.add_argument("--output_csv", type=str, default="metrics_summary.csv", help="Output CSV path")
    args = parser.parse_args()

    evaluate_predictions(args.pred_dir, args.gt_dir, args.output_csv)

if __name__ == "__main__":
    main()
