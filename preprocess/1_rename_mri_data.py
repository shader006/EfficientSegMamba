"""
Rename BraTS 2023 multi-modal MRI files to standard format.
BraTS filenames typically follow: BraTS-GLI-XXXXX-000-t1n.nii.gz -> t1n.nii.gz
"""

import os
import argparse

def rename_cases(data_dir: str):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    all_cases = sorted(os.listdir(data_dir))
    count = 0
    for case_name in all_cases:
        case_dir = os.path.join(data_dir, case_name)
        if not os.path.isdir(case_dir):
            continue

        for data_name in os.listdir(case_dir):
            if "-" not in data_name or not data_name.endswith(".nii.gz"):
                continue
            new_name = data_name.split("-")[-1]
            old_path = os.path.join(case_dir, data_name)
            new_path = os.path.join(case_dir, new_name)

            if old_path != new_path:
                os.rename(old_path, new_path)
                count += 1

    print(f"Successfully processed and renamed files across cases in: {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename BraTS 2023 MRI data to standard naming convention")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/raw_data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/",
        help="Path to raw BraTS2023 training directory",
    )
    args = parser.parse_args()
    rename_cases(args.data_dir)
