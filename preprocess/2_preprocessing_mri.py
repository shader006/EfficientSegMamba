"""
Preprocessing pipeline for BraTS 2023 multi-modal MRI volumes:
1. Resample to isotropic resolution (default 1.0 x 1.0 x 1.0 mm^3).
2. Non-zero brain foreground bounding-box cropping.
3. Modality-wise intensity normalization (Z-score).
4. Save preprocessed slices/volumes to .npy / .pkl files for high-throughput training.
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS 2023 dataset")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="./data/raw_data/BraTS2023/",
        help="Root directory containing raw BraTS data",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        help="Subfolder name containing patient volumes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/fullres/train/",
        help="Directory to save preprocessed files",
    )
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Target isotropic voxel spacing (e.g., 1.0 1.0 1.0)",
    )
    args = parser.parse_args()

    data_filenames = ["t2w.nii.gz", "t2f.nii.gz", "t1n.nii.gz", "t1c.nii.gz"]
    seg_filename = "seg.nii.gz"

    print(f"Initializing MultiModalityPreprocessor from: {os.path.join(args.raw_dir, args.image_folder)}")
    preprocessor = MultiModalityPreprocessor(
        base_dir=args.raw_dir,
        image_dir=args.image_folder,
        data_filenames=data_filenames,
        seg_filename=seg_filename,
    )

    print("Running dataset planning...")
    preprocessor.run_plan()

    print(f"Running preprocessing with output spacing {args.spacing} -> {args.output_dir}...")
    preprocessor.run(
        output_spacing=args.spacing,
        output_dir=args.output_dir,
        all_labels=[1, 2, 3],
    )
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()
