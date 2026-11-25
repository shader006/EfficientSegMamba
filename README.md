# BraTS 2023 3D Brain Tumor Segmentation (SegMamba Pipeline)

Self-contained training, preprocessing, dataset analysis, and evaluation pipeline for **BraTS 2023 Adult Glioma (GLI)** built around the official [ge-xing/SegMamba](https://github.com/ge-xing/SegMamba) architecture (`bimamba_type="v3"`).

---

## Repository Structure

```text
.
├── settings.py                  # Centralized paths, hyperparameters, optimizer/scheduler, reproducibility
├── losses.py                    # MONAI DiceCELoss wrapper supporting `multiclass` (4-class) and `regions` (WT/TC/ET)
├── train.py                     # Training & validation pipeline (Top-K saving, train.log, history.csv, ETA)
├── select_best_checkpoint.py    # Post-training Top-K validation selector (Dice + HD95 tie-break)
├── evaluate.py                  # Test/val evaluation with Dice & HD95 summary, optional TTA & NIfTI export
├── model/
│   ├── __init__.py
│   ├── segmamba.py              # Official SegMamba 3D architecture (~67.42M params)
│   └── mamba_ssm/               # Local Mamba SSM modules supporting bimamba_type="v3"
└── script/
    ├── analyze_dataset.py       # Full 1,251-case NIfTI integrity, intensity, morphology & CCA analysis
    ├── create_patient_split.py  # Patient-level leakage-free 70/10/20 & 5-fold split generator
    ├── preprocess_dataset.py    # Raw .nii.gz -> cropped, normalized uncompressed .npy + .pkl metadata
    ├── benchmark_io.py          # I/O speed benchmark (RAM cache vs /dev/shm mmap vs NFS mmap vs .npz)
    ├── benchmark_formats.py     # Storage format benchmark across preprocessing/loading backends
    ├── dataset.py               # High-throughput PyTorch Dataset (ram / mmap cache, foreground oversampling)
    ├── metrics.py               # Standard BraTS WT/TC/ET Dice & tight-bbox HD95 metrics (no CCA)
    └── output_brats23_analysis/ # Pre-computed splits, summary reports, benchmarks & diagnostic figures
```

---

## Dataset & Leakage-Free Splits

Summary from [`script/output_brats23_analysis/summary_report.md`](script/output_brats23_analysis/summary_report.md):

- **Dataset:** BraTS 2023 GLI TrainingData — **1,251 cases** across **1,133 unique patients** (1,015 single-scan + 118 two-scan patients).
- **Geometry:** Uniform `240x240x155` volume shape, `1.0x1.0x1.0` mm spacing, `LPS` orientation, 4 MRI modalities (`t1n`, `t1c`, `t2w`, `t2f`).
- **Patient-Disjoint 70/10/20 Split** ([`brats23_patient_split_70_10_20.json`](script/output_brats23_analysis/brats23_patient_split_70_10_20.json)):
  - `train`: **876** cases (793 patients)
  - `val`: **126** cases (114 patients)
  - `test`: **249** cases (226 patients)
  - Zero patient overlap across splits (`train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0`).
- **Patient-Grouped 5-Fold Split:** [`brats23_patient_5fold.json`](script/output_brats23_analysis/brats23_patient_5fold.json).

---

## Preprocessing & I/O Design

Guided by [`script/output_brats23_analysis/io_benchmark.md`](script/output_brats23_analysis/io_benchmark.md) (`10` cases, `128^3` patch reads):

| Rank | Loading Backend | Mean (ms/case) | Speedup vs `.npz` |
| :---: | :--- | ---: | :---: |
| #1 | RAM cache slice (patch) | **10.26** | **49.33x** |
| #2 | `/dev/shm` `.npy` mmap (patch) | **13.24** | **38.23x** |
| #3 | NFS `.npy` mmap (patch) | **17.37** | **29.13x** |
| #6 | NFS `.npz` compressed (full) | 506.03 | 1.00x |

[`script/preprocess_dataset.py`](script/preprocess_dataset.py) converts raw `.nii.gz` scans into uncompressed `.npy` + `.pkl` inside `data/`:
1. Tight brain-foreground cropping (`data > 0` or `seg > 0`).
2. Artifact-safe per-case, per-modality intensity clipping (default `[0.5, 99.5]` percentiles) and nonzero-foreground Z-score normalization (keeping background strictly `0.0`).
3. Pre-sampled foreground voxel coordinates (`label_1`, `label_2`, `label_3`, `wt`) saved in `<case_id>.pkl` for fast tumor-centered patch cropping without runtime `np.argwhere`.

---

## Environment Requirements

- Linux + CUDA GPU (required by `mamba_ssm` CUDA kernels for model forward/backward passes)
- Python 3.10+
- PyTorch + CUDA, `monai`, `nibabel`, `medpy`, `scipy`, `numpy`, `pandas`, `tqdm`

---

## Quick Start

### 1. Preprocess Raw BraTS 2023 Data

Override default paths with `--data-dir` and `--output-dir` if running on a new machine:

```bash
python script/preprocess_dataset.py \
  --data-dir /path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
  --output-dir ./data
```

Verify the DataLoader:

```bash
python script/dataset.py --sanity-check --cache-mode mmap --batch-size 4 --num-workers 2
```

Verify loss self-test:

```bash
python losses.py
```

### 2. Dry-Run Verification (CPU or GPU)

Validates split loading, DataLoader batch shapes, model instantiation, optimizer, and loss without running the CUDA training loop:

```bash
python train.py --dry-run --cache-mode mmap --num-workers 2
```

### 3. Training

All defaults live in [`settings.py`](settings.py) (`INPUT_SIZE=(128, 128, 128)`, `LABEL_MODE="multiclass"`, `CACHE_MODE="ram"`, `AUG_MODE="strong"`, `EPOCHS=300`, `BATCH_SIZE=2`, `LOSS_NAME="DiceCE"`, `OPTIMIZER_NAME="AdamW"`, `LEARNING_RATE=1e-4`, `SCHEDULER_NAME="cosine_with_warmup"`, `TOP_K_CHECKPOINTS=3`).

```bash
# Quick overfit sanity check on GPU (4 cases, 20 epochs)
python train.py --overfit-cases 4 --epochs 20 --val-every 5 --cache-mode mmap

# Full training run (uses RAM cache by default; pass --cache-mode mmap on low-RAM nodes)
python train.py --run-name segmamba_brats23_baseline

# Enable per-epoch validation HD95 logging (WT, TC, ET, and Mean HD95)
python train.py --run-name segmamba_brats23_baseline --val-hd95

# Resume training from last checkpoint
python train.py --run-name segmamba_brats23_baseline \
  --resume runs/segmamba_brats23_baseline/checkpoints/last_model.pt
```

Each run saves under `runs/<run_name>/`:
- `config.json` — CLI arguments + exported `settings.py` snapshot
- `train.log` — append-only timestamped training log (`[INIT]`, `[DATA]`, `[RESUME]`, `[Epoch ...]`, `[BEST]`, `[TOP-K]`, `[DONE]`, and crash tracebacks)
- `history.csv` — per-epoch `train_loss`, `lr`, `val_mean_dice`, `val_wt_dice`, `val_tc_dice`, `val_et_dice`, `val_mean_hd95`, `val_wt_hd95`, `val_tc_hd95`, `val_et_hd95`, `best_val_dice`, and `epoch_sec`
- `checkpoints/best_model.pt`, `checkpoints/last_model.pt`, and `checkpoints/topk/` (managed via `topk_manifest.json`)

### 4. Post-Training Top-K Checkpoint Selection (Dice + HD95 Tie-Break)

After training, evaluate all saved Top-K candidates (plus `best_model.pt`) with full validation Dice & HD95 and pick the lowest-HD95 model within `HD95_TIEBREAK_TOLERANCE` (default `0.002`) of the highest validation Mean Dice:

```bash
python select_best_checkpoint.py --run-dir runs/segmamba_brats23_baseline --cache-mode mmap
```

Outputs `checkpoints/best_model_selected.pt`, `val_topk_selection.json`, and `val_topk_selection.csv`.

### 5. Test-Set Evaluation

Evaluate the selected checkpoint (automatically resolves `best_model_selected.pt`, falling back to `best_model.pt` / `last_model.pt`) on the `test` split with BraTS WT/TC/ET/Mean Dice & HD95:

```bash
# Standard evaluation
python evaluate.py --run-dir runs/segmamba_brats23_baseline --cache-mode mmap

# With 8-view flip TTA and uncropped .nii.gz prediction export
python evaluate.py --run-dir runs/segmamba_brats23_baseline --cache-mode mmap --tta --save-nii
```

Outputs `per_case_metrics.csv`, `summary.json`, and (when `--save-nii` is set) `predictions/<case_id>.nii.gz` inside `runs/<run_name>/eval_test_<ckpt_stem>/`.
