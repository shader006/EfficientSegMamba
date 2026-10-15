# EfficientSegMamba: A Lightweight Mamba-Based Framework for 3D Brain Tumor Segmentation

Official implementation of **EfficientSegMamba** for volumetric brain tumor segmentation on the BraTS 2023 Adult Glioma dataset.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Toolkit](https://img.shields.io/badge/MONAI-1.3%2B-blueviolet.svg)](https://monai.io/)
[![Hardware](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 📌 Overview

While State Space Models (SSMs) like Mamba have established linear-time sequence modeling for medical vision, standard 3D Mamba architectures still incur substantial computational overhead from dense 3D convolutions and high-channel decoders.

**EfficientSegMamba** is designed to achieve state-of-the-art segmentation accuracy while substantially reducing FLOPs and parameter counts through four synergetic architectural innovations:

1. **Pseudo-3D (P3D) Bottlenecks**: Decomposes computationally dense full 3D convolutions ($3\times3\times3$) into orthogonal in-plane spatial $(1\times3\times3)$ and through-plane depth $(3\times1\times1)$ operations.
2. **Shifted Softplus KAN (SKAN)**: Replaces parameter-heavy feed-forward MLPs with parameter-efficient Kolmogorov-Arnold Networks using smooth Shifted Softplus activation functions ($y = \text{softplus}(kx) - \ln 2$).
3. **Morton Z-Order Tokenization**: Flattens 3D spatial feature volumes along a space-filling Morton Z-curve, preserving 3D spatial locality before feeding tokens into sequence models.
4. **Adaptive Onsampling Decoder**: Replaces conventional Transposed Convolutions with content-aware dynamic upsampling kernels.
5. **Tri-directional Mamba (TSmamba+)**: Captures forward, backward, and cross-axis spatial dependencies with linear memory complexity.

---

## 🛠️ Environment and Dependencies

### System Requirements
- OS: Linux (Ubuntu 20.04 / 22.04 LTS tested)
- Python $\ge$ 3.10
- CUDA $\ge$ 11.8 or 12.x
- NVIDIA GPU with $\ge$ 16 GB VRAM (RTX 3090, 4090, A100, A6000)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/shader006/EfficientSegMamba.git
cd EfficientSegMamba

# 2. Create conda environment
conda create -n efficient_segmamba python=3.10 -y
conda activate efficient_segmamba

# 3. Install PyTorch with CUDA matching your system
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install Mamba acceleration kernels (requires ninja and C++ compiler)
pip install ninja
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.2.0

# 5. Install medical imaging dependencies
pip install -r requirements.txt
```

---

## 📂 Data Structure

Download the official dataset from the **BraTS 2023 Adult Glioma Challenge** ([Synapse platform](https://www.synapse.org/#!Synapse:syn51156910/wiki/622351)).

Organize your raw directory as follows:

```text
data/
├── raw_data/
│   └── BraTS2023/
│       └── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
│           ├── BraTS-GLI-00001-000/
│           │   ├── BraTS-GLI-00001-000-t1n.nii.gz
│           │   ├── BraTS-GLI-00001-000-t1c.nii.gz
│           │   ├── BraTS-GLI-00001-000-t2w.nii.gz
│           │   ├── BraTS-GLI-00001-000-t2f.nii.gz
│           │   └── BraTS-GLI-00001-000-seg.nii.gz
│           └── ...
└── fullres/
    └── train/                  # Will store preprocessed .npy and .pkl volumes
```

The 4 multi-modal MRI scans per patient correspond to:
- `t1n`: T1-weighted native
- `t1c`: T1-weighted post-contrast (gadolinium)
- `t2w`: T2-weighted
- `t2f`: T2 Fluid Attenuated Inversion Recovery (FLAIR)
- `seg`: Ground-truth segmentation mask with labels $\{0: \text{BG}, 1: \text{NCR}, 2: \text{ED}, 3: \text{ET}\}$

---

## 🔄 Preprocessing Pipeline

The volumetric preprocessing consists of:
1. **Filename standardization**: Stripping subject prefixes.
2. **Spatial Resampling**: Resampling all volumes to isotropic $1.0 \times 1.0 \times 1.0\text{ mm}^3$ resolution.
3. **Non-zero Bounding Box Cropping**: Eliminating uninformative background air voxels.
4. **Modality Intensity Normalization**: Applying per-channel Z-score intensity normalization ($(\mathbf{x} - \mu) / \sigma$) exclusively over non-zero brain tissue.

Run the preprocessing commands:

```bash
# Step 1: Standardize file names
python preprocess/1_rename_mri_data.py \
    --data_dir ./data/raw_data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/

# Step 2: Run isotropic resampling, cropping, and normalization
python preprocess/2_preprocessing_mri.py \
    --raw_dir ./data/raw_data/BraTS2023/ \
    --image_folder ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
    --output_dir ./data/fullres/train/ \
    --spacing 1.0 1.0 1.0
```

---

## 🎯 Loss Configuration (Manuscript Specification)

To alleviate severe class imbalance and improve the sensitivity for small and fragmented lesions (such as Enhancing Tumor - ET), the model is optimized using a **Hybrid Dice + Focal Tversky Objective**:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}} + \lambda_{\text{FT}} \mathcal{L}_{\text{FT}}$$

### 1. Dice Loss ($\mathcal{L}_{\text{Dice}}$)
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}$$

### 2. Focal Tversky Loss ($\mathcal{L}_{\text{FT}}$)
$$\text{TI} = \frac{\sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i g_i + \alpha \sum_{i} p_i(1 - g_i) + \beta \sum_{i} (1 - p_i)g_i + \epsilon}$$

$$\mathcal{L}_{\text{FT}} = (1 - \text{TI})^{1/\gamma}$$

### Hyperparameters (Configured in `configs/settings.py`):
- $\lambda_{\text{Dice}} = 1.0$: Dice loss weight
- $\lambda_{\text{FT}} = 1.0$: Focal Tversky loss weight
- $\alpha = 0.3$: Penalty weight for False Positives (FP)
- $\beta = 0.7$: Penalty weight for False Negatives (FN) — prioritizing recall for subtle lesions
- $\gamma = 4/3 \approx 1.333$: Focusing exponent emphasizing hard voxels
- `include_background = False`: Loss is computed strictly on tumor sub-regions

---

## 🚀 Training and Inference Commands

### Training
Train EfficientSegMamba with 70/10/20 train/validation/test split or K-Fold:

```bash
python train.py \
    --data_dir ./data/fullres/train \
    --split_json ./splits/brats23_split_70_10_20.json \
    --epochs 500 \
    --batch_size 2 \
    --patch_size 96 96 96 \
    --lr 1e-4 \
    --logdir ./logs/efficient_segmamba \
    --amp
```

### Inference
Run sliding window inference on the held-out test split:

```bash
python predict.py \
    --checkpoint ./logs/efficient_segmamba/checkpoints/best_model_dice_*.pt \
    --data_dir ./data/fullres/train \
    --split_json ./splits/brats23_split_70_10_20.json \
    --save_path ./prediction_results/efficient_segmamba \
    --patch_size 96 96 96 \
    --overlap 0.5
```

---

## 📏 Evaluation Protocol

In accordance with the official BraTS Challenge protocol, metrics are reported for **three clinically meaningful composite tumor sub-regions**:

1. **Whole Tumor (WT)**: Comprises all abnormal tumor tissues — necrotic core (1), non-enhancing/peritumoral edema (2), and active enhancing tumor (3) ($1 \cup 2 \cup 3$).
2. **Tumor Core (TC)**: Comprises necrotic core and active enhancing tumor ($1 \cup 3$).
3. **Enhancing Tumor (ET)**: Active enhancing rim ($3$).

### Evaluation Metrics
- **Dice Similarity Coefficient (DSC / %)**: Overlap accuracy between prediction and ground truth.
- **95% Hausdorff Distance ($HD_{95}$ / mm)**: Boundary surface distance capturing contour fidelity.

Run the evaluation script:

```bash
python compute_metrics.py \
    --pred_dir ./prediction_results/efficient_segmamba \
    --gt_dir ./data/fullres/train \
    --output_csv ./metrics_summary.csv
```

---

## 📦 Checkpoints and Pre-trained Weights

Pre-trained model checkpoints for BraTS 2023 will be made available upon paper publication:
- **EfficientSegMamba Checkpoint**: [Download Link / Hugging Face Placeholder](#)
- Place downloaded weights under `./checkpoints/`.

---

## 📜 License and Conditions of Use

- **Code License**: This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
- **BraTS Data Disclaimer**: Dati and scans utilized originate from the BraTS Challenge. Use of the BraTS dataset must comply with the [BraTS Data Use Agreement](https://www.med.upenn.edu/cbica/brats/).

### Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{efficientsegmamba2026,
  title={EfficientSegMamba: A Lightweight Mamba-Based Framework for 3D Brain Tumor Segmentation},
  author={Duong, Tran Bui Nguyen and Guidance, Professor},
  journal={arXiv preprint},
  year={2026}
}
```

### Acknowledgements
We acknowledge the authors and open-source communities of:
- [SegMamba](https://github.com/ge-xing/SegMamba)
- [Project-MONAI](https://github.com/Project-MONAI/MONAI)
- [U-Mamba](https://github.com/bowang-lab/U-Mamba)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
