"""
Configuration settings for EfficientSegMamba.
Matches the published manuscript specifications for 3D Brain Tumor Segmentation (BraTS 2023).
"""

import os
import random
import numpy as np
import torch

# ==============================================================================
# Reproducibility & Global Seeds
# ==============================================================================
REPRO_SEED = 42

def set_global_reproducibility(seed: int = REPRO_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==============================================================================
# EfficientSegMamba Architecture Configuration
# ==============================================================================
SEGMAMBA_IN_CHANS = 4          # 4 MRI modalities: T1n, T1c, T2w, T2f (FLAIR)
SEGMAMBA_OUT_CHANS = 4         # Output channels (Background + 3 tumor regions)
SEGMAMBA_DEPTHS = [1, 1, 1, 1] # Hierarchical Mamba blocks per stage
SEGMAMBA_FEAT_SIZE = [48, 96, 192, 384]
SEGMAMBA_HIDDEN_SIZE = 768
SEGMAMBA_DROP_PATH_RATE = 0
SEGMAMBA_LAYER_SCALE_INIT_VALUE = 1e-6
SEGMAMBA_NORM_NAME = "instance"
SEGMAMBA_SPATIAL_DIMS = 3

# Key Architectural Enhancements
SEGMAMBA_3D_CONV = False       # False: Use Pseudo-3D (P3D: (1x3x3) + (3x1x1)) convolutions
SEGMAMBA_RES_BLOCK = True      # Enable residual connections
SEGMAMBA_KAN = False           # Standard B-spline KAN
SEGMAMBA_SKAN = True           # Shifted Softplus KAN (SKAN) for parameter-efficient boundary modeling

# Token scanning / spatial ordering mode:
# Choices:
#   - "mortonZ": Single-branch 3D space-filling Morton Z-order curve token scanning
#   - "raster_scan": Single-branch 1D sequential raster scan (D x H x W)
#   - "tri_directional": 3 parallel orthogonal scan branches along 3 axes (D, W, H) with raster scan
SEGMAMBA_SCAN_MODE = "mortonZ"  # choices: "mortonZ", "raster_scan", "tri_directional"
SEGMAMBA_KAN_MORTON_Z = (SEGMAMBA_SCAN_MODE == "mortonZ")   # 3D Morton Z-order space-filling curve tokenization

SEGMAMBA_ONSAMPLING = True     # Adaptive Onsampling Decoder
SEGMAMBA_MAMBA_STAGES = [0, 1, 2]
ADVANCED_SEGMAMBA_MAMBA_IMPL = "mamba3"  # Tri-directional State Space Model (TSmamba+)

# ==============================================================================
# Loss Configuration (Matches Manuscript: Dice + Focal Tversky Loss)
# ==============================================================================
SEGMAMBA_LOSS_NAME = "DiceFocalTversky"
SEGMAMBA_DICE_LOSS_WEIGHT = 1.0
SEGMAMBA_CE_LOSS_WEIGHT = 1.0
SEGMAMBA_FOCAL_TVERSKY_LOSS_WEIGHT = 1.0

# Focal Tversky hyperparameters (penalize False Negatives for small enhancing tumors)
SEGMAMBA_FOCAL_TVERSKY_ALPHA = 0.3    # FP weight
SEGMAMBA_FOCAL_TVERSKY_BETA = 0.7     # FN weight (higher sensitivity for small targets like ET)
SEGMAMBA_FOCAL_TVERSKY_GAMMA = 4.0 / 3.0 # Focusing parameter (gamma = 1.333)
SEGMAMBA_LOSS_INCLUDE_BACKGROUND = False # Compute loss exclusively over tumor targets

# ==============================================================================
# Optimizer & Learning Rate Schedule
# ==============================================================================
OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SGD_MOMENTUM = 0.99
SGD_NESTEROV = True

SCHEDULER_NAME = "cosine_with_warmup"
T_MAX = 500
ETA_MIN = 1e-6
SCHEDULER_WARMUP = 0.05

# ==============================================================================
# Hardware, Acceleration & Data Augmentation
# ==============================================================================
PATCH_SIZE = [96, 96, 96]
BATCH_SIZE = 2
VAL_BATCH_SIZE = 1

SEGMAMBA_AMP_ENABLED = True
SEGMAMBA_AMP_PRECISION = "fp16"  # choices: fp16, bf16, fp32
SEGMAMBA_TF32_ENABLED = True

SEGMAMBA_AUGMENTER_BACKEND = "nondet-multiprocess"
SEGMAMBA_TRAIN_PROCESS = 2
SEGMAMBA_VAL_PROCESS = 2
SEGMAMBA_AUGMENTER_SEED_COUNT = 32

SEGMAMBA_RAM_CACHE_ENABLED = True
SEGMAMBA_RAM_CACHE_PRELOAD = False
SEGMAMBA_GPU_TRANSFORMS_ENABLED = True
SEGMAMBA_CHANNELS_LAST_3D_ENABLED = False

AUGMENTATION = "nomirror"
SEGMAMBA_MODALITY_DROPOUT_ENABLED = True
SEGMAMBA_MODALITY_DROPOUT_PROB = 0.3
SEGMAMBA_MODALITY_DROPOUT_MAX_CHANNELS = 1

# ==============================================================================
# Logging & Tracking
# ==============================================================================
SEGMAMBA_WANDB_ENABLED = False
SEGMAMBA_WANDB_PROJECT = "EfficientSegMamba-BraTS23"
SEGMAMBA_WANDB_RUN_NAME = None
SEGMAMBA_LOG_EVERY_N_STEPS = 10
SEGMAMBA_BEST_MIN_DICE_DELTA = 1e-4
SEGMAMBA_BEST_MIN_HD95_DELTA = 1e-4

import math
from torch.optim.lr_scheduler import LambdaLR

def build_optimizer(model):
    """Build optimizer for project training."""
    if OPTIMIZER_NAME == "Adam":
        return torch.optim.Adam(
            params=model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER_NAME == "AdamW":
        return torch.optim.AdamW(
            params=model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER_NAME == "SGD":
        return torch.optim.SGD(
            params=model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")

def _build_cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def build_scheduler(optimizer, num_training_steps=None, num_epochs=None):
    """Build LR scheduler paired with optimizer from project settings."""
    if SCHEDULER_NAME in (None, "none", "None"):
        return None
    if SCHEDULER_NAME == "cosine_with_warmup":
        if num_training_steps is None:
            num_training_steps = 1000
        warmup_steps = int(num_training_steps * SCHEDULER_WARMUP)
        return _build_cosine_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    return None
