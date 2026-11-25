#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized training configuration for BraTS 2023 experiments inside `all/`.

Provides paths, data/augmentation options, official SegMamba parameters,
DiceCELoss settings, optimizer/scheduler builders, and reproducibility helpers.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


# =============================================================================
# 1. Paths (self-contained inside `all/`)
# =============================================================================
ALL_DIR = Path(__file__).resolve().parent
DATA_DIR = ALL_DIR.parent / "data"
SPLIT_JSON = ALL_DIR / "script" / "output_brats23_analysis" / "brats23_patient_split_70_10_20.json"
RUNS_DIR = ALL_DIR / "runs"


# =============================================================================
# 2. DataLoader & Augmentation
# =============================================================================
INPUT_SIZE = (128, 128, 128)
LABEL_MODE = "multiclass"      # "multiclass" (4 classes, SegMamba official) | "regions" (3 channels WT/TC/ET)
CACHE_MODE = "ram"             # "ram" (~10 ms/case, uses ~55GB RAM) | "mmap" (~13-17 ms/case)
AUG_MODE = "strong"            # "strong" | "light" | "none"
FG_PROB = 0.67                 # Probability of tumor-centered patch sampling
MODALITY_DROPOUT_PROB = 0.0    # Keep 0.0 (disabled) for standard BraTS training
NUM_WORKERS = 8
FOLD = 0                       # Used only when SPLIT_JSON is a K-fold JSON


# =============================================================================
# 3. Model (Official ge-xing/SegMamba)
# =============================================================================
MODEL_NAME = "segmamba"
IN_CHANS = 4                   # t1n, t1c, t2w, t2f
OUT_CHANS = 4 if LABEL_MODE == "multiclass" else 3
SEGMAMBA_DEPTHS = [2, 2, 2, 2]
SEGMAMBA_FEAT_SIZE = [48, 96, 192, 384]
SEGMAMBA_HIDDEN_SIZE = 768
SEGMAMBA_NORM_NAME = "instance"
SEGMAMBA_DROP_PATH_RATE = 0.0


# =============================================================================
# 4. Training & Precision
# =============================================================================
EPOCHS = 300
BATCH_SIZE = 1
VAL_EVERY = 1
DEVICE = "cuda:0"
AMP_ENABLED = False
AMP_PRECISION = "fp32"         # "bf16" | "fp16" | "fp32"
GRAD_CLIP_NORM = 1.0


# =============================================================================
# 5. Loss (DiceCELoss)
# =============================================================================
LOSS_NAME = "DiceCE"
DICE_LOSS_WEIGHT = 1.0
CE_LOSS_WEIGHT = 1.0
LOSS_INCLUDE_BACKGROUND = False  # Applies to multiclass mode (False = focus dice on tumor classes 1,2,3)
LOSS_SMOOTH_NR = 1e-5
LOSS_SMOOTH_DR = 1e-5


# =============================================================================
# 6. Optimizer & LR Scheduler
# =============================================================================
OPTIMIZER_NAME = "AdamW"       # "AdamW" | "Adam" | "SGD"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SGD_MOMENTUM = 0.99
SGD_NESTEROV = True

SCHEDULER_NAME = "cosine_with_warmup"  # "cosine_with_warmup" | "poly" | "none"
SCHEDULER_WARMUP = 0.05
ETA_MIN = 1e-6


# =============================================================================
# 7. Validation & Sliding-Window Inference
# =============================================================================
SW_BATCH_SIZE = 4
SW_OVERLAP = 0.5
SW_MODE = "gaussian"
EMPTY_HD95_PENALTY = 373.13    # Diagonal of 240x240x155 volume (mm)
TOP_K_CHECKPOINTS = 3          # Keep top-K checkpoints by val Mean Dice for post-train HD95 selection
HD95_TIEBREAK_TOLERANCE = 0.002  # Among top-K within this Dice margin of best, pick lowest val HD95


# =============================================================================
# 8. Reproducibility
# =============================================================================
REPRO_SEED = 42
DETERMINISTIC = False
DETERMINISTIC_WARN_ONLY = True
TF32_ENABLED = True


def set_global_reproducibility(
    seed: int = REPRO_SEED,
    deterministic: bool = DETERMINISTIC,
    warn_only: bool = DETERMINISTIC_WARN_ONLY,
    tf32_enabled: bool = TF32_ENABLED,
) -> None:
    """Fix Python, NumPy, and PyTorch random seeds and backend flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = tf32_enabled

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except Exception:
            pass


def build_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = OPTIMIZER_NAME,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> torch.optim.Optimizer:
    """Create optimizer according to settings."""
    name = optimizer_name.lower()
    if name == "adamw":
        params = list(model.parameters())
        use_fused = torch.cuda.is_available() and bool(params) and all(p.is_cuda for p in params)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=use_fused)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
        )
    raise ValueError(f"Unsupported OPTIMIZER_NAME: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    scheduler_name: Optional[str] = SCHEDULER_NAME,
    warmup_ratio: float = SCHEDULER_WARMUP,
    eta_min: float = ETA_MIN,
) -> Optional[LambdaLR]:
    """Create step-wise LR scheduler paired with `build_optimizer`."""
    if scheduler_name is None or str(scheduler_name).lower() == "none":
        return None

    name = str(scheduler_name).lower()
    total_steps = max(1, int(total_steps))
    warmup_steps = int(total_steps * warmup_ratio)
    base_lr = float(optimizer.defaults["lr"])
    min_ratio = min(1.0, max(0.0, eta_min / max(base_lr, 1e-12)))

    if name == "cosine_with_warmup":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda)

    if name == "poly":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return max(min_ratio, (1.0 - progress) ** 0.9)

        return LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unsupported SCHEDULER_NAME: {scheduler_name}")


def get_amp_dtype(precision: str = AMP_PRECISION) -> Optional[torch.dtype]:
    """Map precision string (`bf16`, `fp16`, `fp32`) to torch.dtype."""
    p = str(precision).lower()
    if p == "bf16":
        return torch.bfloat16
    if p == "fp16":
        return torch.float16
    if p == "fp32":
        return None
    raise ValueError(f"Unsupported AMP_PRECISION: {precision}")


def export_config_dict() -> Dict[str, Any]:
    """Collect all uppercase configuration constants into a JSON-serializable dict."""
    cfg: Dict[str, Any] = {}
    for k, v in globals().items():
        if not k.isupper():
            continue
        if isinstance(v, Path):
            cfg[k] = str(v)
        elif isinstance(v, (str, int, float, bool, list, tuple, dict, type(None))):
            cfg[k] = list(v) if isinstance(v, tuple) else v
    return cfg
