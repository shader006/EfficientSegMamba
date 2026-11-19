#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions for BraTS 2023 3D segmentation inside `all/`.

Wraps MONAI's `DiceCELoss` to handle both:
1. `label_mode="multiclass"` (default for official SegMamba with `out_chans=4`):
   Automatically unsqueezes integer target `(B, X, Y, Z)` -> `(B, 1, X, Y, Z)`
   and computes Softmax + One-Hot Dice + Cross-Entropy loss.
2. `label_mode="regions"` (for `out_chans=3` nested BraTS targets `[WT, TC, ET]`):
   Computes Sigmoid Dice + Binary Cross-Entropy loss directly on `(B, 3, X, Y, Z)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.losses import DiceCELoss

import settings


class BraTSDiceCELoss(nn.Module):
    """Unified DiceCELoss wrapper matching `dataset.py` label modes."""

    def __init__(
        self,
        label_mode: str = settings.LABEL_MODE,
        lambda_dice: float = settings.DICE_LOSS_WEIGHT,
        lambda_ce: float = settings.CE_LOSS_WEIGHT,
        include_background: bool = settings.LOSS_INCLUDE_BACKGROUND,
        smooth_nr: float = settings.LOSS_SMOOTH_NR,
        smooth_dr: float = settings.LOSS_SMOOTH_DR,
    ) -> None:
        super().__init__()
        if label_mode not in ("multiclass", "regions"):
            raise ValueError(f"Unsupported label_mode: {label_mode}")

        self.label_mode = label_mode
        if label_mode == "multiclass":
            self.loss_fn = DiceCELoss(
                include_background=include_background,
                to_onehot_y=True,
                softmax=True,
                lambda_dice=lambda_dice,
                lambda_ce=lambda_ce,
                smooth_nr=smooth_nr,
                smooth_dr=smooth_dr,
            )
        else:
            # Multi-label BraTS regions (WT, TC, ET): all 3 channels are foreground.
            self.loss_fn = DiceCELoss(
                include_background=True,
                to_onehot_y=False,
                sigmoid=True,
                lambda_dice=lambda_dice,
                lambda_ce=lambda_ce,
                smooth_nr=smooth_nr,
                smooth_dr=smooth_dr,
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_mode == "multiclass":
            if targets.ndim == 4:
                targets = targets.unsqueeze(1)
            return self.loss_fn(logits, targets)

        return self.loss_fn(logits, targets.float())


def build_loss(
    label_mode: str = settings.LABEL_MODE,
    lambda_dice: float = settings.DICE_LOSS_WEIGHT,
    lambda_ce: float = settings.CE_LOSS_WEIGHT,
    include_background: bool = settings.LOSS_INCLUDE_BACKGROUND,
) -> nn.Module:
    """Instantiate the configured loss function from `settings.py`."""
    loss_name = str(settings.LOSS_NAME).lower()
    if loss_name in ("dicece", "dice_ce", "diceceloss"):
        return BraTSDiceCELoss(
            label_mode=label_mode,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            include_background=include_background,
        )
    raise ValueError(f"Unsupported LOSS_NAME in settings.py: {settings.LOSS_NAME}")


if __name__ == "__main__":
    # Quick self-test for both multiclass (4-class) and regions (3-channel) DiceCELoss.
    logits_mc = torch.randn(2, 4, 32, 32, 32, requires_grad=True)
    target_mc = torch.randint(0, 4, (2, 32, 32, 32), dtype=torch.int64)
    loss_mc = build_loss(label_mode="multiclass")
    val_mc = loss_mc(logits_mc, target_mc)
    val_mc.backward()
    assert torch.isfinite(val_mc) and logits_mc.grad is not None

    logits_reg = torch.randn(2, 3, 32, 32, 32, requires_grad=True)
    target_reg = torch.randint(0, 2, (2, 3, 32, 32, 32), dtype=torch.float32)
    loss_reg = build_loss(label_mode="regions")
    val_reg = loss_reg(logits_reg, target_reg)
    val_reg.backward()
    assert torch.isfinite(val_reg) and logits_reg.grad is not None

    print(f"[LOSS SELF-TEST PASSED] multiclass={val_mc.item():.4f} | regions={val_reg.item():.4f}")
