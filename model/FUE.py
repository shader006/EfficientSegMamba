"""Feature Uncertainty Enhancement (FUE) skip-connection gating module."""

from __future__ import annotations

import torch
import torch.nn as nn


class FUE(nn.Module):
    """Reweight encoder features using channel-mean Shannon entropy uncertainty."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_bar = self.activation(x.mean(dim=1, keepdim=True)).clamp(min=self.eps)
        uncertainty = -z_bar * torch.log(z_bar)
        return x + x * (1.0 - uncertainty)
