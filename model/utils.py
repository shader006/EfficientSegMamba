"""Shared utilities, normalization, activation, and Morton Z-order helpers for `all/model/`."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_project_settings():
    """Load `all/settings.py` if present in `sys.modules` or parent directory."""
    if "settings" in sys.modules:
        return sys.modules["settings"]

    settings_path = Path(__file__).resolve().parents[1] / "settings.py"
    if not settings_path.exists():
        return None

    settings_dir = str(settings_path.parent)
    added_to_path = settings_dir not in sys.path
    if added_to_path:
        sys.path.insert(0, settings_dir)

    try:
        import settings
        return settings
    except ImportError:
        try:
            spec = importlib.util.spec_from_file_location("settings", settings_path)
            if spec is None or spec.loader is None:
                return None
            settings_mod = importlib.util.module_from_spec(spec)
            sys.modules["settings"] = settings_mod
            spec.loader.exec_module(settings_mod)
            return settings_mod
        except Exception:
            return None
    finally:
        if added_to_path and settings_dir in sys.path:
            sys.path.remove(settings_dir)


def _setting_or_default(settings_obj: Any, name: str, default: Any) -> Any:
    """Return `getattr(settings_obj, name, default)` when `settings_obj` is loaded."""
    return getattr(settings_obj, name, default) if settings_obj is not None else default


def _make_activation(kind: str) -> nn.Module:
    """Instantiate an activation module from string identifier."""
    if kind == "relu":
        return nn.ReLU()
    if kind == "gelu":
        return nn.GELU()
    if kind == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation kind: {kind}")


class LayerNorm(nn.Module):
    """LayerNorm supporting both `channels_last` and `channels_first` 3D tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data_format: {data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


def _norm3d(channels: int, norm_name: str = "instance") -> nn.Module:
    """Create 3D InstanceNorm or BatchNorm layer."""
    if isinstance(norm_name, str) and norm_name.lower() in {"batch", "batchnorm", "batchnorm3d"}:
        return nn.BatchNorm3d(channels)
    return nn.InstanceNorm3d(channels)


def _valid_group_count(*channels: int, requested: int = 16) -> int:
    """Find largest group count <= `requested` dividing all given channel dimensions."""
    requested = max(1, int(requested))
    max_group = min(requested, *(int(channel) for channel in channels))
    for group in range(max_group, 0, -1):
        if all(int(channel) % group == 0 for channel in channels):
            return group
    return 1


def _part1by2(n: torch.Tensor) -> torch.Tensor:
    """Spread bits of a 10-bit integer so there are 2 zeros between each bit."""
    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8)) & 0x0300F00F
    n = (n | (n << 4)) & 0x030C30C3
    n = (n | (n << 2)) & 0x09249249
    return n


def _morton3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compute 3D Morton Z-order key from integer voxel coordinates."""
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


@torch.no_grad()
def _morton_perm_3d(
    d: int, h: int, w: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(perm, inv_perm)` indices mapping 3D grid `(D, H, W)` to Morton Z-order."""
    dd = torch.arange(d, device=device, dtype=torch.int32)
    hh = torch.arange(h, device=device, dtype=torch.int32)
    ww = torch.arange(w, device=device, dtype=torch.int32)
    try:
        zz, yy, xx = torch.meshgrid(dd, hh, ww, indexing="ij")
    except TypeError:
        zz, yy, xx = torch.meshgrid(dd, hh, ww)
    keys = _morton3d(zz.reshape(-1), yy.reshape(-1), xx.reshape(-1))
    perm = torch.argsort(keys)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=device)
    return perm.long(), inv.long()


def _conv_output_size(
    size: int, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1
) -> int:
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _feature_depths_from_input_size(
    input_size: Optional[Union[int, Sequence[int]]], num_stages: int = 4
) -> List[int]:
    """Compute encoder stage spatial depths from `input_size`."""
    if input_size is None:
        return [64, 32, 16, 8]
    if isinstance(input_size, int):
        depth = input_size
    else:
        if len(input_size) < 1:
            raise ValueError("input_size must contain at least the depth dimension.")
        depth = int(input_size[0])

    feature_depths: List[int] = []
    depth = _conv_output_size(depth, kernel_size=7, stride=2, padding=3)
    feature_depths.append(depth)
    for _ in range(1, num_stages):
        depth = _conv_output_size(depth, kernel_size=2, stride=2)
        feature_depths.append(depth)
    return feature_depths
