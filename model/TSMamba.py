"""TSMamba encoder blocks (GSC, MambaLayer with Mamba1/2/3 + MortonZ, TSMambaLayer, MambaEncoder)."""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .utils import LayerNorm, _feature_depths_from_input_size, _make_activation, _morton_perm_3d


def _import_advanced_mamba_classes() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Import `Mamba`, `Mamba2`, and `Mamba3` from environment `mamba_ssm` (v2.3+)
    without shadowing or colliding with `all/model/mamba_ssm` (v1.0.1 for official SegMamba).
    """
    model_dir = str(Path(__file__).resolve().parent)
    fallback_mamba_dir = Path(__file__).resolve().parents[2] / "enhanted-model" / "mamba"

    old_path = list(sys.path)
    old_mamba_modules = {
        k: sys.modules.pop(k)
        for k in list(sys.modules.keys())
        if k == "mamba_ssm" or k.startswith("mamba_ssm.")
    }

    try:
        sys.path = [p for p in old_path if Path(p or ".").resolve() != Path(model_dir).resolve()]
        try:
            from mamba_ssm import Mamba, Mamba2, Mamba3
            return Mamba, Mamba2, Mamba3
        except ImportError:
            for k in list(sys.modules.keys()):
                if k == "mamba_ssm" or k.startswith("mamba_ssm."):
                    sys.modules.pop(k, None)
            if fallback_mamba_dir.exists():
                sys.path.insert(0, str(fallback_mamba_dir))
                try:
                    from mamba_ssm import Mamba, Mamba2, Mamba3
                    return Mamba, Mamba2, Mamba3
                except ImportError:
                    pass
        return None, None, None
    finally:
        for k in list(sys.modules.keys()):
            if k == "mamba_ssm" or k.startswith("mamba_ssm."):
                sys.modules.pop(k, None)
        sys.modules.update(old_mamba_modules)
        sys.path = old_path


EnvMamba, Mamba2, Mamba3 = _import_advanced_mamba_classes()


class GSC(nn.Module):
    """Gated Spatial Convolution block used prior to Mamba token scanning."""

    def __init__(self, in_channles: int, mamba_impl: str = "mamba1") -> None:
        super().__init__()
        self.mamba_impl = str(mamba_impl).lower()

        if self.mamba_impl == "mamba3":
            self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
            self.norm = nn.InstanceNorm3d(in_channles)
            self.nonliner = nn.ReLU()

            self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
            self.norm2 = nn.InstanceNorm3d(in_channles)
            self.nonliner2 = nn.ReLU()

            self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
            self.norm3 = nn.InstanceNorm3d(in_channles)
            self.nonliner3 = nn.ReLU()

            self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
            self.norm4 = nn.InstanceNorm3d(in_channles)
            self.nonliner4 = nn.ReLU()
        else:
            self.conv3 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
            self.conv1 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
            self.out_conv3 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mamba_impl == "mamba3":
            x_residual = x
            x1 = self.nonliner(self.norm(self.proj(x)))
            x1 = self.nonliner2(self.norm2(self.proj2(x1)))
            x2 = self.nonliner3(self.norm3(self.proj3(x)))
            x = self.nonliner4(self.norm4(self.proj4(x1 + x2)))
            return x + x_residual
        return x + self.out_conv3(self.conv3(x) * self.conv1(x))


class MlpChannel(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = _make_activation("gelu")
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ParameterFreeIdentity(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MambaLayer(nn.Module):
    """3D Mamba token layer supporting Mamba1/2/3 and `mortonZ`/`raster_scan`/`tri_directional`."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_slices: Optional[int] = None,
        mamba3_headdim: int = 64,
        mamba3_chunk_size: int = 64,
        mamba3_mimo_enabled: bool = False,
        mamba3_mimo_rank: int = 4,
        mamba3_rope_fraction: float = 0.5,
        mamba3_outproj_norm_enabled: bool = False,
        mamba_impl: str = "mamba1",
        morton_z_enabled: bool = False,
        scan_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba_impl = str(mamba_impl).lower()

        if scan_mode is not None:
            sm = str(scan_mode).strip().lower()
            if sm in {"mortonz", "morton_z", "morton"}:
                self.scan_mode = "mortonZ"
                self.morton_z_enabled = True
            elif sm in {"raster_scan", "raster", "single_raster", "single"}:
                self.scan_mode = "raster_scan"
                self.morton_z_enabled = False
            elif sm in {"tri_directional", "tri_axis", "3_branches", "tri_raster", "tri"}:
                self.scan_mode = "tri_directional"
                self.morton_z_enabled = False
            else:
                self.scan_mode = str(scan_mode)
                self.morton_z_enabled = bool(morton_z_enabled)
        else:
            self.morton_z_enabled = bool(morton_z_enabled)
            if self.morton_z_enabled:
                self.scan_mode = "mortonZ"
            elif self.mamba_impl in {"mamba", "mamba1", "simple"}:
                self.scan_mode = "tri_directional"
            else:
                self.scan_mode = "raster_scan"

        self._morton_cache: Dict[Tuple[int, ...], Tuple[torch.Tensor, torch.Tensor]] = {}
        if self.mamba_impl == "mamba3":
            if Mamba3 is None:
                raise ImportError("Mamba3 could not be imported from environment `mamba_ssm`.")
            self.mamba = Mamba3(
                d_model=dim,
                d_state=d_state,
                expand=expand,
                headdim=self._resolve_headdim(dim=dim, expand=expand, preferred_headdim=mamba3_headdim),
                chunk_size=self._resolve_chunk_size(mamba3_chunk_size, mamba3_mimo_enabled, mamba3_mimo_rank),
                rope_fraction=mamba3_rope_fraction,
                is_outproj_norm=mamba3_outproj_norm_enabled,
                is_mimo=mamba3_mimo_enabled,
                mimo_rank=mamba3_mimo_rank,
            )
        elif self.mamba_impl == "mamba2":
            if Mamba2 is None:
                raise ImportError("Mamba2 could not be imported from environment `mamba_ssm`.")
            self.mamba = Mamba2(
                d_model=dim,
                d_state=max(d_state, 16),
                expand=expand,
                headdim=self._resolve_headdim(dim=dim, expand=expand, preferred_headdim=mamba3_headdim),
                chunk_size=max(mamba3_chunk_size, 64),
                use_mem_eff_path=False,
            )
        elif self.mamba_impl in {"mamba", "mamba1", "simple"}:
            mamba_cls = EnvMamba
            if mamba_cls is None:
                from mamba_ssm import Mamba as LocalMamba
                mamba_cls = LocalMamba
            sig = inspect.signature(mamba_cls.__init__)
            mamba_kwargs: Dict[str, Any] = {
                "d_model": dim,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
            }
            if "bimamba_type" in sig.parameters:
                mamba_kwargs["bimamba_type"] = "none"
            if "nslices" in sig.parameters:
                mamba_kwargs["nslices"] = num_slices
            self.mamba = mamba_cls(**mamba_kwargs)
        else:
            raise ValueError(f"Unsupported mamba implementation: {mamba_impl}")

    @staticmethod
    def _resolve_headdim(dim: int, expand: int, preferred_headdim: int) -> int:
        d_inner = int(dim * expand)
        preferred_headdim = min(preferred_headdim, d_inner)
        for candidate in (preferred_headdim, 64, 32, 16, 8):
            if candidate <= d_inner and d_inner % candidate == 0 and candidate % 8 == 0:
                return candidate
        for candidate in (64, 48, 32, 24, 16, 12, 8, 4, 2, 1):
            if candidate <= d_inner and d_inner % candidate == 0:
                return candidate
        gcd = math.gcd(d_inner, preferred_headdim)
        if gcd > 0:
            return gcd
        raise ValueError(f"Could not find a valid Mamba3 headdim for dim={dim}, expand={expand}")

    @staticmethod
    def _resolve_chunk_size(chunk_size: int, mimo_enabled: bool, mimo_rank: int) -> int:
        chunk_size = max(1, int(chunk_size))
        if not mimo_enabled:
            return chunk_size
        return max(1, chunk_size // max(1, int(mimo_rank)))

    def _get_morton_perm(
        self, spatial_shape: Tuple[int, ...], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = tuple(spatial_shape)
        if key not in self._morton_cache:
            self._morton_cache[key] = _morton_perm_3d(*spatial_shape, device=torch.device("cpu"))
        perm, inv = self._morton_cache[key]
        if perm.device != device:
            perm = perm.to(device=device)
            inv = inv.to(device=device)
        return perm, inv

    def _mamba_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        if c != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {c}")
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(b, c, n_tokens).transpose(-1, -2)

        if self.morton_z_enabled:
            perm, inv = self._get_morton_perm(img_dims, x.device)
            x_flat = x_flat[:, perm, :].contiguous()

        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        if self.morton_z_enabled:
            x_mamba = x_mamba[:, inv, :].contiguous()

        return x_mamba.transpose(-1, -2).reshape(b, c, *img_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        if self.scan_mode in {"tri_directional", "tri_axis", "3_branches", "tri_raster", "tri"}:
            from einops import rearrange

            out_x_1 = self._mamba_forward(x)

            x_2 = rearrange(x, "b c d w h -> b c w d h")
            out_x_2 = rearrange(self._mamba_forward(x_2), "b c w d h -> b c d w h")

            x_3 = rearrange(x, "b c d w h -> b c h w d")
            out_x_3 = rearrange(self._mamba_forward(x_3), "b c h w d -> b c d w h")

            return out_x_1 + out_x_2 + out_x_3 + x_skip

        return self._mamba_forward(x) + x_skip


class TSMambaLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_slices: Optional[int] = None,
        mlp_ratio: int = 2,
        mamba3_d_state: int = 16,
        mamba3_headdim: int = 64,
        mamba3_chunk_size: int = 64,
        mamba3_mimo_enabled: bool = False,
        mamba3_mimo_rank: int = 4,
        mamba3_rope_fraction: float = 0.5,
        mamba3_outproj_norm_enabled: bool = False,
        mamba_impl: str = "mamba1",
        morton_z_enabled: bool = False,
        scan_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.gsc = GSC(dim, mamba_impl=mamba_impl)
        self.tom = MambaLayer(
            dim=dim,
            num_slices=num_slices,
            d_state=mamba3_d_state,
            mamba3_headdim=mamba3_headdim,
            mamba3_chunk_size=mamba3_chunk_size,
            mamba3_mimo_enabled=mamba3_mimo_enabled,
            mamba3_mimo_rank=mamba3_mimo_rank,
            mamba3_rope_fraction=mamba3_rope_fraction,
            mamba3_outproj_norm_enabled=mamba3_outproj_norm_enabled,
            mamba_impl=mamba_impl,
            morton_z_enabled=morton_z_enabled,
            scan_mode=scan_mode,
        )
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MlpChannel(dim, mlp_ratio * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gsc(x)
        x = self.tom(x)
        return x + self.mlp(self.norm(x))


class MambaEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        depths: Sequence[int] = (2, 2, 2, 2),
        dims: Sequence[int] = (48, 96, 192, 384),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        input_size: Optional[Sequence[int]] = None,
        mamba_stages: Optional[Sequence[int]] = None,
        mamba3_d_state: int = 16,
        mamba3_headdim: int = 64,
        mamba3_chunk_size: int = 64,
        mamba3_mimo_enabled: bool = False,
        mamba3_mimo_rank: int = 4,
        mamba3_rope_fraction: float = 0.5,
        mamba3_outproj_norm_enabled: bool = False,
        mamba_impl: str = "mamba1",
        morton_z_enabled: bool = False,
        scan_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3, groups=in_chans),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        num_slices_list = _feature_depths_from_input_size(input_size, num_stages=4)
        self.num_slices_list = num_slices_list
        valid_stage_indices = set(range(len(dims)))
        if mamba_stages is None:
            mamba_stage_set = valid_stage_indices
        else:
            mamba_stage_set = {int(stage_idx) for stage_idx in mamba_stages}
            invalid_indices = sorted(mamba_stage_set - valid_stage_indices)
            if invalid_indices:
                raise ValueError(f"mamba_stages contains invalid stage indices: {invalid_indices}")
        self.mamba_stages = sorted(mamba_stage_set)
        for i in range(4):
            block_cls = TSMambaLayer if i in mamba_stage_set else ParameterFreeIdentity
            stage = nn.Sequential(
                *[
                    block_cls(
                        dim=dims[i],
                        num_slices=num_slices_list[i],
                        mamba3_d_state=mamba3_d_state,
                        mamba3_headdim=mamba3_headdim,
                        mamba3_chunk_size=mamba3_chunk_size,
                        mamba3_mimo_enabled=mamba3_mimo_enabled,
                        mamba3_mimo_rank=mamba3_mimo_rank,
                        mamba3_rope_fraction=mamba3_rope_fraction,
                        mamba3_outproj_norm_enabled=mamba3_outproj_norm_enabled,
                        mamba_impl=mamba_impl,
                        morton_z_enabled=morton_z_enabled,
                        scan_mode=scan_mode,
                    )
                    if block_cls is TSMambaLayer
                    else block_cls(dim=dims[i])
                    for _ in range(depths[i])
                ]
            )
            self.stages.append(stage)

        self.out_indices = list(out_indices)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs: List[torch.Tensor] = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.forward_features(x)
