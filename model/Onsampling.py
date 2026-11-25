"""Dynamic content-aware 2D/3D upsampling (`Onsampling`, adapted from Swin-DER)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Conv
from monai.networks.utils import pixelshuffle


def normal_init(module: nn.Module, mean: float = 0.0, std: float = 1.0, bias: float = 0.0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module: nn.Module, val: float, bias: float = 0.0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Onsampling(nn.Module):
    """Deformable content-aware upsampling predicting sampling offsets & neighbor weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int,
        mid_channels: int = 64,
        scale: int = 2,
        kernel_size_encoder: int = 3,
        dyscope: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.spatial_dims = spatial_dims

        if in_channels != out_channels:
            self.preconv = Conv[Conv.CONV, self.spatial_dims](
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
            in_channels = out_channels

        self.comp = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=out_channels,
            out_channels=mid_channels,
            kernel_size=1,
            strides=1,
            act="RELU",
            norm="INSTANCE",
        )
        self.enc = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=mid_channels,
            out_channels=(scale * 2) ** self.spatial_dims,
            kernel_size=kernel_size_encoder,
            strides=1,
            act=None,
            norm="INSTANCE",
        )

        self.offset = Conv[Conv.CONV, self.spatial_dims](
            in_channels=in_channels,
            out_channels=self.spatial_dims * scale ** self.spatial_dims,
            kernel_size=1,
        )
        normal_init(self.offset, std=0.001)

        if dyscope:
            self.scope = Conv[Conv.CONV, self.spatial_dims](
                in_channels=in_channels,
                out_channels=self.spatial_dims * scale ** self.spatial_dims,
                kernel_size=1,
            )
            constant_init(self.scope, val=0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> torch.Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        if self.spatial_dims == 2:
            return torch.stack(torch.meshgrid(h, h, indexing="ij")).transpose(1, 2).reshape(1, -1, 1, 1)
        if self.spatial_dims == 3:
            return torch.stack(torch.meshgrid(h, h, h, indexing="ij")).transpose(1, 3).reshape(1, -1, 1, 1, 1)
        raise ValueError(f"Unsupported spatial_dims: {self.spatial_dims}")

    def pixelshuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange channel elements into spatial upsampled dimensions."""
        dim, factor = self.spatial_dims, self.scale
        input_size = list(x.size())
        keeped_dim = input_size[: -(dim + 1)]
        channels = input_size[-(dim + 1)]

        if len(keeped_dim) == 2 and dim != keeped_dim[1]:
            raise ValueError(
                f"The data has a dimension of {dim}, while the coordinate dimension is {keeped_dim[1]}."
            )
        scale_divisor = factor ** dim
        if channels % scale_divisor != 0:
            raise ValueError(
                f"Number of input channels ({channels}) must be evenly "
                f"divisible by scale_factor ** dimensions ({factor}**{dim}={scale_divisor})."
            )

        spatial_start_idx = len(keeped_dim) + 1
        org_channels = int(channels // scale_divisor)
        output_size = keeped_dim + [org_channels] + [d * factor for d in input_size[spatial_start_idx:]]

        indices = list(range(spatial_start_idx, spatial_start_idx + 2 * dim))
        indices = indices[dim:] + indices[:dim]
        permute_indices = list(range(spatial_start_idx))
        for idx in range(dim):
            permute_indices.extend(indices[idx::dim])

        x = x.reshape(keeped_dim + [org_channels] + [factor] * dim + input_size[spatial_start_idx:])
        return x.permute(permute_indices).reshape(output_size)

    def get_grid(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos

        if self.spatial_dims == 2:
            b, _, h, w = offset.shape
            offset = offset.view(b, 2, -1, h, w)
            coords_h = torch.arange(h) + 0.5
            coords_w = torch.arange(w) + 0.5
            coords = (
                torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij"))
                .transpose(1, 2)
                .unsqueeze(1)
                .unsqueeze(0)
                .type(x.dtype)
                .to(x.device)
            )
            return self.pixelshuffle(coords + offset).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        if self.spatial_dims == 3:
            b, _, d, h, w = offset.shape
            offset = offset.view(b, 3, -1, d, h, w)
            coords_d = torch.arange(d) + 0.5
            coords_h = torch.arange(h) + 0.5
            coords_w = torch.arange(w) + 0.5
            coords = (
                torch.stack(torch.meshgrid(coords_w, coords_h, coords_d, indexing="ij"))
                .transpose(1, 3)
                .unsqueeze(1)
                .unsqueeze(0)
                .type(x.dtype)
                .to(x.device)
            )
            return self.pixelshuffle(coords + offset).permute(0, 2, 3, 4, 5, 1).contiguous().flatten(0, 1)

        raise ValueError(f"Unsupported spatial_dims: {self.spatial_dims}")

    def weighted_neighbor_sample(
        self, x: torch.Tensor, grid: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        if self.spatial_dims == 2:
            _, _, h, w = x.shape
            offsets = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=x.dtype, device=x.device)
            normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 1, 1, 2)
            out = None
            for k in range(offsets.shape[0]):
                coords = 2 * (grid + offsets[k].view(1, 1, 1, 2)) / normalizer - 1
                sampled = F.grid_sample(x, coords, mode="bilinear", align_corners=False, padding_mode="border")
                weighted = sampled * weights[:, k : k + 1]
                out = weighted if out is None else out + weighted
            return out

        if self.spatial_dims == 3:
            _, _, d, h, w = x.shape
            offsets = torch.tensor(
                [
                    [-1, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, 1, 1],
                ],
                dtype=x.dtype,
                device=x.device,
            )
            normalizer = torch.tensor([w, h, d], dtype=x.dtype, device=x.device).view(1, 1, 1, 1, 3)
            out = None
            for k in range(offsets.shape[0]):
                coords = 2 * (grid + offsets[k].view(1, 1, 1, 1, 3)) / normalizer - 1
                sampled = F.grid_sample(x, coords, mode="bilinear", align_corners=False, padding_mode="border")
                weighted = sampled * weights[:, k : k + 1]
                out = weighted if out is None else out + weighted
            return out

        raise ValueError(f"Unsupported spatial_dims: {self.spatial_dims}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "preconv"):
            x = self.preconv(x)
        w = self.comp(x)
        w = self.enc(w)
        w = pixelshuffle(w, spatial_dims=self.spatial_dims, scale_factor=self.scale)
        w = F.softmax(w, dim=1)

        grid = self.get_grid(x)
        return self.weighted_neighbor_sample(x, grid, w)
