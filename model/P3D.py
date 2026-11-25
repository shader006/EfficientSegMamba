"""Pseudo-3D (P3D) bottleneck encoder/decoder blocks and pointwise-depthwise 3D conv."""

from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from .Onsampling import Onsampling
from .utils import _make_activation, _norm3d


class Pseudo3DBottleneckBlock(nn.Module):
    """Factorized Pseudo-3D bottleneck: 1x1x1 -> 1x3x3 -> 3x1x1 -> 1x1x1 + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_name: str = "instance",
        bottleneck_ratio: int = 4,
        res_block: bool = True,
        morton_z_enabled: bool = False,
    ) -> None:
        super().__init__()
        hidden_channels = max(out_channels // bottleneck_ratio, 8)
        self.res_block = res_block
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            _norm3d(hidden_channels, norm_name),
            _make_activation("gelu"),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            _norm3d(hidden_channels, norm_name),
            _make_activation("gelu"),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            _norm3d(hidden_channels, norm_name),
            _make_activation("gelu"),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=False),
            _norm3d(out_channels, norm_name),
        )
        self.act = _make_activation("gelu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.res_block:
            out = out + self.proj(x)
        return self.act(out)


class Pseudo3DUpBlock(nn.Module):
    """3D decoder upsampling block pairing `Onsampling` or `ConvTranspose3d` with P3D."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        upsample_kernel_size: int,
        norm_name: str = "instance",
        res_block: bool = True,
        upsample_mode: str = "transconv",
    ) -> None:
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("Pseudo3DUpBlock only supports spatial_dims=3.")
        if upsample_mode == "onsampling":
            self.upsample: nn.Module = Onsampling(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                dyscope=True,
            )
        else:
            self.upsample = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_kernel_size,
            )
        self.conv_block = Pseudo3DBottleneckBlock(
            out_channels + out_channels,
            out_channels,
            norm_name=norm_name,
            res_block=res_block,
        )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.upsample(inp)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)


class PWDWConv3D(nn.Module):
    """Pointwise-then-depthwise 3D convolution block."""

    def __init__(self, channels: int, norm_name: str = "batch") -> None:
        super().__init__()
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.pw_norm = _norm3d(channels, norm_name)
        self.pw_act = nn.ReLU(inplace=True)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True, groups=channels)
        self.dw_norm = _norm3d(channels, norm_name)
        self.dw_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw_act(self.pw_norm(self.pwconv(x)))
        return self.dw_act(self.dw_norm(self.dwconv(x)))


def _make_encoder_block(
    use_unet_3d_conv: bool,
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    norm_name: str,
    res_block: bool,
) -> nn.Module:
    if use_unet_3d_conv:
        return UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
    return Pseudo3DBottleneckBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        norm_name=norm_name,
        res_block=res_block,
    )


def _make_decoder_block(
    use_unet_3d_conv: bool,
    upsample_mode: str,
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    norm_name: str,
    res_block: bool,
) -> nn.Module:
    if use_unet_3d_conv:
        return UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
    return Pseudo3DUpBlock(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=res_block,
        upsample_mode=upsample_mode,
    )
