"""
Efficient / Advanced SegMamba 3D architecture inside `all/model/`.

Integrates the modular components split across `all/model/`:
- `TSMamba.py`    : `MambaEncoder` (`Mamba3` / `Mamba2` / `Mamba1`, `GSC`, Morton Z-order scan)
- `P3D.py`        : `Pseudo3DBottleneckBlock`, `Pseudo3DUpBlock`, `PWDWConv3D`
- `Onsampling.py` : `Onsampling` dynamic content-aware 3D upsampling
- `SKAN.py`       : `TokenSKANPseudo3DBlock`, `TokenKANPseudo3DBlock`, `TokenGroupKANPseudo3DBlock`
- `FUE.py`        : `FUE` (Feature Uncertainty Enhancement) skip-connection gating
- `utils.py`      : `LayerNorm`, Morton Z-curve permutation, normalization & settings helpers
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock

from .FUE import FUE
from .Onsampling import Onsampling
from .P3D import (
    PWDWConv3D,
    Pseudo3DBottleneckBlock,
    Pseudo3DUpBlock,
    _make_decoder_block,
    _make_encoder_block,
)
from .SKAN import (
    GroupedKANActivation,
    GroupedKANTransform,
    KANLinear,
    SKANLinear,
    TokenGroupKANPseudo3DBlock,
    TokenGroupKANPseudo3DUpBlock,
    TokenKANPseudo3DBlock,
    TokenKANPseudo3DUpBlock,
    TokenSKANPseudo3DBlock,
    TokenSKANPseudo3DUpBlock,
)
from .TSMamba import (
    GSC,
    MambaEncoder,
    MambaLayer,
    MlpChannel,
    ParameterFreeIdentity,
    TSMambaLayer,
)
from .utils import _load_project_settings, _setting_or_default


class EfficientSegMamba(nn.Module):
    """
    Advanced SegMamba 3D segmentation network.

    Defaults match the advanced model configuration in `enhanted-model/settings.py`:
    - `depths=[1, 1, 1, 1]`, `feat_size=[48, 96, 192, 384]`, `hidden_size=768`
    - `mamba_impl="mamba3"`, `mamba_stages=[0, 1, 2]`, `scan_mode="mortonZ"`
    - `use_unet_3d_conv=False` (Pseudo-3D), `skan_enabled=True`, `onsampling_enabled=True`
    - `FUE` gating on all 6 encoder skip/bottleneck outputs
    """

    def __init__(
        self,
        in_chans: Optional[int] = None,
        out_chans: Optional[int] = None,
        depths: Optional[Sequence[int]] = None,
        feat_size: Optional[Sequence[int]] = None,
        drop_path_rate: Optional[float] = None,
        layer_scale_init_value: Optional[float] = None,
        hidden_size: Optional[int] = None,
        norm_name: Optional[str] = None,
        conv_block: Optional[bool] = None,
        res_block: Optional[bool] = None,
        spatial_dims: Optional[int] = None,
        input_size: Optional[Sequence[int]] = None,
        mamba_stages: Optional[Sequence[int]] = None,
        kan_enabled: Optional[bool] = None,
        skan_enabled: Optional[bool] = None,
        groupkan_enabled: Optional[bool] = None,
        groupkan_active_group: Optional[int] = None,
        groupkan_channel_group: Optional[int] = None,
        groupkan_spatial_mixer: Optional[str] = None,
        kan_morton_z_enabled: Optional[bool] = None,
        scan_mode: Optional[str] = None,
        use_unet_3d_conv: Optional[bool] = None,
        onsampling_enabled: Optional[bool] = None,
        mamba_impl: Optional[str] = None,
        use_settings: bool = True,
    ) -> None:
        super().__init__()

        settings = _load_project_settings() if use_settings else None

        in_chans = (
            in_chans
            if in_chans is not None
            else _setting_or_default(settings, "IN_CHANS", _setting_or_default(settings, "SEGMAMBA_IN_CHANS", 4))
        )
        out_chans = (
            out_chans
            if out_chans is not None
            else _setting_or_default(settings, "OUT_CHANS", _setting_or_default(settings, "SEGMAMBA_OUT_CHANS", 4))
        )
        depths = (
            list(depths)
            if depths is not None
            else list(_setting_or_default(settings, "EFFICIENT_SEGMAMBA_DEPTHS", [1, 1, 1, 1]))
        )
        feat_size = (
            list(feat_size)
            if feat_size is not None
            else list(_setting_or_default(settings, "SEGMAMBA_FEAT_SIZE", [48, 96, 192, 384]))
        )
        drop_path_rate = (
            drop_path_rate
            if drop_path_rate is not None
            else _setting_or_default(settings, "SEGMAMBA_DROP_PATH_RATE", 0.0)
        )
        layer_scale_init_value = (
            layer_scale_init_value
            if layer_scale_init_value is not None
            else _setting_or_default(settings, "SEGMAMBA_LAYER_SCALE_INIT_VALUE", 1e-6)
        )
        hidden_size = (
            hidden_size if hidden_size is not None else _setting_or_default(settings, "SEGMAMBA_HIDDEN_SIZE", 768)
        )
        norm_name = norm_name if norm_name is not None else _setting_or_default(settings, "SEGMAMBA_NORM_NAME", "instance")
        conv_block = conv_block if conv_block is not None else _setting_or_default(settings, "SEGMAMBA_CONV_BLOCK", True)
        res_block = res_block if res_block is not None else _setting_or_default(settings, "SEGMAMBA_RES_BLOCK", True)

        kan_enabled = kan_enabled if kan_enabled is not None else _setting_or_default(settings, "SEGMAMBA_KAN", False)
        skan_enabled = skan_enabled if skan_enabled is not None else _setting_or_default(settings, "SEGMAMBA_SKAN", True)
        groupkan_enabled = (
            groupkan_enabled
            if groupkan_enabled is not None
            else _setting_or_default(settings, "SEGMAMBA_GROUPKAN", False)
        )
        groupkan_active_group = (
            groupkan_active_group
            if groupkan_active_group is not None
            else _setting_or_default(settings, "SEGMAMBA_GROUPKAN_ACTIVE_GROUP", 16)
        )
        groupkan_channel_group = (
            groupkan_channel_group
            if groupkan_channel_group is not None
            else _setting_or_default(settings, "SEGMAMBA_GROUPKAN_CHANNEL_GROUP", 16)
        )
        groupkan_spatial_mixer = (
            groupkan_spatial_mixer
            if groupkan_spatial_mixer is not None
            else _setting_or_default(settings, "SEGMAMBA_GROUPKAN_SPATIAL_MIXER", "pseudo3d")
        )

        mamba_impl = (
            mamba_impl
            if mamba_impl is not None
            else _setting_or_default(settings, "ADVANCED_SEGMAMBA_MAMBA_IMPL", "mamba3")
        )

        raw_scan_mode = (
            scan_mode if scan_mode is not None else _setting_or_default(settings, "SEGMAMBA_SCAN_MODE", "mortonZ")
        )
        if raw_scan_mode is not None:
            sm = str(raw_scan_mode).strip().lower()
            if sm in {"mortonz", "morton_z", "morton"}:
                self.scan_mode = "mortonZ"
                kan_morton_z_enabled = True
            elif sm in {"raster_scan", "raster", "single_raster", "single"}:
                self.scan_mode = "raster_scan"
                kan_morton_z_enabled = False
            elif sm in {"tri_directional", "tri_axis", "3_branches", "tri_raster", "tri"}:
                self.scan_mode = "tri_directional"
                kan_morton_z_enabled = False
            else:
                self.scan_mode = str(raw_scan_mode)
                kan_morton_z_enabled = bool(kan_morton_z_enabled) if kan_morton_z_enabled is not None else False
        else:
            kan_morton_z_enabled = (
                kan_morton_z_enabled
                if kan_morton_z_enabled is not None
                else _setting_or_default(settings, "SEGMAMBA_KAN_MORTON_Z", True)
            )
            if kan_morton_z_enabled:
                self.scan_mode = "mortonZ"
            elif str(mamba_impl).lower() in {"mamba", "mamba1", "simple"}:
                self.scan_mode = "tri_directional"
            else:
                self.scan_mode = "raster_scan"

        spatial_dims = (
            spatial_dims if spatial_dims is not None else _setting_or_default(settings, "SEGMAMBA_SPATIAL_DIMS", 3)
        )
        input_size = (
            list(input_size)
            if input_size is not None
            else list(_setting_or_default(settings, "INPUT_SIZE", [128, 128, 128]))
        )
        mamba_stages = (
            list(mamba_stages)
            if mamba_stages is not None
            else list(_setting_or_default(settings, "SEGMAMBA_MAMBA_STAGES", [0, 1, 2]))
        )

        mamba_impl_lower = str(mamba_impl).lower()
        if mamba_impl_lower == "mamba3":
            mamba3_d_state = _setting_or_default(settings, "ADVANCED_SEGMAMBA_MAMBA3_D_STATE", 64)
        elif mamba_impl_lower == "mamba2":
            mamba3_d_state = _setting_or_default(settings, "ADVANCED_SEGMAMBA_MAMBA2_D_STATE", 128)
        else:
            mamba3_d_state = _setting_or_default(settings, "ADVANCED_SEGMAMBA_MAMBA1_D_STATE", 16)

        mamba3_headdim = _setting_or_default(settings, "ADVANCED_SEGMAMBA_MAMBA3_HEADDIM", 64)
        mamba3_chunk_size = _setting_or_default(settings, "ADVANCED_SEGMAMBA_MAMBA3_CHUNK_SIZE", 64)
        mamba3_mimo_enabled = _setting_or_default(settings, "ADVANCED_SEGMAMBA_MAMBA3_MIMO_ENABLED", False)
        mamba3_mimo_rank = _setting_or_default(settings, "ADVANCED_SEGMAMBA_MAMBA3_MIMO_RANK", 4)
        mamba3_rope_fraction = _setting_or_default(settings, "ADVANCED_SEGMAMBA_MAMBA3_ROPE_FRACTION", 0.5)
        mamba3_outproj_norm_enabled = _setting_or_default(
            settings, "ADVANCED_SEGMAMBA_MAMBA3_OUTPROJ_NORM_ENABLED", False
        )

        use_unet_3d_conv = (
            use_unet_3d_conv
            if use_unet_3d_conv is not None
            else _setting_or_default(settings, "SEGMAMBA_3D_CONV", False)
        )
        onsampling_enabled = (
            onsampling_enabled
            if onsampling_enabled is not None
            else _setting_or_default(settings, "SEGMAMBA_ONSAMPLING", True)
        )
        upsample_mode = "onsampling" if onsampling_enabled else "transconv"

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.input_size = input_size
        self.upsample_mode = upsample_mode
        self.mamba_stages = list(mamba_stages)
        self.kan_enabled = bool(kan_enabled)
        self.skan_enabled = bool(skan_enabled)
        self.groupkan_enabled = bool(groupkan_enabled)
        self.groupkan_active_group = int(groupkan_active_group)
        self.groupkan_channel_group = int(groupkan_channel_group)
        self.groupkan_spatial_mixer = str(groupkan_spatial_mixer).lower()
        self.kan_morton_z_enabled = bool(kan_morton_z_enabled)
        self.use_unet_3d_conv = bool(use_unet_3d_conv)
        self.mamba3_mimo_enabled = bool(mamba3_mimo_enabled)
        self.mamba3_mimo_rank = int(mamba3_mimo_rank)
        self.mamba3_rope_fraction = float(mamba3_rope_fraction)
        self.mamba3_outproj_norm_enabled = bool(mamba3_outproj_norm_enabled)
        self.spatial_dims = spatial_dims

        self.vit = MambaEncoder(
            in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            input_size=input_size,
            mamba_stages=mamba_stages,
            mamba3_d_state=mamba3_d_state,
            mamba3_headdim=mamba3_headdim,
            mamba3_chunk_size=mamba3_chunk_size,
            mamba3_mimo_enabled=self.mamba3_mimo_enabled,
            mamba3_mimo_rank=self.mamba3_mimo_rank,
            mamba3_rope_fraction=self.mamba3_rope_fraction,
            mamba3_outproj_norm_enabled=self.mamba3_outproj_norm_enabled,
            mamba_impl=mamba_impl,
            morton_z_enabled=self.kan_morton_z_enabled,
            scan_mode=self.scan_mode,
        )

        self.encoder1 = _make_encoder_block(
            use_unet_3d_conv=self.use_unet_3d_conv,
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = _make_encoder_block(
            use_unet_3d_conv=self.use_unet_3d_conv,
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = _make_encoder_block(
            use_unet_3d_conv=self.use_unet_3d_conv,
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = _make_encoder_block(
            use_unet_3d_conv=self.use_unet_3d_conv,
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            norm_name=norm_name,
            res_block=res_block,
        )

        use_token_kan = (self.groupkan_enabled or self.kan_enabled or self.skan_enabled) and not self.use_unet_3d_conv
        late_encoder_kwargs: Dict[str, Any] = {}
        if self.use_unet_3d_conv:
            late_encoder_block = None
        elif self.groupkan_enabled:
            late_encoder_block = TokenGroupKANPseudo3DBlock
            groupkan_linear_cls = SKANLinear if self.skan_enabled else KANLinear
            groupkan_linear_name = "SKANLinear_pure" if self.skan_enabled else "KANLinear"
            late_encoder_kwargs = {
                "active_group": self.groupkan_active_group,
                "channel_group": self.groupkan_channel_group,
                "spatial_mixer": self.groupkan_spatial_mixer,
                "linear_cls": groupkan_linear_cls,
                "linear_name": groupkan_linear_name,
            }
        elif self.skan_enabled:
            late_encoder_block = TokenSKANPseudo3DBlock
        elif self.kan_enabled:
            late_encoder_block = TokenKANPseudo3DBlock
        else:
            late_encoder_block = Pseudo3DBottleneckBlock

        if self.use_unet_3d_conv:
            self.encoder5 = _make_encoder_block(
                use_unet_3d_conv=True,
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[3],
                out_channels=self.feat_size[3],
                norm_name=norm_name,
                res_block=res_block,
            )
        else:
            self.encoder5 = late_encoder_block(
                in_channels=self.feat_size[3],
                out_channels=self.feat_size[3],
                norm_name=norm_name,
                res_block=res_block,
                morton_z_enabled=self.kan_morton_z_enabled,
                **late_encoder_kwargs,
            )

        self.bottleneck_downsample = nn.Sequential(
            nn.InstanceNorm3d(self.feat_size[3]),
            nn.Conv3d(self.feat_size[3], self.hidden_size, kernel_size=2, stride=2),
        )

        if self.use_unet_3d_conv:
            self.encoder6 = _make_encoder_block(
                use_unet_3d_conv=True,
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                norm_name=norm_name,
                res_block=res_block,
            )
        else:
            self.encoder6 = late_encoder_block(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                norm_name=norm_name,
                res_block=res_block,
                morton_z_enabled=self.kan_morton_z_enabled,
                **late_encoder_kwargs,
            )

        # Matches advanced_model: when skan_enabled=True, decoder5/decoder4 use P3D + Onsampling.
        if use_token_kan and not self.skan_enabled:
            decoder_block_cls = (
                TokenGroupKANPseudo3DUpBlock if self.groupkan_enabled else TokenKANPseudo3DUpBlock
            )
            self.decoder5 = decoder_block_cls(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=self.feat_size[3],
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                upsample_mode=upsample_mode,
                morton_z_enabled=self.kan_morton_z_enabled,
                **late_encoder_kwargs,
            )
            self.decoder4 = decoder_block_cls(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[3],
                out_channels=self.feat_size[2],
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                upsample_mode=upsample_mode,
                morton_z_enabled=self.kan_morton_z_enabled,
                **late_encoder_kwargs,
            )
        else:
            self.decoder5 = _make_decoder_block(
                use_unet_3d_conv=self.use_unet_3d_conv,
                upsample_mode=upsample_mode,
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=self.feat_size[3],
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder4 = _make_decoder_block(
                use_unet_3d_conv=self.use_unet_3d_conv,
                upsample_mode=upsample_mode,
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[3],
                out_channels=self.feat_size[2],
                norm_name=norm_name,
                res_block=res_block,
            )

        self.decoder3 = _make_decoder_block(
            use_unet_3d_conv=self.use_unet_3d_conv,
            upsample_mode=upsample_mode,
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = _make_decoder_block(
            use_unet_3d_conv=self.use_unet_3d_conv,
            upsample_mode=upsample_mode,
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = _make_decoder_block(
            use_unet_3d_conv=self.use_unet_3d_conv,
            upsample_mode=upsample_mode,
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            norm_name=norm_name,
            res_block=res_block,
        )

        self.fue1 = FUE()
        self.fue2 = FUE()
        self.fue3 = FUE()
        self.fue4 = FUE()
        self.fue5 = FUE()
        self.fue6 = FUE()
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        outs = self.vit(x_in)
        enc1 = self.fue1(self.encoder1(x_in))
        enc2 = self.fue2(self.encoder2(outs[0]))
        enc3 = self.fue3(self.encoder3(outs[1]))
        enc4 = self.fue4(self.encoder4(outs[2]))
        enc5 = self.fue5(self.encoder5(outs[3]))
        enc_hidden = self.fue6(self.encoder6(self.bottleneck_downsample(enc5)))
        dec3 = self.decoder5(enc_hidden, enc5)
        dec2 = self.decoder4(dec3, enc4)
        dec1 = self.decoder3(dec2, enc3)
        dec0 = self.decoder2(dec1, enc2)
        out = self.decoder1(dec0, enc1)
        return self.out(out)


SegMamba = EfficientSegMamba

__all__ = [
    "EfficientSegMamba",
    "SegMamba",
    "FUE",
    "Onsampling",
    "Pseudo3DBottleneckBlock",
    "Pseudo3DUpBlock",
    "PWDWConv3D",
    "TokenKANPseudo3DBlock",
    "TokenSKANPseudo3DBlock",
    "TokenGroupKANPseudo3DBlock",
    "TokenKANPseudo3DUpBlock",
    "TokenSKANPseudo3DUpBlock",
    "TokenGroupKANPseudo3DUpBlock",
    "GroupedKANActivation",
    "GroupedKANTransform",
    "GSC",
    "MambaLayer",
    "TSMambaLayer",
    "MambaEncoder",
    "MlpChannel",
    "ParameterFreeIdentity",
]
