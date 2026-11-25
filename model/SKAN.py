"""Self-contained SKAN / KAN linear basis and Token-Pseudo3D blocks."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Onsampling import Onsampling
from .P3D import Pseudo3DBottleneckBlock, PWDWConv3D
from .utils import _make_activation, _morton_perm_3d, _norm3d, _valid_group_count


# -----------------------------------------------------------------------------
# Basis functions (from unikan.basis)
# -----------------------------------------------------------------------------

def lrelu(x: torch.Tensor, k: Any) -> torch.Tensor:
    return torch.clamp(k[0] * x, min=0)


def lleaky_relu(x: torch.Tensor, k: Any) -> torch.Tensor:
    return torch.max(k[0] * x, x)


def lswish(x: torch.Tensor, k: Any) -> torch.Tensor:
    return x / (1 + torch.exp(-k[0] * x))


def lmish(x: torch.Tensor, k: Any) -> torch.Tensor:
    return x * torch.tanh(torch.log(1 + torch.exp(k[0] * x)))


def lsoftplus(x: torch.Tensor, k: Any) -> torch.Tensor:
    return torch.log(1 + torch.exp(k[0] * x))


def lhard_sigmoid(x: torch.Tensor, k: Any) -> torch.Tensor:
    return torch.clamp(k[0] * x + 0.5, min=0, max=1)


def lelu(x: torch.Tensor, k: Any) -> torch.Tensor:
    return torch.where(x < 0, k[0] * (torch.exp(x / k[0]) - 1), x)


def lshifted_softplus(x: torch.Tensor, k: Any) -> torch.Tensor:
    return F.softplus(k[0] * x) - np.log(2)


def lgelup(x: torch.Tensor, k: Any) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (k[0] * x + 0.044715 * k[0] ** 3 * x[0] ** 3)))


def lsin(x: torch.Tensor, k: Any) -> torch.Tensor:
    return torch.sin(k[0] * x)


def lcos(x: torch.Tensor, k: Any) -> torch.Tensor:
    return torch.cos(k[0] * x)


def larctan(x: torch.Tensor, k: Any) -> torch.Tensor:
    return torch.arctan(k[0] * x)


# -----------------------------------------------------------------------------
# UniversalKAN / SKANLinear_pure (from unikan.universal_KAN & unikan.skan)
# -----------------------------------------------------------------------------

class UniversalKANSubLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sublayer_type: str = "add",
        function: Callable[..., torch.Tensor] = lshifted_softplus,
        param_num: int = 1,
        use_bias: bool = True,
        device: str = "cpu",
        init_method: Any = "default",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sublayer_type = sublayer_type
        self.function = function
        self.param_num = param_num
        self.use_bias = use_bias
        self.init_method = init_method

        cols = self.in_features + 1 if self.use_bias else self.in_features
        self.weight = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.out_features, cols, device=device))
                for _ in range(self.param_num)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.param_num):
            if self.init_method == "default":
                nn.init.kaiming_uniform_(self.weight[i], a=5 ** 0.5)
                if self.use_bias:
                    nn.init.constant_(self.weight[i][:, -1], 0)
            elif callable(self.init_method):
                self.init_method(self.weight[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)
        output = self.function(x, self.weight)
        if self.sublayer_type == "add":
            return torch.sum(output, dim=-1)
        return torch.prod(output, dim=-1)


class UniversalKANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        function_list: Optional[List[Dict[str, Any]]] = None,
        device: str = "cpu",
        init_method: Any = "default",
    ) -> None:
        super().__init__()
        if not function_list:
            function_list = [
                {
                    "function": lshifted_softplus,
                    "param_num": 1,
                    "sublayer_type": "add",
                    "node_num": out_features,
                    "use_bias": True,
                }
            ]
        self.in_features = in_features
        self.out_features = out_features
        self.basis_function = function_list
        self.sublayers = nn.ModuleList()
        if sum(fn["node_num"] for fn in self.basis_function) != self.out_features:
            raise AssertionError("The sum of node_num should be equal to out_features")
        for fn in self.basis_function:
            sublayer_type = "mul" if fn["sublayer_type"] == "mul" else "add"
            self.sublayers.append(
                UniversalKANSubLayer(
                    self.in_features,
                    fn["node_num"],
                    sublayer_type,
                    fn["function"],
                    fn["param_num"],
                    fn["use_bias"],
                    device=device,
                    init_method=init_method,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([sublayer(x) for sublayer in self.sublayers], dim=-1)


class SKANLinear_pure(UniversalKANLinear):
    """Single-basis Shifted-Softplus KAN linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        basis_function: Callable[..., torch.Tensor] = lshifted_softplus,
        sublayer_type: str = "add",
        device: str = "cpu",
        init_method: Any = "default",
    ) -> None:
        function_list = [
            {
                "function": basis_function,
                "param_num": 1,
                "sublayer_type": sublayer_type,
                "node_num": out_features,
                "use_bias": bias,
            }
        ]
        super().__init__(
            in_features,
            out_features,
            function_list,
            device=device,
            init_method=init_method,
        )


class SKAN_pure(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        basis_function: Callable[..., torch.Tensor] = lshifted_softplus,
        sublayer_type: str = "add",
        bias: bool = True,
        device: str = "cpu",
        init_method: Any = "default",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SKANLinear_pure(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    bias=bias,
                    basis_function=basis_function,
                    sublayer_type=sublayer_type,
                    device=device,
                    init_method=init_method,
                )
                for i in range(len(layer_sizes) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


SKANLinear = SKANLinear_pure


def _load_optional_kanlinear() -> Optional[Any]:
    """Load optional B-spline `KANLinear` from sibling `enhanted-model/UKAN-EP` if present."""
    kan_path = Path(__file__).resolve().parents[2] / "enhanted-model" / "UKAN-EP" / "main" / "kannet" / "kannet.py"
    if not kan_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("brats23_ukan_kannet", kan_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "KANLinear", None)


KANLinear = _load_optional_kanlinear()


# -----------------------------------------------------------------------------
# Token KAN / SKAN Pseudo-3D blocks (from advanced_model/model_segmamba/skan.py)
# -----------------------------------------------------------------------------

def _make_group_kan_linear(
    linear_cls: Any, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3
) -> nn.Module:
    if KANLinear is not None and linear_cls is KANLinear:
        return linear_cls(
            in_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
        )
    return linear_cls(in_features, out_features)


def _make_groupkan_spatial_mixer(
    channels: int,
    norm_name: str = "instance",
    bottleneck_ratio: int = 4,
    spatial_mixer: str = "pseudo3d",
) -> nn.Module:
    if str(spatial_mixer).lower() == "pwdw":
        return PWDWConv3D(channels, norm_name=norm_name)
    return Pseudo3DBottleneckBlock(
        channels,
        channels,
        norm_name=norm_name,
        bottleneck_ratio=bottleneck_ratio,
        res_block=False,
    )


class TokenKANPseudo3DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_name: str = "instance",
        bottleneck_ratio: int = 4,
        res_block: bool = True,
        morton_z_enabled: bool = False,
        linear_cls: Optional[Any] = None,
        linear_name: str = "KANLinear",
    ) -> None:
        super().__init__()
        linear_cls = KANLinear if linear_cls is None else linear_cls
        if linear_cls is None:
            raise ModuleNotFoundError(
                f"{linear_name} could not be loaded for TokenKANPseudo3DBlock."
            )

        self.res_block = res_block
        self.linear_cls = linear_cls
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.token_norm1 = nn.LayerNorm(in_channels)
        self.fc1 = self.linear_cls(in_channels, out_channels)
        self.token_norm2 = nn.LayerNorm(out_channels)
        self.mixer1 = Pseudo3DBottleneckBlock(
            out_channels,
            out_channels,
            norm_name=norm_name,
            bottleneck_ratio=bottleneck_ratio,
            res_block=False,
        )
        self.token_norm3 = nn.LayerNorm(out_channels)
        self.fc2 = self.linear_cls(out_channels, out_channels)
        self.mixer2 = Pseudo3DBottleneckBlock(
            out_channels,
            out_channels,
            norm_name=norm_name,
            bottleneck_ratio=bottleneck_ratio,
            res_block=False,
        )
        self.out_norm = _norm3d(out_channels, norm_name)
        self.act = _make_activation("gelu")
        self.morton_z_enabled = bool(morton_z_enabled)
        self._morton_cache: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _volume_to_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        b, c, d, h, w = x.shape
        tokens = x.reshape(b, c, d * h * w).transpose(1, 2).contiguous()
        if not self.morton_z_enabled:
            return tokens, {"mode": "global", "spatial_shape": (d, h, w)}

        perm, inv = self._get_morton_perm((d, h, w), x.device)
        tokens = tokens[:, perm, :].contiguous()
        return tokens, {
            "mode": "morton_z",
            "spatial_shape": (d, h, w),
            "inverse_perm": inv,
        }

    def _get_morton_perm(
        self, spatial_shape: Tuple[int, int, int], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = tuple(spatial_shape)
        if key not in self._morton_cache:
            self._morton_cache[key] = _morton_perm_3d(*spatial_shape, device=torch.device("cpu"))
        perm, inv = self._morton_cache[key]
        if perm.device != device:
            perm = perm.to(device=device)
            inv = inv.to(device=device)
        return perm, inv

    @staticmethod
    def _global_tokens_to_volume(
        tokens: torch.Tensor, spatial_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        b, _, c = tokens.shape
        d, h, w = spatial_shape
        return tokens.transpose(1, 2).reshape(b, c, d, h, w).contiguous()

    @staticmethod
    def _morton_tokens_to_volume(tokens: torch.Tensor, token_meta: Dict[str, Any]) -> torch.Tensor:
        inverse_perm = token_meta["inverse_perm"]
        spatial_shape = token_meta["spatial_shape"]
        tokens = tokens[:, inverse_perm, :].contiguous()
        return TokenKANPseudo3DBlock._global_tokens_to_volume(tokens, spatial_shape)

    @staticmethod
    def _tokens_to_volume(tokens: torch.Tensor, token_meta: Dict[str, Any]) -> torch.Tensor:
        if token_meta["mode"] == "global":
            return TokenKANPseudo3DBlock._global_tokens_to_volume(tokens, token_meta["spatial_shape"])
        if token_meta["mode"] == "morton_z":
            return TokenKANPseudo3DBlock._morton_tokens_to_volume(tokens, token_meta)
        raise ValueError(f"Unsupported token reconstruction mode: {token_meta['mode']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        tokens, token_meta = self._volume_to_tokens(x)
        tokens = self.token_norm1(tokens)
        tokens = self.fc1(tokens.reshape(-1, tokens.shape[-1])).reshape(tokens.shape[0], tokens.shape[1], -1)
        tokens = self.token_norm2(tokens)
        out = self._tokens_to_volume(tokens, token_meta)
        out = self.mixer1(out)

        tokens, token_meta = self._volume_to_tokens(out)
        tokens = self.token_norm3(tokens)
        tokens = self.fc2(tokens.reshape(-1, tokens.shape[-1])).reshape(tokens.shape[0], tokens.shape[1], -1)
        out = self._tokens_to_volume(tokens, token_meta)
        out = self.mixer2(out)
        out = self.out_norm(out)
        if self.res_block:
            out = out + residual
        return self.act(out)


class TokenSKANPseudo3DBlock(TokenKANPseudo3DBlock):
    """Token-mixing Pseudo-3D block backed by self-contained `SKANLinear_pure`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, linear_cls=SKANLinear, linear_name="SKANLinear_pure", **kwargs)


class GroupedKANActivation(nn.Module):
    def __init__(
        self,
        channels: int,
        group: int = 16,
        linear_cls: Optional[Any] = None,
        linear_name: str = "KANLinear",
    ) -> None:
        super().__init__()
        linear_cls = KANLinear if linear_cls is None else linear_cls
        if linear_cls is None:
            raise ModuleNotFoundError(f"{linear_name} could not be loaded for GroupedKANActivation.")

        self.channels = int(channels)
        self.group = _valid_group_count(self.channels, requested=group)
        self.channels_per_group = self.channels // self.group
        self.vectorized_lss = linear_cls is SKANLinear
        if self.vectorized_lss:
            self.weight = nn.Parameter(torch.empty(self.group, 2))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            with torch.no_grad():
                self.weight[:, -1].zero_()
            self.register_buffer("log2", torch.tensor(math.log(2.0)), persistent=False)
        else:
            self.group_kan = nn.ModuleList([linear_cls(1, 1) for _ in range(self.group)])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        if self.vectorized_lss:
            grouped = tokens.view(b, n, self.group, self.channels_per_group)
            scale = self.weight[:, 0].view(1, 1, self.group, 1)
            bias = self.weight[:, 1].view(1, 1, self.group, 1)
            grouped = F.softplus(grouped * scale) - self.log2
            grouped = grouped + F.softplus(bias) - self.log2
            return grouped.reshape(b, n, c)

        group_outputs = []
        for group_idx, kan in enumerate(self.group_kan):
            start = group_idx * self.channels_per_group
            end = start + self.channels_per_group
            group_tokens = tokens[:, :, start:end].reshape(-1, 1)
            group_tokens = kan(group_tokens).view(b, n, self.channels_per_group)
            group_outputs.append(group_tokens)
        return torch.cat(group_outputs, dim=2)


class GroupedKANTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group: int = 16,
        grid_size: int = 5,
        spline_order: int = 3,
        linear_cls: Optional[Any] = None,
        linear_name: str = "KANLinear",
    ) -> None:
        super().__init__()
        linear_cls = KANLinear if linear_cls is None else linear_cls
        if linear_cls is None:
            raise ModuleNotFoundError(f"{linear_name} could not be loaded for GroupedKANTransform.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.group = _valid_group_count(self.in_channels, self.out_channels, requested=group)
        self.in_per_group = self.in_channels // self.group
        self.out_per_group = self.out_channels // self.group
        self.vectorized_lss = linear_cls is SKANLinear
        if self.vectorized_lss:
            self.weight = nn.Parameter(torch.empty(self.group, self.out_per_group, self.in_per_group + 1))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            with torch.no_grad():
                self.weight[:, :, -1].zero_()
            self.register_buffer("log2", torch.tensor(math.log(2.0)), persistent=False)
        else:
            self.group_kan = nn.ModuleList(
                [
                    _make_group_kan_linear(
                        linear_cls,
                        self.in_per_group,
                        self.out_per_group,
                        grid_size=grid_size,
                        spline_order=spline_order,
                    )
                    for _ in range(self.group)
                ]
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        if self.vectorized_lss:
            grouped = tokens.view(b, n, self.group, self.in_per_group)
            input_weight = self.weight[:, :, :-1].view(1, 1, self.group, self.out_per_group, self.in_per_group)
            bias_weight = self.weight[:, :, -1].view(1, 1, self.group, self.out_per_group)
            grouped = grouped.unsqueeze(3)
            grouped = (F.softplus(grouped * input_weight) - self.log2).sum(dim=-1)
            grouped = grouped + F.softplus(bias_weight) - self.log2
            return grouped.reshape(b, n, self.out_channels)

        group_outputs = []
        for group_idx, kan in enumerate(self.group_kan):
            start = group_idx * self.in_per_group
            end = start + self.in_per_group
            group_tokens = tokens[:, :, start:end].reshape(b * n, self.in_per_group)
            group_tokens = kan(group_tokens).view(b, n, self.out_per_group)
            group_outputs.append(group_tokens)
        return torch.cat(group_outputs, dim=2)


class TokenGroupKANPseudo3DBlock(TokenKANPseudo3DBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_name: str = "instance",
        bottleneck_ratio: int = 4,
        res_block: bool = True,
        morton_z_enabled: bool = False,
        active_group: int = 16,
        channel_group: int = 16,
        spatial_mixer: str = "pseudo3d",
        linear_cls: Optional[Any] = None,
        linear_name: str = "KANLinear",
    ) -> None:
        nn.Module.__init__(self)
        linear_cls = KANLinear if linear_cls is None else linear_cls
        if linear_cls is None:
            raise ModuleNotFoundError(f"{linear_name} could not be loaded for TokenGroupKANPseudo3DBlock.")

        self.res_block = res_block
        self.linear_cls = linear_cls
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.token_norm1 = nn.LayerNorm(in_channels)
        self.gka1 = GroupedKANActivation(
            in_channels,
            group=active_group,
            linear_cls=linear_cls,
            linear_name=linear_name,
        )
        self.fc1 = GroupedKANTransform(
            in_channels,
            out_channels,
            group=channel_group,
            linear_cls=linear_cls,
            linear_name=linear_name,
        )
        self.spatial_mixer1 = _make_groupkan_spatial_mixer(
            out_channels,
            norm_name=norm_name,
            bottleneck_ratio=bottleneck_ratio,
            spatial_mixer=spatial_mixer,
        )
        self.token_norm2 = nn.LayerNorm(out_channels)
        self.gka2 = GroupedKANActivation(
            out_channels,
            group=active_group,
            linear_cls=linear_cls,
            linear_name=linear_name,
        )
        self.fc2 = GroupedKANTransform(
            out_channels,
            out_channels,
            group=channel_group,
            linear_cls=linear_cls,
            linear_name=linear_name,
        )
        self.spatial_mixer2 = _make_groupkan_spatial_mixer(
            out_channels,
            norm_name=norm_name,
            bottleneck_ratio=bottleneck_ratio,
            spatial_mixer=spatial_mixer,
        )
        self.out_norm = _norm3d(out_channels, norm_name)
        self.act = _make_activation("gelu")
        self.morton_z_enabled = bool(morton_z_enabled)
        self._morton_cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)

        tokens, token_meta = self._volume_to_tokens(x)
        tokens = self.token_norm1(tokens)
        tokens = self.gka1(tokens)
        tokens = self.fc1(tokens)
        out = self._tokens_to_volume(tokens, token_meta)
        out = self.spatial_mixer1(out)

        tokens, token_meta = self._volume_to_tokens(out)
        tokens = self.token_norm2(tokens)
        tokens = self.gka2(tokens)
        tokens = self.fc2(tokens)
        out = self._tokens_to_volume(tokens, token_meta)
        out = self.spatial_mixer2(out)

        out = self.out_norm(out)
        if self.res_block:
            out = out + residual
        return self.act(out)


def _build_upsampler(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    upsample_kernel_size: int,
    upsample_mode: str,
) -> nn.Module:
    if upsample_mode == "onsampling":
        return Onsampling(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            dyscope=True,
        )
    return nn.ConvTranspose3d(
        in_channels,
        out_channels,
        kernel_size=upsample_kernel_size,
        stride=upsample_kernel_size,
    )


class TokenKANPseudo3DUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        upsample_kernel_size: int,
        norm_name: str = "instance",
        res_block: bool = True,
        upsample_mode: str = "transconv",
        morton_z_enabled: bool = False,
    ) -> None:
        super().__init__()
        if spatial_dims != 3:
            raise ValueError("TokenKANPseudo3DUpBlock only supports spatial_dims=3.")
        self.upsample = _build_upsampler(
            spatial_dims, in_channels, out_channels, upsample_kernel_size, upsample_mode
        )
        self.conv_block = TokenKANPseudo3DBlock(
            out_channels + out_channels,
            out_channels,
            norm_name=norm_name,
            res_block=res_block,
            morton_z_enabled=morton_z_enabled,
        )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.upsample(inp)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)


class TokenSKANPseudo3DUpBlock(TokenKANPseudo3DUpBlock):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        upsample_kernel_size: int,
        norm_name: str = "instance",
        res_block: bool = True,
        upsample_mode: str = "transconv",
        morton_z_enabled: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        if spatial_dims != 3:
            raise ValueError("TokenSKANPseudo3DUpBlock only supports spatial_dims=3.")
        self.upsample = _build_upsampler(
            spatial_dims, in_channels, out_channels, upsample_kernel_size, upsample_mode
        )
        self.conv_block = TokenSKANPseudo3DBlock(
            out_channels + out_channels,
            out_channels,
            norm_name=norm_name,
            res_block=res_block,
            morton_z_enabled=morton_z_enabled,
        )


class TokenGroupKANPseudo3DUpBlock(TokenKANPseudo3DUpBlock):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        upsample_kernel_size: int,
        norm_name: str = "instance",
        res_block: bool = True,
        upsample_mode: str = "transconv",
        morton_z_enabled: bool = False,
        active_group: int = 16,
        channel_group: int = 16,
        spatial_mixer: str = "pseudo3d",
        linear_cls: Optional[Any] = None,
        linear_name: str = "KANLinear",
    ) -> None:
        nn.Module.__init__(self)
        if spatial_dims != 3:
            raise ValueError("TokenGroupKANPseudo3DUpBlock only supports spatial_dims=3.")
        self.upsample = _build_upsampler(
            spatial_dims, in_channels, out_channels, upsample_kernel_size, upsample_mode
        )
        self.conv_block = TokenGroupKANPseudo3DBlock(
            out_channels + out_channels,
            out_channels,
            norm_name=norm_name,
            res_block=res_block,
            morton_z_enabled=morton_z_enabled,
            active_group=active_group,
            channel_group=channel_group,
            spatial_mixer=spatial_mixer,
            linear_cls=linear_cls,
            linear_name=linear_name,
        )
