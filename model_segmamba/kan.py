import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Basis functions for Shifted Softplus KAN (SKAN)
# ---------------------------------------------------------------------------
def lshifted_softplus(x, k):
    """Shifted Softplus activation: y = log(1 + e^(kx)) - log(2)"""
    return F.softplus(k[0] * x) - np.log(2)


class UniversalKANSubLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        sublayer_type="add",
        function=lshifted_softplus,
        param_num=1,
        use_bias=True,
        device="cpu",
        init_method="default",
    ):
        super(UniversalKANSubLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sublayer_type = sublayer_type
        self.function = function
        self.param_num = param_num
        self.use_bias = use_bias
        self.init_method = init_method

        if self.use_bias:
            self.weight = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(self.out_features, self.in_features + 1, device=device))
                    for _ in range(self.param_num)
                ]
            )
        else:
            self.weight = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(self.out_features, self.in_features, device=device))
                    for _ in range(self.param_num)
                ]
            )
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.param_num):
            if self.init_method == "default":
                nn.init.kaiming_uniform_(self.weight[i], a=5 ** 0.5)
                if self.use_bias:
                    nn.init.constant_(self.weight[i][:, -1], 0)
            elif callable(self.init_method):
                self.init_method(self.weight[i])

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)
        output = self.function(x, self.weight)
        if self.sublayer_type == "add":
            return torch.sum(output, dim=-1)
        else:
            return torch.prod(output, dim=-1)


class UniversalKANLinear(nn.Module):
    def __init__(self, in_features, out_features, function_list=None, device="cpu", init_method="default"):
        super(UniversalKANLinear, self).__init__()
        if function_list is None or len(function_list) == 0:
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
        assert sum([f["node_num"] for f in self.basis_function]) == self.out_features, (
            "The sum of node_num should be equal to out_features"
        )
        for i in range(len(self.basis_function)):
            stype = self.basis_function[i].get("sublayer_type", "add")
            self.sublayers.append(
                UniversalKANSubLayer(
                    self.in_features,
                    self.basis_function[i]["node_num"],
                    stype,
                    self.basis_function[i]["function"],
                    self.basis_function[i]["param_num"],
                    self.basis_function[i]["use_bias"],
                    device=device,
                    init_method=init_method,
                )
            )

    def forward(self, x):
        return torch.cat([sublayer(x) for sublayer in self.sublayers], dim=-1)


class SKANLinear_pure(UniversalKANLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        basis_function=lshifted_softplus,
        sublayer_type="add",
        device="cpu",
        init_method="default",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.function_list = [
            {
                "function": basis_function,
                "param_num": 1,
                "sublayer_type": sublayer_type,
                "node_num": out_features,
                "use_bias": bias,
            }
        ]
        super(SKANLinear_pure, self).__init__(
            in_features,
            out_features,
            self.function_list,
            device=device,
            init_method=init_method,
        )

    def forward(self, x):
        return super(SKANLinear_pure, self).forward(x)


# Alias for SKANLinear
SKANLinear = SKANLinear_pure


# ---------------------------------------------------------------------------
# B-spline KANLinear (from UKAN / Efficient-KAN)
# ---------------------------------------------------------------------------
class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=None,
    ):
        super(KANLinear, self).__init__()
        if grid_range is None:
            grid_range = [-1, 1]
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output
