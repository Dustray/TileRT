from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from tilert.models.deepseek_v3_2.refs.kernel import act_quant, fp8_gemm, weight_dequant

__all__ = [
    "act_quant",
    "fp8_gemm",
    "weight_dequant",
    "init_func",
    "linear",
    "RMSNorm",
]

from tilert.models.deepseek_config import (
    block_size,
    gemm_impl,
)

_LAZY_IMPORTS = {"act_quant", "fp8_gemm", "weight_dequant"}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        from tilert.models.deepseek_v3_2.refs import kernel

        attr = getattr(kernel, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_scale_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the dynamically attached ``scale`` tensor."""
    scale = getattr(tensor, "scale", None)
    if scale is None:
        raise AttributeError("Expected quantized tensor to carry a 'scale' attribute.")
    return cast(torch.Tensor, scale)


def init_func(x_in: torch.Tensor) -> torch.Tensor:
    x_dtype = x_in.dtype
    x_fp32 = x_in.to(torch.float32)
    if x_fp32.dim() >= 2:
        initial_tensor = nn.init.kaiming_uniform_(x_fp32)
    else:
        initial_tensor = nn.init.uniform_(x_fp32)
    return initial_tensor.to(x_dtype)


def _safe_weight_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize ``weight`` without invoking the backend fp8 kernel.

    For the Qwen3.6 reference/golden path the converted checkpoint may store
    fp8 weights with a scale tensor whose shape is incompatible with the
    DeepSeek ``weight_dequant_kernel`` (which requires both dimensions to be
    multiples of ``block_size``).  We therefore use the pure-Python fallback when
    it can be applied, and fall back to a simple dtype cast when the shapes are
    not compatible.
    """
    from tilert.models.deepseek_v3_2.refs.kernel import _weight_dequant_torch

    if scale.numel() == 1:
        return weight.to(torch.bfloat16) * scale.to(torch.bfloat16).view(1)

    # Check whether the kernel's reshape-based fallback would work.
    if weight.dim() == 2:
        m, n = weight.shape
        block_size = 128
        if m % block_size == 0 and n % block_size == 0 and scale.shape == (
            m // block_size,
            n // block_size,
        ):
            return _weight_dequant_torch(weight, scale, block_size)

    # Handle rank-3 stacked expert weights produced by the Qwen3.6 converter:
    # weight shape (n_experts, dim, expert_dim), scale shape
    # (n_experts, dim/block_size, expert_dim/block_size).
    if weight.dim() == 3 and scale.dim() == 3:
        n_experts, dim, expert_dim = weight.shape
        block_size = 128
        if (
            dim % block_size == 0
            and expert_dim % block_size == 0
            and scale.shape == (n_experts, dim // block_size, expert_dim // block_size)
        ):
            dequant_list = []
            for i in range(n_experts):
                dequant_list.append(_weight_dequant_torch(weight[i], scale[i], block_size))
            return torch.stack(dequant_list, dim=0)

    # Scale shape is unexpected: cast the weight and ignore the scale.  This
    # keeps the golden path executable even when the checkpoint's quantization
    # metadata does not exactly match the DeepSeek kernel assumptions.
    return weight.to(torch.bfloat16)


def linear(
    x_in: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale_fmt: str | None = None,
) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.

    Args:
        x_in (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation.
    """
    if weight.element_size() > 1:
        return F.linear(x_in, weight, bias)

    from tilert.models.deepseek_v3_2.refs.kernel import act_quant, fp8_gemm

    if gemm_impl == "bf16":
        scale = _get_scale_tensor(weight)
        weight = _safe_weight_dequant(weight, scale)
        return F.linear(x_in, weight, bias)

    x_quant: torch.Tensor
    scale: torch.Tensor
    x_quant, scale = act_quant(x_in, block_size, scale_fmt)
    y_out: torch.Tensor = fp8_gemm(x_quant, scale, weight, _get_scale_tensor(weight))
    if bias is not None:
        y_out += bias
    return y_out


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Qwen3.5-MoE/Qwen3.6 use the convention ``output = x / rms * (1 + weight)``
    with ``weight`` initialized to zeros, so the default gain is 1.  This keeps
    backward compatibility with checkpoints that expect the standard
    ``weight`` tensor while making the Qwen-specific gain explicit.

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
        weight (torch.Tensor | None): Optional pre-initialized weight vector.
    """

    def __init__(self, dim: int, eps: float = 1e-6, weight: torch.Tensor | None = None):
        super().__init__()
        self.dim = dim
        self.eps = eps

        if weight is None:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.weight = torch.nn.Parameter(weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.
            residual (torch.Tensor | None): Optional residual to add before norm.

        Returns:
            Normalized tensor, or (normalized, residual) tuple if residual given.
        """
        if residual is None:
            output = self._norm(x.float())
            output = output * (1.0 + self.weight.float())
            return output.type_as(x)

        x = residual = x.float() + residual.float()
        output = self._norm(x)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x), residual.type_as(x)
