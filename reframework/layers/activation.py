"""Gated activations — pure PyTorch.

FreeToken has flashinfer/triton ``silu_and_mul`` / ``gelu_and_mul``. Re computes
the same thing with plain ops; on Pascal the elementwise work is bandwidth
bound and PyTorch's vectorised kernels are fine. The [gate; up] halves are
*uninterleaved* (first half = gate, second half = up) — matches HF Qwen/Llama.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``silu(gate) * up`` where ``x = [gate; up]`` along the last dim."""
    d = x.shape[-1]
    assert d % 2 == 0
    gate, up = x[..., : d // 2], x[..., d // 2 :]
    res = F.silu(gate) * up
    if out is not None:
        out.copy_(res)
        return out
    return res


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    d = x.shape[-1]
    assert d % 2 == 0
    gate, up = x[..., : d // 2], x[..., d // 2 :]
    res = F.gelu(gate) * up
    if out is not None:
        out.copy_(res)
        return out
    return res


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    d = x.shape[-1]
    assert d % 2 == 0
    gate, up = x[..., : d // 2], x[..., d // 2 :]
    res = F.gelu(gate, approximate="tanh") * up
    if out is not None:
        out.copy_(res)
        return out
    return res


GATED_ACTIVATIONS = ("silu", "gelu", "gelu_tanh")


def gated_act_and_mul(activation: str, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if activation == "silu":
        return silu_and_mul(x, out)
    if activation == "gelu":
        return gelu_and_mul(x, out)
    if activation == "gelu_tanh":
        return gelu_tanh_and_mul(x, out)
    raise ValueError(f"unknown gated activation {activation!r}; known: {GATED_ACTIVATIONS}")


__all__ = [
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "gated_act_and_mul",
    "GATED_ACTIVATIONS",
]
