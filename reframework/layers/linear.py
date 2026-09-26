"""Dense linear layer.

FreeToken routes every Linear through a ``quant_method`` (fp8-block, marlin,
nvfp4, bf16). Pascal has no tensor cores and no fp8, so a *quantised* GEMM
would dequantise to fp16/fp32 and then run a slow fp32 matmul anyway — the
quantisation only adds VRAM overhead and load-time cost with no speed win.
Re therefore keeps a plain fp32 (or fp16-storage) weight and a single
``nn.functional.linear`` call. ``column/row`` parallelism is dropped too
(single-GPU by design on a 6-8 GB card).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOP


class Linear(BaseOP):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # weight is [out, in] (HF layout); nn.functional.linear multiplies x @ W^T
        # nn.Parameter (not a plain tensor) so BaseOP.state_dict()/load_state_dict
        # (which walk named_parameters()) actually see it.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    @staticmethod
    def _to_compute(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match x and weight to the compute dtype (fp32 on Pascal)."""
        if x.dtype == weight.dtype:
            return x, weight
        return x.to(weight.dtype), weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(x.device)
        # fp16 storage on Pascal: compute the GEMM in fp32 for accuracy
        compute = torch.float32 if (x.dtype == torch.float16 or weight.dtype == torch.float16) else x.dtype
        xf, wf = x.to(compute), weight.to(compute)
        out = F.linear(xf, wf)
        if self.bias is not None:
            out = out + self.bias.to(compute)
        return out.to(x.dtype)

    def load_state_dict(self, state_dict, *, prefix: str = "") -> None:
        w_key = f"{prefix}.weight"
        if w_key in state_dict:
            self.weight.copy_(state_dict[w_key].to(dtype=self.weight.dtype))
        b_key = f"{prefix}.bias"
        if self.bias is not None and b_key in state_dict:
            self.bias.copy_(state_dict[b_key].to(dtype=self.bias.dtype))


__all__ = ["Linear"]
