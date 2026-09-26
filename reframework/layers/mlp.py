"""Dense MLP (SwiGLU / GELU).

FreeToken's MLP is a thin wrapper over its quantised fused-Linear (gate/up/down
in one bank). Re uses three plain :class:`Linear` layers with the gated
activation between gate and up. On Pascal the fused triton kernel would not
exist, so this is the honest implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .activation import GATED_ACTIVATIONS, gated_act_and_mul
from .base import BaseOP
from .linear import Linear


class MLP(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        assert activation in GATED_ACTIVATIONS, f"unsupported activation {activation!r}"
        self.activation = activation
        # gate & up are concatenated into one GEMM (matches HF's gate_proj/up_proj
        # being the same shape), then split by the gated activation.
        self.gate_up = Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up.forward(x)
        act = gated_act_and_mul(self.activation, gate_up)
        return self.down.forward(act)

    def state_dict(self, *, prefix: str = "", result=None):
        result = {} if result is None else result
        # HF-style names (gate_proj/up_proj are the two halves of the fused
        # gate_up bank, each [I, H]) so a Llama/Qwen checkpoint loads as-is.
        w = self.gate_up.weight  # [2I, H]
        result[f"{prefix}.gate_proj.weight"] = w[: self.gate_up.out_features // 2]
        result[f"{prefix}.up_proj.weight"] = w[self.gate_up.out_features // 2 :]
        result[f"{prefix}.down_proj.weight"] = self.down.weight
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "") -> None:
        with torch.no_grad():
            g = f"{prefix}.gate_proj.weight"
            u = f"{prefix}.up_proj.weight"
            if g in state_dict and u in state_dict:
                # fused gate_up is the vertical concat of the two HF projections
                self.gate_up.weight.copy_(
                    torch.cat(
                        [
                            state_dict[g].detach().to(dtype=self.gate_up.weight.dtype),
                            state_dict[u].detach().to(dtype=self.gate_up.weight.dtype),
                        ]
                    )
                )
            d = f"{prefix}.down_proj.weight"
            if d in state_dict:
                self.down.weight.copy_(state_dict[d].detach().to(dtype=self.down.weight.dtype))


__all__ = ["MLP"]
