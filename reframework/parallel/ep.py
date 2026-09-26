"""Expert Parallelism (EP) — shard a MoE layer's experts across ranks.

A :class:`~reframework.moe.FusedMoE` holds every expert on every rank. EP splits
that: rank *r* owns a contiguous block of the experts, the routing ``gate`` is
replicated (identical on every rank), and the per-token expert contribution is
computed locally then combined with a SUM all-reduce.

Why the math is exact. ``FusedMoE.forward`` is a scatter-add over experts::

    out[m] = sum_{k} topk_weight[m,k] * expert_out(topk_ids[m,k], m)

If rank r keeps only experts in ``shard_r`` and runs ``fused_experts`` with the
non-local slots weight-masked to 0, its partial output is exactly the sum over
``shard_r``. Because the shards are disjoint and cover every expert, the SUM
all-reduce over ranks reconstructs the full resident output bit-for-bit (up to
fp32 summation order, which is why the test uses ``allclose``).

This is the EP half of Re's PP+EP mode: the pipeline stages (see ``pp.py``)
partition the *layers*, and each stage's MoE layer is further expert-sharded by
EP across the ``ep_size`` ranks of that stage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from reframework.moe import FusedMoE
from reframework.moe.fused import fused_experts, torch_fused_topk
from reframework.models.config import ModelConfig

from . import dist as _dist

__all__ = ["shard_range", "EPFusedMoE", "ep_enabled"]


def ep_enabled() -> bool:
    """True when a live process group has ``ep_size > 1``."""
    return _dist.is_initialized() and _dist.current().ep_size > 1


def shard_range(num_experts: int, ep_size: int, ep_rank: int) -> Tuple[int, int]:
    """Contiguous ``[start, end)`` expert block for ``ep_rank``.

    Divides ``num_experts`` into ``ep_size`` roughly-equal blocks; the first
    ``num_experts % ep_size`` ranks take one extra expert. Example: 8 experts,
    ep_size 2 -> rank0 ``[0,4)``, rank1 ``[4,8)``.
    """
    base, rem = divmod(num_experts, ep_size)
    size = base + (1 if ep_rank < rem else 0)
    start = ep_rank * base + min(ep_rank, rem)
    return start, start + size


class EPFusedMoE(nn.Module):
    """An expert-sharded ``FusedMoE`` that is a drop-in for the resident one.

    Exposes the same ``forward(x) -> x`` contract and the same HF-style
    ``state_dict``/``load_state_dict`` names (a rank stores only the experts it
    owns, so a checkpoint sharded per-rank loads cleanly). Build from a config
    for real deployment, or from a fully-loaded :class:`FusedMoE` via
    :meth:`from_full` (the verification path).
    """

    def __init__(self, cfg: ModelConfig, ep_size: int, ep_rank: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.moe_inter
        self.activation = cfg.moe_activation
        self.renormalize = cfg.moe_renormalize
        self.dtype = cfg.dtype

        # Replicated gate (identical on every rank) — routing must agree.
        self.gate = nn.Parameter(torch.empty(cfg.num_experts, cfg.hidden_size, dtype=cfg.dtype))

        start, end = shard_range(cfg.num_experts, ep_size, ep_rank)
        self.local_start = start
        self.local_end = end
        self.local_count = end - start
        n = self.local_count
        I, H = cfg.moe_inter, cfg.hidden_size
        self.experts_w1 = nn.ParameterList(
            [nn.Parameter(torch.empty(2 * I, H, dtype=cfg.dtype)) for _ in range(n)]
        )
        self.experts_w2 = nn.ParameterList(
            [nn.Parameter(torch.empty(H, I, dtype=cfg.dtype)) for _ in range(n)]
        )

    # --------------------------------------------------------------- build
    @classmethod
    def from_full(cls, base: FusedMoE, ep_size: int, ep_rank: int) -> "EPFusedMoE":
        """Shard a fully-loaded :class:`FusedMoE` (the verification / test path)."""
        ep = cls(
            ModelConfig(
                vocab_size=0, hidden_size=base.hidden_size, num_hidden_layers=0,
                intermediate_size=0, num_attention_heads=1, num_key_value_heads=1,
                head_dim=1, max_position_embeddings=1, rms_norm_eps=1e-6,
                use_moe=True, num_experts=base.num_experts, num_experts_per_tok=base.top_k,
                moe_intermediate_size=base.intermediate_size,
                moe_activation=base.activation, moe_renormalize=base.renormalize, dtype=base.dtype,
            ),
            ep_size, ep_rank,
        )
        with torch.no_grad():
            ep.gate.copy_(base.gate)
            for loc, g in enumerate(range(ep.local_start, ep.local_end)):
                ep.experts_w1[loc].copy_(base.experts_w1[g])
                ep.experts_w2[loc].copy_(base.experts_w2[g])
        return ep

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, self.hidden_size)
        dev = x2.device
        # Replicated routing (same math as FusedMoE.route): every rank makes the
        # identical topk decision because the gate is replicated.
        gate_out = F.linear(x2.to(self.dtype), self.gate.to(dev))
        topk_weights, topk_ids = torch_fused_topk(gate_out, self.top_k, self.renormalize)

        w1 = torch.stack([p.to(dev) for p in self.experts_w1], dim=0) if self.local_count \
            else torch.zeros((0, 2 * self.intermediate_size, self.hidden_size), dtype=self.dtype, device=dev)
        w2 = torch.stack([p.to(dev) for p in self.experts_w2], dim=0) if self.local_count \
            else torch.zeros((0, self.hidden_size, self.intermediate_size), dtype=self.dtype, device=dev)

        if self.local_count == 0:
            local_out = torch.zeros_like(x2)
        else:
            # Remap global ids -> local rows; non-local experts map to row 0
            # with their weight masked to 0, so they contribute exactly 0.
            remap = torch.zeros(self.num_experts, dtype=torch.long, device=dev)
            for loc, g in enumerate(range(self.local_start, self.local_end)):
                remap[g] = loc
            local_mask = (topk_ids >= self.local_start) & (topk_ids < self.local_end)
            local_ids = remap[topk_ids]
            local_weights = torch.where(local_mask, topk_weights, torch.zeros_like(topk_weights))
            local_out = fused_experts(x2, w1, w2, local_weights, local_ids, self.activation)

        # Combine the disjoint per-rank partial sums -> full MoE output.
        if ep_enabled():
            local_out = local_out.contiguous()
            _dist.all_reduce_sum(local_out)
        return local_out.reshape(x.shape[:1] + (self.hidden_size,)) if x.dim() > 1 else local_out

    # --------------------------------------------------------------- (de)serial
    def state_dict(self, *, prefix: str = "", result: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        result = {} if result is None else result
        result[f"{prefix}.gate.weight" if prefix else "gate.weight"] = self.gate
        I = self.intermediate_size
        for loc, g in enumerate(range(self.local_start, self.local_end)):
            p = f"{prefix}.experts.{g}" if prefix else f"experts.{g}"
            result[f"{p}.gate_proj.weight"] = self.experts_w1[loc][:I]
            result[f"{p}.up_proj.weight"] = self.experts_w1[loc][I:]
            result[f"{p}.down_proj.weight"] = self.experts_w2[loc]
        return result

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], *, prefix: str = "") -> None:
        with torch.no_grad():
            g = f"{prefix}.gate.weight" if prefix else "gate.weight"
            if g in state_dict:
                self.gate.copy_(state_dict[g].detach().to(dtype=self.dtype))
            I = self.intermediate_size
            for loc, gg in enumerate(range(self.local_start, self.local_end)):
                p = f"{prefix}.experts.{gg}" if prefix else f"experts.{gg}"
                for half, param in (
                    ("gate_proj", self.experts_w1[loc][:I]),
                    ("up_proj", self.experts_w1[loc][I:]),
                    ("down_proj", self.experts_w2[loc]),
                ):
                    key = f"{p}.{half}.weight"
                    if key in state_dict:
                        param.copy_(state_dict[key].detach().to(dtype=self.dtype))
