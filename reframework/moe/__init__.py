"""MoE: routing + grouped expert compute + host-offload LRU cache.

Re's MoE is the pure-torch, Pascal-safe stand-in for FreeToken's triton/sgl
fused_moe kernels, plus the small-VRAM superpower: experts staged in pinned
host memory with a VRAM LRU window (:class:`ExpertOffloadCache`).
"""

from reframework.moe.fused import torch_fused_topk, fused_experts
from reframework.moe.moe_layer import FusedMoE
from reframework.moe.offload_cache import ExpertOffloadCache, OffloadStats

__all__ = [
    "torch_fused_topk",
    "fused_experts",
    "FusedMoE",
    "ExpertOffloadCache",
    "OffloadStats",
]
