"""PhiMoE (microsoft/Phi-tiny-MoE-instruct) — Llama-shape with MoE FFN.

Differences from Qwen3/OLMoE:
  * head_dim = 128 (NOT hidden//heads, which would be 256)
  * All attention projections have biases
  * LayerNorm (RMSNorm) has bias
  * No QK-Norm
  * MoE keys: block_sparse_moe.experts.{e}.w1/w3/w2 -> mlp.experts.{e}.gate_proj/up_proj/down_proj
"""
from __future__ import annotations

import torch

from .base import LlamaModel
from .config import ModelConfig

__all__ = ["PhiMoEConfig", "PhiMoEModel", "PhiMoEForCausalLM", "build_phimoe"]


class PhiMoEConfig(ModelConfig):
    """ModelConfig subclass for PhiMoE."""


class PhiMoEModel(LlamaModel):
    """PhiMoE decoder-only MoE."""


class PhiMoEForCausalLM(PhiMoEModel):
    """PhiMoE causal LM."""


def build_phimoe(
    config_json: dict,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> PhiMoEForCausalLM:
    cfg = PhiMoEConfig.from_hf(config_json, dtype=dtype, device=device)
    return PhiMoEForCausalLM(cfg)
