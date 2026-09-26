"""OLMoE (allenai/OLMoE-1B-7B-0125-Instruct) — thin alias over the shared Llama shape.

OLMoE is Llama-shaped: pre-norm, RoPE, RMSNorm. Its MoE layer has 64 routed
experts with top-8 routing, which the shared FusedMoE handles as-is
(``use_moe`` flips on when ``num_experts`` appears in the config).

Differences from Qwen3MoE:
  * No shared expert (only routed experts).
  * All layers are MoE (no dense layers).
  * Per-expert intermediate_size = config's ``intermediate_size``.
"""
from __future__ import annotations

import torch

from .base import LlamaModel
from .config import ModelConfig

__all__ = ["OlmoeConfig", "OlmoeModel", "OlmoeForCausalLM", "build_olmoe"]


class OlmoeConfig(ModelConfig):
    """ModelConfig subclass for OLMoE."""


class OlmoeModel(LlamaModel):
    """OLMoE decoder-only MoE."""


class OlmoeForCausalLM(OlmoeModel):
    """OLMoE causal LM."""


def build_olmoe(
    config_json: dict,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> OlmoeForCausalLM:
    cfg = OlmoeConfig.from_hf(config_json, dtype=dtype, device=device)
    return OlmoeForCausalLM(cfg)
