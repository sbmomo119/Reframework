"""Qwen3 (dense + A-series MoE) — thin aliases over the shared Llama shape.

Qwen3 checkpoints are Llama-shaped (pre-norm, GQA, RoPE, RMSNorm 1+eps);
the MoE variants (Qwen3-30B-A3B ...) just flip ``use_moe`` in the config and
:class:`~reframework.models.base.DecoderLayer` swaps in a
:class:`~reframework.moe.FusedMoE`. No new modules are needed.

Out of scope for this skeleton: shared experts (Qwen3-235B's
``mlp.shared_expert``) — only routed experts (``mlp.experts.*``) are modeled.
"""
from __future__ import annotations

import torch

from .base import LlamaModel
from .config import ModelConfig

__all__ = ["Qwen3Config", "Qwen3Model", "Qwen3ForCausalLM", "build_qwen3"]


class Qwen3Config(ModelConfig):
    """ModelConfig subclass so a Qwen3 config is *is-a* model config;
    ``from_hf`` maps Qwen3's config.json 1:1 (MoE fields included)."""


class Qwen3Model(LlamaModel):
    """Decoder-only Qwen3 (dense or MoE per the config)."""


class Qwen3ForCausalLM(Qwen3Model):
    """Qwen3 causal LM (tie_word_embeddings honored by the base)."""


def build_qwen3(
    config_json: dict,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Qwen3ForCausalLM:
    """One-stop constructor from an HF ``config.json`` dict.

    >>> model = build_qwen3(json.load(open("config.json")), dtype=torch.float16)

    MoE host offload is an engine-side step, done after construction:
    ``layer.mlp.build_offload_cache(device, capacity)`` per MoE layer, with
    the capacity from :meth:`ModelConfig.size_moe_lru` (env-gated by
    ``RE_MOE_OFFLOAD`` / ``RE_MOE_CACHE_SIZE``).
    """
    cfg = Qwen3Config.from_hf(config_json, dtype=dtype, device=device)
    return Qwen3ForCausalLM(cfg)
