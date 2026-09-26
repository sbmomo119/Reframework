"""reframework.models — model definitions + the ModelConfig contract.

``ModelConfig`` (config.py) is the single source of truth every module reads:
attention heads/heads-per-token via the ``num_qo_heads``/``num_kv_heads``
properties, MoE geometry via the ``use_moe``/``num_experts``/``moe_*`` fields,
and the offload LRU budget via ``size_moe_lru``. FusedMoE/ExpertOffloadCache
each expose a ``from_config`` factory that maps those fields onto their
constructors, so a DecoderLayer never hardcodes expert dimensions.

Lazy subpackage imports follow the top-level ``reframework/__init__.py``
pattern (each subpackage is imported on demand, not at package init).
"""
from __future__ import annotations

import importlib

from .config import ModelConfig

_LAZY = {
    "LlamaModel": "base",
    "DecoderLayer": "base",
    "Attention": "base",
    "Qwen3Config": "qwen3",
    "Qwen3Model": "qwen3",
    "Qwen3ForCausalLM": "qwen3",
    "build_qwen3": "qwen3",
    "OlmoeConfig": "olmoe",
    "OlmoeModel": "olmoe",
    "OlmoeForCausalLM": "olmoe",
    "build_olmoe": "olmoe",
    "PhiMoEConfig": "phimoe",
    "PhiMoEModel": "phimoe",
    "PhiMoEForCausalLM": "phimoe",
    "build_phimoe": "phimoe",
}


def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(f"reframework.models.{_LAZY[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'reframework.models' has no attribute {name!r}")


__all__ = ["ModelConfig", *sorted(_LAZY)]
