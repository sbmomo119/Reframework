"""Third-party / cross-framework integration backends for reframework.

方案 A 约定：外部框架（如 litemoe）只复用其**存储 / 缓存 / 量化**层，
compute 一律走 reframework 原生路径（``reframework.moe.fused.fused_experts``），
routing 完全在 reframework 侧。见 :mod:`reframework.integrations.litemoe_adapter`。
"""
from reframework.integrations.litemoe_adapter import (
    LitemoeExpertBackend,
    LayerMeta,
    model_meta_to_config,
)

__all__ = ["LitemoeExpertBackend", "LayerMeta", "model_meta_to_config"]
