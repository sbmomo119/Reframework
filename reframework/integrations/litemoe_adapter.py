"""litemoe storage/cache/quantization backend for reframework's FusedMoE.

方案 A：litemoe 只负责**存储 / 缓存 / 量化**（专家权重的 fetch seam 是
``cache.fetch_many``）；**compute 走 reframework 原生的 ``fused_experts``**，
routing 完全使用 reframework 侧的 ``topk_ids`` / ``topk_weights``。litemoe 的
``MoELayer.route`` / 其自有 compute 路径**从不被调用** —— 这样数值等价性可以
直接对 reframework 的原生路径验证（同一组专家权重，两条路输出应逐元素一致）。

契约要点（对齐真实源码，勿凭记忆）：

  * ``cache.fetch_many(layer, eids)`` 返回 ``{eid: bank}``，
    ``bank = {"gate":[I,H], "up":[I,H], "down":[H,I]}``（**每专家**，未堆叠）。
    int4 bank 在 litemoe store 内部已反量化为 fp16，本适配器**不做**
    scale/zero-point 拼接，只操作反量化后的张量。
  * ``fused_experts`` 直接 ``w1[e]`` 索引，行布局 ``gate|up``（Qwen3，gate 在前）：
        w1 = stack([cat([bank[e]["gate"], bank[e]["up"]], dim=0) for e])  -> [n, 2I, H]
        w2 = stack([bank[e]["down"] for e])                              -> [n, H, I]
    因此必须把全局专家 id（topk_ids）**重映射**到堆叠 bank 的本地行号，
    否则 ``w1[e]`` 会越界。
  * ``CacheStats`` 是扁平的（``hits/misses/evictions/transfer_bytes/...``），
    归一到 reframework profiler 的 ``OffloadStats.as_dict`` 键
    （``hits/misses/evictions/subs/predicts/transfer_mb``）。

用法（引擎侧，未接入主循环前先在单测里用 FakeStore 跑通）::

    store  = load_expert_store(...)          # litemoe ExpertStore（int4 / fp16）
    cache  = make_cache(store, "lru", max_experts=..., device=...)
    meta   = LayerMeta(w1=native_w1, w2=native_w2, activation="silu")
    backend = LitemoeExpertBackend(store, cache, layer_id, meta)
    out = backend.compute(hidden, topk_weights, topk_ids, layer_id)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import torch

from reframework.moe.fused import fused_experts
from reframework.models.config import ModelConfig

# 延迟导入 litemoe（避免 reframework 在无 litemoe 环境下硬依赖它）。
# 这里仅用于类型标注；运行时只鸭子类型调用 .fetch_many / .stats。
if TYPE_CHECKING:
    from litemoe.interface import ExpertStore
    from litemoe.model.loader import ModelMeta


@dataclass
class LayerMeta:
    """reframework 侧每层的静态信息。

    * ``w1`` / ``w2``：原生预打包专家权重（``[E,2I,H]`` gate-first / ``[E,H,I]``），
      作为 ``fetch_many`` 失败时的**回退权重**（与 ``FusedMoE._banks`` resident
      分支同形）。
    * ``activation``：传给 ``fused_experts`` 的激活名（默认 ``"silu"``）。
    * ``fused_moe``：可选的 ``FusedMoE`` 实例（备用取原生权重），默认不使用。
    """
    w1: torch.Tensor
    w2: torch.Tensor
    activation: str = "silu"
    fused_moe: Optional[object] = None


class LitemoeExpertBackend:
    """Bridge: reframework routing -> litemoe fetch -> reframework compute."""

    def __init__(
        self,
        store: "ExpertStore",
        cache,
        layer: int,
        meta: LayerMeta,
    ) -> None:
        self.store = store
        self.cache = cache
        self.layer = int(layer)
        self.meta = meta
        self._fused_experts = fused_experts

    # ------------------------------------------------------------ fetch seam
    def fetch_experts(self, eids) -> "Dict[int, Dict[str, torch.Tensor]]":
        """litemoe 只负责这一步：从缓存 / store 拿专家权重（不打包）。

        返回 ``{eid: {"gate":[I,H], "up":[I,H], "down":[H,I]}}``（每专家）。
        """
        return self.cache.fetch_many(self.layer, list(eids))

    def _pack_banks(self, banks: "Dict[int, Dict[str, torch.Tensor]]", dev: torch.device):
        """{eid: bank} -> (w1 [n,2I,H] gate-first, w2 [n,H,I], global_ids)。"""
        order = sorted(banks)
        w1 = torch.stack(
            [torch.cat([banks[e]["gate"], banks[e]["up"]], dim=0) for e in order],
            dim=0,
        ).to(dev)  # [n, 2I, H]
        w2 = torch.stack([banks[e]["down"] for e in order], dim=0).to(dev)  # [n, H, I]
        return w1, w2, order

    # ------------------------------------------------------------ main path
    def compute(
        self,
        hidden: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        layer: Optional[int] = None,
    ) -> torch.Tensor:
        """reframework 主路径。

        ``hidden`` [M,H]；``topk_weights``/``topk_ids`` [M,K]（reframework 侧路由，
        全局专家 id）。返回 ``fused_experts`` 的输出（与 ``hidden`` 同形状）。
        fetch 失败时回退到原生权重（:meth:`fallback`）。
        """
        if layer is not None:
            self.layer = int(layer)
        ids = topk_ids.reshape(-1) if topk_ids.dim() == 1 else topk_ids
        try:
            banks = self.fetch_experts(ids.unique().tolist())
        except Exception:
            return self.fallback(hidden, topk_ids, topk_weights)
        w1, w2, order = self._pack_banks(banks, hidden.device)
        # 全局专家 id -> 本地 bank 行号（fused_experts 直接 w1[e] 索引）
        remap = torch.full(
            (max(order) + 1,), -1, dtype=torch.long, device=hidden.device
        )
        for loc, g in enumerate(order):
            remap[g] = loc
        local_ids = remap[ids]
        out = self._fused_experts(
            hidden, w1, w2, topk_weights, local_ids, self.meta.activation
        )
        return out.reshape(hidden.shape) if hidden.dim() > 1 else out

    # ----------------------------------------------------------- fallback
    def fallback(
        self, hidden: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """fetch 失败时回退到 reframework 原生（预打包）专家权重。

        原生 bank 覆盖**全部**专家，全局 id 无需重映射（恒等）。
        """
        out = self._fused_experts(
            hidden, self.meta.w1.to(hidden.device), self.meta.w2.to(hidden.device),
            topk_weights, topk_ids, self.meta.activation,
        )
        return out.reshape(hidden.shape) if hidden.dim() > 1 else out

    # ------------------------------------------------------------- stats
    def get_cache_stats(self) -> dict:
        """litemoe ``CacheStats`` -> reframework profiler（``OffloadStats`` 键）。"""
        s = self.cache.stats()
        return {
            # reframework profiler 键（与 OffloadStats.as_dict 对齐）
            "hits": int(s.hits),
            "misses": int(s.misses),
            "evictions": int(s.evictions),
            "subs": 0,          # reframework 专属（近似专家替换），litemoe 无此概念
            "predicts": 0,      # reframework 专属（lookahead 预取计数），litemoe 无
            "transfer_mb": s.transfer_bytes / (1024 * 1024),
            # litemoe 侧额外观测项（profiler 可忽略）
            "hit_rate": round(float(s.hit_rate), 4),
            "miss_rate": round(float(s.miss_rate), 4),
            "load_ms": round(float(s.load_ms), 3),
            "transfer_ms": round(float(s.transfer_ms), 3),
        }


def model_meta_to_config(meta: "ModelMeta", *, dtype: torch.dtype = torch.float32,
                        device: str = "cpu", arch: str = "") -> "ModelConfig":
    """litemoe ``ModelMeta`` -> reframework ``ModelConfig``。

    MoE 几何 1:1 映射（``n_layers->num_hidden_layers``、``n_experts->num_experts``、
    ``top_k->num_experts_per_tok``、``expert_inter->moe_intermediate_size``、
    ``hidden_size->hidden_size``、``vocab_size->vocab_size``）。注意力几何
    （heads / head_dim）``ModelMeta`` 未完整提供，这里给安全默认值；需要真实值时
    应由 checkpoint 的 ``config.json``（``ModelConfig.from_hf``）补齐覆盖。
    """
    hidden = int(meta.hidden_size)
    heads = int(getattr(meta, "n_heads", 0) or 1) or 1
    kv = int(getattr(meta, "n_kv_heads", 0) or 1) or 1
    head_dim = int(getattr(meta, "head_dim", 0) or 0) or (hidden // heads)
    return ModelConfig(
        vocab_size=int(meta.vocab_size),
        hidden_size=hidden,
        num_hidden_layers=int(meta.n_layers),
        intermediate_size=0,  # MoE 模型：moe_inter 走 moe_intermediate_size 回退
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        use_moe=True,
        num_experts=int(meta.n_experts),
        num_experts_per_tok=int(meta.top_k),
        moe_intermediate_size=int(meta.expert_inter),
        moe_activation="silu",
        moe_renormalize=True,
        rope_theta=float(_rope_theta_from_extra(getattr(meta, "extra", {}))),
        dtype=dtype,
        device=device,
        arch=arch or str(getattr(meta, "arch", "")),
    )


def _rope_theta_from_extra(extra) -> float:
    """从 ``ModelMeta.extra`` 取 ``rope_freq_base``（缺失/非法时回退 10000.0）。"""
    if isinstance(extra, dict):
        v = extra.get("rope_freq_base")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 10000.0
    return 10000.0


__all__ = [
    "LitemoeExpertBackend",
    "LayerMeta",
    "model_meta_to_config",
]
