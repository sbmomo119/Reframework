"""The four core litemoe interfaces — decided up front, implemented by all backends.

This module is the contract. Every backend (GGUF-int4, safetensors-fp16,
safetensors-int4) and every cache/predictor/scheduler strategy plugs into it,
so the rest of the runtime never special-cases a backend.

    ExpertStore   — where weights live & how to load expert banks
    Cache         — the resident (VRAM) expert window + stats
    Predictor     — predicts which experts fire in future steps
    Scheduler     — per-step fetch/evict policy (bandwidth-aware)

PDE (Predictive-Deepest-Use Eviction + Predictive Prefetch)
----------------------------------------------------------
The pluggable cache strategy that generalizes LRU. Two coupled rules:

  1. Predictive prefetch — at step ``t``, before the demand fetch for step
     ``t``'s experts is consumed, prefetch the expert set predicted by the
     ``Predictor`` to fire at ``t+1`` (and optionally ``t+2..t+H``) into the
     cache, subject to the per-step bandwidth budget. Overlaps the H2D/
     disk->RAM transfer with step ``t``'s compute.

  2. Predicted-deepest-use eviction — when a slot must be freed, evict the
     resident expert whose *predicted next activation* is farthest in the
     future (largest ``Predictor.next_use``). If the predictor has no signal
     for an expert, it falls back to the LRU timestamp.

LRU is recovered as the special case of PDE with a predictor that returns
"no future use" for every expert (=> pure recency order). So PDE strictly
dominates LRU when the predictor is useful and never hurts when it isn't.

Naming note: "deepest" = farthest predicted next use = safest to evict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from litemoe.model.loader import ModelMeta

from .cache.base import (  # noqa: F401  (re-export: cache 契约与统计)
    ActivationRecord, CacheStats, ExpertBank, ExpertCache,
)


# --------------------------------------------------------------------------- #
# shared value objects                                                        #
# --------------------------------------------------------------------------- #
ExpertBanks = Dict[int, Dict[str, torch.Tensor]]  # eid -> {gate,up,down}


@dataclass
class SchedulerStats:
    prefetch_bytes: int = 0
    demand_bytes: int = 0
    budget_exhausted: int = 0
    plans: int = 0

    def to_dict(self) -> dict:
        import dataclasses

        return dataclasses.asdict(self)


@dataclass(frozen=True)
class FetchPlan:
    """A single scheduler decision for one (step, layer)."""

    fetch: Tuple[Tuple[int, int], ...] = ()      # (layer, eid) to load now
    evict: Tuple[Tuple[int, int], ...] = ()      # (layer, eid) to drop now
    transfer_bytes: int = 0

    def empty(self) -> bool:
        return not self.fetch and not self.evict


# --------------------------------------------------------------------------- #
# 1) ExpertStore                                                              #
# --------------------------------------------------------------------------- #
class ExpertStore(ABC):
    """Backend-agnostic expert weight provider.

    A store is *stateless about residency*: it only knows how to hand out an
    expert bank on demand and how much that costs in transferred bytes. The
    :class:`Cache` owns which experts are actually resident in RAM/VRAM.

    Concrete backends: ``GGUFExpertStore`` (int4 GGUF), ``SafetensorsExpertStore``
    (fp16 or int4-packed safetensors).
    """

    meta: ModelMeta

    @abstractmethod
    def load_dense(self, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
        """All non-expert tensors (embeddings, norms, attention, shared experts)."""

    @abstractmethod
    def expert_banks(self, layer: int, eids: Iterable[int], dtype: torch.dtype = torch.float16) -> ExpertBanks:
        """Dequantized fp expert banks ``eid -> {gate,up,down}`` for ``layer``."""

    @abstractmethod
    def gate(self, layer: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Router projection ``[E, hidden]`` for ``layer``."""

    def has_shared_expert(self, layer: int) -> bool:
        return bool(self.meta.has_shared_expert)

    def shared_experts(self, layer: int, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
        return {}

    @abstractmethod
    def expert_payload_bytes_for(self, layer: int, eids: Iterable[int]) -> int:
        """Bytes actually moved from the store to RAM on a miss (the quant payload
        for GGUF, the packed payload for int4 safetensors, fp16 for fp16)."""

    def drop(self, layer: int, eids: Iterable[int]) -> None:
        """Optional: release any per-expert scratch the backend kept. Default no-op."""
        return None

    # -- spec 适配层（统一接口，所有 backend 免费获得） ------------------------ #
    @property
    def num_layers(self) -> int:
        """层数（= ``meta.n_layers``）。"""
        return int(self.meta.n_layers)

    @property
    def num_experts(self) -> int:
        """每层路由专家数（= ``meta.n_experts``）。"""
        return int(self.meta.n_experts)

    @property
    def expert_payload_bytes(self) -> int:
        """单个专家从 store 传输的字节数（真实量化 payload）。

        后端实现 :meth:`expert_payload_bytes_for(layer, eids)` 返回真实 on-disk
        字节数；此 property 取层 0 的单专家值作为统一量纲（GGUF 下逐层一致）。
        """
        try:
            return int(self.expert_payload_bytes_for(0, [0]))
        except Exception:
            m = self.meta
            return 3 * int(m.expert_inter) * int(m.hidden_size) * 2  # fp16 回退

    def get_expert(self, layer: int, expert: int,
                   dtype: torch.dtype = torch.float16) -> "ExpertBank":
        """返回单个专家的权重 bank ``{'gate': [I,H], 'up': [I,H], 'down': [H,I]}``。

        spec 统一入口；内部委托 :meth:`expert_banks`（backend 真正的加载路径）。
        """
        return self.expert_banks(layer, [int(expert)], dtype=dtype)[int(expert)]


# --------------------------------------------------------------------------- #
# 2) Cache                                                                    #
# --------------------------------------------------------------------------- #
class Cache(ExpertCache, ABC):
    """Resident expert window (the simulated/real VRAM slot pool).

    在 :class:`ExpertCache` 协议之上追加批量与状态查询接口，供模型 /
    executor / PDE 使用:

    * :meth:`get` — 唯一同步接口（契约）
    * :meth:`fetch_many` — ``get`` 的 batch wrapper（循环 get，统一 stats/日志/驱逐）
    * :meth:`record_activation` / :meth:`set_step` — 激活日志（PDE 继承用）
    """

    max_experts: int

    @abstractmethod
    def contains(self, layer: int, eid: int) -> bool: ...

    @abstractmethod
    def fetch_many(
        self, layer: int, eids: Iterable[int], dtype: torch.dtype = torch.float16
    ) -> ExpertBanks:
        """批量获取（``get`` 的 batch wrapper，循环 :meth:`get`）。"""

    @abstractmethod
    def evict(self, layer: int, eid: int) -> None: ...

    @abstractmethod
    def resident_ids(self) -> Dict[int, set]:
        """``{layer: {eid,...}}`` currently resident."""

    @abstractmethod
    def reset(self) -> None:
        """Clear residency (keeps stats)."""

    # -- eviction policy hook ------------------------------------------------
    def choose_eviction(self, layer: int, eids: Iterable[int]) -> Optional[int]:
        """Pick which resident eid to evict. Default: LRU (oldest access)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 3) Predictor                                                                #
# --------------------------------------------------------------------------- #
class Predictor(ABC):
    """Predicts which experts will be activated in future decode steps.

    Trained online: :meth:`observe` is called with the *actual* routing of each
    completed step; :meth:`predict` is called one (or more) steps ahead.
    """

    @abstractmethod
    def observe(self, step: int, layer: int, eids: Iterable[int]) -> None:
        """Record the actual expert set that fired at ``step`` for ``layer``."""

    @abstractmethod
    def predict(self, step: int, layer: int, horizon: int = 1) -> List[int]:
        """Experts predicted to fire ``horizon`` steps after ``step`` (i.e. at
        ``step + horizon``). Empty list = no prediction."""

    def next_use(self, layer: int, eid: int) -> Optional[float]:
        """Predicted steps until ``eid`` next activates. ``None`` = unknown.
        Used by PDE eviction (largest = evict first)."""
        return None

    @abstractmethod
    def ready(self) -> bool:
        """Has enough history to make predictions?"""

    @abstractmethod
    def reset(self) -> None: ...


# --------------------------------------------------------------------------- #
# 4) Scheduler                                                                #
# --------------------------------------------------------------------------- #
class Scheduler(ABC):
    """Per-step, per-layer fetch/evict policy.

    Bridges :class:`Cache` + :class:`Predictor` + a bandwidth budget. The
    runtime calls :meth:`plan` after each step's router fires, applies the
    returned :class:`FetchPlan` to the cache (this is the prefetch/overlap
    window), then does the demand fetch for the current step's experts.
    """

    @abstractmethod
    def plan(self, step: int, layer: int, actual_eids: Iterable[int],
             predicted_eids: Iterable[int]) -> FetchPlan:
        """Decide what to prefetch/evict now so the predicted next step is served."""

    @abstractmethod
    def on_result(self, step: int, layer: int, plan: FetchPlan, actual_eids: Iterable[int]) -> None:
        """Feedback after the step completes (updates budget accounting)."""

    def budget_bytes(self, step: int) -> float:
        """Bytes that can be transferred this step. Default: unbounded."""
        return float("inf")

    @abstractmethod
    def stats(self) -> SchedulerStats: ...


# --------------------------------------------------------------------------- #
# factory                                                                     #
# --------------------------------------------------------------------------- #
def make_cache(
    store: ExpertStore,
    strategy: str = "lru",
    max_experts: int = 32,
    device: str = "cpu",
    predict_window: int = 32,
    predictor: Optional[Predictor] = None,
    per_layer_cap: Optional[Dict[int, int]] = None,
):
    """Build a cache by strategy name (``lru`` | ``pde``).

    Args:
        store: 后端 :class:`ExpertStore`
        strategy: ``lru`` | ``pde``
        max_experts: 常驻专家槽位数（全局 LRU 的容量）
        device: 缓存 bank 所在设备（"cpu" / "cuda"），CPU-only 环境传 "cpu"
        predict_window / predictor: PDE 用的预测器（LRU 忽略）
    """
    from litemoe.cache.lru import LRUCache

    strategy = (strategy or "lru").lower()
    if strategy == "lru":
        return LRUCache(store, capacity=max_experts, device=device)
    if strategy == "pde":
        from litemoe.cache.pde import PDECache

        if predictor is None:
            from litemoe.predictor.ngram import NgramPredictor

            predictor = NgramPredictor(store.meta.n_layers, window=predict_window)
        return PDECache(store, max_experts=max_experts, predictor=predictor, device=device)
    if strategy == "per_layer_lru":
        from litemoe.cache.lru import PerLayerLRUCache

        caps = per_layer_cap or {i: max_experts for i in range(store.num_layers)}
        return PerLayerLRUCache(store, per_layer_cap=caps, device=device)
    raise ValueError(
        f"unknown cache strategy: {strategy!r} (use lru|pde|per_layer_lru)")
