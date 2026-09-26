# litemoe/cache/lru.py
"""LRU 专家缓存基线。

全局 LRU: OrderedDict[(layer, expert), ExpertBank]
容量满时驱逐最久未使用的。
可选记录每一步每一层的激活日志。

适配说明（vs 原始 spec）:
  - store API: 实际 ExpertStore 提供
        expert_banks(layer, eids, dtype) -> {eid: {gate,up,down}}
        expert_payload_bytes_for(layer, eids) -> int
    而非 spec 里的 get_expert / num_layers / num_experts。
    传输字节数用 store.expert_payload_bytes_for（真实量化 payload），
    而非 bank_nbytes（反量化后的 fp 尺寸）——后者在 int4 GGUF 下会高估。
  - ``get()`` 是唯一同步主路径：命中 / 加载 / 搬移 / 驱逐 / 统计都在这里。
    ``fetch_many()`` 是它的 batch wrapper（循环 get），模型侧
    MoELayer._banks_for 走这条路，保证 stats / 激活日志 / 驱逐口径一致。
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, Iterable, TYPE_CHECKING

import torch

from .base import (
    CacheStats, ActivationRecord, ExpertBank, ExpertCache,
    _device_sync, bank_nbytes, move_bank_to,
)

if TYPE_CHECKING:  # 仅注解用；运行时不 import 避免循环
    from ..interface import ExpertStore


class LRUCache(ExpertCache):
    """全局 LRU。

    Args:
        store: 后端专家存储
        capacity: 最大缓存专家数（跨所有层共享）
        device: 目标设备，如 "cuda" / "cpu"
    """

    def __init__(
        self,
        store: "ExpertStore",
        capacity: int,
        device: str = "cpu",
    ):
        self.store = store
        self.capacity = max(1, int(capacity))
        self.max_experts = self.capacity  # interface.Cache 契约字段
        self.device = str(device)

        self._cache: OrderedDict[tuple[int, int], ExpertBank] = OrderedDict()
        self._stats = CacheStats()

        # 激活日志
        self._log_enabled = False
        self._log: list[ActivationRecord] = []
        self._current_step = 0

    # -------------------------------------------------------------- 主接口

    def get(self, layer: int, expert: int) -> ExpertBank:
        """单专家同步获取（ExpertCache 契约；唯一主路径）。"""
        expert = int(expert)
        key = (int(layer), expert)

        # 命中
        if key in self._cache:
            self._stats.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        # miss: 加载
        self._stats.misses += 1
        t0 = time.perf_counter()
        loaded = self.store.expert_banks(layer, [expert])
        bank = loaded[expert]
        self._stats.load_ms += (time.perf_counter() - t0) * 1000.0

        # 搬移（CPU -> device）
        t0 = time.perf_counter()
        bank = move_bank_to(bank, self.device)
        _device_sync(self.device)
        self._stats.transfer_ms += (time.perf_counter() - t0) * 1000.0
        # 传输字节按真实量化 payload 计（int4 下远小于 fp bank 尺寸）
        try:
            self._stats.transfer_bytes += int(
                self.store.expert_payload_bytes_for(layer, [expert]))
        except Exception:
            self._stats.transfer_bytes += bank_nbytes(bank)

        # 驱逐：腾出 1 个槽位（最久未用先走）
        while len(self._cache) + 1 > self.capacity:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

        self._cache[key] = bank
        return bank

    def fetch_many(
        self,
        layer: int,
        eids: Iterable[int],
        dtype: torch.dtype = torch.float16,
    ) -> dict[int, ExpertBank]:
        """批量获取 = get 的 batch wrapper（循环 get，统一 stats/驱逐）。

        返回 {eid: bank}。dtype 仅为契约兼容（bank 在 store 侧反量化，
        cache 侧不做二次转换）。
        """
        out: dict[int, ExpertBank] = {}
        for e in dict.fromkeys(int(x) for x in eids):  # 去重保序
            out[e] = self.get(layer, e)
        return out

    def _evict_for(self, need: int) -> None:
        """驱逐最久未使用的专家，直到腾出 need 个槽位。"""
        while len(self._cache) + need > self.capacity:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

    # ------------------------------------------------- interface.Cache 契约

    def contains(self, layer: int, eid: int) -> bool:
        return (int(layer), int(eid)) in self._cache

    def evict(self, layer: int, eid: int) -> None:
        self._cache.pop((int(layer), int(eid)), None)

    def resident_ids(self) -> Dict[int, set]:
        out: Dict[int, set] = {}
        for (L, e) in self._cache:
            out.setdefault(L, set()).add(e)
        return out

    def reset(self) -> None:
        self._cache.clear()  # 保留 stats

    # -------------------------------------------------------------- 预取

    def prefetch(self, layer: int, experts: list[int]) -> None:
        """LRU 无预测，空实现。PDE 里会覆盖。"""
        pass

    # -------------------------------------------------------------- 状态

    def stats(self) -> CacheStats:
        return self._stats

    def clear(self) -> None:
        self._cache.clear()
        self._stats.reset()

    # ------------------------------------------------------------ 激活日志

    def enable_log(self) -> None:
        self._log_enabled = True
        self._log = []

    def set_step(self, step: int) -> None:
        self._current_step = int(step)

    def record_activation(
        self,
        layer: int,
        expert_ids: list[int],
        weights: list[float],
    ) -> None:
        """executor 在 MoELayer 里调用，记录这一步这一层的路由。"""
        if not self._log_enabled:
            return
        hit_mask = [(layer, e) in self._cache for e in expert_ids]
        self._log.append(ActivationRecord(
            step=self._current_step,
            layer=int(layer),
            expert_ids=[int(e) for e in expert_ids],
            weights=[float(w) for w in weights],
            hit_mask=hit_mask,
        ))

    def get_log(self) -> list[ActivationRecord]:
        return list(self._log)

    def dump_log(self, path: str) -> None:
        """保存为 JSONL。"""
        import json
        with open(path, "w", encoding="utf-8") as f:
            for rec in self._log:
                f.write(json.dumps({
                    "step": rec.step,
                    "layer": rec.layer,
                    "expert_ids": rec.expert_ids,
                    "weights": rec.weights,
                    "hit_mask": rec.hit_mask,
                }, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------- 调试

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._cache

    def cached_keys(self) -> list[tuple[int, int]]:
        return list(self._cache.keys())


# ------------------------------------------------------------------ 每层独立容量

class PerLayerLRUCache(LRUCache):
    """每层独立容量的 LRU。

    给 bandwidth scheduler 用：每层分配的容量不同。
    get() 覆写为层内 LRU；fetch_many 仍走循环 get。
    """

    def __init__(
        self,
        store: "ExpertStore",
        per_layer_cap: dict[int, int],
        device: str = "cpu",
        default_cap: int = 1,
    ):
        # 基类容量取各层之和（全局上限），实际驱逐按层独立
        super().__init__(store, capacity=sum(per_layer_cap.values()) or 1, device=device)
        self.per_layer_cap = dict(per_layer_cap)
        self.default_cap = max(1, int(default_cap))
        self._caches: dict[int, OrderedDict[int, ExpertBank]] = {}

    def _get_cache(self, layer: int) -> "OrderedDict[int, ExpertBank]":
        if layer not in self._caches:
            self._caches[layer] = OrderedDict()
        return self._caches[layer]

    def get(self, layer: int, expert: int) -> ExpertBank:
        cache = self._get_cache(layer)
        expert = int(expert)
        if expert in cache:
            self._stats.hits += 1
            cache.move_to_end(expert)
            return cache[expert]

        self._stats.misses += 1
        t0 = time.perf_counter()
        loaded = self.store.expert_banks(layer, [expert])
        self._stats.load_ms += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        bank = move_bank_to(loaded[expert], self.device)
        _device_sync(self.device)
        self._stats.transfer_ms += (time.perf_counter() - t0) * 1000.0
        try:
            self._stats.transfer_bytes += int(
                self.store.expert_payload_bytes_for(layer, [expert]))
        except Exception:
            self._stats.transfer_bytes += bank_nbytes(bank)

        cap = self.per_layer_cap.get(layer, self.default_cap)
        while len(cache) >= cap:
            cache.popitem(last=False)
            self._stats.evictions += 1
        cache[expert] = bank
        return bank

    def prefetch(self, layer: int, experts: list[int]) -> None:
        pass

    # ---- 基类内省方法按 self._caches 覆写（基类看的是 self._cache）---- #
    def contains(self, layer: int, eid: int) -> bool:
        return int(eid) in self._caches.get(int(layer), {})

    def __contains__(self, key: tuple[int, int]) -> bool:
        (layer, eid) = key
        return self.contains(layer, eid)

    def __len__(self) -> int:
        return sum(len(c) for c in self._caches.values())

    def resident_ids(self) -> Dict[int, set]:
        return {L: set(c.keys()) for L, c in self._caches.items()}

    def cached_keys(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for L, c in self._caches.items():
            out.extend((L, e) for e in c)
        return out

    def reset(self) -> None:
        for c in self._caches.values():
            c.clear()  # 保留 stats

    def clear(self) -> None:
        for c in self._caches.values():
            c.clear()
        self._stats.reset()
