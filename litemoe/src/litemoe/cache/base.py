# litemoe/cache/base.py
"""Cache 契约 + 统计数据结构。

值类型: 单个专家的权重 bank
    {
        "gate": torch.Tensor,  # [I, H]
        "up":   torch.Tensor,  # [I, H]
        "down": torch.Tensor,  # [H, I]
    }

所有缓存后端（LRU / PDE / 预测式）实现 ExpertCache 协议。
executor 只依赖 ExpertCache，不关心具体策略。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional
import torch


# ------------------------------------------------------------------ 类型别名

ExpertBank = dict[str, torch.Tensor]
"""单个专家的权重 bank: {"gate": [I,H], "up": [I,H], "down": [H,I]}"""


# ------------------------------------------------------------------ 统计

@dataclass
class CacheStats:
    """缓存运行统计。

    六项指标:
        hits / misses:    get() 命中与未命中次数
        evictions:        被驱逐的专家数
        transfer_bytes:   CPU→GPU 传输字节
        transfer_ms:      传输累计耗时
        load_ms:          从 store 加载 + 反量化累计耗时
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    transfer_bytes: int = 0
    transfer_ms: float = 0.0
    load_ms: float = 0.0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate

    @property
    def transfer_mb(self) -> float:
        return self.transfer_bytes / 1e6

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "transfer_mb": round(self.transfer_mb, 3),
            "transfer_ms": round(self.transfer_ms, 2),
            "load_ms": round(self.load_ms, 2),
        }

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.transfer_bytes = 0
        self.transfer_ms = 0.0
        self.load_ms = 0.0


# ------------------------------------------------------------------ 激活日志

@dataclass
class ActivationRecord:
    """单步单层的路由记录。"""
    step: int
    layer: int
    expert_ids: list[int]
    weights: list[float]
    hit_mask: list[bool] = field(default_factory=list)


# ------------------------------------------------------------------ 契约

class ExpertCache(Protocol):
    """专家缓存契约。

    get() 是唯一同步接口；prefetch() 是可选的非阻塞接口。
    """

    def get(self, layer: int, expert: int) -> ExpertBank:
        """返回专家的权重 bank，已在 device 上。

        miss 时从 store 加载并可能驱逐其他专家。
        """
        ...

    def prefetch(self, layer: int, experts: list[int]) -> None:
        """异步预取（可选实现，LRU 可空实现）。"""
        ...

    def stats(self) -> CacheStats:
        ...

    def clear(self) -> None:
        ...


# ------------------------------------------------------------------ 工具

def _device_sync(device: str | torch.device) -> None:
    """安全的 CUDA 同步（CPU-only 时不报错）。"""
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def bank_nbytes(bank: ExpertBank) -> int:
    """一个 bank 的总字节数。"""
    return sum(t.numel() * t.element_size() for t in bank.values())


def move_bank_to(bank: ExpertBank, device: str, non_blocking: bool = True) -> ExpertBank:
    """把整个 bank 搬到 device。"""
    return {k: v.to(device, non_blocking=non_blocking) for k, v in bank.items()}
