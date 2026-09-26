"""Cache package: 常驻专家窗口 + 驱逐策略。

契约（:class:`ExpertCache` 协议 / :class:`CacheStats` / :class:`ActivationRecord` /
:class:`ExpertBank`）定义在 :mod:`litemoe.cache.base`；具体策略:

* :class:`LRUCache`        — 全局 LRU 基线（``get`` 为唯一同步接口）
* :class:`PerLayerLRUCache` — 每层独立容量（bandwidth scheduler 预留）
* ``PDECache``             — 预测式（Day 8-10, ``cache/pde.py``，覆写 ``get``）

``get()`` 是唯一同步路径；``fetch_many()`` 是它的 batch wrapper（循环 get），
这样所有 stats / 激活日志 / 驱逐路径统一。
"""

from litemoe.cache.base import (
    ActivationRecord,
    CacheStats,
    ExpertBank,
    ExpertCache,
    bank_nbytes,
    move_bank_to,
    _device_sync,
)
from litemoe.cache.lru import LRUCache, PerLayerLRUCache
from litemoe.cache.pde import PDECache

__all__ = [
    "ActivationRecord",
    "CacheStats",
    "ExpertBank",
    "ExpertCache",
    "bank_nbytes",
    "move_bank_to",
    "_device_sync",
    "LRUCache",
    "PerLayerLRUCache",
    "PDECache",
]
