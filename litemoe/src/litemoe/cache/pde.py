"""PDE cache — P0 placeholder (Day 8-10).

Full PDE = predictive prefetch + predicted-deepest-use eviction (see
:mod:`litemoe.interface` docstring). Until the predictor/scheduler land,
``PDECache`` is behaviorally identical to :class:`LRUCache`; ``prefetch``
is a no-op and the eviction order stays pure LRU. ``make_cache(strategy="pde")``
therefore already works end-to-end.
"""
from __future__ import annotations

from typing import Optional

from .lru import LRUCache


class PDECache(LRUCache):
    """PDE 占位：继承 LRU 全部行为；prefetch 空实现。

    Day 8-10 将覆写 ``get``/驱逐顺序（predicted-deepest-use）并实现
    带宽预算下的 prefetch；届时本文件是主要改动点。
    """

    def __init__(
        self,
        store,
        max_experts: int = 32,
        predictor=None,
        device: str = "cpu",
        predict_window: int = 32,
    ):
        super().__init__(store, capacity=max_experts, device=device)
        self.predictor: Optional[object] = predictor
        self.predict_window: int = int(predict_window)

    # 占位：PDE 的预测预取在 Day 8-10 实现
    def prefetch(self, layer: int, experts: list[int]) -> None:
        pass
