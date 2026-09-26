"""Paged token allocator — the physical slot bookkeeping behind the KV pool.

FreeToken keeps this inside its pools (MHAKVCache & co.). Re factors it out
because the MoE host-offload cache also needs *physical slots* in the same
address space, and one allocator serving both is what makes the VRAM budget
math stay in one place.
"""

from __future__ import annotations

import torch

from reframework.utils import div_ceil


class TokenPool:
    """Flat slot allocator over ``num_slots`` physical positions.

    Slot 0 is reserved as the *dummy slot* (mirrors FreeToken's dummy page):
    every freed slot is replaced by 0 in the page table so stale rows can never
    be accidentally read.
    """

    def __init__(self, num_slots: int, device: torch.device) -> None:
        assert num_slots >= 2
        self.num_slots = num_slots
        self.device = device
        # free list as a python list of ints: O(1) alloc/free, trivially debuggable
        self._free: list[int] = list(range(1, num_slots))
        self._free_tensor = torch.empty(0, dtype=torch.int64, device=device)

    def available(self) -> int:
        return len(self._free)

    def full(self) -> bool:
        return len(self._free) == 0

    def alloc(self, n: int) -> torch.Tensor:
        """Allocate ``n`` slots, returning them as a flat int64 tensor."""
        if n > len(self._free):
            raise RuntimeError(f"TokenPool exhausted: asked for {n}, {len(self._free)} free")
        slots = self._free[-n:]
        del self._free[-n:]
        return torch.tensor(slots, dtype=torch.int64, device=self.device)

    def free(self, slots: torch.Tensor) -> None:
        if slots.numel() == 0:
            return
        self._free.extend(slots.tolist())

    def num_alloc(self) -> int:
        return self.num_slots - 1 - len(self._free)
