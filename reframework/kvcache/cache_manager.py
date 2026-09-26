"""CacheManager — the engine-facing allocator + page table (FreeToken parity).

FreeToken's CacheManager owns the token pool, the per-request page tables and
the prefix-cache policy. Re keeps that split, minus the radix tree (the naive
cache is the default; a radix cache can slot in later behind the same
``BasePrefixCache`` interface).

Responsibilities:
  * allocate/free physical KV slots (via :class:`TokenPool`)
  * grow/shrink the per-request page table rows (``[max_reqs, max_pages]``)
  * expose the flat write-loc tensor for a batch's new tokens
"""

from __future__ import annotations

import torch

from reframework.kvcache.token_pool import TokenPool
from reframework.utils import init_logger, mem_GB

logger = init_logger(__name__)

DUMMY_SLOT = 0


class PageTable:
    """Per-request rows of flat slot ids. Row ``r`` covers request ``r``'s KV.

    ``_lens[r]`` tracks how many leading slots of row ``r`` are live. Slot id 0
    is the DUMMY and is a *valid stored value*, so liveness must never be
    inferred from the tensor contents — the lengths list is the single source
    of truth.
    """

    def __init__(self, device: torch.device, max_rows: int = 16, row_len: int = 256) -> None:
        self.device = device
        self._table = torch.zeros((max_rows, row_len), dtype=torch.int64, device=device)
        self._lens: list[int] = [0] * max_rows

    @property
    def table(self) -> torch.Tensor:
        return self._table

    @property
    def num_rows(self) -> int:
        return len(self._lens)

    def row(self, r: int) -> torch.Tensor:
        return self._table[r, : self._lens[r]]

    def alloc_row(self, initial_slots: torch.Tensor) -> int:
        """Append a row seeded with ``initial_slots`` (e.g. prefix-cached slots)."""
        if len(self._lens) >= self._table.shape[0]:
            self._grow_rows(self._table.shape[0] * 2)
        r = len(self._lens)
        n = initial_slots.numel()
        if n > 0:
            self._table[r, :n] = initial_slots
        self._lens.append(n)
        return r

    def append_slots(self, r: int, slots: torch.Tensor) -> None:
        n = slots.numel()
        if n == 0:
            return
        end = self._lens[r] + n
        if end > self._table.shape[1]:
            self._grow_cols(self._table.shape[1] * 2)
        self._table[r, self._lens[r] : end] = slots
        self._lens[r] = end

    def clear_row(self, r: int) -> torch.Tensor:
        """Zero a row and return the live slots it held (for freeing)."""
        held = self._table[r, : self._lens[r]].clone()
        if self._lens[r] > 0:
            self._table[r, : self._lens[r]].zero_()
        self._lens[r] = 0
        return held

    def _grow_rows(self, new_rows: int) -> None:
        t = torch.zeros((new_rows, self._table.shape[1]), dtype=torch.int64, device=self.device)
        t[: len(self._lens)] = self._table[: len(self._lens)]
        self._table = t

    def _grow_cols(self, new_cols: int) -> None:
        t = torch.zeros((self._table.shape[0], new_cols), dtype=torch.int64, device=self.device)
        t[:, : self._table.shape[1]] = self._table
        self._table = t


class CacheManager:
    def __init__(self, pool, device: torch.device, page_size: int) -> None:
        self.pool = pool
        self.device = device
        self.page_size = page_size
        self.tokens = TokenPool(pool.num_slots, device)
        self.page_table = PageTable(device, max_rows=16, row_len=page_size * 64)

    def available_tokens(self) -> int:
        return self.tokens.available()

    def admit_request(self, cached_slots: torch.Tensor | None = None) -> int:
        """Reserve a page-table row for a new request. ``cached_slots`` are slots
        already holding its prefix (from the prefix cache); they are *adopted*,
        not allocated."""
        if cached_slots is None:
            cached_slots = torch.empty(0, dtype=torch.int64, device=self.device)
        return self.page_table.alloc_row(cached_slots)

    def alloc_kv_slots(self, r: int, n_tokens: int) -> torch.Tensor:
        """Allocate ``n_tokens`` fresh slots, append them to row ``r`` and return
        the flat write locations (in slot order)."""
        slots = self.tokens.alloc(n_tokens)
        self.page_table.append_slots(r, slots)
        return slots

    def free_request(self, r: int) -> None:
        held = self.page_table.clear_row(r)
        self.tokens.free(held)

    def num_alloc_tokens(self) -> int:
        return self.tokens.num_alloc()

    def stats(self) -> str:
        return (
            f"kv: {self.num_alloc_tokens()}/{self.pool.num_slots} tokens in use "
            f"({mem_GB(self.num_alloc_tokens() * 2 * self.pool.num_kv_heads * self.pool.head_dim * self.pool.dtype.itemsize * self.pool.num_layers)})"
        )
