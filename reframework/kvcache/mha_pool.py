"""KV cache pool base + MHA (GQA) paged pool.

Mirrors FreeToken's ``kvcache/base.py`` + ``mha_pool.py``: a per-layer slab of
``[num_slots, num_kv_heads, head_dim]`` for K and the same for V, addressed
through a page table of flat slot ids. Re implements *only* the FULL/MHA family
(what Llama and Qwen3 use); FreeToken's SWA/MLA/DSA/BSA/QSA families are
multi-GB-VRAM / new-arch features that a 6-8 GB Pascal card never needs.

Sizing is the same template as FreeToken's ``solve_num_pages``: the engine
measures free VRAM after weights, the pool prices a page, and the largest
fitting page count wins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from reframework.utils import init_logger

logger = init_logger(__name__)


class BaseKVCachePool(ABC):
    """Interface shared by all Re pools (FreeToken's BaseKVCachePool, slimmed)."""

    @classmethod
    @abstractmethod
    def kv_cost(cls, num_kv_heads: int, head_dim: int, num_layers: int, page_size: int,
                dtype: torch.dtype) -> tuple[int, int, int]:
        """``(bytes_per_page, fixed_bytes, tokens_per_page)`` for this family."""

    @classmethod
    def solve_num_pages(cls, available_memory: int, num_kv_heads: int, head_dim: int,
                        num_layers: int, page_size: int, dtype: torch.dtype,
                        num_page_override: int | None = None) -> int:
        per_page, fixed, _ = cls.kv_cost(num_kv_heads, head_dim, num_layers, page_size, dtype)
        if num_page_override is not None:
            num_pages = num_page_override
        else:
            num_pages = (available_memory - fixed) // per_page
        assert num_pages > 1, "Not enough memory for KV cache; reduce --max-model-len or --num-pages"
        logger.info(
            "KV pool: %d slots (%.2f GB) for %d layers x %d kv-heads x dim %d, page_size=%d",
            num_pages * page_size, (num_pages * per_page + fixed) / 1024**3,
            num_layers, num_kv_heads, head_dim, page_size,
        )
        return num_pages

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor,
                 layer_id: int) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


class MHAKVCache(BaseKVCachePool):
    """Uniform causal MHA/GQA pool: K/V slabs per layer over a shared slot space.

    Layout: ``k`` and ``v`` are each ``[num_layers, num_slots, num_kv_heads,
    head_dim]``. A forward pass writes the new tokens at ``out_loc`` (flat slot
    ids) and reads them back through the page table.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._num_slots = num_slots
        self._dtype = dtype
        self._device = device
        self.k = torch.zeros(
            (num_layers, num_slots, num_kv_heads, head_dim), dtype=dtype, device=device
        )
        self.v = torch.zeros(
            (num_layers, num_slots, num_kv_heads, head_dim), dtype=dtype, device=device
        )

    # -- sizing (classmethod surface the engine calls before the pool exists) --
    @classmethod
    def kv_cost(cls, num_kv_heads: int, head_dim: int, num_layers: int, page_size: int,
                dtype: torch.dtype) -> tuple[int, int, int]:
        per_slot = 2 * num_kv_heads * head_dim * dtype.itemsize * num_layers
        return per_slot * page_size, 0, page_size

    # -- accessors --
    def k_cache(self, index: int) -> torch.Tensor:
        return self.k[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self.v[index]

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor,
                 layer_id: int) -> None:
        """Write ``k``/``v`` (each ``[num_new_tokens, kv_heads, head_dim]``) into
        the slots named by ``out_loc`` (``[num_new_tokens]`` flat ids)."""
        if k.numel() == 0:
            return
        # scatter via index_copy_: fastest portable path (no triton on Pascal)
        self.k[layer_id].index_copy_(0, out_loc, k.to(self._dtype, non_blocking=True))
        self.v[layer_id].index_copy_(0, out_loc, v.to(self._dtype, non_blocking=True))

    def gather_kv(
        self, layer_id: int, loc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for the flat slot list ``loc`` -> each ``[len, kv_heads, head_dim]``."""
        if loc.numel() == 0:
            empty = self.k[layer_id][0:0]
            return empty, empty
        return self.k[layer_id].index_select(0, loc), self.v[layer_id].index_select(0, loc)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_slots(self) -> int:
        return self._num_slots


def create_kv_pool(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_slots: int,
    dtype: torch.dtype,
    device: torch.device,
) -> MHAKVCache:
    """Single factory (FreeToken's kvcache/__init__ has one per family; Re has one)."""
    return MHAKVCache(num_layers, num_kv_heads, head_dim, num_slots, dtype, device)


__all__ = ["BaseKVCachePool", "MHAKVCache", "create_kv_pool"]
