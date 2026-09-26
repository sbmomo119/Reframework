"""Prefix caches.

FreeToken offers ``naive`` (per-request, no cross-request reuse) and ``radix``
(shared prefix tree). Re ships the same two interfaces; only ``naive`` is
functional in v0.1 — a radix tree is a nice-to-have for Pascal (where the KV
budget is tiny) and will land behind this same ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int


class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle


@dataclass(frozen=True)
class InsertResult:
    cached_len: int
    handle: BaseCacheHandle


class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None: ...

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult: ...

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult: ...

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor: ...

    @abstractmethod
    def reset(self) -> None: ...

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo: ...

    @abstractmethod
    def check_integrity(self) -> None: ...


class _NaiveHandle(BaseCacheHandle):
    def __init__(self, cached_len: int) -> None:
        super().__init__()
        self.cached_len = cached_len


class NaivePrefixCache(BasePrefixCache):
    """No cross-request reuse: every match is length 0.

    Kept as a first-class citizen (not a no-op) so the engine code path —
    match -> adopt slots -> insert -> evict — is exercised from day one and the
    radix implementation can be swapped in without touching the engine.
    """

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        return None

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(cuda_handle=_NaiveHandle(0))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return InsertResult(cached_len=0, handle=_NaiveHandle(0))

    def evict(self, size: int) -> torch.Tensor:
        return torch.empty(0, dtype=torch.int64)

    def reset(self) -> None:
        return None

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(0, 0)

    def check_integrity(self) -> None:
        return None


SUPPORTED_CACHE_MANAGER = {"naive": NaivePrefixCache}


def create_prefix_cache(type: str, device: torch.device) -> BasePrefixCache:
    if type not in SUPPORTED_CACHE_MANAGER:
        raise ValueError(f"unknown cache type {type!r}; supported: {sorted(SUPPORTED_CACHE_MANAGER)}")
    if type == "naive":
        return NaivePrefixCache()
    raise AssertionError("unreachable")


__all__ = [
    "BasePrefixCache",
    "BaseCacheHandle",
    "MatchResult",
    "InsertResult",
    "SizeInfo",
    "NaivePrefixCache",
    "create_prefix_cache",
    "SUPPORTED_CACHE_MANAGER",
]
