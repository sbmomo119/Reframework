"""Core data structures: requests, batches and the global context.

FreeToken's ``core.py`` is the seam between the engine and the model: the model
only ever sees a ``ForwardBatch`` (``get_global_ctx().batch``), never the engine
internals. Re keeps the same shape (Req -> Batch -> Context) but drops the
multi-GPU / MLA / DSV4 machinery that Pascal never needs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from reframework.kvcache import KVCache, CacheManager
    from reframework.attention import AttentionBackend


class SamplingParams:
    """Per-request sampling configuration (greedy defaults)."""

    __slots__ = ("temperature", "top_k", "top_p", "max_new_tokens", "stop", "seed",
                 "repetition_penalty", "no_repeat_ngram_size")

    def __init__(
        self,
        temperature: float = 0.0,
        top_k: int = 1,
        top_p: float = 1.0,
        max_new_tokens: int = 1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.stop = stop or []
        self.seed = seed
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 1e-6


@dataclass
class Req:
    """One sequence in flight. ``output_ids`` is appended to as tokens are
    produced; the scheduler reads ``len`` against the sampling budget."""

    rid: str
    origin_input_ids: List[int]
    output_ids: List[int] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    # number of tokens already in the KV cache (prefill progress)
    num_computed_tokens: int = 0
    finished: bool = False
    finished_reason: str = ""
    # page-table row owned by the CacheManager (-1 until admitted)
    table_idx: int = -1
    # tokens processed by the *next* forward pass (set by the scheduler)
    extend_len: int = 0

    def __post_init__(self) -> None:
        if self.rid == "":
            self.rid = uuid.uuid4().hex

    @property
    def all_ids(self) -> List[int]:
        return self.origin_input_ids + self.output_ids

    @property
    def len(self) -> int:
        return len(self.all_ids)

    @property
    def last_token_id(self) -> int:
        return self.all_ids[-1]

    def remaining_tokens(self) -> int:
        return self.sampling_params.max_new_tokens - len(self.output_ids)


@dataclass
class AttnMetadata:
    """Everything the attention backend needs for one forward pass.

    Padded-layout bookkeeping:
      * ``query_lens``   : per-request query length (prefill) or 1 (decode)
      * ``seq_lens``     : per-request full KV length AFTER this pass
      * ``positions``    : absolute positions of the queried tokens
      * ``req_pool_indices`` / ``page_table`` : paged-KV indirection
      * ``out_cache_loc``: flat slot indices to *write* the new K/V into
    """

    is_prefill: bool
    query_lens: torch.Tensor  # [bs] int32
    seq_lens: torch.Tensor  # [bs] int32 (cumulative incl. new tokens)
    positions: torch.Tensor  # [num_tokens] int64 (flat across the batch)
    req_pool_indices: torch.Tensor  # [bs] int64 (row in the page table)
    page_table: torch.Tensor  # [bs, max_pages] int32 (flat slot ids, padded)
    out_cache_loc: torch.Tensor  # [num_tokens] int64 (write targets)
    extend_start_loc: Optional[torch.Tensor] = None  # [bs] int64 (prefill offsets)
    kv_indices: Optional[torch.Tensor] = None  # [total_kv] int64, flat slot ids per request (set by the backend)
    q_to_req: Optional[torch.Tensor] = None  # [num_tokens] int32, request id of each query token

    @property
    def bs(self) -> int:
        return self.query_lens.shape[0]

    @classmethod
    def empty(cls, is_prefill: bool, num_tokens: int, device: torch.device) -> "AttnMetadata":
        """Placeholder the engine assigns to the batch; the backend's
        ``prepare_metadata`` overwrites every field except ``out_cache_loc``
        (which the engine fills from the CacheManager)."""
        return cls(
            is_prefill=is_prefill,
            query_lens=torch.empty(0, dtype=torch.int32, device=device),
            seq_lens=torch.empty(0, dtype=torch.int32, device=device),
            positions=torch.empty(0, dtype=torch.int64, device=device),
            req_pool_indices=torch.empty(0, dtype=torch.int64, device=device),
            page_table=torch.empty(0, 0, dtype=torch.int32, device=device),
            out_cache_loc=torch.empty(num_tokens, dtype=torch.int64, device=device),
        )


@dataclass
class Batch:
    """A forward batch: one model call. ``is_prefill`` is True when any request
    is still extending its prompt (Re runs prefill and decode in the same call
    only when the whole batch is one phase — the scheduler keeps them apart)."""

    reqs: List[Req]
    is_prefill: bool
    input_ids: torch.Tensor  # [num_tokens] int64 (flat)
    attn_metadata: Optional[AttnMetadata] = None  # filled by the backend's prepare_metadata

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def num_tokens(self) -> int:
        return self.input_ids.shape[0]

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """Flat indices of each request's *last* token (used by the LM head to
        take only the positions that produce the next token)."""
        idx = []
        offset = 0
        for q in self.attn_metadata.query_lens.tolist():
            idx.append(offset + q - 1)
            offset += q
        return torch.tensor(idx, dtype=torch.int64, device=self.input_ids.device)


class Context:
    """Global, process-wide context (FreeToken keeps one in a module global via
    ``set_global_ctx`` so layers can reach the current batch without plumbing)."""

    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        self.batch: Optional[Batch] = None
        self.kv_cache: Optional[KVCache] = None
        self.cache_manager: Optional[CacheManager] = None
        self.attn_backend: Optional[AttentionBackend] = None
        # the MoE host-offload cache is attached here so layers can look it up
        self.moe_offload_cache = None


_global_ctx: Optional[Context] = None


def set_global_ctx(ctx: Context) -> None:
    global _global_ctx
    _global_ctx = ctx


def get_global_ctx() -> Context:
    if _global_ctx is None:
        raise RuntimeError("Global context not set — are you calling a layer outside the engine?")
    return _global_ctx


__all__ = [
    "SamplingParams",
    "Req",
    "AttnMetadata",
    "Batch",
    "Context",
    "set_global_ctx",
    "get_global_ctx",
]
