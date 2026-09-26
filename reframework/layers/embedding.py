"""Embedding + LM head.

FreeToken's ``VocabParallelEmbedding`` shards the vocab across TP ranks and
all-reduces. Re is single-GPU, so it is just a gather over the full vocab. The
LM head is a :class:`Linear` (or the tied embedding weight) and, like
FreeToken, it reads only each request's *last* token during prefill
(``batch.attn_metadata.get_last_indices``) to avoid a full-vocab GEMM over
every prompt position.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reframework.core import get_global_ctx

from .base import BaseOP
from .linear import Linear


class VocabParallelEmbedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float | None = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = num_embeddings  # no TP: full vocab
        self.vocab_range = (0, num_embeddings)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.weight.to(x.device)[x]
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(self._embed_scale, dtype=y.dtype, device=y.device)
            y = y * self._embed_scale_t
        return y


class ParallelLMHead(BaseOP):
    """The LM head is a Linear over the full vocab unless tied to the embedding."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.has_bias = bias
        self.in_features = embedding_dim
        self.out_features = num_embeddings
        self.tied_embedding = tied_embedding
        self.linear: Linear | None = None
        if tied_embedding is None:
            self.linear = Linear(embedding_dim, num_embeddings, bias=bias)
        self.bias = None  # handled by Linear / tied weight

    @staticmethod
    def _last_token_indices() -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        return batch.get_last_indices(batch.size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        if batch.is_prefill:
            x = x[self._last_token_indices()].contiguous()
        if self.linear is not None:
            return self.linear.forward(x)
        # tied: use the embedding weight as the projection
        return F.linear(x, self.tied_embedding.weight.to(x.device))

    def state_dict(self, *, prefix: str = "", result=None):
        result = {} if result is None else result
        if self.linear is not None:
            self.linear.state_dict(prefix=prefix, result=result)
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "") -> None:
        if self.tied_embedding is not None:
            # tied head: drop any head weight in the checkpoint
            for k in (f"{prefix}.weight", f"{prefix}.bias"):
                state_dict.pop(k, None)
            return
        if self.linear is not None:
            self.linear.load_state_dict(state_dict, prefix=prefix)


__all__ = ["VocabParallelEmbedding", "ParallelLMHead"]
