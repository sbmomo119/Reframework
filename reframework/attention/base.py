"""Attention backend base (mirrors FreeToken's attention/base.py).

FreeToken's taxonomy (FULL/SWA/MLA/DSA/DSV4/BSA/QSA/LINEAR) exists because its
pools differ per family. Re serves one family — uniform causal MHA/GQA — so the
ABC is a single ``BaseAttnBackend`` with no AttnType enum. The ``AttentionSpec``
dataclass is kept verbatim (sliding_window / sm_scale) so per-layer specs from
the model config still work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch
    from reframework.core import Batch


@dataclass
class AttentionSpec:
    sliding_window: int | None = None
    sm_scale: float | None = None


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        """Run one attention layer. ``q``/``k``/``v`` are ``[num_tokens, heads, dim]``
        (flat across the batch); the return has the same shape."""

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """Fill ``batch.attn_metadata`` with whatever this backend needs."""
