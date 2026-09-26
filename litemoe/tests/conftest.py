"""Shared fixtures: an in-memory FakeStore so LRU/cache tests don't need a checkpoint."""
from __future__ import annotations

import pytest
import torch

from litemoe.interface import ExpertStore
from litemoe.model.loader import ModelMeta


class FakeStore(ExpertStore):
    """In-memory ExpertStore: 1x1-weight banks, deterministic payload size."""

    PAYLOAD_BYTES = 128  # fake quantized on-disk size per expert

    def __init__(self, n_layers: int = 2, n_experts: int = 8):
        self.meta = ModelMeta(
            backend="fake", arch="fake", hidden_size=4,
            n_layers=n_layers, n_experts=n_experts, top_k=2,
            expert_inter=2, head_dim=2, n_heads=2, n_kv_heads=1,
        )
        self.calls = 0  # count expert_banks() invocations

    def _bank(self, layer: int, expert: int) -> dict:
        # tag weights so identity is observable
        tag = float(layer * 100 + expert)
        return {
            "gate": torch.full((2, 4), tag, dtype=torch.float16),
            "up": torch.full((2, 4), tag, dtype=torch.float16),
            "down": torch.full((4, 2), tag, dtype=torch.float16),
        }

    def load_dense(self, dtype=torch.float16):
        return {}

    def expert_banks(self, layer, eids, dtype=torch.float16):
        self.calls += 1
        eids = list(dict.fromkeys(int(e) for e in eids))
        return {e: self._bank(layer, e) for e in eids}

    def gate(self, layer, dtype=torch.float16) -> torch.Tensor:
        return torch.zeros(self.meta.n_experts, 4, dtype=dtype)

    def expert_payload_bytes_for(self, layer, eids) -> int:
        return self.PAYLOAD_BYTES * len(dict.fromkeys(int(e) for e in eids))


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()
