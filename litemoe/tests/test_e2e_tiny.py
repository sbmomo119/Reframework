"""End-to-end: full executor pipeline on the tiny-qwen3-moe checkpoint.

Skipped when the checkpoint is absent (CI without the model dir).
Covers: config -> make_store (directory) -> Qwen3MoEModel -> MoELayer
3-tuple -> LRUCache get()/fetch_many -> profiler report -> activation jsonl.
"""
from __future__ import annotations

import json
import os

import pytest

TINY = "/home/samuel/Re/.hf/tiny-qwen3-moe"
pytestmark = pytest.mark.skipif(
    not (os.path.isdir(TINY) and os.path.exists(os.path.join(TINY, "model.safetensors"))),
    reason="tiny-qwen3-moe checkpoint not present",
)


def _run(max_new_tokens: int = 8, tmp_path=None):
    from litemoe.config import LitemoeConfig
    from litemoe.runtime.executor import Executor

    act = os.path.join(str(tmp_path), "act.jsonl") if tmp_path else "./_e2e_act.jsonl"
    cfg = LitemoeConfig.from_dict({
        "model": {"path": TINY},
        "compute": {"dtype": "fp16", "device": "cpu"},
        "cache": {"strategy": "lru", "max_experts": 8},
        "metrics": {"log_activations": True, "activations_file": act,
                    "print_per_step": False},
        "prompt": "The capital of France is",
        "max_new_tokens": max_new_tokens,
        "seed": 0,
    })
    rep = Executor(cfg).run(prompt=cfg.prompt, max_new_tokens=max_new_tokens)
    return rep, act


def test_tiny_e2e_report_fields(tmp_path):
    rep, act = _run(tmp_path=tmp_path)
    assert rep.new_tokens == 8
    assert rep.prompt_tokens >= 1
    assert rep.hit_rate > 0.0
    assert rep.cache_hits + rep.cache_misses > 0
    assert rep.transfer_bytes > 0
    assert rep.tok_per_s > 0.0
    assert rep.ttft_ms > 0.0
    # summary() must render without error (uses the renamed fields)
    assert "cache hit-rate" in rep.summary()


def test_tiny_e2e_activation_log(tmp_path):
    rep, act = _run(tmp_path=tmp_path)
    assert os.path.exists(act)
    with open(act) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # (1 prefill + N decode) steps x 2 layers
    assert len(rows) == (1 + rep.new_tokens) * 2
    for r in rows:
        assert set(r) == {"step", "layer", "experts"}
        assert len(r["experts"]) == 2  # top_k = 2
        assert 0 <= r["layer"] < 2


def test_tiny_e2e_pde_placeholder(tmp_path):
    """strategy=pde (placeholder) runs identically end-to-end."""
    from litemoe.config import LitemoeConfig
    from litemoe.runtime.executor import Executor

    cfg = LitemoeConfig.from_dict({
        "model": {"path": TINY},
        "compute": {"dtype": "fp16", "device": "cpu"},
        "cache": {"strategy": "pde", "max_experts": 8, "predict_window": 8},
        "metrics": {"log_activations": False, "print_per_step": False},
        "prompt": "The capital of France is",
        "max_new_tokens": 4, "seed": 0,
    })
    rep = Executor(cfg).run(prompt=cfg.prompt, max_new_tokens=4)
    assert rep.new_tokens == 4
    assert rep.hit_rate > 0.0
