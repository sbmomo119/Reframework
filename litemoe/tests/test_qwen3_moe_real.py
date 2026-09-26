"""Regression: real Qwen3-MoE checkpoint must not degenerate."""
import os
import pytest

CKPT = "/home/samuel/huihui-ai--Huihui-MoE-1.2B-A0.6B/snapshots/master"
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "model.safetensors")),
    reason="Qwen3-MoE checkpoint not present",
)


def _run(prompt="The capital of France is", max_new_tokens=8,
         strategy="lru", max_experts=84):
    from litemoe.config import LitemoeConfig
    from litemoe.runtime.executor import Executor
    cfg = LitemoeConfig.from_dict({
        "model": {"path": CKPT, "tokenizer": CKPT},
        "compute": {"dtype": "fp16", "device": "cpu"},
        "cache": {"strategy": strategy, "max_experts": max_experts,
                  "predict_window": 8},
        "metrics": {"log_activations": False, "print_per_step": False},
        "prompt": prompt, "max_new_tokens": max_new_tokens, "seed": 42,
    })
    return Executor(cfg).run(prompt=prompt, max_new_tokens=max_new_tokens)


def test_real_qwen3_moe_coherent():
    rep = _run()
    text = rep.output_text.strip()
    assert text, "output is empty"
    assert "Paris" in text, f"unexpected output: {text!r}"
    for bad in ("useruser", "is is is", "the the the"):
        assert bad not in text, f"degenerate: {text!r}"


def test_real_qwen3_moe_cache_hits():
    rep = _run()
    assert rep.hit_rate > 0.5, f"hit-rate too low: {rep.hit_rate}"
    assert rep.cache_hits + rep.cache_misses > 0


def test_real_qwen3_moe_pde_runs():
    rep = _run(strategy="pde", max_experts=84)
    assert rep.new_tokens == 8
    assert rep.hit_rate > 0.0


def test_qk_norm_and_rope_theta_loaded():
    """Guard against regression: QK-Norm weights and rope_theta must be applied."""
    import torch
    from litemoe.config import LitemoeConfig
    from litemoe.runtime.executor import Executor

    cfg = LitemoeConfig.from_dict({
        "model": {"path": CKPT, "tokenizer": CKPT},
        "compute": {"dtype": "fp16", "device": "cpu"},
        "cache": {"strategy": "lru", "max_experts": 84},
        "metrics": {"log_activations": False, "print_per_step": False},
        "prompt": "x", "max_new_tokens": 1, "seed": 0,
    })
    ex = Executor(cfg)
    store, cache, model = ex._load()

    # rope_theta 必须来自 config，不是默认 10000
    assert store.meta.extra.get("rope_theta") == 1000000, \
        f"rope_theta not propagated: {store.meta.extra.get('rope_theta')}"

    # QK-Norm 权重必须加载
    for L in range(store.meta.n_layers):
        attn = model.attns[L]
        assert attn.q_norm is not None, f"layer {L} q_norm missing"
        assert attn.k_norm is not None, f"layer {L} k_norm missing"
        assert attn.q_norm.shape == (store.meta.head_dim,)
