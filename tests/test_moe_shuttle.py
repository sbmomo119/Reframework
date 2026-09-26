"""Tests for the three-way PCIe competition modeling (traffic A).

The lookahead's ``reserve()`` used to size its prefetch window against the
*entire* PCIe link — but in a dual-GPU MoE deployment the link also carries
the token hidden-state transfers of the CPU-split path.  These tests cover
the mock shuttle (bandwidth model), the ``inflight_bytes`` budget deduction
in ``reserve``/``RecordStats``/``Prediction``, and the plumbing that feeds
the shuttle's in-flight token bytes into ``record_step``.
"""
import torch

import reframework.moe.offload_cache as oc
from reframework.moe.moe_layer import CpuShuttle
from reframework.moe.offload_cache import Prediction, RecordStats, ReserveResult, _Budget

# ---------------------------------------------------------------------------
# CpuShuttle: the bandwidth model for token (traffic A) transfers
# ---------------------------------------------------------------------------


def test_shuttle_send_and_inflight():
    s = CpuShuttle(n_steps=2)
    assert s.inflight_bytes == 0.0
    s.send(1000.0)
    assert s.inflight_bytes == 1000.0


def test_shuttle_noop_on_zero_bytes():
    s = CpuShuttle(n_steps=2)
    s.send(0.0)
    s.step()
    s.step()
    assert s.inflight_bytes == 0.0


def test_shuttle_drains_over_n_steps():
    s = CpuShuttle(n_steps=2)
    s.send(800.0)  # 400 bytes/step
    assert s.inflight_bytes == 800.0
    s.step()
    assert s.inflight_bytes == 400.0
    s.step()
    assert s.inflight_bytes == 0.0
    s.step()  # past the horizon: stays 0, never negative
    assert s.inflight_bytes == 0.0


def test_shuttle_multiple_transfers_stack():
    s = CpuShuttle(n_steps=1)
    s.send(100.0)
    s.send(50.0)
    assert s.inflight_bytes == 150.0
    s.step()  # both drain (1 step each)
    assert s.inflight_bytes == 0.0


# ---------------------------------------------------------------------------
# RecordStats / ReserveResult: the deduction itself
# ---------------------------------------------------------------------------


def test_stats_inflight_deducted():
    st = RecordStats(bandwidth_gbps=5.0, forward_ms=10.0, inflight_bytes=100.0)
    assert st.budget_bytes == max(0.0, 5000.0 - 100.0)


def test_stats_inflight_clamped_to_zero():
    st = RecordStats(bandwidth_gbps=5.0, forward_ms=10.0, inflight_bytes=999999.0)
    assert st.budget_bytes == 0.0


def test_stats_no_inflight_unchanged():
    st = RecordStats(bandwidth_gbps=5.0, forward_ms=10.0)
    assert st.budget_bytes == 5000.0


def test_reserve_inflight_shrinks_window():
    cache = oc.LiteMoEOffloadCache(
        w1q=torch.zeros(16, 64, dtype=torch.int8),
        w2q=torch.zeros(16, 32, dtype=torch.int8),
        scale_w1=torch.ones(16), scale_w2=torch.ones(16),
    )
    res = cache.reserve(oc._Budget(5000.0, 5.0, 10.0),
                        inflight_bytes=2000.0)
    # Budget after deduction: 3000 bytes; each expert row-pair = 64+32 = 96 B
    # so the window holds 31 experts (the budget used to hold 50).
    assert res.experts_budget == 31
    assert res.budget_bytes == 3000.0


def test_reserve_inflight_covers_all():
    cache = oc.LiteMoEOffloadCache(
        w1q=torch.zeros(16, 64, dtype=torch.int8),
        w2q=torch.zeros(16, 32, dtype=torch.int8),
        scale_w1=torch.ones(16), scale_w2=torch.ones(16),
    )
    res = cache.reserve(oc._Budget(5000.0, 5.0, 10.0),
                        inflight_bytes=5000.0)
    assert res.experts_budget == 0
    assert res.budget_bytes == 0.0


def test_reserve_zero_inflight_matches_baseline():
    cache = oc.LiteMoEOffloadCache(
        w1q=torch.zeros(16, 64, dtype=torch.int8),
        w2q=torch.zeros(16, 32, dtype=torch.int8),
        scale_w1=torch.ones(16), scale_w2=torch.ones(16),
    )
    a = cache.reserve(oc._Budget(5000.0, 5.0, 10.0), inflight_bytes=0.0)
    b = cache.reserve(oc._Budget(5000.0, 5.0, 10.0))
    assert a.experts_budget == b.experts_budget == 50


# ---------------------------------------------------------------------------
# Prediction.record_step threads the inflight bytes into reserve
# ---------------------------------------------------------------------------


def test_record_step_passes_inflight_to_reserve():
    cache = oc.LiteMoEOffloadCache(
        w1q=torch.zeros(16, 64, dtype=torch.int8),
        w2q=torch.zeros(16, 32, dtype=torch.int8),
        scale_w1=torch.ones(16), scale_w2=torch.ones(16),
    )
    pred = Prediction(cache, window_budget_bytes=100.0)
    pred.record_step([1], is_prefill=True, forward_ms=10.0, inflight_bytes=100.0)
    assert pred.stats[-1].inflight_bytes == 100.0
    assert pred.stats[-1].budget_bytes == 0.0


# ---------------------------------------------------------------------------
# FusedMoE plumbing: token bytes recorded on the shuttle flow into record_step
# ---------------------------------------------------------------------------


def test_forward_split_feeds_shuttle_inflight_into_record_step():
    import torch
    from reframework.moe.moe_layer import FusedMoE

    torch.manual_seed(0)
    E, I, H = 8, 16, 16
    mlp = FusedMoE(num_experts=E, inter=I, hidden=H, top_k=2, cpu_split=True)
    mlp.attach_predictor(window_budget_bytes=200.0)
    assert mlp._predictor is not None

    calls = []
    orig_reserve = mlp._predictor.reserve

    def spy(*args, **kwargs):
        calls.append(kwargs.get("inflight_bytes", 0.0))
        return orig_reserve(*args, **kwargs)

    mlp._predictor.reserve = spy

    class _FakeShuttle:
        def __init__(self, inflight):
            self._i = inflight
            self.sent = []

        @property
        def inflight_bytes(self):
            return self._i

        def send(self, n_bytes):
            self.sent.append(n_bytes)

        def step(self):
            self._i = 0.0

    mlp.set_shuttle(_FakeShuttle(inflight=0.0))
    shuttle = mlp._shuttle

    x = torch.randn(4, H, dtype=torch.float32)
    # Every routed expert misses on step 1: all traffic goes through the
    # shuttle (4 tokens x H x 2 bytes fp16).
    mlp(x, torch.tensor([1, 2, 3, 4]))
    assert len(shuttle.sent) == 1
    assert shuttle.sent[0] == 4 * H * 2  # token hidden-state traffic (A)

    # Step 2: the predictor's reserve must see the shuttle's in-flight
    # bytes (the shuttle still holds the step-1 transfer, which drains
    # over 2 steps by default).
    mlp(x, torch.tensor([5, 6, 7, 8]))
    assert len(calls) == 1  # first step had no history -> no reserve yet
    assert calls[0] == 4 * H * 2