"""Tests for the MoE lookahead prefetch (predict next-step experts -> LRU).

Covers the three moving parts added for the PCIe-latency-hiding lookahead:

  * :class:`reframework.moe.offload_cache.Prediction`
      - ``predict_next`` ranks non-resident experts by activation frequency
        (with a recency tiebreak) and filters out what is already resident.
      - ``record_step`` on a prefill batch resets history; on a decode batch it
        ages history, appends, and triggers ``reserve``.
  * :meth:`ExpertOffloadCache.reserve`
      - dynamic ``k``: bounded by (a) the measured PCIe bandwidth budget over
        the inter-forward window, and (b) the LRU's evictable room.
  * :meth:`ExpertOffloadCache.prefetch` / :meth:`_evict`
      - two-tier protection: the current step's set is *hard* (never evicted),
        the next-step lookahead set is *soft* (evicted only once stale experts
        are gone).

Everything runs on CPU (``device=cpu``): on a CPU-only torch build
``measure_bandwidth`` returns None, so the bandwidth cap is exercised by
setting ``_bw_gb_s`` directly. No CUDA required.
"""

from __future__ import annotations

import torch

from reframework.moe.offload_cache import ExpertOffloadCache, Prediction
from reframework.moe.moe_layer import FusedMoE

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------- helpers
def _cache(E: int, cap: int, H: int = 4, I: int = 2) -> ExpertOffloadCache:
    """Build a CPU offload cache with ``E`` experts of a known byte size."""
    cache = ExpertOffloadCache(device=DEVICE, lru_capacity=cap, dtype=torch.float16)
    experts = []
    for _ in range(E):
        w1 = torch.randn(2 * I, H, dtype=torch.float16)
        w2 = torch.randn(H, I, dtype=torch.float16)
        experts.append({"w1": w1, "w2": w2})
    cache.load(experts)
    return cache


def _expert_bytes(cache: ExpertOffloadCache) -> int:
    w1, w2 = cache._host[0]
    return (w1.numel() + w2.numel()) * cache.dtype.itemsize


# ------------------------------------------------------------- Prediction unit
def test_predict_next_ranks_by_frequency_and_filters_resident():
    cache = _cache(E=6, cap=6)
    # Make experts 0 and 1 resident (they should be filtered out of predictions).
    cache.prefetch([0, 1])
    pred = Prediction(cache, window=16)
    # Feed history directly (not record_step, which auto-reserves and would fill
    # the LRU so that every expert becomes resident). Expert 2 fires every step,
    # expert 3 twice, expert 4 once; 0 and 1 also fire but are resident.
    pred._history = [
        frozenset({2, 4, 0}),
        frozenset({2, 3, 1}),
        frozenset({2, 3}),
    ]
    pred._age = {2: 0, 3: 1, 4: 2}
    ranked = pred.predict_next()
    # 0 and 1 are resident -> excluded.
    assert 0 not in ranked and 1 not in ranked
    # Non-resident survivors: 2 (freq 3), 3 (freq 2), 4 (freq 1).
    assert set(ranked) == {2, 3, 4}
    # Frequency ordering: 2 first, then 3, then 4.
    assert ranked[0] == 2
    assert ranked.index(3) < ranked.index(4)


def test_predict_next_recency_tiebreak():
    cache = _cache(E=4, cap=8)  # nothing resident yet
    pred = Prediction(cache, window=16)
    # Both fire once, but expert 1 was seen *more recently*. Feed history
    # directly (not record_step, which would auto-reserve and fill the LRU).
    pred._history.append(frozenset({0}))
    pred._history.append(frozenset({1}))
    pred._age = {0: 1, 1: 0}
    ranked = pred.predict_next()
    # Equal frequency (1 each); the just-seen expert 1 must outrank 0.
    assert ranked[0] == 1
    assert ranked.index(1) < ranked.index(0)


def test_prefill_resets_history():
    cache = _cache(E=4, cap=8)
    pred = Prediction(cache, window=16)
    pred.record_step([0], is_prefill=False, forward_ms=0.0)
    pred.record_step([1], is_prefill=False, forward_ms=0.0)
    assert pred._history  # has decode history
    # A prefill batch clears history and does not look ahead.
    pred.record_step([0, 1, 2, 3], is_prefill=True, forward_ms=0.0)
    assert list(pred._history) == []
    assert pred._age == {}
    # With no history, nothing is predicted.
    assert pred.predict_next() == []


# ------------------------------------------------------------- reserve dynamic k
def test_reserve_bandwidth_caps_k():
    E = 8
    cache = _cache(E=E, cap=16)  # LRU room is generous; bandwidth must bind.
    b = _expert_bytes(cache)
    cache._bw_headroom = 1.0
    cache._bw_gb_s = 1e9  # 1 GB/s
    # budget = bw_gb_s * 1e9 * (forward_ms/1000) * headroom = 1e15 * forward_ms
    # bytes. Pick forward_ms so the budget covers *just* 2 experts (2.5*b):
    # the top 2 fit, the 3rd breaks the window.
    forward_ms = 2.5 * b / 1e15
    n = cache.reserve([0, 1, 2, 3], forward_ms=forward_ms, protect_current=[])
    assert n == 2
    assert set(cache._lru.keys()) == {0, 1}
    # The lookahead set is remembered for prefetch's soft protection.
    assert cache._next_reserve == frozenset({0, 1})
    assert cache.stats.predicts == 2


def test_reserve_lru_room_caps_k_and_evicts_stale():
    # LRU is full of stale experts; reserve must evict them (never the
    # protect_current set) to make room, up to the predicted list.
    cache = _cache(E=8, cap=4)
    cache.prefetch([0, 1, 2, 3])  # LRU full of "stale" experts
    # Current step used {2,3}; predict {4,5} for next.
    n = cache.reserve([4, 5], forward_ms=0.0, protect_current=[2, 3])
    assert n == 2
    assert set(cache._lru.keys()) == {2, 3, 4, 5}
    # Stale {0,1} were evicted; current {2,3} kept; predicted {4,5} loaded.
    assert 0 not in cache._lru and 1 not in cache._lru
    assert cache._next_reserve == frozenset({4, 5})


def test_reserve_skips_resident_and_noop():
    cache = _cache(E=4, cap=4)
    cache.prefetch([0])  # 0 resident
    # 0 is resident -> skipped; only 1,2 loaded.
    n = cache.reserve([0, 1, 2], forward_ms=0.0, protect_current=[0])
    assert n == 2
    assert set(cache._lru.keys()) == {0, 1, 2}
    # Empty predicted list is a no-op.
    assert cache.reserve([], forward_ms=0.0) == 0


# ------------------------------------------------------- prefetch two-tier evict
def test_prefetch_evicts_stale_not_soft():
    # LRU holds soft {0,1} (next-step lookahead) plus a stale expert 7.
    cache = _cache(E=8, cap=4)
    cache.prefetch([0, 1, 7])
    cache._next_reserve = frozenset({0, 1})
    # Current step needs {2,3}; this pushes len to 5 > cap=4.
    cache.prefetch([2, 3])
    # The stale expert 7 must be evicted; soft {0,1} and hard {2,3} stay.
    assert set(cache._lru.keys()) == {0, 1, 2, 3}
    assert 7 not in cache._lru


def test_prefetch_evicts_stale_first_then_soft():
    # No stale expert available: current load overflows and must give up a
    # *soft* entry (oldest soft) rather than a hard (current) one.
    cache = _cache(E=6, cap=4)
    cache.prefetch([0, 1])
    cache._next_reserve = frozenset({0, 1})
    # Current {2,3,4}: after fill len=5 > cap=4, and there is no stale expert,
    # so the oldest soft expert (0) is sacrificed.
    cache.prefetch([2, 3, 4])
    assert {2, 3, 4} <= set(cache._lru.keys())  # current set intact
    assert 0 not in cache._lru                  # oldest soft evicted
    assert 1 in cache._lru                      # newer soft kept


def test_prefetch_hard_never_evicted_when_only_hard_exceeds():
    # Current set alone exceeds capacity and is all hard-protected: the LRU may
    # exceed cap (you cannot evict the current step), which is the intended
    # degenerate behaviour.
    cache = _cache(E=6, cap=2)
    cache.prefetch([0, 1, 2])
    # All three are the current (hard) set -> none evicted even though len>cap.
    assert set(cache._lru.keys()) == {0, 1, 2}


def test_reset_lru_clears_reserve_and_history():
    cache = _cache(E=6, cap=4)
    cache.attach_predictor(window=4)
    cache._predictor.record_step([0], is_prefill=False, forward_ms=0.0)
    cache.reserve([3], forward_ms=0.0, protect_current=[0])
    assert cache._next_reserve  # non-empty
    cache.reset_lru()
    assert len(cache._lru) == 0
    assert cache._next_reserve == frozenset()
    assert list(cache._predictor._history) == []


# ------------------------------------------------------- FusedMoE integration
def _routed_moe(E: int = 8) -> FusedMoE:
    """A MoE whose gate routes a token to expert == the spiked input dim.

    gate[e, e] = 10 (rest 0); a token with a spike of 5.0 at dim d produces
    gate_out[d] = 50 >> gate_out[others] = 0, so top-1 routes to expert d.
    Deterministic and CPU-friendly. (Values kept small: 5*100 would overflow
    fp16 -> inf -> NaN -> topk collapses to expert 0.)
    """
    torch.manual_seed(0)
    mlp = FusedMoE(
        num_experts=E,
        top_k=1,
        hidden_size=16,
        intermediate_size=8,
        dtype=torch.float16,
    )
    with torch.no_grad():
        mlp.gate.fill_(0.0)
        for e in range(E):
            mlp.gate[e, e] = 10.0
    return mlp


def test_forward_drives_lookahead_reserve():
    mlp = _routed_moe(E=8)
    cache = mlp.build_offload_cache(DEVICE, lru_capacity=4)
    mlp.attach_predictor(window=8)
    assert mlp._predictor is not None
    assert cache._predictor is mlp._predictor

    def x_spike(d: int) -> torch.Tensor:
        x = torch.zeros(1, 16, dtype=torch.float16)
        x[0, d] = 5.0
        return x

    # Decode through experts 0,1,2,3,4 (M=1 => not prefill). Each forward
    # prefetched that expert; with cap=4 expert 0 was evicted mid-sequence.
    for d in range(5):
        out = mlp(x_spike(d))
        assert out.shape == (1, 16)
    # The 5th forward's record_step predicted expert 0 (the only non-resident
    # expert with history) and reserved it, evicting the oldest stale expert.
    # The lookahead therefore pulled expert 0 *back* into the LRU — that is the
    # whole point: the next step needs 0 and it is already resident.
    assert cache._next_reserve == frozenset({0})
    assert 0 in cache._lru
    assert 4 in cache._lru  # just-used expert still resident
    assert cache.stats.predicts >= 1


def test_forward_prefill_does_not_lookahead():
    mlp = _routed_moe(E=8)
    cache = mlp.build_offload_cache(DEVICE, lru_capacity=8)
    mlp.attach_predictor(window=8)
    # A prefill batch (M>1) resets history and must not reserve anything.
    x = torch.zeros(4, 16, dtype=torch.float16)
    for d in range(4):
        x[d, d] = 5.0
    mlp(x)
    assert list(mlp._predictor._history) == []
    assert cache.stats.predicts == 0
