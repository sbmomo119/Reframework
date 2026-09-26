"""Tests for the MoE expert-cache eviction policy (``RE_CACHE_POLICY``).

Covers :meth:`reframework.moe.offload_cache.ExpertOffloadCache._evict` under
both policies:

  * ``lru`` (default, backward-compatible): stale candidates are evicted
    LRU-first; the two-tier protection (hard = current step, soft = next-step
    lookahead) is unchanged.
  * ``least_stale``: the candidate *least likely to be routed again* is
    evicted first — highest routing staleness from ``Prediction._age`` (most
    decode steps since last activation; never-seen = maximally stale) — with
    LRU order as the tie-break, and a graceful degradation to plain LRU when
    no predictor is attached.
"""

from __future__ import annotations

import os

import pytest
import torch

from reframework.moe.offload_cache import ExpertOffloadCache, Prediction

DEVICE = torch.device("cpu")


def _cache(E: int, cap: int, H: int = 4, I: int = 2) -> ExpertOffloadCache:
    cache = ExpertOffloadCache(device=DEVICE, lru_capacity=cap, dtype=torch.float16)
    experts = []
    for _ in range(E):
        w1 = torch.randn(2 * I, H, dtype=torch.float16)
        w2 = torch.randn(H, I, dtype=torch.float16)
        experts.append({"w1": w1, "w2": w2})
    cache.load(experts)
    return cache


@pytest.fixture
def policy(monkeypatch):
    """Set RE_CACHE_POLICY for the duration of a test; restore afterwards."""

    def _set(value: str) -> None:
        monkeypatch.setenv("RE_CACHE_POLICY", value)

    return _set


# ------------------------------------------------------------------- default lru
def test_lru_is_default_when_unset(policy, monkeypatch):
    monkeypatch.delenv("RE_CACHE_POLICY", raising=False)
    cache = _cache(E=6, cap=3)
    cache.prefetch([0, 1, 2])
    cache.prefetch([3])  # pushes len to 4; LRU-oldest (0) goes
    assert set(cache._lru.keys()) == {1, 2, 3}
    assert 0 not in cache._lru


def test_lru_invalid_value_falls_back(policy):
    policy("bogus_policy")
    cache = _cache(E=6, cap=3)
    cache.prefetch([0, 1, 2])
    cache.prefetch([3])
    assert set(cache._lru.keys()) == {1, 2, 3}  # behaves exactly like lru


# ------------------------------------------------------------- least_stale: stale by age
def test_least_stale_evicts_most_stale_not_lru_oldest(policy):
    """LRU-oldest (0) is the *freshest* by routing age; the *newest* LRU
    entry (2) is the most stale and must go instead."""
    policy("least_stale")
    cache = _cache(E=8, cap=3)
    cache.prefetch([0, 1, 2])  # LRU order [0, 1, 2]
    pred = Prediction(cache, window=16)
    cache._predictor = pred
    pred._age = {0: 1, 1: 4, 2: 8}  # 0 = used 1 step ago, 2 = used 8 steps ago
    cache.prefetch([3])  # len 4 > cap 3; stale candidates {0,1,2}
    assert 2 not in cache._lru   # most stale evicted
    assert set(cache._lru.keys()) == {0, 1, 3}


def test_least_stale_without_predictor_degrades_to_lru(policy):
    policy("least_stale")
    cache = _cache(E=6, cap=3)  # no attach_predictor -> no routing signal
    cache.prefetch([0, 1, 2])
    cache.prefetch([3])
    assert set(cache._lru.keys()) == {1, 2, 3}  # identical to plain lru


def test_least_stale_never_seen_counts_as_maximally_stale(policy):
    policy("least_stale")
    cache = _cache(E=8, cap=3)
    cache.prefetch([0, 1, 2])
    pred = Prediction(cache, window=16)
    cache._predictor = pred
    pred._age = {0: 1, 2: 3}  # expert 1 was never routed -> age = window (16)
    cache.prefetch([3])
    assert 1 not in cache._lru
    assert set(cache._lru.keys()) == {0, 2, 3}


def test_least_stale_tie_breaks_by_lru_order(policy):
    policy("least_stale")
    cache = _cache(E=8, cap=3)
    cache.prefetch([0, 1, 2])  # LRU order [0, 1, 2]
    pred = Prediction(cache, window=16)
    cache._predictor = pred
    pred._age = {0: 5, 1: 5, 2: 2}  # 0 and 1 tied at max staleness
    cache.prefetch([3])
    # Tie -> oldest LRU entry (0) wins.
    assert 0 not in cache._lru
    assert set(cache._lru.keys()) == {1, 2, 3}


# ------------------------------------------------------- least_stale: two-tier protection
def test_least_stale_hard_never_evicted(policy):
    policy("least_stale")
    cache = _cache(E=6, cap=2)
    cache.prefetch([0, 1])
    pred = Prediction(cache, window=16)
    cache._predictor = pred
    pred._age = {0: 10, 1: 10, 2: 0}  # the *new* expert 2 looks freshest
    cache.prefetch([2])  # hard = {2}; must evict a stale one, never 2
    assert 2 in cache._lru
    assert len(cache._lru) == 2


def test_least_stale_soft_evicted_only_after_stale(policy):
    policy("least_stale")
    cache = _cache(E=8, cap=4)
    cache.prefetch([0, 1, 7])
    cache._next_reserve = frozenset({0, 1})  # soft (next-step lookahead)
    pred = Prediction(cache, window=16)
    cache._predictor = pred
    # stale = {7}; soft = {0,1}. Stale 7 must go first even though 0/1 are
    # staler by age.
    pred._age = {0: 9, 1: 9, 7: 1}
    cache.prefetch([2, 3])  # len 5 > cap 4
    assert 7 not in cache._lru
    assert {0, 1, 2, 3} <= set(cache._lru.keys())


def test_least_stale_picks_stalest_soft_when_no_stale_left(policy):
    policy("least_stale")
    cache = _cache(E=6, cap=3)
    cache.prefetch([0, 1])
    cache._next_reserve = frozenset({0, 1})  # both soft; no stale expert
    pred = Prediction(cache, window=16)
    cache._predictor = pred
    pred._age = {0: 2, 1: 7}  # 1 is staler -> evict 1, not LRU-oldest 0
    cache.prefetch([2, 3])  # len 4 > cap 3 -> one soft must go
    assert 1 not in cache._lru
    assert set(cache._lru.keys()) == {0, 2, 3}


# ------------------------------------------------------------------ lru control (same shape as least_stale tests)
def test_lru_oldest_wins_same_scenario(policy):
    """Control for test_least_stale_evicts_most_stale_not_lru_oldest: under
    ``lru`` the LRU-oldest expert 0 is evicted instead of 2."""
    policy("lru")
    cache = _cache(E=8, cap=3)
    cache.prefetch([0, 1, 2])
    pred = Prediction(cache, window=16)
    cache._predictor = pred
    pred._age = {0: 1, 1: 4, 2: 8}  # ages are irrelevant under lru
    cache.prefetch([3])
    assert 0 not in cache._lru
    assert set(cache._lru.keys()) == {1, 2, 3}


# ------------------------------------------------------------- env getter directly
def test_env_getter_normalizes():
    from reframework import env

    cases = {"least_stale": "least_stale", "LEAST_STALE": "least_stale", "lru": "lru"}
    for raw, expected in cases.items():
        os.environ["RE_CACHE_POLICY"] = raw
        try:
            assert env.get_moe_cache_policy() == expected
        finally:
            del os.environ["RE_CACHE_POLICY"]
    os.environ["RE_CACHE_POLICY"] = "garbage"
    try:
        assert env.get_moe_cache_policy() == "lru"
    finally:
        del os.environ["RE_CACHE_POLICY"]
