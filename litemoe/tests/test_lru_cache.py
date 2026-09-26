"""LRUCache: hit/miss/eviction/stats, fetch_many wrapper, activation log."""
from __future__ import annotations

import torch

from litemoe.cache.lru import LRUCache, PerLayerLRUCache


def test_hit_miss_and_identity(fake_store):
    c = LRUCache(fake_store, capacity=4, device="cpu")
    b1 = c.get(0, 3)
    b2 = c.get(0, 3)  # hit: same object, no extra store call
    assert b1 is b2
    assert fake_store.calls == 1
    s = c.stats()
    assert (s.hits, s.misses) == (1, 1)


def test_lru_eviction_order(fake_store):
    c = LRUCache(fake_store, capacity=2, device="cpu")
    c.get(0, 0)
    c.get(0, 1)
    c.get(0, 0)      # touch 0 -> LRU tail is now 1
    c.get(0, 2)      # evicts 1
    assert (0, 0) in c and (0, 2) in c
    assert (0, 1) not in c
    assert c.stats().evictions == 1


def test_transfer_bytes_use_payload_not_fp_size(fake_store):
    from tests.conftest import FakeStore
    c = LRUCache(fake_store, capacity=8, device="cpu")
    c.get(0, 0)
    c.get(1, 5)
    # 2 experts x PAYLOAD_BYTES fake quant bytes (NOT the fp16 bank size)
    assert c.stats().transfer_bytes == 2 * FakeStore.PAYLOAD_BYTES


def test_fetch_many_is_get_wrapper(fake_store):
    c = LRUCache(fake_store, capacity=8, device="cpu")
    out = c.fetch_many(0, [3, 1, 3])  # dedup -> [3, 1]
    assert set(out) == {1, 3}
    s = c.stats()
    # 2 unique experts -> 2 store calls (one per get miss), 2 misses
    assert fake_store.calls == 2 and (s.hits, s.misses) == (0, 2)
    c.fetch_many(0, [3])  # now a hit, no extra store call
    assert c.stats().hits == 1 and fake_store.calls == 2


def test_stats_properties_hit_rate_transfer_mb(fake_store):
    c = LRUCache(fake_store, capacity=8, device="cpu")
    c.get(0, 0)
    c.get(0, 0)
    s = c.stats()
    assert s.total == 2
    assert s.hit_rate == 0.5
    assert s.miss_rate == 0.5
    assert abs(s.transfer_mb - (128 / 1e6)) < 1e-12


def test_activation_log(fake_store):
    c = LRUCache(fake_store, capacity=8, device="cpu")
    c.get(0, 2)  # resident
    c.record_activation(0, [2, 7], [0.6, 0.4])  # 7 not yet resident
    assert c.get_log() == []  # logging disabled by default
    c.enable_log()
    c.set_step(5)
    c.record_activation(0, [2, 7], [0.6, 0.4])
    recs = c.get_log()
    assert len(recs) == 1
    assert recs[0].step == 5 and recs[0].layer == 0
    assert recs[0].expert_ids == [2, 7]
    assert recs[0].hit_mask == [True, False]


def test_reset_vs_clear(fake_store):
    c = LRUCache(fake_store, capacity=4, device="cpu")
    c.get(0, 0)
    c.reset()      # drops residency, keeps stats
    assert len(c) == 0 and c.stats().misses == 1
    c.clear()      # drops residency AND stats
    assert len(c) == 0 and c.stats().misses == 0


def test_per_layer_lru_caps(fake_store):
    c = PerLayerLRUCache(fake_store, per_layer_cap={0: 2, 1: 2}, device="cpu")
    c.get(0, 0)
    c.get(0, 1)
    c.get(0, 2)   # evicts expert 0 in layer 0 only
    assert (0, 2) in c
    assert (0, 0) not in c
    assert c.stats().evictions == 1
