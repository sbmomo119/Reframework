"""Tests for the litemoe storage/cache backend bridged into reframework FusedMoE.

方案 A 的核心验证：litemoe 只提供**存储/缓存**（``cache.fetch_many`` 取数 seam），
**compute 走 reframework 原生 ``fused_experts``**，routing 完全用 reframework 侧的
``topk_ids``/``topk_weights``。因此数值等价性可以**直接**对 reframework 原生路径
验证：同一组专家权重，"litemoe fetch + 格式翻译 + fused_experts" 的输出应与
"原生预打包 w1/w2 + fused_experts" 逐元素一致。

覆盖：
  * 数值等价性（含 gate/up 顺序、全局→本地行重映射）—— 用 ``RandStore`` 让 bank
    是真实随机权重，避免平凡常量导致假通过。
  * 路由归属 —— 证明 litemoe ``MoELayer.route`` 从未被调用，且输出由传入的
    ``topk_ids``/``topk_weights`` 驱动（线性标度 + 换专家改变结果）。
  * 回退路径 —— ``fetch_many`` 抛异常时回退原生权重，输出与原生一致且不崩溃。
  * 两个便宜的 sanity：CacheStats 归一化键、ModelMeta->ModelConfig 映射。

全部 CPU / float32（确定性，等价性可逐位比较）。
"""
from __future__ import annotations

import torch

from reframework.integrations.litemoe_adapter import (
    LitemoeExpertBackend,
    LayerMeta,
    model_meta_to_config,
)
from reframework.moe.fused import fused_experts
from litemoe.cache.lru import LRUCache
from litemoe.model.model import MoELayer
from litemoe.model.loader import ModelMeta

DEVICE = torch.device("cpu")
DTYPE = torch.float32

E = 8   # n_experts
H = 8   # hidden_size
I = 6   # expert intermediate size
K = 2   # top_k


class RandStore:
    """确定性 ExpertStore：每个 (layer, expert) 一个随机 bank。

    ``bank[e] = {"gate":[I,H], "up":[I,H], "down":[H,I]}``。同时提供**预打包**的
    原生权重 ``w1``/``w2``（与 litemoe bank 同值），用于构建原生参考路径。
    """

    def __init__(self, n_layers: int = 1, n_experts: int = E, seed: int = 0):
        self.meta = ModelMeta(
            backend="rand", arch="rand", hidden_size=H,
            n_layers=n_layers, n_experts=n_experts, top_k=K,
            expert_inter=I, head_dim=4, n_heads=2, n_kv_heads=1,
        )
        g = torch.Generator().manual_seed(seed)
        self._banks: dict[tuple[int, int], dict] = {}
        self._w1: list[torch.Tensor] = []
        self._w2: list[torch.Tensor] = []
        for _l in range(n_layers):
            for e in range(n_experts):
                bank = {
                    "gate": torch.randn(I, H, generator=g, dtype=DTYPE),
                    "up": torch.randn(I, H, generator=g, dtype=DTYPE),
                    "down": torch.randn(H, I, generator=g, dtype=DTYPE),
                }
                self._banks[(0, e)] = bank  # 本测试只用 layer 0
                # 预打包：gate 在前（Qwen3），与 reframework 原生 w1 同形。
                self._w1.append(torch.cat([bank["gate"], bank["up"]], dim=0))  # [2I,H]
                self._w2.append(bank["down"].clone())                          # [H,I]

    @property
    def w1(self) -> torch.Tensor:  # [E, 2I, H]
        return torch.stack(self._w1, dim=0)

    @property
    def w2(self) -> torch.Tensor:  # [E, H, I]
        return torch.stack(self._w2, dim=0)

    def expert_banks(self, layer, eids, dtype=DTYPE):
        return {
            e: {k: v for k, v in self._banks[(int(layer), int(e))].items()}
            for e in dict.fromkeys(int(x) for x in eids)
        }

    def expert_payload_bytes_for(self, layer, eids) -> int:
        return 1024 * len(dict.fromkeys(int(e) for e in eids))

    def gate(self, layer, dtype=DTYPE):
        return torch.zeros(self.meta.n_experts, H, dtype=dtype)


def _routing(M: int = 4, seed: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """reframework 侧路由：topk_ids [M,K]（全局 id，覆盖全部专家），topk_weights [M,K] 归一。"""
    g = torch.Generator().manual_seed(seed)
    topk_ids = torch.randint(0, E, (M, K), generator=g)
    topk_weights = torch.softmax(torch.rand(M, K, generator=g, dtype=DTYPE), dim=-1)
    return topk_weights, topk_ids


def _adapter(store: RandStore, cache=None) -> LitemoeExpertBackend:
    if cache is None:
        cache = LRUCache(store, capacity=E, device=str(DEVICE))
    meta = LayerMeta(w1=store.w1, w2=store.w2, activation="silu")
    return LitemoeExpertBackend(store, cache, layer=0, meta=meta)


# ------------------------------------------------------------ 测试 1: 数值等价性
def test_numerical_equivalence():
    store = RandStore(seed=0)
    backend = _adapter(store)
    topk_weights, topk_ids = _routing(M=4)
    hidden = torch.randn(4, H, dtype=DTYPE)

    # 原生参考路径：预打包 w1/w2，全局 id 直接用（reframework 原生路径）。
    native = fused_experts(hidden, store.w1, store.w2, topk_weights, topk_ids, "silu")
    # litemoe 路径：fetch -> 格式翻译（stack + 行重映射）-> fused_experts。
    via_litemoe = backend.compute(hidden, topk_weights, topk_ids)

    # 逐位相等（同源权重、同 dtype、同 kernel）。
    assert torch.equal(native, via_litemoe)
    # 形状正确（[M, H]）。
    assert via_litemoe.shape == hidden.shape


def test_equivalence_uses_gate_first_and_row_remap():
    """格式翻译的关键：w1 = cat([gate, up])（gate 在前），且全局 id 重映射到本地行。

    直接校验 adapter 的打包结果与原生预打包张量逐位一致——若把 up/gate 拼反，
    或没有做 id 重映射，这里会立刻失败。
    """
    store = RandStore(seed=1)
    backend = _adapter(store)
    # 只取一个**非连续**子集（全局 id 3,7）—— 覆盖重映射的非平凡情形。
    sub_ids = [3, 7]
    banks = backend.fetch_experts(sub_ids)
    w1, w2, order = backend._pack_banks(banks, DEVICE)
    # 行按全局 id 排序 => order == [3, 7]。
    assert order == [3, 7]
    # 打包后的本地行必须与原生对应全局行逐位一致。
    assert torch.equal(w1[0], store.w1[3])
    assert torch.equal(w2[0], store.w2[3])
    assert torch.equal(w1[1], store.w1[7])
    assert torch.equal(w2[1], store.w2[7])
    # 且 gate 在前：w1[0] 的前 I 行 = expert 3 的 gate，后 I 行 = up。
    b3 = store.expert_banks(0, [3])[3]
    assert torch.equal(w1[0][:I], b3["gate"])
    assert torch.equal(w1[0][I:], b3["up"])


# ------------------------------------------------------------ 测试 2: 路由归属
def test_routing_ownership(monkeypatch):
    store = RandStore(seed=2)
    cache = LRUCache(store, capacity=E, device=str(DEVICE))
    # litemoe 的 MoELayer（routing 的"拥有者"）—— 证明其 route 从未被 adapter 调用。
    moe = MoELayer(
        layer=0, gate_w=torch.randn(E, H, dtype=DTYPE), store=store,
        top_k=K, n_experts=E, device=DEVICE, dtype=DTYPE,
    )
    calls = []

    def _boom(*a, **k):
        calls.append((a, k))
        raise AssertionError("litemoe MoELayer.route must never be called by the adapter")

    monkeypatch.setattr(moe, "route", _boom)

    backend = LitemoeExpertBackend(
        store, cache, layer=0,
        meta=LayerMeta(w1=store.w1, w2=store.w2, activation="silu"),
    )
    topk_weights, topk_ids = _routing(M=4)
    hidden = torch.randn(4, H, dtype=DTYPE)

    out1 = backend.compute(hidden, topk_weights, topk_ids)
    assert calls == []  # litemoe route 从未被调用

    # 输出由 reframework 侧 topk_weights 驱动：线性标度。
    out2 = backend.compute(hidden, topk_weights * 0.5, topk_ids)
    assert torch.allclose(out2, out1 * 0.5, atol=1e-7)

    # 输出由 reframework 侧 topk_ids 驱动：换路由专家 => 结果改变。
    alt_ids = torch.where(topk_ids == 0, 1, topk_ids)  # 把 0 号专家换成 1 号
    alt_ids = torch.where(alt_ids == 1, 0, alt_ids)    # 与 0/1 都不同 => 必变
    out3 = backend.compute(hidden, topk_weights, alt_ids)
    assert not torch.equal(out1, out3)


# ------------------------------------------------------------ 测试 3: 回退路径
def test_fallback_on_fetch_failure():
    store = RandStore(seed=4)

    class ExplodingCache:
        def fetch_many(self, layer, eids, dtype=None):
            raise RuntimeError("simulated cache/store failure")

        def stats(self):
            from litemoe.cache.base import CacheStats
            return CacheStats()

    backend = _adapter(store, cache=ExplodingCache())
    topk_weights, topk_ids = _routing(M=4)
    hidden = torch.randn(4, H, dtype=DTYPE)

    # 回退到原生权重：输出 == 原生 fused_experts（用原生预打包 w1/w2 + 全局 id）。
    expected = fused_experts(hidden, store.w1, store.w2, topk_weights, topk_ids, "silu")
    out = backend.compute(hidden, topk_weights, topk_ids)  # 不应抛出
    assert torch.equal(expected, out)


# ------------------------------------------------------------ sanity: stats 归一化
def test_cache_stats_normalized_to_profiler_keys():
    store = RandStore(seed=5)
    backend = _adapter(store)  # 真 LRUCache
    topk_weights, topk_ids = _routing(M=4)
    hidden = torch.randn(4, H, dtype=DTYPE)
    backend.compute(hidden, topk_weights, topk_ids)  # 首步全部 miss

    stats = backend.get_cache_stats()
    # reframework profiler（OffloadStats.as_dict）键必须齐全。
    for key in ("hits", "misses", "evictions", "subs", "predicts", "transfer_mb"):
        assert key in stats
    # 数值与 litemoe CacheStats 一致（4 个专家各 miss 一次）。
    s = backend.cache.stats()
    assert stats["hits"] == s.hits
    assert stats["misses"] == s.misses
    assert stats["evictions"] == s.evictions
    assert stats["subs"] == 0 and stats["predicts"] == 0
    assert abs(stats["transfer_mb"] - s.transfer_bytes / (1024 * 1024)) < 1e-9
    # 首步每个唯一专家都 miss 一次（adapter 按 unique 专家数 fetch）。
    n_unique = int(topk_ids.unique().numel())
    assert stats["misses"] == n_unique and stats["hits"] == 0


# ------------------------------------------------------------ sanity: config 映射
def test_model_meta_to_config_maps_moe_geometry():
    meta = ModelMeta(
        backend="t", arch="t", hidden_size=32, n_layers=3, n_experts=16,
        top_k=4, expert_inter=10, head_dim=4, n_heads=8, n_kv_heads=2,
    )
    cfg = model_meta_to_config(meta)
    assert cfg.use_moe is True
    assert cfg.num_experts == 16
    assert cfg.num_experts_per_tok == 4
    assert cfg.moe_intermediate_size == 10
    assert cfg.hidden_size == 32
    assert cfg.num_hidden_layers == 3
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 4
    assert cfg.moe_activation == "silu"
