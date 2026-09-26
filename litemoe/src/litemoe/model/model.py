"""Qwen3-MoE forward pass (tiny ``qwen3`` and 14GB ``qwen35moe``).

This is the *model* layer on top of the :class:`~litemoe.interface.ExpertStore`
/ :class:`~litemoe.interface.Cache`. It contains NO weight ``nn.Parameters`` of
its own except the dense tensors (embeddings, norms, attention, shared experts);
the per-expert MoE banks live in the :class:`ExpertStore` and are handed to
:meth:`MoELayer.forward` through the :class:`Cache` (the simulated VRAM window).

Two architectures share the same :class:`MoELayer`:

* ``Qwen3MoEModel``  — tiny ``qwen3``: every block is a full GQA attention +
  sparse MoE.  (No q/k norm, no shared expert, tied embeddings.)
* ``Qwen35MoEModel`` — 14GB ``qwen35moe``: a mix of full-attention blocks and
  linear-attention (GatedDeltaNet / SSM) blocks, every block with a shared
  expert.  (See :class:`Qwen35FullAttention` / :class:`Qwen35SSM`.)

Routing is replicated exactly from the official HF
``Qwen3MoeTopKRouter`` / ``Qwen3MoeExperts``::

    logits      = h @ gate_w.T                       # (T, E)
    probs       = softmax(logits, fp32, dim=-1)
    v, idx      = topk(probs, top_k)                 # (T, top_k)
    if norm_topk: v  = v / v.sum(-1, keepdim)
    out[t]     += w[t, pos] * expert_idx(h_t)

"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from litemoe.interface import Cache, ExpertBanks, ExpertStore
from litemoe.model.loader import ModelMeta


# --------------------------------------------------------------------------- #
# small building blocks (no parameters — weights are supplied by the store)   #
# --------------------------------------------------------------------------- #
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3 RMSNorm (fp32 accumulate)."""
    dtype = x.dtype
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * xf).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_cache(
    n_positions: int, head_dim: int, n_rot: int, rope_theta: float,
    base: int = 1, device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for a rotary embedding (HF-style, first n_rot dims).

    Returns ``(cos, sin)`` each of shape ``(n_positions, n_rot)``.  The
    ``base`` argument supports the Qwen3.5 *partial rotary* layout where only the
    first ``n_rot`` of ``head_dim`` channels are rotated.
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    inv_freq = inv_freq[: n_rot // 2]
    t = torch.arange(n_positions, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)   # (T, head_dim)  == (T, 2*(head_dim/2))
    emb = emb[:, : n_rot]                     # keep only the rotated leading part
    return emb.cos(), emb.sin()


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding.  ``q``/``k``: (B, n_heads, T, head_dim)."""
    B, nh, T, hd = q.shape
    n_rot = cos.shape[-1]

    def _apply(x: torch.Tensor, n_heads: int) -> torch.Tensor:
        if n_rot == hd:
            return x * cos + _rotate_half(x) * sin
        # partial rotary: rotate the leading n_rot channels, keep the rest
        rot, pas = x[..., :n_rot], x[..., n_rot:]
        out = rot * cos + _rotate_half(rot) * sin
        return torch.cat((out, pas), dim=-1)

    cos_b = cos[None, None, :, :]  # (1, 1, T, n_rot)
    sin_b = sin[None, None, :, :]
    return _apply(q, nh), _apply(k, nh)


# --------------------------------------------------------------------------- #
# MoELayer — the shared MoE block (routed + optional shared expert)           #
# --------------------------------------------------------------------------- #
class MoELayer(nn.Module):
    """Sparse MoE FFN backed by an :class:`ExpertStore` (+ :class:`Cache`).

    Parameters
    ----------
    gate_w : (E, H)
        Router projection (``mlp.gate.weight`` / ``ffn_gate_inp``).
    store : ExpertStore
        Provider of the routed expert banks (``store.expert_banks``).
    shared : dict {gate,up,down}, optional
        The dense *shared* expert (``ffn_*_shexp`` / ``mlp.shared_expert``).
        Added unconditionally when present (Qwen3.5 style).
    cache : Cache, optional
        If given, routed experts are fetched through the cache (the simulated
        VRAM window); otherwise they are loaded directly from the store.
    """

    def __init__(
        self,
        *,
        layer: int,
        gate_w: torch.Tensor,
        store: ExpertStore,
        shared: Optional[Dict[str, torch.Tensor]] = None,
        cache: Optional[Cache] = None,
        top_k: int = 2,
        n_experts: int = 0,
        norm_topk_prob: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.layer = int(layer)
        self.gate_w = gate_w.to(dtype).to(device)
        self.store = store
        self.shared = {k: v.to(dtype).to(device) for k, v in (shared or {}).items()}
        self.cache = cache
        self.top_k = int(top_k)
        self.n_experts = int(n_experts) or int(store.meta.n_experts)
        self.norm_topk_prob = bool(norm_topk_prob)
        self._device = device
        self._dtype = dtype

    # -- routing (matches HF Qwen3MoeTopKRouter) ---------------------------
    def route(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``h``: (T, H) -> (weights (T, top_k), idx (T, top_k))."""
        logits = F.linear(h, self.gate_w)                    # (T, E)
        probs = F.softmax(logits, dtype=torch.float32, dim=-1)
        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_val = top_val / top_val.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return top_val.to(self._dtype), top_idx

    # -- expert fetch (cache-aware) ----------------------------------------
    def _banks_for(self, eids: Iterable[int]) -> ExpertBanks:
        eids = list(dict.fromkeys(int(e) for e in eids))  # unique, stable order
        if self.cache is not None:
            return self.cache.fetch_many(self.layer, eids, dtype=self._dtype)
        return self.store.expert_banks(self.layer, eids, dtype=self._dtype)

    # -- forward ------------------------------------------------------------
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``h``: (..., H) -> (out (..., H), top_w (T, top_k), top_ids (T, top_k)).

        ``top_w`` / ``top_ids`` are the router's normalized weights and expert
        indices, returned for the activation log / predictor.
        """
        leading = h.shape[:-1]
        H = h.shape[-1]
        h2 = h.reshape(-1, H)
        weights, idx = self.route(h2)         # (T, top_k) each
        out = torch.zeros_like(h2)
        eids = [int(e) for e in idx.reshape(-1).tolist()]
        banks = self._banks_for(eids)

        for e in banks:
            b = banks[e]
            mask = (idx == e)                 # (T, top_k)
            tok = mask.any(dim=1).nonzero(as_tuple=True)[0]
            if tok.numel() == 0:
                continue
            # per-token routing weight contributed by expert e
            wsum = (weights * mask.to(weights.dtype)).sum(dim=-1)  # (T,)
            x = h2[tok]                       # (nt, H)
            act = F.silu(F.linear(x, b["gate"], None))
            act = act * F.linear(x, b["up"], None)
            y = F.linear(act, b["down"], None)           # (nt, H)
            y = y * wsum[tok, None]
            out.index_add_(0, tok, y)

        if self.shared:
            s = self.shared
            act = F.silu(F.linear(h2, s["gate"], None)) * F.linear(h2, s["up"], None)
            out = out + F.linear(act, s["down"], None)

        return out.reshape(leading + (H,)), weights, idx


# --------------------------------------------------------------------------- #
# full GQA attention (tiny ``qwen3``) — no q/k norm, tied or not embeddings   #
# --------------------------------------------------------------------------- #
class Qwen3Attention(nn.Module):
    def __init__(self, *, layer, W, head_dim, n_heads, n_kv_heads, eps, device, dtype):
        super().__init__()
        self.layer = layer
        self.q = W[f"model.layers.{layer}.self_attn.q_proj.weight"].to(dtype).to(device)
        self.k = W[f"model.layers.{layer}.self_attn.k_proj.weight"].to(dtype).to(device)
        self.v = W[f"model.layers.{layer}.self_attn.v_proj.weight"].to(dtype).to(device)
        self.o = W[f"model.layers.{layer}.self_attn.o_proj.weight"].to(dtype).to(device)
        # Qwen3 QK-Norm (per-head RMSNorm on head_dim)
        qn_key = f"model.layers.{layer}.self_attn.q_norm.weight"
        kn_key = f"model.layers.{layer}.self_attn.k_norm.weight"
        self.q_norm = W[qn_key].to(dtype).to(device) if qn_key in W else None
        self.k_norm = W[kn_key].to(dtype).to(device) if kn_key in W else None
        self.head_dim = int(head_dim)
        self.n_heads = int(n_heads)
        self.n_kv = int(n_kv_heads)
        self.group = self.n_heads // self.n_kv
        self.eps = float(eps)
        self.scaling = self.head_dim ** -0.5
        self.device = device
        self.dtype = dtype

    def forward(self, h, cos, sin, kv_cache=None, last_pos: int = -1):
        B, T, _ = h.shape
        q = F.linear(h, self.q).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = F.linear(h, self.k).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = F.linear(h, self.v).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        # Qwen3 QK-Norm: RMSNorm over head_dim, before RoPE
        if self.q_norm is not None:
            q = rms_norm(q, self.q_norm, self.eps)
        if self.k_norm is not None:
            k = rms_norm(k, self.k_norm, self.eps)
        q, k = apply_rope(q, k, cos, sin)
        if kv_cache is not None:
            k, v = kv_cache.append(self.layer, k, v)
        if self.group > 1:
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        Tq, Tk = q.size(2), k.size(2)
        # causal mask: token at query index i (absolute pos last_pos - Tq + 1 + i)
        # may attend to keys with absolute pos <= that position.
        if Tq > 1 or Tk > Tq:
            diag = (last_pos + 1) - Tq
            mask = torch.triu(
                torch.full((Tq, Tk), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=diag + 1,
            )
            attn = attn + mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return F.linear(out, self.o)


class Qwen3MoEModel(nn.Module):
    """Tiny ``qwen3`` MoE: all-full GQA + sparse MoE, cache-aware experts."""

    def __init__(
        self,
        store: ExpertStore,
        *,
        cache: Optional[Cache] = None,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        rope_theta: float = 10000.0,
        rms_eps: float = 1e-6,
        max_positions: int = 8192,
    ) -> None:
        super().__init__()
        device = device or torch.device("cpu")
        m = store.meta
        self.meta = m
        self.device = device
        self.dtype = dtype
        self.norm_topk_prob = bool(m.extra.get("norm_topk_prob", True)) if m.extra else True
        W = store.load_dense(dtype=dtype)
        self.embed = W["model.embed_tokens.weight"].to(dtype).to(device)
        self.lm_head = W["lm_head.weight"].to(dtype).to(device) if "lm_head.weight" in W else self.embed
        self.final_norm = W["model.norm.weight"].to(dtype).to(device)
        self.n_heads = int(m.n_heads)
        self.n_kv = int(m.n_kv_heads)
        self.head_dim = int(m.head_dim)
        cos, sin = _build_rope_cache(max_positions, self.head_dim, self.head_dim,
                                     rope_theta, device=device, dtype=dtype)
        self.cos, self.sin = cos, sin

        self.attns: List[Qwen3Attention] = []
        self.input_ln: List[torch.Tensor] = []
        self.post_ln: List[torch.Tensor] = []
        self.moe: List[MoELayer] = []
        for L in range(m.n_layers):
            self.attns.append(Qwen3Attention(
                layer=L, W=W, head_dim=self.head_dim, n_heads=self.n_heads,
                n_kv_heads=self.n_kv, eps=rms_eps, device=device, dtype=dtype))
            self.input_ln.append(W[f"model.layers.{L}.input_layernorm.weight"].to(dtype).to(device))
            self.post_ln.append(W[f"model.layers.{L}.post_attention_layernorm.weight"].to(dtype).to(device))
            self.moe.append(MoELayer(
                layer=L, gate_w=store.gate(L, dtype=dtype), store=store,
                shared=(store.shared_experts(L, dtype=dtype) or None),
                cache=cache, top_k=int(m.top_k), n_experts=int(m.n_experts),
                norm_topk_prob=self.norm_topk_prob, device=device, dtype=dtype))
        # KV cache (simple list-of-arrays)
        self.kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * m.n_layers

    def reset(self) -> None:
        self.kv_cache = [None] * self.meta.n_layers

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """KV cache append (called by :class:`Qwen3Attention`)."""
        old = self.kv_cache[layer]
        if old is None:
            self.kv_cache[layer] = (k.contiguous(), v.contiguous())
        else:
            self.kv_cache[layer] = (
                torch.cat([old[0], k], dim=2).contiguous(),
                torch.cat([old[1], v], dim=2).contiguous(),
            )
        return self.kv_cache[layer]

    def forward(self, input_ids: torch.Tensor,
                positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, T = input_ids.shape
        h = F.embedding(input_ids, self.embed)
        if positions is None:
            positions = torch.arange(T, device=input_ids.device)
        cos = self.cos[positions].to(h.dtype).to(h.device)
        sin = self.sin[positions].to(h.dtype).to(h.device)
        last_pos = int(positions.max().item())
        routed = []
        for L in range(self.meta.n_layers):
            resid = h
            h = rms_norm(h, self.input_ln[L], 1e-6)
            h = self.attns[L](h, cos, sin, kv_cache=self, last_pos=last_pos)
            h = resid + h
            resid = h
            h = rms_norm(h, self.post_ln[L], 1e-6)
            h2, top_w, top_ids = self.moe[L](h)
            routed.append(top_ids)
            # 激活日志：LRU 有 record_activation，其它 cache 策略可能没有
            rec = getattr(self.moe[L].cache, "record_activation", None)
            if rec is not None and top_ids.dim() == 2:
                rec(L, top_ids[0].tolist(), top_w[0].tolist())
            h = resid + h2
        h = rms_norm(h, self.final_norm, 1e-6)
        logits = F.linear(h, self.lm_head)
        return logits, routed

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 do_sample: bool = True, temperature: float = 1.0,
                 top_p: float = 0.9, seed: int = 0) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        if seed:
            torch.manual_seed(seed)
        self.reset()
        ids = input_ids.to(self.device)
        positions = torch.arange(ids.size(1), device=self.device)
        routed_all: List[List[torch.Tensor]] = [[] for _ in range(self.meta.n_layers)]
        logits, routed = self.forward(ids, positions)
        for L in range(self.meta.n_layers):
            routed_all[L].append(routed[L])
        next_logits = logits[:, -1, :]
        for step in range(max_new_tokens):
            if do_sample:
                nxt = _sample(next_logits, temperature, top_p)
            else:
                nxt = next_logits.argmax(-1, keepdim=True)
            new_tok = nxt.to(self.device)
            ids = torch.cat([ids, new_tok], dim=1)
            pos = torch.tensor([ids.size(1) - 1], device=self.device)
            logits, routed = self.forward(new_tok, pos)
            for L in range(self.meta.n_layers):
                routed_all[L].append(routed[L])
            if do_sample and new_tok.item() == int(getattr(self, "eos_id", -1)):
                break
            next_logits = logits[:, -1, :]
        return ids, routed_all


# --------------------------------------------------------------------------- #
# sampling helpers                                                            #
# --------------------------------------------------------------------------- #
def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    x = logits / max(temperature, 1e-5)
    if top_p and top_p < 1.0:
        v, s = torch.sort(x, dim=-1, descending=True)
        c = torch.cumsum(F.softmax(v, dim=-1), dim=-1)
        c = c - torch.roll(c, 1, dims=-1)
        c[..., 0] = 1.0
        keep = c > top_p
        x = x.masked_fill(keep, float("-inf"))
    p = F.softmax(x, dim=-1)
    return torch.multinomial(p, num_samples=1)
