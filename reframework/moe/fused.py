"""MoE routing + grouped expert compute (pure torch, Pascal-safe).

FreeToken routes through triton/sgl kernels (``fused_topk_softmax`` + the
fused_moe grouped-GEMM). None of those exist for sm_61, so Re does the same
math with plain torch:

  * ``torch_fused_topk`` — softmax -> top-k -> renormalize (identical
    reference to FreeToken's, so routing decisions match kernel-for-kernel)
  * ``fused_experts``    — per-expert loop: gather the tokens routed to each
    expert, run the two GEMMs through ``torch.mm`` (cuBLAS fp32 on Pascal),
    apply the gated activation, scatter-add back.

The per-expert loop looks naive but is the right shape on Pascal: each active
expert's GEMM is a clean [M_e, H] x [H, 2I] cuBLAS call, and cuBLAS fp32 on
GP104 is exactly as fast as any hand-written kernel would be without tensor
cores. The bottleneck is PCIe (moving experts) and HBM (reading weights), not
the loop overhead.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from reframework.layers import gated_act_and_mul
from reframework.utils import init_logger

logger = init_logger(__name__)


def torch_fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax over all experts, keep the top-k, renormalize the kept weights.

    Same convention as FreeToken's ``_torch_fused_topk`` (the reference its
    triton kernel is tested against), so a model loaded in FreeToken and in Re
    routes identically."""
    probs = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.contiguous(), topk_ids.contiguous()


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [E, 2I, H] gate|up rows (Qwen3 style, gate first)
    w2: torch.Tensor,  # [E, H, I]
    topk_weights: torch.Tensor,  # [M, K] fp32
    topk_ids: torch.Tensor,  # [M, K] int64
    activation: str = "silu",
) -> torch.Tensor:
    """Run the routed experts and return the summed output ``[M, H]``.

    ``w1`` rows are ``[gate | up]`` (Qwen3 / Llama-MoE convention; FreeToken's
    gpt-oss interleaving is not supported — no gpt-oss on Pascal anyway).
    """
    M, H = hidden_states.shape
    E, gate_up, _ = w1.shape
    I = gate_up // 2
    K = topk_ids.shape[1]
    dev = hidden_states.device

    out = torch.zeros((M, H), dtype=hidden_states.dtype, device=dev)

    # group token ids by expert (host-side; K and M are tiny at decode)
    flat_expert = topk_ids.reshape(-1)  # [M*K]
    flat_weight = topk_weights.reshape(-1)
    flat_token = torch.arange(M, device=dev).repeat_interleave(torch.ones(M, dtype=torch.int32, device=dev) * K)

    counts = torch.bincount(flat_expert, minlength=E)
    for e in torch.nonzero(counts, as_tuple=False).flatten().tolist():
        sel = (flat_expert == e)
        tok = flat_token[sel]           # token ids routed to e (dups possible? no: topk is distinct per token)
        w = flat_weight[sel].to(hidden_states.dtype)
        x = hidden_states[tok]          # [m_e, H]
        gu = x.to(w1.dtype) @ w1[e].t()  # [m_e, 2I] (dtype-align: x can be fp32, experts are fp16)
        inter = gated_act_and_mul(activation, gu)  # [m_e, I]
        y = inter @ w2[e].t()           # [m_e, H]
        if w.dim() == 1:
            y = y * w.unsqueeze(-1)
        out.index_add_(0, tok, y)
    return out


def fused_experts_masked(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [E, 2I, H] — rows cover ONLY the kept experts
    w2: torch.Tensor,  # [E, H, I]
    topk_weights: torch.Tensor,  # [M, K] fp32
    topk_ids: torch.Tensor,  # [M, K] int64, already remapped to w1/w2 local rows
    active: torch.Tensor,  # [M, K] bool — which (token, expert) slots THIS device computes
    activation: str = "silu",
) -> torch.Tensor:
    """Like :func:`fused_experts` but only the ``(token, expert)`` slots where
    ``active`` is True are computed; the rest contribute nothing (the other
    device owns them). ``topk_ids`` must already be remapped to the local rows
    of ``w1``/``w2`` (which contain exactly the experts this device is given).

    This is the GPU half of the CPU/GPU split: ``w1``/``w2`` hold only the
    LRU-resident ("hit") experts and ``active`` masks out the routed slots that
    route to a non-resident (CPU-computed) expert, so a token routed to both a
    hit and a miss still gets its full output once the two halves are summed.
    """
    M, H = hidden_states.shape
    E = w1.shape[0]
    K = topk_ids.shape[1]
    dev = hidden_states.device

    out = torch.zeros((M, H), dtype=hidden_states.dtype, device=dev)
    flat_expert = topk_ids.reshape(-1)
    flat_weight = topk_weights.reshape(-1)
    flat_active = active.reshape(-1)
    flat_token = torch.arange(M, device=dev).repeat(K)

    # Count only *active* slots, so an expert with no active token is skipped
    # and we never index a bank row that doesn't exist.
    active_ids = torch.where(flat_active, flat_expert, torch.full_like(flat_expert, -1))
    counts = torch.bincount(active_ids[flat_active], minlength=E)
    for e in torch.nonzero(counts, as_tuple=False).flatten().tolist():
        sel = (flat_expert == e) & flat_active
        if not bool(sel.any()):
            continue
        tok = flat_token[sel]
        w = flat_weight[sel].to(hidden_states.dtype)
        x = hidden_states[tok]
        gu = x.to(w1.dtype) @ w1[e].t()
        inter = gated_act_and_mul(activation, gu)
        y = inter @ w2[e].t()
        if w.dim() == 1:
            y = y * w.unsqueeze(-1)
        out.index_add_(0, tok, y)
    return out


def fused_experts_cpu_int8(
    hidden_states: torch.Tensor,  # [M, H] (any device; copied to CPU fp32)
    w1q: torch.Tensor,  # [E, 2I, H] int8 (sym, per-tensor scale)
    w2q: torch.Tensor,  # [E, H, I] int8
    scales1: torch.Tensor,  # [E] fp32 — w1 dequant scale
    scales2: torch.Tensor,  # [E] fp32 — w2 dequant scale
    topk_weights: torch.Tensor,  # [M, K] fp32
    topk_ids: torch.Tensor,  # [M, K] int64, remapped to w1q/w2q local rows
    active: torch.Tensor,  # [M, K] bool — slots the CPU computes
    activation: str = "silu",
) -> torch.Tensor:
    """CPU-side expert compute for the non-resident (miss) experts, using
    symmetric int8 GEMMs (``torch._int_mm``) so fp32 weights are never touched
    on the CPU — that is what makes the CPU side fast enough to overlap the GPU.

    Activations are quantized per step-block to int8, both GEMMs run as int8
    (``_int_mm`` -> int32) and are dequantized by the product of the activation
    and weight scales. Only the slots where ``active`` is True are computed (the
    same contract as :func:`fused_experts_masked`); the result is a ``[M, H]``
    fp32 tensor on the CPU, to be summed with the GPU half by the caller.
    """
    from reframework.moe.offload_cache import _sym_int8_quantize  # local: no import cycle

    M, H = hidden_states.shape
    E = w1q.shape[0]
    K = topk_ids.shape[1]

    x2 = hidden_states.detach().cpu().float()
    wts = topk_weights.detach().cpu().float()
    ids = topk_ids.detach().cpu().long()
    flat_active = active.detach().cpu().reshape(-1)

    out = torch.zeros((M, H), dtype=torch.float32)
    flat_expert = ids.reshape(-1)
    flat_weight = wts.reshape(-1)
    flat_token = torch.arange(M).repeat(K)

    active_ids = torch.where(flat_active, flat_expert, torch.full_like(flat_expert, -1))
    counts = torch.bincount(active_ids[flat_active], minlength=E)
    for e in torch.nonzero(counts, as_tuple=False).flatten().tolist():
        sel = (flat_expert == e) & flat_active
        if not bool(sel.any()):
            continue
        tok = flat_token[sel]
        w = flat_weight[sel]
        x = x2[tok]  # [m_e, H] fp32
        # GEMM1: gu = x @ w1q[e]^T, as int8 x int8 -> int32, then dequant.
        xq, xs = _sym_int8_quantize(x)
        gu_i32 = torch._int_mm(xq, w1q[e].t().contiguous())
        gu = gu_i32.float() * (xs * float(scales1[e]))
        inter = gated_act_and_mul(activation, gu)  # [m_e, I] fp32
        # GEMM2: y = inter @ w2q[e]^T, int8 x int8 -> int32, then dequant.
        iq, iscale = _sym_int8_quantize(inter)
        y_i32 = torch._int_mm(iq, w2q[e].t().contiguous())
        y = y_i32.float() * (iscale * float(scales2[e]))
        y = y * w.unsqueeze(-1)
        out.index_add_(0, tok, y)
    return out


__all__ = ["torch_fused_topk", "fused_experts", "fused_experts_masked", "fused_experts_cpu_int8"]
