"""SDPA attention backend — the only backend that works on sm_61.

FreeToken registers trtllm / flashinfer / flash-attention / triton; every one of
those needs sm_70+ (flash-attn) or sm_80+ (cp.async, tensor cores). On Pascal,
attention is computed with ``torch.nn.functional.scaled_dot_product_attention``
in fp32, per request, with an explicit causal mask (SDPA's ``is_causal`` only
covers the square case; our prefill rows are ragged).

Why per-request instead of a fused paged kernel:
  * a fused paged SDPA kernel for sm_61 does not exist in the PyTorch
    distribution — writing one is out of scope for v0.1
  * decode (qlen=1) is mask-free and the gather is the only extra cost
  * prefill on a 6-8 GB card is short (long prompts are exactly what the MoE
    offload + small max-model-len target), so the loop is not the bottleneck
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from reframework.attention.base import AttentionSpec, BaseAttnBackend
from reframework.core import Batch, get_global_ctx
from reframework.utils import init_logger

logger = init_logger(__name__)


class SDPAAttentionBackend(BaseAttnBackend):
    def __init__(self, config) -> None:
        self.config = config
        ctx = get_global_ctx()
        self.pool = ctx.kv_cache
        self.cache_manager = ctx.cache_manager
        self.device = self.pool.device
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

    # ------------------------------------------------------------------ meta
    def prepare_metadata(self, batch: Batch) -> None:
        from reframework.core import AttnMetadata

        device = self.device
        reqs = batch.reqs
        query_lens = [r.extend_len for r in reqs]
        seq_lens = [r.num_computed_tokens + r.extend_len for r in reqs]
        num_tokens = batch.num_tokens

        # flat KV slot ids: concat of each request's page-table row (live prefix)
        kv_parts = []
        for r in reqs:
            row = self.cache_manager.page_table.table[r.table_idx]
            kv_parts.append(row[: r.num_computed_tokens + r.extend_len])
        kv_indices = torch.cat(kv_parts) if kv_parts else torch.empty(0, dtype=torch.int64, device=device)

        # which request owns each query token (for get_last_indices bookkeeping)
        q_to_req = torch.repeat_interleave(
            torch.arange(len(reqs), dtype=torch.int32),
            torch.tensor(query_lens, dtype=torch.int64),
        ).to(device)

        positions = torch.cat([
            torch.arange(r.num_computed_tokens, r.num_computed_tokens + r.extend_len)
            for r in reqs
        ]).to(device)

        meta = batch.attn_metadata
        assert meta is not None, "engine must construct AttnMetadata before prepare_metadata"
        meta.query_lens = torch.tensor(query_lens, dtype=torch.int32, device=device)
        meta.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        meta.positions = positions
        meta.kv_indices = kv_indices
        meta.q_to_req = q_to_req
        meta.req_pool_indices = torch.tensor([r.table_idx for r in reqs], dtype=torch.int64, device=device)
        meta.page_table = self.cache_manager.page_table.table[: len(reqs)].to(torch.int32)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        meta = batch.attn_metadata
        self.pool.store_kv(k, v, meta.out_cache_loc, layer_id)
        k_g, v_g = self.pool.gather_kv(layer_id, meta.kv_indices)  # [total_kv, kv_h, dim]

        # Pascal: fp16 storage is allowed, fp16 *math* is not — upcast to fp32.
        compute = torch.float32
        qf = q.to(compute)
        k_g = k_g.to(compute)
        v_g = v_g.to(compute)

        spec = attn_spec or AttentionSpec()
        scale = spec.sm_scale if spec.sm_scale is not None else self.head_dim ** -0.5
        window = spec.sliding_window

        # GQA: repeat kv heads up to q heads. (num_qo // num_kv) is 1 for MHA.
        groups = self.num_qo_heads // self.num_kv_heads
        if groups > 1:
            # broadcast 而不是复制：显存省 groups 倍
            k_g = k_g.unsqueeze(2).expand(-1, -1, groups, -1).reshape(k_g.shape[0], -1, k_g.shape[-1])
            v_g = v_g.unsqueeze(2).expand(-1, -1, groups, -1).reshape(v_g.shape[0], -1, v_g.shape[-1])

        out_chunks = []
        kv_offset = 0
        q_offset = 0
        for r_idx, (qlen, klen) in enumerate(zip(meta.query_lens.tolist(), meta.seq_lens.tolist())):
            q_i = qf[q_offset : q_offset + qlen]               # [qlen, h, d]
            k_i = k_g[kv_offset : kv_offset + klen]            # [klen, h, d]
            v_i = v_g[kv_offset : kv_offset + klen]
            q_offset += qlen
            kv_offset += klen
            # SDPA wants (N, H, L, E); the model layout is (L, H, E) — permute +
            # add a size-1 batch dim. The ragged mask stays (qlen, klen) and
            # broadcasts over the batch/head dims.
            q4 = q_i.transpose(0, 1).unsqueeze(0)               # [1, h, qlen, d]
            k4 = k_i.transpose(0, 1).unsqueeze(0)               # [1, h, klen, d]
            v4 = v_i.transpose(0, 1).unsqueeze(0)

            if qlen == 1:
                # decode: every kv position is valid, no mask needed
                o = F.scaled_dot_product_attention(
                    q4, k4, v4, scale=scale, is_causal=False
                )
            else:
                # ragged causal mask: query row i (abs pos = klen-qlen+i) sees j <= that
                i_idx = torch.arange(qlen, device=q.device).unsqueeze(1)
                j_idx = torch.arange(klen, device=q.device).unsqueeze(0)
                base = klen - qlen
                mask = j_idx <= (base + i_idx)
                if window is not None:
                    mask &= j_idx >= (base + i_idx + 1 - window)
                o = F.scaled_dot_product_attention(
                    q4, k4, v4, attn_mask=mask, scale=scale, is_causal=False
                )
            out_chunks.append(o.squeeze(0).transpose(0, 1))    # back to [qlen, h, d]

        return torch.cat(out_chunks).to(q.dtype)


def create_sdpa_backend(config) -> SDPAAttentionBackend:
    return SDPAAttentionBackend(config)


__all__ = ["SDPAAttentionBackend", "create_sdpa_backend"]
