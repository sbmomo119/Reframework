"""Model checkpoint loaders (the ``io/`` layer).

Two backends, one contract. Each loader turns a checkpoint file into a flat
``name -> torch.Tensor`` dict (fp32, torch layout ``[out, in]`` / ``[E, out, in]``)
plus a small ``ModelMeta`` describing the MoE geometry so the model and the
expert cache can be built without re-parsing the file.

* :class:`GGUFLoader`         — dequantizes every tensor (any GGML quant) and
                                separates experts so the cache can hold them
                                individually.
* :class:`SafetensorsLoader`  — loads ``.safetensors`` (dense or int4-packed).

Neither loader runs the model; they only materialize weights + metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from litemoe.quantization.gguf_int4 import load_gguf_tensor


@dataclass
class ModelMeta:
    """MoE geometry inferred from the checkpoint (backend-agnostic)."""

    backend: str = ""                 # 'gguf' | 'safetensors'
    arch: str = ""                    # e.g. 'qwen35moe' | 'qwen3_moe'
    hidden_size: int = 0
    n_layers: int = 0
    n_experts: int = 0                # routed experts per MoE block
    top_k: int = 0
    expert_inter: int = 0             # per-expert intermediate dim (I)
    shared_inter: int = 0             # shared-expert intermediate dim (0 if none)
    has_shared_expert: bool = False
    layer_types: List[str] = field(default_factory=list)  # 'linear' | 'full'
    vocab_size: int = 0
    # --- attention geometry ---
    head_dim: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    # --- linear-attention / SSM (GatedDeltaNet) geometry ---
    ssm_state_size: int = 0           # n (per-head recurrent state width)
    ssm_group_count: int = 0          # g (num_v_heads = g * num_k_heads)
    ssm_conv_kernel: int = 0
    ssm_time_step_rank: int = 0
    ssm_inner_size: int = 0           # V = num_v_heads * head_v_dim
    head_k_dim: int = 0
    head_v_dim: int = 0
    # raw arch-specific field blob (rope_theta, layer_norm eps, ...)
    extra: Dict[str, object] = field(default_factory=dict)

    def is_moe_layer(self, i: int) -> bool:
        return True  # qwen35moe: every block is MoE; tiny qwen3-moe: also every block

    def linear_blocks(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear"]

    def full_blocks(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full"]


# --------------------------------------------------------------------------- #
# GGUF                                                                        #
# --------------------------------------------------------------------------- #
class GGUFLoader:
    """Load + dequantize a GGUF checkpoint, separating routed experts.

    Expert tensors are split out of their ``[E, out, in]`` stacks so the
    :class:`~litemoe.cache.ExpertCache` can hold each expert's ``w1/w2``
    individually. Non-expert (attention/SSM/norm/dense) tensors are kept whole.
    """

    def __init__(self, path: str) -> None:
        import gguf

        self.path = path
        self._gguf = gguf
        self.reader = gguf.GGUFReader(path)
        self.meta = self._read_meta()

    # ------------------------------------------------------------------ meta
    def _read_meta(self) -> ModelMeta:
        import gguf

        r = self.reader
        f = r.fields

        def gf(name):
            e = f.get(name)
            return e.contents() if e is not None else None

        arch = str(gf("general.architecture") or "unknown")

        # Architecture-prefixed fields, e.g. <arch>.block_count. The values are
        # ReaderField objects (NOT gguf.GGUFValue) — call .contents() on each.
        afields: Dict[str, Any] = {}
        for k, v in f.items():
            if not isinstance(k, str) or not k.startswith(arch + "."):
                continue
            try:
                afields[k[len(arch) + 1:]] = v.contents()
            except Exception:
                pass

        def garch(key):
            return afields.get(key)

        hidden = int(garch("embedding_length") or 0)
        n_layers = int(garch("block_count") or 0)
        vocab = int(garch("vocab_size") or 0)
        n_heads = int(garch("attention.head_count") or 0)
        n_kv_heads = int(garch("attention.head_count_kv") or 0)
        head_dim = int(garch("attention.key_length") or 0)
        # expert geometry: prefer the arch fields, fall back to tensor scan
        n_experts = int(garch("expert_count") or 0)
        top_k = int(garch("expert_used_count") or 0)
        expert_inter = int(garch("expert_feed_forward_length") or 0)
        shared_inter = int(garch("expert_shared_feed_forward_length") or 0)
        has_shared = bool(garch("expert_shared_feed_forward_length"))
        # SSM / linear-attention geometry
        ssm_state_size = int(garch("ssm.state_size") or 0)
        ssm_group_count = int(garch("ssm.group_count") or 0)
        ssm_conv_kernel = int(garch("ssm.conv_kernel") or 0)
        ssm_time_step_rank = int(garch("ssm.time_step_rank") or 0)
        ssm_inner_size = int(garch("ssm.inner_size") or 0)
        # full-attention interval (which blocks are full vs linear)
        full_interval = int(garch("full_attention_interval") or 0)

        # ---- tensor-scan fallbacks / cross-checks ----
        for t in r.tensors:
            n = t.name
            shape = [int(x) for x in t.shape]  # innermost-first
            if n_experts == 0 and n.endswith(".ffn_gate_exps.weight"):
                # [E, in, out] -> out is last, in is first, E is first-most
                n_experts = shape[0]
                expert_inter = expert_inter or shape[2]
            if not has_shared and "shexp" in n:
                has_shared = True
            if not head_dim and n.endswith("attn_k") and len(shape) == 2 and n_kv_heads:
                # full-attention k_proj: [in, n_kv*head] -> head = last / n_kv
                head_dim = shape[1] // n_kv_heads

        # ---- classify blocks: 'linear' (has ssm_) vs 'full' ----
        ssm_names = [t.name for t in r.tensors if ".ssm_" in t.name]
        layer_types: List[str] = []
        for i in range(n_layers):
            prefix = f"blk.{i}."
            has_ssm = any(nm.startswith(prefix) for nm in ssm_names)
            if full_interval:
                layer_types.append("full" if (i + 1) % full_interval == 0 else "linear")
            else:
                layer_types.append("linear" if has_ssm else "full")

        return ModelMeta(
            backend="gguf",
            arch=arch,
            hidden_size=hidden,
            n_layers=n_layers,
            n_experts=n_experts,
            top_k=top_k,
            expert_inter=expert_inter,
            shared_inter=shared_inter,
            has_shared_expert=has_shared,
            layer_types=layer_types,
            vocab_size=vocab,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ssm_state_size=ssm_state_size,
            ssm_group_count=ssm_group_count,
            ssm_conv_kernel=ssm_conv_kernel,
            ssm_time_step_rank=ssm_time_step_rank,
            ssm_inner_size=ssm_inner_size,
            head_k_dim=head_dim,
            head_v_dim=head_dim,
            extra={
                "rope_freq_base": garch("rope.freq_base"),
                "rope_dim": garch("rope.dimension_count"),
                "rope_sections": garch("rope.dimension_sections"),
                "context_length": garch("context_length"),
                "expert_used": top_k,
                "norm_eps": garch("attention.layer_norm_rms_epsilon"),
                "full_attention_interval": full_interval,
                "activation": garch("activation"),
            },
        )

    # ----------------------------------------------------------------- load
    def load_dense(self, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
        """All non-expert tensors (attention/SSM/norm/embed) as torch, in ``dtype``."""
        out: Dict[str, torch.Tensor] = {}
        for t in self.reader.tensors:
            n = t.name
            if ".ffn_gate_exps." in n or ".ffn_up_exps." in n or ".ffn_down_exps." in n:
                continue  # routed experts handled separately
            ten, _, _, _ = load_gguf_tensor(self.reader, n)
            out[n] = ten.to(dtype)
        return out

    def load_expert(self, layer: int, expert: int, kind: str) -> torch.Tensor:
        """One routed expert's projection. ``kind`` in {'gate','up','down'}."""
        name = f"blk.{layer}.ffn_{kind}_exps.weight"
        ten, _, _, _ = load_gguf_tensor(self.reader, name)
        # ten is [E, out, in]; index the expert axis
        return ten[expert].contiguous()

    def load_expert_banks(
        self, layer: int, experts: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float16,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Packed banks for a block's experts: ``{'w1':[N,2I,H] or per-expert, ...}``.

        Returns a dict keyed by expert id -> ``{'gate','up','down'}`` (each fp32
        [out,in] moved to ``dtype``). The cache decides how to pack them.
        """
        eids = experts if experts is not None else list(range(self.meta.n_experts))
        banks: Dict[int, Dict[str, torch.Tensor]] = {}
        for e in eids:
            banks[e] = {
                k: self.load_expert(layer, e, k).to(dtype) for k in ("gate", "up", "down")
            }
        return banks

    def tensor_nbytes(self, name: str) -> int:
        """On-disk (raw, still-quantized) byte size of one tensor. 0 if absent.

        Used for accurate transfer-cost accounting (the cache moves the raw
        quantized payload, not the dequantized fp32).
        """
        for t in self.reader.tensors:
            if t.name == name:
                return int(t.n_bytes)
        return 0

    def close(self) -> None:
        try:
            self.reader.close()
        except Exception:
            pass
