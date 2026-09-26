"""safetensors loader (HF naming) with expert separation + optional int4.

Supports the standard Qwen3-MoE checkpoint layout used by ``tiny-qwen3-moe``::

    model.embed_tokens.weight
    model.layers.<L>.{input_layernorm,post_attention_layernorm}.weight
    model.layers.<L>.self_attn.{q,k,v,o}_proj.weight
    model.layers.<L>.mlp.gate.weight
    model.layers.<L>.mlp.experts.<E>.{gate,up,down}_proj.weight
    model.norm.weight

Routed experts (``.experts.<E>.*``) are split into per-expert banks; everything
else is returned dense. If the file was written with the int4 payload layout
(see :mod:`litemoe.quantization.safetensors_int4`), expert banks are unpacked to
fp32 transparently.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch

from litemoe.model.loader import ModelMeta

_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")


class SafetensorsLoader:
    """Load an HF-named ``.safetensors`` checkpoint, separating experts."""

    def __init__(self, path: str, config: Optional[dict] = None) -> None:
        from safetensors.torch import load_file

        self.path = path
        _raw = load_file(path)  # torch tensors (bf16 supported)

        def _to_numpy(t: torch.Tensor):
            # numpy has no bfloat16; downcast to fp16 (matches config dtype)
            if t.dtype == torch.bfloat16:
                t = t.half()
            return t.numpy()

        self._np = {k: _to_numpy(v) for k, v in _raw.items()}
        self._cfg = dict(config or {})
        self.meta = self._infer_meta()

    # ------------------------------------------------------------------ meta
    def _infer_meta(self) -> ModelMeta:
        c = self._cfg
        keys = list(self._np.keys())
        n_layers = int(c.get("num_hidden_layers", 0))
        n_experts = int(c.get("num_experts", 0))
        # fall back to scanning tensor names
        if not n_layers:
            ls = {int(m.group(1)) for k in keys if (m := re.search(r"layers\.(\d+)\.", k))}
            n_layers = (max(ls) + 1) if ls else 0
        if not n_experts:
            es = {int(m.group(1)) for k in keys if (m := _EXPERT_RE.search(k))}
            n_experts = (max(es) + 1) if es else 0
        hidden = int(c.get("hidden_size", 0))
        n_heads = int(c.get("num_attention_heads", 0))
        n_kv_heads = int(c.get("num_key_value_heads", 0))
        head_dim = int(c.get("head_dim", 0)) or (hidden // n_heads if n_heads else 0)
        # linear-attention / SSM geometry (qwen3.5-moe / qwen3-next); 0 for dense GQA
        ssm_state_size = int(c.get("linear_ssm_state_size", 0) or
                             c.get("linear_value_head_dim", 0) or 0)  # n (recurrent width)
        ssm_conv_kernel = int(c.get("linear_conv_kernel_dim", 0) or 0)
        ssm_time_step_rank = int(c.get("linear_time_step_rank", 0) or 0)
        ssm_inner_size = int(c.get("linear_inner_size", 0) or 0)
        ssm_group_count = 0  # not in config; derived from heads in model
        head_k_dim = int(c.get("linear_key_head_dim", 0) or 0)
        head_v_dim = int(c.get("linear_value_head_dim", 0) or 0)
        # full_attention_interval drives linear/full block split
        full_interval = int(c.get("full_attention_interval", 0) or 0)
        # layer_types: honor an explicit list if provided, else derive
        lt = c.get("layer_types")
        if isinstance(lt, list) and len(lt) == n_layers:
            layer_types = [str(x) for x in lt]
        elif full_interval:
            layer_types = ["full" if (i + 1) % full_interval == 0 else "linear"
                           for i in range(n_layers)]
        else:
            layer_types = ["full"] * n_layers  # tiny qwen3-moe: all full GQA
        return ModelMeta(
            backend="safetensors",
            arch=c.get("model_type", "qwen3_moe"),
            hidden_size=hidden,
            n_layers=n_layers,
            n_experts=n_experts,
            top_k=int(c.get("num_experts_per_tok", 0)),
            expert_inter=int(c.get("moe_intermediate_size", 0)),
            shared_inter=int(c.get("moe_shared_intermediate_size", 0) or 0),
            has_shared_expert=bool(c.get("mlp_only_layers", None) or
                                   c.get("moe_shared_intermediate_size", 0)),
            layer_types=layer_types,
            vocab_size=int(c.get("vocab_size", 0)),
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ssm_state_size=ssm_state_size,
            ssm_group_count=ssm_group_count,
            ssm_conv_kernel=ssm_conv_kernel,
            ssm_time_step_rank=ssm_time_step_rank,
            ssm_inner_size=ssm_inner_size,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            extra={k: c[k] for k in c if k not in ("num_hidden_layers", "num_experts")},
        )

    # ----------------------------------------------------------------- access
    @staticmethod
    def _as_torch(a) -> torch.Tensor:
        return torch.from_numpy(a)

    def load_dense(self, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
        """All non-expert tensors as torch in ``dtype``."""
        out: Dict[str, torch.Tensor] = {}
        for k, a in self._np.items():
            if _EXPERT_RE.search(k):
                continue
            out[k] = self._as_torch(a).to(dtype)
        return out

    def load_expert_banks(
        self, layer: int, experts: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float16,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Per-expert ``{'gate','up','down'}`` banks for a layer."""
        eids = experts if experts is not None else list(range(self.meta.n_experts))
        banks: Dict[int, Dict[str, torch.Tensor]] = {}
        for e in eids:
            row: Dict[str, torch.Tensor] = {}
            for kind in ("gate", "up", "down"):
                k = f"model.layers.{layer}.mlp.experts.{e}.{kind}_proj.weight"
                if k in self._np:
                    row[kind] = self._as_torch(self._np[k]).to(dtype)
            banks[e] = row
        return banks

    def gate(self, layer: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return self._as_torch(self._np[f"model.layers.{layer}.mlp.gate.weight"]).to(dtype)
