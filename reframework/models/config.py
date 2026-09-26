"""Model config — single source of truth for model geometry.

The HF ``config.json`` of a Llama/Qwen3 (MoE) model carries everything a
constructor needs. ``ModelConfig`` is the Re-side dataclass: the engine builds
it from the checkpoint config and hands it to the models / attention backend.

Two consumer contracts are already in the tree and pin the field names:
  * :class:`reframework.attention.sdpa.SDPAAttentionBackend` reads
    ``num_qo_heads`` / ``num_kv_heads`` / ``head_dim``
    (``attention/sdpa.py:36-38``)
  * :class:`reframework.moe.FusedMoE.from_config` maps ``num_experts`` /
    ``num_experts_per_tok`` / ``hidden_size`` / ``moe_inter`` /
    ``moe_activation`` / ``moe_renormalize`` / ``dtype`` onto the FusedMoE
    constructor (``moe/moe_layer.py``), and
    :class:`reframework.moe.ExpertOffloadCache.from_config` reads ``device`` /
    ``dtype`` plus the env LRU override.

FreeToken's equivalent is ``ModelArgs``; Re keeps the same field names so the
mapping is 1:1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class ModelConfig:
    # --- core geometry ------------------------------------------------------
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float

    # --- optional model knobs ----------------------------------------------
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    # HF rope_scaling dict, as a hashable tuple-of-pairs for the @cache in
    # reframework.layers.rotary.get_rope (None when the model is not long-context)
    rope_scaling: Optional[Tuple[Tuple[str, Any], ...]] = None
    sliding_window: Optional[int] = None

    # --- MoE (absent on dense models -> use_moe False) ----------------------
    use_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    # Qwen3-30B-A3B config.json has no moe_intermediate_size; fall back to
    # intermediate_size in that case (moe_inter property below).
    moe_intermediate_size: int = 0
    moe_activation: str = "silu"
    moe_renormalize: bool = True  # HF: norm_topk_prob

    # --- QK-Norm scope ------------------------------------------------------
    # 0 = per-head (Qwen3 / Llama 4 style): RMSNorm(head_dim), applied AFTER
    #     splitting the projection into heads.
    # >0 = pre-head (OLMoE / Mixtral-era): RMSNorm(qk_norm_size), applied to
    #      the FULL projection output BEFORE the head reshape.
    qk_norm_size: int = 0

    # --- bias flags (PhiMoE: all True; Llama/Qwen3: all False) -------------
    attention_bias: bool = False
    norm_bias: bool = False
    # q_norm/k_norm presence: Qwen3/OLMoE True, PhiMoE False
    has_qk_norm: bool = True
    # 熵感知自适应 top-k：低熵 token 只用少量专家
    entropy_adaptive_k: bool = False
    entropy_k_min: int = 2

    # --- runtime (engine-filled, not in the checkpoint) ----------------------
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    arch: str = ""

    # ----------------------------------------------------------------- helpers
    @property
    def num_qo_heads(self) -> int:
        """Query/output head count — the name the attention backend reads."""
        return self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def has_moe(self) -> bool:
        return self.use_moe and self.num_experts > 0

    @property
    def moe_inter(self) -> int:
        """Expert FFN width (moe_intermediate_size or the dense fallback)."""
        return self.moe_intermediate_size or self.intermediate_size

    def expert_bytes(self) -> int:
        """VRAM bytes of one expert (w1 [2I,H] + w2 [H,I]) in the storage dtype.

        Used by the engine to auto-size the offload LRU from the free-VRAM
        budget (``env.get_moe_cache_ratio``).
        """
        h, i = self.hidden_size, self.moe_inter
        return (2 * i * h + h * i) * self.dtype.itemsize

    def size_moe_lru(self, budget_bytes: int) -> int:
        """LRU capacity for one layer's ExpertOffloadCache from a byte budget."""
        return max(1, budget_bytes // max(1, self.expert_bytes()))

    # ------------------------------------------------------------------- build
    @classmethod
    def from_hf(
        cls,
        cfg: Dict[str, Any],
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> "ModelConfig":
        """Build from a raw HF ``config.json`` dict."""
        hidden = int(cfg["hidden_size"])
        heads = int(cfg["num_attention_heads"])
        kv = int(cfg.get("num_key_value_heads", heads))
        arch_lower = str(cfg.get("architectures", [""])[0]).lower()
        is_phimoe = arch_lower.startswith("phimoe")
        # PhiMoE: head_dim is 128 regardless of hidden//heads (hidden=4096,
        # heads=16 -> hidden//heads=256 which is wrong; q_proj is 2048 = 16*128)
        hd_cfg = cfg.get("head_dim")
        if hd_cfg:
            head_dim = int(hd_cfg)
        elif is_phimoe:
            head_dim = 128
        else:
            head_dim = hidden // heads
        rs = cfg.get("rope_scaling")
        num_experts = int(cfg.get("num_experts") or cfg.get("num_local_experts") or 0)
        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=hidden,
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_attention_heads=heads,
            num_key_value_heads=kv,
            head_dim=head_dim,
            max_position_embeddings=int(cfg.get("max_position_embeddings", 32768)),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            rope_scaling=tuple(sorted(rs.items())) if rs else None,
            sliding_window=cfg.get("sliding_window"),
            use_moe=num_experts > 0,
            num_experts=num_experts,
            num_experts_per_tok=int(cfg.get("num_experts_per_tok", 0)),
            moe_intermediate_size=int(cfg.get("moe_intermediate_size") or 0),
            moe_activation=str(cfg.get("moe_activation", "silu")),
            moe_renormalize=bool(cfg.get("norm_topk_prob", True)),
            qk_norm_size=(
                int(cfg["hidden_size"])
                if str(cfg.get("architectures", [""])[0]).lower().startswith("olmoe")
                else 0
            ),
            attention_bias=is_phimoe or bool(cfg.get("attention_bias", False)),
            norm_bias=is_phimoe or bool(cfg.get("norm_bias", False)),
            has_qk_norm=not is_phimoe,
            entropy_adaptive_k=bool(cfg.get("entropy_adaptive_k", False)),
            entropy_k_min=int(cfg.get("entropy_k_min", 2)),
            dtype=dtype or torch.float32,
            device=device or "cpu",
            arch=str(cfg.get("architectures", [""])[0]).lower(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Round-trip helper for logging / the server config endpoint."""
        rs = dict(self.rope_scaling) if self.rope_scaling else None
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": rs,
            "use_moe": self.use_moe,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_inter,
            "arch": self.arch,
        }


__all__ = ["ModelConfig"]
