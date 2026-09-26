"""Rotary position embedding — pure PyTorch (NeoX / half-rotation default).

FreeToken ships a triton/flashinfer inplace rope kernel. Re applies RoPE with
plain tensor ops, which on a memory-bound Pascal card is not the bottleneck.
Supports the standard NeoX layout (the one Qwen/Llama use) plus the partial
rotary variant (rotary_dim < head_size, e.g. Qwen3.5) and llama3-style
interpolation scaling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process=None,
        proportional: bool = False,
        attention_factor: float = 1.0,
        is_neox: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.is_neox = is_neox
        assert rotary_dim % 2 == 0

        if proportional:
            assert 0 < rotary_dim <= head_size
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
            )
            if rotary_dim < head_size:
                inv_freq[rotary_dim // 2 :] = 0.0
        else:
            assert 0 < rotary_dim <= head_size
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
            )
        if post_process is not None:
            inv_freq = post_process(inv_freq)

        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * attention_factor
        sin = freqs.sin() * attention_factor
        # buffer (not a parameter): excluded from state_dict
        self.register_buffer("_cos_sin_cache", torch.cat((cos, sin), dim=-1), persistent=False)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # positions: [num_tokens]  (flat across the batch)
        # query/key: [num_tokens, num_heads, head_size]
        cos_sin = self._cos_sin_cache.to(query.device)[positions]  # [n, 2*rotary_dim/2]
        cos, sin = cos_sin.chunk(2, dim=-1)  # each [n, rotary_dim/2]

        def apply(x: torch.Tensor) -> torch.Tensor:
            # split the rotary portion and the untouched tail
            rd = self.rotary_dim
            x_rot = x[..., :rd]
            x_pass = x[..., rd:]
            x1, x2 = x_rot.chunk(2, dim=-1)  # neoX: first half / second half
            c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)  # [n,1,rd/2]
            # neoX rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
            new1 = x1 * c - x2 * s
            new2 = x2 * c + x1 * s
            rot = torch.cat((new1, new2), dim=-1)
            if rd < x.shape[-1]:
                return torch.cat((rot, x_pass), dim=-1)
            return rot

        return apply(query), apply(key)


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
    is_neox: bool = True,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base, is_neox=is_neox)
    rtype = rope_scaling.get("rope_type")
    if rtype == "default":
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base, is_neox=is_neox)
    if rtype == "proportional":
        return RotaryEmbedding(
            head_dim, rotary_dim, max_position, base, proportional=True, is_neox=is_neox
        )
    if rtype == "llama3":
        factor = rope_scaling["factor"]
        low_freq_factor = rope_scaling["low_freq_factor"]
        high_freq_factor = rope_scaling["high_freq_factor"]
        orig_max_pos = rope_scaling["original_max_position_embeddings"]

        def post_process(inv_freq):
            wave_len = 2 * math.pi / inv_freq
            if low_freq_factor == high_freq_factor:
                return torch.where(
                    wave_len < orig_max_pos / high_freq_factor, inv_freq, inv_freq / factor
                )
            delta = high_freq_factor - low_freq_factor
            smooth = (orig_max_pos / wave_len - low_freq_factor) / delta
            smooth = torch.clamp(smooth, 0, 1)
            coeff = (1 - smooth) / factor + smooth
            return coeff * inv_freq

        return RotaryEmbedding(
            head_dim, rotary_dim, max_position, base, post_process, is_neox=is_neox
        )
    if rtype == "yarn":
        factor = rope_scaling["factor"]
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        orig_max_pos = rope_scaling["original_max_position_embeddings"]

        def get_mscale(scale, mscale=1.0):
            return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0

        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is None:
            mscale = rope_scaling.get("mscale")
            mscale_all_dim = rope_scaling.get("mscale_all_dim")
            if mscale and mscale_all_dim:
                attention_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            else:
                attention_factor = get_mscale(factor)

        def _find_correction_dim(num_rotations):
            return (
                rotary_dim
                * math.log(orig_max_pos / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(_find_correction_dim(beta_fast)), 0)
        high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim - 1)
        if low == high:
            high += 0.001

        def post_process(inv_freq):
            ramp = torch.clamp(
                (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
            )
            return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

        return RotaryEmbedding(
            head_dim,
            rotary_dim,
            max_position,
            base,
            post_process,
            attention_factor=float(attention_factor),
            is_neox=is_neox,
        )
    raise ValueError(f"Unsupported rope_type {rtype!r}")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device) -> None:
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


from functools import cache  # noqa: E402


@cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
    is_neox: bool = True,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    # rope caches are real tensors; build them on a concrete device. If we were
    # asked on meta (model init) fall back to the configured rope device.
    t = torch.tensor([])
    if t.device == torch.device("meta") and _ROPE_DEVICE is not None:
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map, is_neox)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map, is_neox)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
