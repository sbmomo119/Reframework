"""QuantScheme interface — the abstraction the loader/cache program against.

A ``QuantScheme`` knows how to (a) pack a dense fp weight into a compact
payload, (b) unpack it back to fp, and (c) report its bytes-per-element so the
engine can size caches and predict transfer cost. Two built-ins:

* ``F16``  — passthrough (no compression). Weights stored/kept as fp16.
* ``Int4`` — symmetric per-group 4-bit integer codes (2 packed per int8 byte)
             plus one fp16 scale per group. ``~4.5 bits/weight`` including scales.

The GGUF path does *not* round-trip through this interface: GGUF tensors are
dequantized once on load (see :mod:`litemoe.quantization.gguf_int4`) and the
cache then holds the resulting fp16/fp32 banks. ``Int4`` is used for the
safetensors path and for shrinking the host pool.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import numpy as np


class QuantScheme(ABC):
    """Pack/unpack a dense weight tensor between fp and a compact form."""

    name: str = "base"

    @abstractmethod
    def pack(self, w: np.ndarray) -> Dict[str, np.ndarray]:
        """``w`` fp32/fp16 [*, in] -> payload dict (numpy arrays)."""

    @abstractmethod
    def unpack(self, payload: Dict[str, np.ndarray], shape: tuple) -> np.ndarray:
        """payload + logical shape -> fp32 ndarray of ``shape``."""

    @property
    @abstractmethod
    def bytes_per_element(self) -> float:
        """Approximate on-disk/on-host bytes per weight element."""


# --------------------------------------------------------------------------- #
# F16 passthrough                                                            #
# --------------------------------------------------------------------------- #
class F16(QuantScheme):
    """No compression: store fp16, unpack to fp32 for compute."""

    name = "f16"

    def pack(self, w: np.ndarray) -> Dict[str, np.ndarray]:
        return {"data": np.ascontiguousarray(w, dtype=np.float16)}

    def unpack(self, payload: Dict[str, np.ndarray], shape: tuple) -> np.ndarray:
        return payload["data"].astype(np.float32).reshape(shape)

    @property
    def bytes_per_element(self) -> float:
        return 2.0


# --------------------------------------------------------------------------- #
# Int4 (per-group symmetric)                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Int4Payload:
    """Compact int4 storage: 4-bit codes packed 2-per-byte + per-group scales.

    ``codes``  : int8 [ceil(N/2)] where each byte holds two codes (hi, lo).
    ``scales`` : fp16 [G] one symmetric scale per ``group`` elements.
    """

    codes: np.ndarray
    scales: np.ndarray
    group: int
    shape: tuple  # original logical shape (row-major)


def int4_pack(w: np.ndarray, group: int = 128) -> Int4Payload:
    """Symmetric per-group int4 quantize (host, fp32 math).

    Groups are taken over the *flattened* tensor in row-major order, so a
    [out, in] weight groups along the last axis first — the usual scheme for
    weight-only quantization.
    """
    flat = np.ascontiguousarray(w, dtype=np.float32).ravel()
    N = flat.size
    n_groups = (N + group - 1) // group
    codes = np.zeros(N, dtype=np.uint8)
    scales = np.zeros(n_groups, dtype=np.float32)
    for g in range(n_groups):
        s = flat[g * group : (g + 1) * group]
        amax = float(np.abs(s).max()) if s.size else 0.0
        if amax <= 0:
            scales[g] = 1.0
            continue
        scale = amax / 7.0  # 4-bit symmetric range [-7, 7] (leave 127/15 for hi/lo)
        scales[g] = scale
        q = np.clip(np.round(s / scale), -7, 7).astype(np.uint8) + 8
        codes[g * group : g * group + s.size] = q
    # pack 2 codes per byte (hi = code[2i], lo = code[2i+1]); pad one zero if odd
    pad = (2 - N % 2) % 2
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    packed = (codes[0::2] << 4) | codes[1::2]
    return Int4Payload(codes=packed.astype(np.int8), scales=scales.astype(np.float16),
                       group=group, shape=tuple(w.shape))


def int4_unpack(payload: Int4Payload) -> np.ndarray:
    """Unpack an :class:`Int4Payload` back to fp32 of ``payload.shape``."""
    packed = payload.codes.astype(np.uint8)
    N = payload.codes.size * 2
    hi = (packed >> 4).astype(np.uint8)
    lo = (packed & 0x0F).astype(np.uint8)
    codes = np.empty(2 * packed.size, dtype=np.uint8)
    codes[0::2] = hi
    codes[1::2] = lo
    q = codes[: payload.scales.size * payload.group].astype(np.float32) - 8
    scales = payload.scales.astype(np.float32)
    g = payload.group
    n = q.size
    # map each element to its group scale
    grp_idx = np.arange(n) // g
    out = q * scales[grp_idx]
    return out.reshape(payload.shape)


class Int4(QuantScheme):
    """Per-group 4-bit weight-only scheme (see :func:`int4_pack`)."""

    def __init__(self, group: int = 128) -> None:
        self.group = int(group)
        self.name = f"int4_g{self.group}"

    def pack(self, w: np.ndarray) -> Dict[str, np.ndarray]:
        p = int4_pack(w, self.group)
        return {"codes": p.codes, "scales": p.scales, "group": np.array(p.group)}

    def unpack(self, payload: Dict[str, np.ndarray], shape: tuple) -> np.ndarray:
        p = Int4Payload(codes=payload["codes"], scales=payload["scales"],
                        group=int(payload["group"]), shape=shape)
        return int4_unpack(p)

    @property
    def bytes_per_element(self) -> float:
        # 0.5 bytes/code + 2 bytes scale per `group` elements
        return 0.5 + 2.0 / self.group
