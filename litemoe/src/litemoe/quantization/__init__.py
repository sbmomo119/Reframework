"""Quantization: GGUF int4 dequantization + safetensors int4 packing.

- ``gguf_int4``       — dequantize any GGML quant type (Q2_K..Q8_0, IQ2_XXS..IQ4_XS)
                        to fp32 via the ``gguf`` package, reshaped to torch layout.
- ``safetensors_int4`` — pack fp weights into an int4 (4-bit) payload with per-group
                        scales for on-disk / on-host storage, and unpack back.
- ``scheme``          — the ``QuantScheme`` interface (``F16``, ``Int4``) the rest of
                        the runtime programs against.
"""

from litemoe.quantization.scheme import QuantScheme, F16, Int4
from litemoe.quantization.gguf_int4 import (
    dequantize_tensor,
    load_gguf_tensor,
    GGUF_DEQUANTIZABLE,
)
from litemoe.quantization.safetensors_int4 import (
    int4_pack,
    int4_unpack,
    Int4Payload,
)

__all__ = [
    "QuantScheme",
    "F16",
    "Int4",
    "dequantize_tensor",
    "load_gguf_tensor",
    "GGUF_DEQUANTIZABLE",
    "int4_pack",
    "int4_unpack",
    "Int4Payload",
]
