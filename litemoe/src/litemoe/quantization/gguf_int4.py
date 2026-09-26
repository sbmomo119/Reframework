"""GGUF int4 (and any GGML quant) dequantization.

The 14 GB ``qwen35moe`` GGUF stores weights as a mix of IQ2_S / Q3_K / Q4_K /
Q5_K / Q6_K (plus a few F16/F32 tensors). The ``gguf`` package ships a
``dequantize(data, qtype) -> flat fp32`` routine for every GGML quant type, so
we never hand-roll the k-quants. This module wraps it:

* :func:`load_gguf_tensor`   — raw bytes + qtype + raw shape -> torch tensor.
* :func:`dequantize_tensor`  — the numpy-level core (testable without torch).

Shape convention (verified against the GGUF reader): a tensor's raw dims are
stored **innermost-first** (ggml order). ``dequantize`` returns a flat array in
that same order, so the torch layout is ``reversed(raw)`` — i.e. ``[out, in]``
for a linear weight, ``[E, out, in]`` for an expert stack.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _import_gguf():
    import gguf  # local import keeps the module importable without the dep

    return gguf


def _to_qtype(gguf, qtype):
    """Normalize a GGML quant type to its ``GGMLQuantizationType`` enum member.

    Accepts an ``int`` (the raw ``tensor_type`` from the reader) or a ``str``
    name (``"Q4_K"``, ``"F16"``, ...). This version of the ``gguf`` package has
    no ``GGMLQuantizationType.from_str`` — look up by name or construct by int.
    """
    if qtype is None:
        raise ValueError("qtype is required")
    if isinstance(qtype, str):
        try:
            return gguf.GGMLQuantizationType[qtype]
        except KeyError:
            raise ValueError(f"unknown GGML qtype name {qtype!r}") from None
    return gguf.GGMLQuantizationType(int(qtype))


def dequantize_tensor(data: np.ndarray, qtype, raw_shape: Sequence[int]) -> np.ndarray:
    """Dequantize raw GGUF bytes to a flat fp32 numpy array (innermost-first order).

    ``data``    : uint8 array of the tensor's raw (quantized) bytes.
    ``qtype``   : an int GGML qtype (preferred, from ``t.tensor_type``) or a str name.
    ``raw_shape``: innermost-first dims, e.g. ``[in, out]`` for a weight.
    Returns a 1-D fp32 array of ``prod(raw_shape)`` elements.
    """
    gguf = _import_gguf()
    qt = _to_qtype(gguf, qtype)
    flat = np.asarray(gguf.dequantize(np.ascontiguousarray(data, dtype=np.uint8), qt),
                      dtype=np.float32)
    expect = int(np.prod(list(raw_shape))) if len(raw_shape) else data.nbytes
    if flat.size != expect:
        raise ValueError(
            f"dequant size {flat.size} != expected {expect} for {getattr(qt, 'name', qtype)} {list(raw_shape)}"
        )
    return flat


def load_gguf_tensor(reader, name: str):
    """Load one GGUF tensor by name as a torch tensor in ``[out, ... , in]`` order.

    Returns ``(tensor, raw_shape, qtype_name, n_bytes)`` where ``tensor`` is
    ``torch.float32`` shaped to the *reversed* raw dims (torch layout). The
    caller (``GGUFLoader``) moves/converts dtype and slices experts from here.
    """
    import torch

    gguf = _import_gguf()
    for t in reader.tensors:
        if t.name == name:
            raw_shape = [int(x) for x in t.shape]
            qtype_name = gguf.GGMLQuantizationType(int(t.tensor_type)).name
            data = np.frombuffer(t.data, dtype=np.uint8).copy()  # copy off the mmap
            flat = dequantize_tensor(data, int(t.tensor_type), raw_shape)
            # torch layout = reverse the innermost-first raw dims
            tensor = torch.from_numpy(flat).reshape(list(reversed(raw_shape)))
            return tensor, raw_shape, qtype_name, int(t.n_bytes)
    raise KeyError(f"tensor {name!r} not found in GGUF (searched all {len(reader.tensors)})")


# Quant types the gguf dequantizer handles (the 14GB model uses all of these).
GGUF_DEQUANTIZABLE = {
    "F32", "F16",
    "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K", "Q4_K",
    "Q5_K", "Q6_K", "Q8_K",
    "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_XS", "IQ1_S", "IQ4_XS",
}
