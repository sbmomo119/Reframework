"""safetensors int4 packing.

The GGUF path dequantizes to fp on load. The *safetensors* path can instead
store (or shrink the host pool to) an int4 payload produced by :mod:`litemoe.
quantization.scheme`. These helpers serialize/deserialize an :class:`Int4Payload`
to/from the flat dicts that ``safetensors`` ``save_file``/``load_file`` use, so
an int4 expert bank can live on disk as a ``.safetensors`` file.

Layout on disk (all 1-D / dense, torch-friendly):
    ``<key>.codes``   : int8  [ceil(N/2)]
    ``<key>.scales``  : fp16  [G]
    ``<key>.meta``    : fp32  [2] = [group, N]   (N = total elements)
The logical ``shape`` is reconstructed by the loader from the model config
(experts are fixed-shape), so we only persist ``N``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from litemoe.quantization.scheme import Int4Payload, int4_pack, int4_unpack


def int4_to_safetensors_dict(w: np.ndarray, key: str, group: int = 128) -> Dict[str, np.ndarray]:
    """Pack ``w`` and emit the three named arrays ``safetensors.save_file`` wants."""
    p = int4_pack(w, group)
    N = w.size
    return {
        f"{key}.codes": p.codes,
        f"{key}.scales": p.scales,
        f"{key}.meta": np.array([group, N], dtype=np.float32),
    }


def int4_from_safetensors_dict(d: Dict[str, np.ndarray], key: str, shape: tuple) -> np.ndarray:
    """Inverse of :func:`int4_to_safetensors_dict` -> fp32 of ``shape``."""
    p = Int4Payload(
        codes=d[f"{key}.codes"],
        scales=d[f"{key}.scales"],
        group=int(d[f"{key}.meta"][0]),
        shape=tuple(shape),
    )
    return int4_unpack(p)


# Re-export for convenience.
__all__ = ["int4_to_safetensors_dict", "int4_from_safetensors_dict",
           "int4_pack", "int4_unpack", "Int4Payload"]
