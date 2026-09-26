"""Layer package.

FreeToken's ``layers/__init__.py`` re-exports the building blocks; Re does the
same so models can ``from reframework.layers import MLP, RMSNorm, ...``.
"""

from .base import BaseOP, StateLessOP
from .linear import Linear
from .norm import RMSNorm
from .mlp import MLP
from .embedding import ParallelLMHead, VocabParallelEmbedding
from .rotary import get_rope, set_rope_device
from .activation import (
    GATED_ACTIVATIONS,
    gated_act_and_mul,
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
)

__all__ = [
    "BaseOP",
    "StateLessOP",
    "Linear",
    "RMSNorm",
    "MLP",
    "ParallelLMHead",
    "VocabParallelEmbedding",
    "get_rope",
    "set_rope_device",
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "gated_act_and_mul",
    "GATED_ACTIVATIONS",
]
