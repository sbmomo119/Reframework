"""Attention package: registry + factory (FreeToken's attention/__init__ shape).

Re registers a single backend — ``sdpa`` — because that is the only one that
works on sm_61. The registry pattern is kept so a hand-written sm_61 kernel can
be added later (``RE_ATTENTION_BACKEND=custom``) without engine changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

from reframework.utils import init_logger

if TYPE_CHECKING:
    from reframework.models import ModelConfig

logger = init_logger(__name__)

from .base import AttentionSpec, BaseAttnBackend  # noqa: E402

_CREATORS: Dict[str, Callable[["ModelConfig"], BaseAttnBackend]] = {}


def register_backend(name: str):
    def deco(fn: Callable[["ModelConfig"], BaseAttnBackend]):
        _CREATORS[name] = fn
        return fn

    return deco


def supported_backends() -> list[str]:
    return sorted(_CREATORS)


def create_attention_backend(name: str, config: "ModelConfig") -> BaseAttnBackend:
    if name == "auto":
        from reframework.utils import pascal

        if pascal.device_caps().supports_flash_attn:
            logger.info("non-Pascal device with FA support detected; Re still uses sdpa")
        name = "sdpa"
    if name not in _CREATORS:
        raise ValueError(f"unknown attention backend {name!r}; supported: {supported_backends()}")
    return _CREATORS[name](config)


# -- registrations -----------------------------------------------------------
from .sdpa import SDPAAttentionBackend, create_sdpa_backend  # noqa: E402

register_backend("sdpa")(create_sdpa_backend)

__all__ = [
    "AttentionSpec",
    "BaseAttnBackend",
    "SDPAAttentionBackend",
    "create_attention_backend",
    "register_backend",
    "supported_backends",
]
