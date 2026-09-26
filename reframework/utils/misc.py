"""Small numeric / device helpers shared across Re (mirrors FreeToken's utils)."""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Iterator

import torch


def div_ceil(a: int, b: int) -> int:
    return -(-a // b)


def align_ceil(a: int, b: int) -> int:
    return div_ceil(a, b) * b


def mem_GB(num_bytes: float) -> str:
    return f"{num_bytes / 1024**3:.2f} GB"


@contextlib.contextmanager
def torch_dtype(dtype: torch.dtype) -> Iterator[None]:
    prev = torch.get_default_dtype()
    torch.default_dtype = dtype
    try:
        yield
    finally:
        torch.default_dtype = prev


def nvtx_annotate(message: str) -> Callable:
    """FreeToken annotates hot paths with NVTX ranges for nsys profiling.

    On a Pascal box profiling is usually done with ``nsys --trace=cuda``; the
    range is a no-op when there is no CUDA device (CPU fallback path).
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push(f"Re::{message}")
            except Exception:  # noqa: BLE001 - never let profiling break inference
                pass
            try:
                return fn(*args, **kwargs)
            finally:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.nvtx.range_pop()
                except Exception:  # noqa: BLE001
                    pass

        return wrapper

    return decorator
