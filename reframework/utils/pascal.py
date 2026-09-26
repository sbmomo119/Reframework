"""sm_61 (NVIDIA Pascal) capability detection.

This is the module that makes Re *the* Pascal framework. FreeToken assumes
sm_70+ (Volta/Turing/Ampere) and reaches for flash-attention, tensor-core
quantized kernels and 16+ GB of VRAM. Pascal (GTX 1060/1070/1080) has none of
that:

  * BF16:            NOT supported (needs sm_80+)
  * FP8 / FP4:       NOT supported (needs sm_89+)
  * Tensor cores:    NOT supported (no TC on GP102/GP104 at all)
  * flash-attn v2:   NOT supported (needs sm_70+)
  * cp.async:        NOT supported (needs sm_80+)
  * FP16:            *storage* only — the SMs compute fp16->fp32 with no TC

The consequence: Re runs **fp32 compute** (fp16 *storage* is allowed purely as
a VRAM-saver for weights) and uses **PyTorch scaled-dot-product attention**.
Every capability query in Re funnels through :func:`device_caps` so the engine
has a single source of truth about what the GPU can do.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

PASCAL_SM = 61


@dataclass(frozen=True)
class DeviceCaps:
    """Frozen capability snapshot for one device. ``DeviceCaps.cpu()`` models
    the CPU fallback path (no CUDA) so the whole stack is CPU-testable."""

    name: str
    sm: int  # cc_major * 100 + cc_minor, e.g. 61 -> sm_61; 0 -> CPU
    vram_bytes: int = 0

    @property
    def is_cpu(self) -> bool:
        return self.sm == 0

    @property
    def is_pascal(self) -> bool:
        return self.sm == PASCAL_SM

    @property
    def compute_cap(self) -> str:
        return "cpu" if self.is_cpu else f"sm_{self.sm}"

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / 1024**3

    # -- feature gates (the whole "what can Pascal do" story in one place) --
    @property
    def supports_bf16(self) -> bool:
        return self.sm >= 80

    @property
    def supports_fp8(self) -> bool:
        return self.sm >= 89

    @property
    def supports_fp16_tensor_core(self) -> bool:
        # Real fp16 TC multiply-accumulate starts at Turing (sm_75). sm_61
        # stores fp16 but has no TC, so fp16 *compute* is no faster than fp32.
        return self.sm >= 75

    @property
    def supports_flash_attn(self) -> bool:
        return self.sm >= 70

    @property
    def supports_cp_async(self) -> bool:
        return self.sm >= 80

    @property
    def supports_cuda_graph(self) -> bool:
        # Pascal supports graph launch (sm_50+); the engine still gates it off
        # by default for memory-bound workloads.
        return self.sm >= 50

    def preferred_compute_dtype(self) -> torch.dtype:
        """The dtype the engine should actually compute in on this device.

        Pascal: fp32 (fp16 would only halve storage, not speed, without TC).
        Turing/Volta: fp16. Ampere+: bf16 (or fp16). CPU: fp32.
        """
        if self.is_cpu or self.is_pascal or self.sm < 75:
            return torch.float32
        if self.supports_bf16:
            return torch.bfloat16
        return torch.float16

    @classmethod
    def cpu(cls) -> "DeviceCaps":
        return cls(name="cpu", sm=0, vram_bytes=0)

    @classmethod
    def cuda(cls, index: int = 0) -> "DeviceCaps":
        props = torch.cuda.get_device_properties(index)
        sm = props.major * 100 + props.minor
        return cls(name=props.name, sm=sm, vram_bytes=int(props.total_memory))


@lru_cache(maxsize=4)
def device_caps(index: int = 0) -> DeviceCaps:
    """Detect capabilities for CUDA device ``index`` (CPU fallback if no CUDA)."""
    if not torch.cuda.is_available():
        return DeviceCaps.cpu()
    return DeviceCaps.cuda(index)


def is_pascal(device: torch.device | None = None) -> bool:
    idx = 0 if (device is None or not torch.cuda.is_available()) else device.index or 0
    return device_caps(idx).is_pascal


def is_sm61() -> bool:
    """Alias kept for parity with FreeToken's is_sm90_family() style helpers."""
    return device_caps().is_pascal


def pascal_warning_if_needed(index: int = 0) -> str | None:
    """Return a human-readable warning when the device is NOT the sm_61 target,
    so a user who points Re at a 4090/5090 sees why Re is deliberately
    conservative."""
    caps = device_caps(index)
    if caps.is_cpu:
        return None
    if caps.is_pascal:
        return None
    return (
        f"Re is tuned for sm_61 (Pascal) but detected {caps.name} ({caps.compute_cap}). "
        "It will still run, but you are leaving performance on the table — FreeToken or "
        "vLLM would be a better fit for this GPU."
    )


__all__ = [
    "PASCAL_SM",
    "DeviceCaps",
    "device_caps",
    "is_pascal",
    "is_sm61",
    "pascal_warning_if_needed",
]
