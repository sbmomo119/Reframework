"""Re — a lean, sm_61 (NVIDIA Pascal) aware LLM inference framework.

Re mirrors FreeToken's structure (engine / attention / kvcache / layers / moe /
models / server / cli) but is built around the hardware reality of Pascal
GPUs (GTX 1060/1070/1080, compute capability 6.1):

  * no BF16, no FP8/FP4, no tensor cores  -> fp32 (or fp16-storage) compute
  * no flash-attention fast path          -> PyTorch scaled-dot-product attention
  * 6-8 GB of VRAM                         -> host-offloaded MoE expert cache

Every capability decision funnels through :mod:`re.utils.pascal`, so the
engine picks the only backend that actually works on the GPU it finds.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
