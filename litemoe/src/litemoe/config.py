"""Unified configuration for litemoe.

One YAML/JSON file drives a whole run: model path (GGUF or safetensors),
compute dtype, expert-cache size and eviction strategy, offload device, and
metrics/logging switches. ``litemoe run --config xxx`` reads this.

Example config (config.yml):

    model:
      path: /home/samuel/Carnice-Qwen3.6-MoE-35B-A3B-APEX-I-Mini.gguf
      # tokenizer: optional path to tokenizer.json/.model (GGUF has none)
    compute:
      dtype: fp16        # fp16 | fp32 (quantized weights are dequantized on load)
      device: cpu        # cpu | cuda:0 (1070ti needs a CUDA-capable torch build)
    cache:
      strategy: lru      # lru | pde (PDE = predictive/deepest-first, pluggable)
      max_experts: 32    # resident expert slots in the "VRAM" LRU
      predict_window: 32 # decode steps of routing history for PDE lookahead
    metrics:
      log_activations: true     # per-layer per-step routed sets -> jsonl
      activations_file: ./activations.jsonl
      print_per_step: true      # TTFT / tok-per-s / transfer / hit-rate
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# dtype / device helpers                                                      #
# --------------------------------------------------------------------------- #
def parse_dtype(name: str):
    import torch

    name = (name or "fp32").lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"unknown dtype: {name!r} (use fp16|bf16|fp32)")


def parse_device(name: str):
    import torch

    name = (name or "cpu").lower()
    if name == "cpu":
        return torch.device("cpu")
    if name.startswith("cuda"):
        return torch.device(name)
    raise ValueError(f"unknown device: {name!r} (use cpu|cuda:0)")


# --------------------------------------------------------------------------- #
# config dataclasses                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    path: str = ""
    tokenizer: Optional[str] = None

    @property
    def kind(self) -> str:
        """'gguf' or 'safetensors' — inferred from the file extension."""
        low = (self.path or "").lower()
        if low.endswith(".gguf"):
            return "gguf"
        if low.endswith((".safetensors", ".st")):
            return "safetensors"
        return os.path.splitext(low)[1].lstrip(".") or "unknown"


@dataclass
class ComputeConfig:
    dtype: str = "fp16"
    device: str = "cpu"


@dataclass
class CacheConfig:
    strategy: str = "lru"          # lru | pde
    max_experts: int = 32          # resident slots in the LRU window
    predict_window: int = 32       # decode steps of history for PDE
    bandwidth_gb_s: Optional[float] = None  # measured/overridden PCIe rate
    bw_headroom: float = 0.9


@dataclass
class MetricsConfig:
    log_activations: bool = True
    activations_file: str = "./litemoe_activations.jsonl"
    print_per_step: bool = True


@dataclass
class LitemoeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    # raw prompt / generation knobs used by the CLI
    prompt: str = ""
    max_new_tokens: int = 64
    seed: int = 0

    # ------------------------------------------------------------------ load
    @classmethod
    def from_file(cls, path: str) -> "LitemoeConfig":
        """Load from .json or .yaml/.yml (yaml via PyYAML if present)."""
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path.lower().endswith(".json"):
            raw = json.loads(text)
        else:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML is required for .yaml configs "
                    "(pip install pyyaml) or use a .json config"
                )
            raw = yaml.safe_load(text)
        return cls.from_dict(raw or {})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LitemoeConfig":
        def mk(dc, sub):
            sub = dict(sub or {})
            keep = {f: sub.pop(f) for f in list(sub) if f in dc.__dataclass_fields__}
            return dc(**keep)

        return cls(
            model=mk(ModelConfig, d.get("model")),
            compute=mk(ComputeConfig, d.get("compute")),
            cache=mk(CacheConfig, d.get("cache")),
            metrics=mk(MetricsConfig, d.get("metrics")),
            prompt=d.get("prompt", ""),
            max_new_tokens=int(d.get("max_new_tokens", 64)),
            seed=int(d.get("seed", 0)),
        )

    # ----------------------------------------------------------------- save
    def to_dict(self) -> Dict[str, Any]:
        import dataclasses

        return dataclasses.asdict(self)

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
