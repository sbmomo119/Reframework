"""LoRA (Low-Rank Adaptation) for Re's linear layers.

FreeToken ships no adapter support; Re adds a minimal, inference-friendly
LoRA so a fine-tuned adapter can be dropped onto a base checkpoint without
touching the base weights on disk.

Design (deliberately minimal, fp32, no raw CUDA):

  * A :class:`LoRALinear` wraps an existing :class:`~reframework.layers.Linear`
    and adds a low-rank pair ``A`` ``[r, in]`` and ``B`` ``[out, r]``.
    ``forward`` computes ``base(x) + (x @ A^T) @ B^T * scaling`` where
    ``scaling = alpha / r``.
  * ``B`` is zero-initialised, so at load time the delta is exactly 0 and the
    wrapped layer is **bit-identical** to the base layer (greedy output
    unchanged) until an adapter is loaded.
  * The base weight is never modified by the wrapper; ``state_dict`` /
    ``load_state_dict`` delegate to the base ``Linear`` with the same prefix,
    so a model's checkpoint I/O is byte-identical to the no-LoRA path. The
    adapter weights are saved/loaded separately (``save_lora`` / ``load_lora``).
  * ``merge_weights()`` folds the delta permanently into the base weight and
    disables the delta path — a one-time cost for a constant speed win when the
    adapter is baked in.

Only :class:`Linear` instances are wrap candidates. MoE expert weights live in
a ``nn.ParameterList`` and the MoE gate is a raw ``nn.Parameter`` — neither is a
``Linear`` — so MoE layers are naturally skipped (LoRA on dense attention +
FFN projections only, which is the common case).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from reframework.layers.linear import Linear


@dataclass
class LoRAConfig:
    """LoRA hyper-parameters + which linear layers to patch.

    ``target_modules`` are *leaf* module names (last path component). The
    defaults cover Llama/Qwen dense attention + FFN projections:
    ``q_proj k_proj v_proj o_proj gate_up down``.
    """

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Set[str] = field(
        default_factory=lambda: {"q_proj", "k_proj", "v_proj", "o_proj", "gate_up", "down"}
    )

    @property
    def scaling(self) -> float:
        return self.alpha / max(1, self.r)


class LoRALinear(nn.Module):
    """A :class:`Linear` plus a low-rank ``B @ A`` delta (zero until loaded)."""

    def __init__(
        self,
        base: Linear,
        config: LoRAConfig,
        name: str = "",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.base = base  # registered submodule -> its params ride .to()/.parameters()
        self.name = name
        self.config = config
        dt = dtype or base.weight.dtype
        self.scale = config.scaling
        in_f = base.in_features
        out_f = base.out_features
        # A: [r, in], B: [out, r]. B zero -> delta is exactly 0 at init.
        self.lora_A = nn.Parameter(torch.empty(config.r, in_f, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, config.r, dtype=dt))
        # kaiming-ish init for A so a freshly-trained adapter has signal once B != 0
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self._merged = False
        self._active = False  # delta path on/off (B==0 at init -> off)

    # ---- transparent delegation so callers treat LoRALinear like a Linear ----
    # MLP.state_dict does ``self.gate_up.weight``; FusedMoE/engine read
    # ``in_features``/``out_features``. Expose the base attrs (read-only) so
    # wrapping a Linear is drop-in without touching the callers.
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base.forward(x)
        if self._merged or not self._active:
            return out
        # fp32 compute (Re constraint). x is already the compute dtype.
        delta = (x @ self.lora_A.to(x.dtype).T) @ self.lora_B.to(x.dtype).T
        return out + delta * self.scale

    # ------------------------------------------------------------- merge / drop
    def merge_weights(self) -> None:
        """Fold ``B @ A * scale`` into the base weight and drop the delta path."""
        if self._merged:
            return
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scale
            self.base.weight.data.add_(delta.to(self.base.weight.dtype))
        self._merged = True

    @property
    def merged(self) -> bool:
        return self._merged

    # ---------------------------------------------------------- adapter (de)serial
    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {"lora_A": self.lora_A.detach().cpu(), "lora_B": self.lora_B.detach().cpu()}

    def load_lora_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            if "lora_A" in sd:
                self.lora_A.copy_(sd["lora_A"].to(dtype=self.lora_A.dtype))
            if "lora_B" in sd:
                self.lora_B.copy_(sd["lora_B"].to(dtype=self.lora_B.dtype))
        self._merged = False  # loading a fresh adapter un-merges
        self._active = bool(self.lora_B.abs().sum().item() != 0)

    # ------------------------------------------------- base checkpoint I/O (unchanged keys)
    def state_dict(self, *, prefix: str = "", result: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        # delegate to base so model save/load keys are identical to the no-LoRA path
        return self.base.state_dict(prefix=prefix, result=result)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], *, prefix: str = "") -> None:
        self.base.load_state_dict(state_dict, prefix=prefix)


# -------------------------------------------------------------------------- apply
def _leaf(path: str) -> str:
    return path.split(".")[-1]


def apply_lora(model: nn.Module, config: LoRAConfig) -> List[str]:
    """Wrap every target :class:`Linear` in ``model`` with a :class:`LoRALinear`.

    Returns the dotted paths of the layers that were wrapped. Two-phase
    (collect ids, then replace) so mutating the tree never races the traversal.
    """
    targets = set(config.target_modules)
    ids: Set[int] = set()
    paths: Dict[int, str] = {}
    for name, m in model.named_modules():
        if isinstance(m, Linear) and _leaf(name) in targets:
            ids.add(id(m))
            paths[id(m)] = name

    wrapped: List[str] = []
    for parent in list(model.modules()):
        for attr, child in list(parent.named_children()):
            cid = id(child)
            if cid in ids:
                setattr(parent, attr, LoRALinear(child, config, name=attr))
                wrapped.append(paths.pop(cid))
                ids.discard(cid)
    return wrapped


def get_lora_modules(model: nn.Module) -> Dict[str, LoRALinear]:
    """Map wrapped-layer path -> :class:`LoRALinear` (for save/load/merge)."""
    out: Dict[str, LoRALinear] = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            out[name] = m
    return out


def save_lora(model: nn.Module, path: str) -> None:
    """Save all adapter weights (A/B per layer) + config to ``path``."""
    mods = get_lora_modules(model)
    if not mods:
        raise RuntimeError("no LoRA layers to save (did you call apply_lora?)")
    cfg = next(iter(mods.values())).config
    payload = {
        "config": {
            "r": cfg.r,
            "alpha": cfg.alpha,
            "target_modules": sorted(cfg.target_modules),
        },
        "adapters": {name: m.lora_state_dict() for name, m in mods.items()},
    }
    torch.save(payload, path)


def load_lora(model: nn.Module, path: str) -> int:
    """Load adapter weights from ``path`` into the wrapped layers. Returns count."""
    payload = torch.load(path, map_location="cpu")
    mods = get_lora_modules(model)
    n = 0
    for name, sd in payload["adapters"].items():
        if name in mods:
            mods[name].load_lora_state_dict(sd)
            n += 1
        else:
            raise KeyError(f"adapter for {name!r} has no matching wrapped layer")
    return n


def merge_all_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight (in-place). Returns count."""
    n = 0
    for m in get_lora_modules(model).values():
        m.merge_weights()
        n += 1
    return n


__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "get_lora_modules",
    "save_lora",
    "load_lora",
    "merge_all_lora",
]
