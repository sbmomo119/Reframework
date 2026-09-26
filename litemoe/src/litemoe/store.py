"""Expert weight backends (the ``ExpertStore`` contract, implemented).

Two concrete stores, one interface (:class:`~litemoe.interface.ExpertStore`):

* :class:`GGUFExpertStore`        — the 14 GB ``qwen35moe`` GGUF (int4/k-quant mix).
* :class:`SafetensorsExpertStore` — the HF-named safetensors checkpoint
  (tiny ``qwen3-moe`` fp16/fp32, or int4-packed).

Design notes
------------
A store is *stateless about residency*: it hands out expert banks on demand and
reports how many bytes a fetch moves (the cache does the transfer accounting).
The store is not the cache — the :class:`~litemoe.interface.Cache` owns which
experts are resident in the simulated VRAM.

GGUF dequantization cost
------------------------
``gguf.dequantize`` operates on a tensor's whole raw blob, so you cannot cheaply
dequantize a *single* expert out of an ``[E, out, in]`` stack (k-quants are
organized in 256-element super-blocks that don't align to per-expert slices).
The store therefore dequantizes each ``(layer, kind)`` stack **once** and keeps
it in a bounded cache, slicing experts out of it. A later fetch of an evicted
expert is still counted as a miss/transfer by the cache (via
:meth:`expert_payload_bytes`); the internal cache only saves the *CPU* dequant
work, it does not short-circuit the transfer model.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from litemoe.interface import ExpertBanks, ExpertStore
from litemoe.model.loader import GGUFLoader, ModelMeta
from litemoe.model.safetensors_loader import SafetensorsLoader
from litemoe.quantization.gguf_int4 import load_gguf_tensor


# --------------------------------------------------------------------------- #
# GGUF int4 backend                                                           #
# --------------------------------------------------------------------------- #
class GGUFExpertStore(ExpertStore):
    """Expert store over a dequantized GGUF checkpoint.

    Tensor naming (``qwen35moe``)::

        blk.<L>.ffn_gate_exps.weight  [E, I, H]  routed gate_proj
        blk.<L>.ffn_up_exps.weight    [E, I, H]  routed up_proj
        blk.<L>.ffn_down_exps.weight  [E, H, I]  routed down_proj
        blk.<L>.ffn_gate_shexp.weight [I, H]     shared gate_proj (dense)
        blk.<L>.ffn_up_shexp.weight   [I, H]
        blk.<L>.ffn_down_shexp.weight [H, I]
        blk.<L>.ffn_gate_inp.weight   [E, H]     router
    """

    def __init__(self, path: str, stack_cache_layers: int = 16) -> None:
        self.loader = GGUFLoader(path)
        self.meta: ModelMeta = self.loader.meta
        # (layer, kind) -> full dequantized stack [E, out, in] (kept in compute dtype)
        self._stacks: "OrderedDict[Tuple[int, str], torch.Tensor]" = OrderedDict()
        self._stack_cache_layers = int(stack_cache_layers)
        self._stack_bytes: int = 0
        # small whole tensors we dequant once and reuse: (layer) -> gate, shared exps
        self._gates: Dict[int, torch.Tensor] = {}
        self._shared: Dict[int, Dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------ dense
    def load_dense(self, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
        return self.loader.load_dense(dtype=dtype)

    # ------------------------------------------------------------------ expert
    def _stack(self, layer: int, kind: str, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize (once) + cache the full ``[E, out, in]`` stack for a kind."""
        key = (layer, kind)
        if key in self._stacks:
            self._stacks.move_to_end(key)
            return self._stacks[key]
        name = f"blk.{layer}.ffn_{kind}_exps.weight"
        ten, _, _, _ = load_gguf_tensor(self.loader.reader, name)
        ten = ten.to(dtype).contiguous()
        self._stacks[key] = ten
        self._stack_bytes += ten.numel() * ten.element_size()
        # bound by number of *layers* (each layer holds up to 3 kinds)
        max_keys = self._stack_cache_layers * 3
        while len(self._stacks) > max_keys:
            _, old = self._stacks.popitem(last=False)
            self._stack_bytes -= old.numel() * old.element_size()
        return ten

    def expert_banks(
        self, layer: int, eids: Iterable[int], dtype: torch.dtype = torch.float16
    ) -> ExpertBanks:
        eids = list(eids)
        gate_s = self._stack(layer, "gate", dtype)
        up_s = self._stack(layer, "up", dtype)
        down_s = self._stack(layer, "down", dtype)
        banks: ExpertBanks = {}
        for e in eids:
            banks[int(e)] = {
                "gate": gate_s[e].contiguous(),
                "up": up_s[e].contiguous(),
                "down": down_s[e].contiguous(),
            }
        return banks

    # ------------------------------------------------------------------- gate
    def gate(self, layer: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        if layer not in self._gates:
            ten, _, _, _ = load_gguf_tensor(self.loader.reader, f"blk.{layer}.ffn_gate_inp.weight")
            self._gates[layer] = ten.to(dtype).contiguous()
        return self._gates[layer]

    # --------------------------------------------------------------- shared exp
    def has_shared_expert(self, layer: int) -> bool:
        return bool(self.meta.has_shared_expert)

    def shared_experts(self, layer: int, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
        if not self.meta.has_shared_expert:
            return {}
        if layer not in self._shared:
            row: Dict[str, torch.Tensor] = {}
            for kind in ("gate", "up", "down"):
                ten, _, _, _ = load_gguf_tensor(self.loader.reader, f"blk.{layer}.ffn_{kind}_shexp.weight")
                row[kind] = ten.to(dtype).contiguous()
            self._shared[layer] = row
        return self._shared[layer]

    # -------------------------------------------------------------- byte cost
    def expert_payload_bytes_for(self, layer: int, eids: Iterable[int]) -> int:
        """Raw on-disk quant bytes for the given experts (the fetch cost).

        Each expert occupies ``1/E`` of each of the three ``[E,...]`` stacks, so
        per-expert payload = ``sum(stack_bytes)/E`` (uniform qtype per stack).
        """
        eids = list(eids)
        if not eids:
            return 0
        E = max(1, int(self.meta.n_experts))
        total = 0
        for kind in ("gate", "up", "down"):
            total += self.loader.tensor_nbytes(f"blk.{layer}.ffn_{kind}_exps.weight")
        per = total / E
        return int(per * len(eids))

    # -------------------------------------------------------------- housekeep
    def drop(self, layer: int, eids: Iterable[int]) -> None:
        """Release the dequant stacks for this layer (next fetch re-dequants)."""
        for kind in ("gate", "up", "down"):
            key = (layer, kind)
            old = self._stacks.pop(key, None)
            if old is not None:
                self._stack_bytes -= old.numel() * old.element_size()

    def close(self) -> None:
        self._stacks.clear()
        self._gates.clear()
        self._shared.clear()
        self.loader.close()


# --------------------------------------------------------------------------- #
# safetensors backend                                                         #
# --------------------------------------------------------------------------- #
class SafetensorsExpertStore(ExpertStore):
    """Expert store over an HF-named safetensors checkpoint (tiny qwen3-moe).

    Tensor naming::

        model.embed_tokens.weight
        model.layers.<L>....{input_layernorm,post_attention_layernorm}.weight
        model.layers.<L>.self_attn.{q,k,v,o}_proj.weight
        model.layers.<L>.mlp.gate.weight
        model.layers.<L>.mlp.experts.<E>.{gate,up,down}_proj.weight
        model.norm.weight

    No shared expert in the standard Qwen3-MoE layout; ``shared_experts`` is {}.
    """

    def __init__(self, path: str, config: Optional[dict] = None) -> None:
        self.loader = SafetensorsLoader(path, config=config)
        self.meta: ModelMeta = self.loader.meta
        self._gates: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ dense
    def load_dense(self, dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
        return self.loader.load_dense(dtype=dtype)

    # ------------------------------------------------------------------ expert
    def expert_banks(
        self, layer: int, eids: Iterable[int], dtype: torch.dtype = torch.float16
    ) -> ExpertBanks:
        return self.loader.load_expert_banks(layer, list(eids), dtype=dtype)

    # ------------------------------------------------------------------- gate
    def gate(self, layer: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        if layer not in self._gates:
            self._gates[layer] = self.loader.gate(layer, dtype=dtype)
        return self._gates[layer]

    # -------------------------------------------------------------- byte cost
    def expert_payload_bytes_for(self, layer: int, eids: Iterable[int]) -> int:
        """On-disk bytes for the experts (each {gate,up,down}_proj)."""
        eids = list(eids)
        total = 0
        for e in eids:
            for kind in ("gate", "up", "down"):
                k = f"model.layers.{layer}.mlp.experts.{e}.{kind}_proj.weight"
                a = self.loader._np.get(k)
                total += a.nbytes if a is not None else 0
        return int(total)

    def close(self) -> None:
        self._gates.clear()


# --------------------------------------------------------------------------- #
# factory                                                                     #
# --------------------------------------------------------------------------- #
def make_store(path: str, config: Optional[dict] = None) -> ExpertStore:
    """Pick the right :class:`ExpertStore` by file extension.

    ``path`` 可以是 checkpoint 文件（``.gguf`` / ``.safetensors``），
    也可以是包含 checkpoint 的目录（自动取目录内第一个 ``*.safetensors``，
    按字典序排序，多文件分片场景取第一个）。
    """
    from pathlib import Path as _P

    _p = _P(path)
    if _p.is_dir():
        cands = sorted(_p.glob("*.safetensors"))
        if not cands:
            raise FileNotFoundError(
                f"no *.safetensors found in directory {path!r}")
        path = str(cands[0])
    low = (path or "").lower()
    if low.endswith(".gguf"):
        return GGUFExpertStore(path)
    if low.endswith((".safetensors", ".st")):
        return SafetensorsExpertStore(path, config=config)
    raise ValueError(f"unknown checkpoint extension for {path!r} (use .gguf or .safetensors)")


__all__ = ["GGUFExpertStore", "SafetensorsExpertStore", "make_store"]
