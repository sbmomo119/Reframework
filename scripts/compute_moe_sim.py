"""Offline per-layer expert similarity table for MoE offload substitution.

For every MoE layer, computes a cosine-similarity matrix over its experts
(flattened gate/up/down projection weights). At runtime, when the prefetcher
predicts an expert that is not resident in the LRU, it substitutes the most
similar *already-resident* expert looked up from this table (see
``reframework/moe/offload_cache.py``).

Weights are read the same way ``build_model`` does -- direct safetensors
iteration over the ``model.layers.{L}.mlp.experts.{e}.*`` keys -- so the
expert ordering matches what the runtime sees.

Usage::

    python scripts/compute_moe_sim.py <ckpt_dir> [--out <file>]

Default output: ``<ckpt_dir>/moe_expert_sim.pt``.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors import safe_open

from reframework.checkpoint.loader import resolve_weight_files

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_table(ckpt_dir: str) -> dict:
    """Return ``{"tables": [FloatTensor[E,E] per MoE layer], "moe_layers":
    [global layer idxs], "num_moe_layers": N, "num_experts": E}``."""
    weights_paths = [str(p) for p in resolve_weight_files(ckpt_dir)]

    # {global layer idx: {expert id: [weight tensors in file order]}}
    layer_experts: dict[int, dict[int, list[torch.Tensor]]] = {}
    for wp in weights_paths:
        with safe_open(wp, framework="pt") as f:
            for key in f.keys():
                if ".mlp.experts." not in key or not key.startswith("model.layers."):
                    continue
                parts = key.split("model.layers.")[1].split(".")
                layer = int(parts[0])   # model.layers.{L}
                eid = int(parts[3])     # model.layers.{L}.mlp.experts.{e}.*
                layer_experts.setdefault(layer, {}).setdefault(eid, []).append(
                    f.get_tensor(key).float())

    if not layer_experts:
        raise SystemExit(f"no MoE experts found under {ckpt_dir!r}")

    moe_layers = sorted(layer_experts)
    E = max(len(v) for v in layer_experts.values())

    tables = []
    for L in moe_layers:
        experts = layer_experts[L]
        eids = sorted(experts)
        assert len(eids) == E, f"layer {L}: expected {E} experts, got {len(eids)}"
        vecs = [torch.cat([w.flatten() for w in experts[eid]]) for eid in eids]
        mat = torch.stack(vecs)                                  # [E, D]
        norm = mat / mat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim = norm @ norm.t()                                    # [E, E]
        tables.append(sim.cpu())
        print(f"  layer {L:3d}: {E} experts, sim range "
              f"[{sim.min().item():.4f}, {sim.max().item():.4f}]")

    return {
        "tables": tables,
        "moe_layers": moe_layers,
        "num_moe_layers": len(moe_layers),
        "num_experts": E,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir", help="directory with checkpoint.json + safetensors")
    ap.add_argument("--out", default=None,
                    help="output .pt path (default: <ckpt_dir>/moe_expert_sim.pt)")
    args = ap.parse_args()

    out = args.out or os.path.join(args.ckpt_dir, "moe_expert_sim.pt")
    print(f"computing MoE similarity table for {args.ckpt_dir} -> {out}")
    table = compute_table(args.ckpt_dir)
    torch.save(table, out)
    print(f"wrote {out} ({len(table['tables'])} layers, "
          f"{table['num_experts']} experts each)")


if __name__ == "__main__":
    main()
