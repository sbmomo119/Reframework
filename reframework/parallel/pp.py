"""Minimal Pipeline Parallelism (PP) - split model.layers across ranks."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from . import dist as _dist

__all__ = ["shard_model_pp", "pp_enabled", "pipeline_forward"]


def pp_enabled() -> bool:
    return _dist.is_initialized() and _dist.current().pp_size > 1


def shard_model_pp(model, pp_size: int, pp_rank: int):
    layers = model.layers
    total = len(layers)
    if pp_size > total:
        raise ValueError(f"pp_size={pp_size} > num_layers={total}")
    base, rem = divmod(total, pp_size)
    sizes = [base + (1 if i < rem else 0) for i in range(pp_size)]
    start = sum(sizes[:pp_rank])
    end = start + sizes[pp_rank]
    model.layers = nn.ModuleList([layers[i] for i in range(start, end)])
    model._pp_rank = pp_rank
    model._pp_size = pp_size
    model._pp_start = start
    model._pp_end = end
    if pp_rank != 0 and hasattr(model, "embed_tokens"):
        model.embed_tokens = None
    if pp_rank != pp_size - 1:
        if hasattr(model, "norm"):
            model.norm = None
        if hasattr(model, "lm_head"):
            model.lm_head = None
    return model


def _recv_hidden(dtype, device, src):
    meta = torch.empty(2, dtype=torch.long, device=device)
    _dist.recv(meta, src=src)
    n_tokens = int(meta[0].item())
    hidden_size = int(meta[1].item())
    h = torch.empty(n_tokens, hidden_size, dtype=dtype, device=device)
    _dist.recv(h, src=src)
    residual = torch.empty_like(h)
    _dist.recv(residual, src=src)
    return h, residual


def _send_hidden(h, residual, dst):
    meta = torch.tensor([h.shape[0], h.shape[1]], dtype=torch.long, device=h.device)
    _dist.send(meta, dst=dst)
    _dist.send(h.contiguous(), dst=dst)
    _dist.send(residual.contiguous(), dst=dst)


@torch.no_grad()
def pipeline_forward(model, input_ids, positions, batch):
    pp_rank = model._pp_rank
    pp_size = model._pp_size
    device = input_ids.device
    dtype = model.cfg.dtype
    if pp_rank == 0:
        h = model.embed_tokens(input_ids)
        residual = torch.zeros_like(h)
    else:
        h, residual = _recv_hidden(dtype, device, src=pp_rank - 1)
    for layer in model.layers:
        h, residual = layer(positions, h, residual, batch)
    if pp_rank == pp_size - 1:
        h = model.norm(h)
        return model.lm_head(h)
    else:
        _send_hidden(h, residual, dst=pp_rank + 1)
        return None
