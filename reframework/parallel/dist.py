"""Distributed process-group context for Re's PP + EP.

FreeToken is single-process by design; Re adds a thin ``torch.distributed``
wrapper so the same model can be split two ways:

  * **PP (pipeline parallelism)** — the decoder layers are partitioned across
    stages; the hidden state + residual stream is handed stage-to-stage.
  * **EP (expert parallelism)** — a MoE layer's experts are sharded across
    ranks; tokens are dispatched to the owning rank and combined back.

This module is the *only* place that touches ``torch.distributed``. Everything
else (pp.py / ep.py) calls the small collective helpers below, which keeps the
parallel code testable on CPU with the ``gloo`` backend and lets a real
multi-GPU box swap in ``nccl`` without touching the model code.

It is intentionally lazy: importing ``reframework.parallel`` and running the
whole engine single-process never initialises a process group. Only an explicit
:func:`init` (or ``run``) does, so the default single-GPU / CPU path is
unaffected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

__all__ = [
    "ParallelContext",
    "init",
    "run",
    "is_initialized",
    "current",
    "get_rank",
    "get_world_size",
    "all_reduce_sum",
    "all_gather",
    "all_to_all",
    "send",
    "recv",
    "barrier",
    "destroy",
]


@dataclass(frozen=True)
class ParallelContext:
    """Frozen handle to the current process group.

    ``pp_size`` is the number of pipeline stages (a subset of the ranks);
    ``ep_size`` the number of expert-parallel ranks. When only one dimension
    is > 1 the other stays 1, so a pure-PP or pure-EP run is just a 1-D group.
    """

    backend: str
    rank: int
    world_size: int
    group: Optional[dist.ProcessGroup] = None
    pp_size: int = 1
    ep_size: int = 1
    pp_rank: int = 0
    ep_rank: int = 0


_current: Optional[ParallelContext] = None


def is_initialized() -> bool:
    return _current is not None and dist.is_initialized()


def current() -> ParallelContext:
    """Return the active context or raise a clear error if none is live."""
    if _current is None:
        raise RuntimeError(
            "No process group is initialised — call parallel.init()/run() first "
            "(or run single-process without PP/EP)."
        )
    return _current


def get_rank() -> int:
    return current().rank


def get_world_size() -> int:
    return current().world_size


def init(
    *,
    backend: str = "gloo",
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    pp_size: int = 1,
    ep_size: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> ParallelContext:
    """Initialise (or re-set) the process group and the Re parallel context.

    ``init_method`` / ``rank`` / ``world_size`` are passed straight through to
    ``torch.distributed.init_process_group``. ``pp_size`` * ``ep_size`` must
    equal ``world_size`` (the 2-D layout is row-major: a rank's ``pp_rank`` is
    its block, ``ep_rank`` its position within the block).
    """
    global _current
    if dist.is_initialized():
        destroy()
    dist.init_process_group(
        backend=backend, init_method=init_method, rank=rank, world_size=world_size
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    # A 2-D (pp, ep) partition is only meaningful when at least one dimension
    # is actually split; a flat group (pp=ep=1) can be any world size (all-to-all).
    if pp_size > 1 or ep_size > 1:
        if pp_size * ep_size != world:
            raise ValueError(f"pp_size({pp_size}) * ep_size({ep_size}) must equal world_size({world})")
        pp_rank = rank // ep_size if ep_size > 1 else 0
        ep_rank = rank % ep_size if ep_size > 1 else 0
    else:
        pp_rank = 0
        ep_rank = 0
    _current = ParallelContext(
        backend=backend, rank=rank, world_size=world, group=group or dist.group.WORLD,
        pp_size=pp_size, ep_size=ep_size, pp_rank=pp_rank, ep_rank=ep_rank,
    )
    return _current


def run(
    target,
    *,
    world_size: int,
    backend: str = "gloo",
    init_method: Optional[str] = None,
    pp_size: int = 1,
    ep_size: int = 1,
    args: tuple = (),
) -> List:
    """Spawn ``world_size`` local processes (mp spawn) each running ``target``.

    ``target(rank, ctx, *args)`` is called inside each process after the group
    is initialised for that rank. This is the ergonomic path for local testing
    (all ranks on one node); a multi-node launcher would instead call :func:`init`
    directly in each process. Uses the ``fork`` start method (single-node CPU
    gloo) so ``target`` need not be picklable.
    """
    import torch.multiprocessing as mp

    def _worker(rank: int) -> None:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        ctx = init(
            backend=backend, init_method=init_method, rank=rank, world_size=world_size,
            pp_size=pp_size, ep_size=ep_size,
        )
        try:
            target(rank, ctx, *args)
        finally:
            destroy()

    ctx_mp = mp.get_context("fork")
    procs = [ctx_mp.Process(target=_worker, args=(r,)) for r in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    if any(c != 0 for c in codes):
        raise RuntimeError(f"parallel run failed with exit codes {codes}")
    return codes


# -------------------------------------------------------------- collectives
def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce (EP expert-output combine, PP debug)."""
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_gather(t: torch.Tensor) -> List[torch.Tensor]:
    """Gather identical-shape tensors from every rank (rank-major)."""
    world = get_world_size()
    out = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(out, t)
    return out


def all_to_all(t: torch.Tensor, scatter_dim: int = 0) -> torch.Tensor:
    """Symmetric all_to_all: ``t`` is split along ``scatter_dim`` into
    ``world_size`` chunks; rank r's r-th chunk is sent to rank r. Returns the
    reassembled tensor (identical shape to the input). Used by EP dispatch and
    the PP hidden-state handoff.
    """
    world = get_world_size()
    chunks = t.chunk(world, dim=scatter_dim)
    out = [torch.empty_like(c) for c in chunks]
    # dist.all_to_all expects flat input/output lists; reshape each chunk flat
    inputs = [c.contiguous().view(-1) for c in chunks]
    dist.all_to_all([o.view(-1) for o in out], inputs)
    return torch.stack(out, dim=scatter_dim).reshape_as(t)


def send(t: torch.Tensor, dst: int) -> None:
    dist.send(t.contiguous(), dst=dst)


def recv(shape, dtype: torch.dtype, src: int) -> torch.Tensor:
    out = torch.empty(tuple(shape), dtype=dtype)
    dist.recv(out, src=src)
    return out


def barrier() -> None:
    dist.barrier()


def destroy() -> None:
    global _current
    if dist.is_initialized():
        dist.destroy_process_group()
    _current = None
