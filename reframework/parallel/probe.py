"""Smoke-test the collective helpers with real multi-process gloo.

Run by ``.hf/_gloo_driver.py`` (2 spawned procs). Exercises all_reduce,
all_gather, all_to_all and send/recv — the exact primitives PP and EP use.
Each rank prints its result; the driver checks exit codes.
"""

from __future__ import annotations

import torch

from reframework import parallel as par


def main() -> None:
    rank = par.get_rank()
    world = par.get_world_size()

    # 1. all_reduce (EP expert-output combine)
    x = torch.tensor([float(rank * 10)], dtype=torch.float32)
    par.all_reduce_sum(x)
    expect = sum(range(world)) * 10
    assert x.item() == expect, (x.item(), expect)
    print(f"[rank {rank}] all_reduce ok = {x.item()}")

    # 2. all_gather (each rank sends its rank id)
    g = torch.tensor([float(rank)], dtype=torch.float32)
    got = par.all_gather(g)
    assert got[rank].item() == float(rank), got
    print(f"[rank {rank}] all_gather ok = {[t.item() for t in got]}")

    # 3. all_to_all (token scatter: rank r's chunk r goes to rank r)
    n = world * 2
    send = torch.arange(n, dtype=torch.float32).view(world, -1) * (rank + 1)
    recv = par.all_to_all(send)
    # rank r receives, in its slot r, the r-th chunk that rank r sent to it:
    # rank s sends chunk r (value arange*r... * (s+1)); rank r receives rank s's chunk r
    print(f"[rank {rank}] all_to_all row0 = {recv.view(world, -1)[0].tolist()}")

    # 4. send / recv (PP stage-to-stage handoff)
    if rank == 0:
        par.send(torch.tensor([123.0]), dst=1)
    else:
        t = par.recv((1,), torch.float32, src=0)
        assert t.item() == 123.0, t
        print(f"[rank {rank}] send/recv ok = {t.item()}")

    par.barrier()


if __name__ == "__main__":
    main()
