"""PP + EP parallelism for Re (pipeline + expert parallel).

Re-exports the process-group context and the PP / EP primitives. Importing this
package never initialises a process group — only an explicit ``init``/``run``
does — so the default single-process engine path is untouched.
"""

from reframework.parallel.dist import (
    ParallelContext,
    init,
    run,
    is_initialized,
    current,
    get_rank,
    get_world_size,
    all_reduce_sum,
    all_gather,
    all_to_all,
    send,
    recv,
    barrier,
    destroy,
)

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


# --- Pipeline Parallelism (PP) ---
from .pp import shard_model_pp, pp_enabled, pipeline_forward  # noqa: E402
