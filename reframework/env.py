"""Environment configuration for Re.

FreeToken gates behaviour on a handful of FREETOKEN_* env vars; Re uses RE_*
equivalents plus a few Pascal-specific ones. All access goes through this
module so the rest of the code never touches ``os.environ`` directly.
"""

from __future__ import annotations

import os
from functools import lru_cache


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# --- logging ----------------------------------------------------------------
def get_log_level() -> int:
    import logging

    return getattr(logging, _get("RE_LOG_LEVEL", "INFO").upper(), logging.INFO)


# --- device / compute -------------------------------------------------------
@lru_cache(maxsize=1)
def get_compute_dtype_name() -> str:
    """Requested compute dtype. On sm_61 this is forced to fp32 by the engine;
    on newer GPUs it may be bf16/fp16."""
    return _get("RE_COMPUTE_DTYPE", "auto").lower()


def get_max_num_seqs() -> int:
    return _get_int("RE_MAX_NUM_SEQS", 64)


def get_mem_fraction() -> float:
    return _get_float("RE_MEM_FRACTION", 0.85)


# --- MoE host offload -------------------------------------------------------
def get_moe_offload_enabled() -> bool:
    return _get_bool("RE_MOE_OFFLOAD", True)


def get_moe_cache_size() -> int:
    """Number of (expert, layer) slots to keep resident in the GPU slot cache.
    0 -> auto-sized by the engine from free VRAM."""
    return _get_int("RE_MOE_CACHE_SIZE", 0)


def get_moe_cache_ratio() -> float:
    """Fraction of the post-weights free-VRAM budget handed to the MoE cache."""
    return _get_float("RE_MOE_CACHE_RATIO", 0.5)


def get_moe_substitute() -> bool:
    """When a predicted expert is not resident in the LRU, substitute the most
    similar *already-resident* expert (from the offline cosine table) instead of
    loading it over PCIe. Falls back to a normal load when nothing is resident.
    No-op unless a table produced by ``scripts/compute_moe_sim.py`` is present."""
    return _get_bool("RE_MOE_SUBSTITUTE", True)


def get_moe_predict() -> bool:
    """Enable the lookahead predictor: after each decode step the layer's
    routing history (a rolling window of routed expert sets) predicts the next
    step's experts and pulls the top-k of them into the LRU during the
    inter-forward window. k is dynamic — bounded by the measured PCIe
    bandwidth and the LRU's evictable room (see ExpertOffloadCache.reserve)."""
    return _get_bool("RE_MOE_PREDICT", True)


def get_moe_predict_window() -> int:
    """Rolling history window (number of past decode steps) the lookahead
    predictor ranks experts by. Bigger = smoother, slower to adapt."""
    return _get_int("RE_MOE_PREDICT_WINDOW", 32)


def get_moe_predict_headroom() -> float:
    """Fraction of the measured PCIe budget reserve() is willing to spend per
    inter-forward window (1.0 = spend it all; default 0.9 leaves ~10% for
    bandwidth variance)."""
    return _get_float("RE_MOE_PREDICT_HEADROOM", 0.9)


def get_moe_prefill_overlap() -> bool:
    """Double-buffer the prefill expert fetch so the PCIe H2D copy overlaps the
    GPU GEMM. Cheap on Pascal (single stream + two buffers) and the main win
    for long-context prefill on a small-VRAM card."""
    return _get_bool("RE_MOE_PREFILL_OVERLAP", True)


def get_moe_cache_policy() -> str:
    """Eviction policy for the VRAM expert LRU (see ExpertOffloadCache._evict).

    ``lru`` (default): evict the least-recently-used resident first — the
    original behavior. ``least_stale``: evict the expert *least likely to be
    routed again* — highest routing staleness, i.e. the most decode steps
    since its last activation (``Prediction._age``; never-seen = maximally
    stale), tie-broken by LRU order. Requires the lookahead predictor to be
    attached (``RE_MOE_PREDICT``); without one there is no routing signal and
    it degrades to plain LRU. Unknown values fall back to ``lru`` so a typo
    never changes behavior.
    """
    p = _get("RE_CACHE_POLICY", "lru").lower()
    return p if p in {"lru", "least_stale"} else "lru"


def get_moe_cpu_split() -> bool:
    """Split MoE expert compute between GPU and CPU in parallel. The GPU
    computes the experts resident in the LRU on a side CUDA stream; experts
    that are *not* resident (the LRU cannot hold them) are computed on the
    CPU against their int8 host bank while the GPU stream is busy. Results
    are joined (torch.cuda.Event barrier) and merged before the weighted
    reduce. Requires RE_MOE_OFFLOAD=1 and an int8 host bank; a no-op
    fallback to the normal path when there are no misses (or no CUDA)."""
    return _get_bool("RE_MOE_CPU_SPLIT", False)


def get_moe_shuttle_steps() -> int:
    """How many decode steps a token-hidden-state transfer (traffic A) is
    modeled as occupying the shared PCIe link. Drives the CpuShuttle mock's
    per-transfer decay for the three-way competition accounting in
    ``reserve()``; 0 disables the shuttle entirely (two-way baseline)."""
    return _get_int("RE_MOE_SHUTTLE_STEPS", 0)


# --- attention / graph ------------------------------------------------------
def get_attention_backend() -> str:
    """sdpa (default, the only Pascal-capable backend) | sdpa_paged. FreeToken
    exposes fa/fi/triton; Re exposes the two SDPA flavours and nothing more."""
    return _get("RE_ATTENTION_BACKEND", "auto").lower()


def get_enable_cuda_graph() -> bool:
    """CUDA graph capture is supported on Pascal but buys little on a
    memory-bound 6-8 GB card; off by default."""
    return _get_bool("RE_ENABLE_CUDA_GRAPH", False)


def get_host_cache_dir() -> str:
    """Where the MoE expert host banks live (defaults to $XDG_CACHE_HOME/re)."""
    import pathlib

    base = os.environ.get("XDG_CACHE_HOME") or str(pathlib.Path.home() / ".cache")
    return _get("RE_HOST_CACHE_DIR", str(pathlib.Path(base) / "re"))


__all__ = [
    "get_log_level",
    "get_compute_dtype_name",
    "get_max_num_seqs",
    "get_mem_fraction",
    "get_moe_offload_enabled",
    "get_moe_cache_size",
    "get_moe_cache_ratio",
    "get_moe_substitute",
    "get_moe_predict",
    "get_moe_predict_window",
    "get_moe_predict_headroom",
    "get_moe_prefill_overlap",
    "get_moe_cache_policy",
    "get_moe_cpu_split",
    "get_moe_shuttle_steps",
    "get_attention_backend",
    "get_enable_cuda_graph",
    "get_host_cache_dir",
]
