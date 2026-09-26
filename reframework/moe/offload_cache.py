"""Host-offload LRU expert cache — the core of the small-VRAM (1070ti) story.

A Qwen3-30B-A3B / Llama-3.1-8B-MoE class model has 128-256 experts whose weights
totally exceed 6-8 GB. FreeToken solves this with a pinned-host expert pool:
all experts live in RAM (pinned for zero-copy HtoD), and only a small LRU window
is resident in VRAM. Before a step the engine prefetches the experts that the
current token batch will route to, so the forward pass never stalls on PCIe.

Re mirrors that, in plain torch:

  * ``load``      — take per-expert weights (HF naming) and stage them in pinned
                    host memory.
  * ``prefetch``  — bring a set of experts into the VRAM LRU (evicting per
                    ``RE_CACHE_POLICY``: LRU, or Least-Stale when a routing
                    predictor is attached),
                    return their packed banks ``[N,2I,H]``/``[N,H,I]`` for
                    :func:`reframework.moe.fused.fused_experts`, plus the
                    global->local index map that function needs.

The host pool is a single contiguous *pinned* buffer (guarded: if the build has
no CUDA/pin support — e.g. a CPU-only dev box — it falls back to page-able
host memory). Each expert's slice is a view into that buffer, so the HtoD copy
on the 1070ti is a true async DMA instead of a synchronous staged copy.

On top of the reactive load-on-miss path there are two optional accelerators:

  * **substitution** — on a prefetch miss, serve the most-similar *resident*
    expert (from the offline cosine table) instead of a PCIe load.
  * **lookahead**    — :class:`Prediction` keeps a rolling window of *this
    layer's* per-step routed sets and, after each forward, prefetches
    (``reserve``) the experts it predicts the *next* step will need. The
    lookahead depth ``k`` is not fixed: it is the number of ranked predicted
    experts that both (a) fit in the LRU's evictable room and (b) can be moved
    over PCIe within the measured inter-forward window at the measured
    bandwidth. See :meth:`ExpertOffloadCache.reserve`.

``moe_layer.forward`` drives the predictor: after routing it records this step's
global routed set via :meth:`Prediction.record_step`, which triggers the next
step's ``reserve``. Prefill batches (large token count) reset the history rather
than look ahead, since their big routed set is not representative of a decode
step.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch

from reframework.utils import init_logger

logger = init_logger(__name__)


def _sym_int8_quantize(w: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Symmetric per-tensor int8 quantize (host, fp32).

    Returns ``(w_q [*,] int8 contiguous, scale)`` with ``w ~= w_q * scale``.
    The scale is chosen so the largest-magnitude weight maps to +/-127 (never
    saturates at the int8 boundary). A tensor whose range is ~0 (e.g. an
    uninit/empty expert) gets scale 1.0 and all-zero codes so it dequantizes
    back to ~0 instead of blowing up on a zero divide.
    """
    amax = w.abs().max().float()
    if amax <= 0:
        return torch.zeros_like(w, dtype=torch.int8), 1.0
    scale = amax / 127.0
    wq = (w.float() / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    return wq, float(scale)


@dataclass
class OffloadStats:
    """Counters for observability / benchmarking."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    transfers_bytes: int = 0
    subs: int = 0  # experts served by a resident similar-expert substitute (no PCIe load)
    predicts: int = 0  # lookahead expert-loads the frequency predictor pulled in
    inflight_bytes: int = 0  # total bytes deducted from the reserve budget by concurrent PCIe traffic

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.transfers_bytes = 0
        self.subs = 0
        self.predicts = 0
        self.inflight_bytes = 0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "subs": self.subs,
            "predicts": self.predicts,
            "inflight_bytes": self.inflight_bytes,
            "transfer_mb": self.transfers_bytes / (1024 * 1024),
        }


class Prediction:
    """Rolling per-step routing history for one MoE layer's offload cache.

    The model routes every token through this layer's gate, so the *routed* set
    for each forward step is the ground truth we can observe. We keep a rolling
    window of the last ``window`` *decode* steps' routed sets and predict the
    next step by historical activation frequency: an expert activated in the
    recent window is likely to fire again, and the more recently / often it
    fired, the higher its rank.

    This is the "simplest" predictor the design calls for (frequency over the
    last N tokens). It is deliberately routing-kernel-agnostic — it only needs
    the routed *set*, not the router activations — so a learned router-pattern
    MLP would replace just :meth:`predict_next` and plug in here.

    Parameters
    ----------
    window:
        Number of past decode steps to consider (``N``). Bigger = smoother but
        slower to adapt; small = reacts fast.
    """

    def __init__(self, cache: "ExpertOffloadCache", window: int = 32) -> None:
        self.cache = cache
        self.window = max(1, int(window))
        self._history: Deque[frozenset] = deque(maxlen=self.window)
        # Per-expert recency: how many decode steps ago it last routed (0 = just now).
        self._age: Dict[int, int] = {}

    # -------------------------------------------------------------- bookkeeping
    def reset(self) -> None:
        """Drop all history (start of a new sequence / prefill)."""
        self._history.clear()
        self._age.clear()

    def record_step(self, routed: Sequence[int], *, is_prefill: bool, forward_ms: float,
                    inflight_bytes: float = 0.0) -> None:
        """Record this step's routed set, then prefetch the predicted next set.

        ``routed`` is the set of global expert ids this layer's gate selected for
        the tokens just forwarded (unique). ``is_prefill`` marks a prefill batch:
        its large routed set would poison a decode-length frequency estimate, so
        a prefill step resets the history (new sequence) and does not look ahead.
        ``forward_ms`` is the measured inter-forward window — the time available
        to hide the lookahead transfer before the next step needs these experts.

        ``inflight_bytes`` is the bytes of *other* PCIe traffic (token
        hidden-state transfers, traffic A) still occupying the shared link when
        this step's ``reserve`` runs. It is passed straight to
        :meth:`ExpertOffloadCache.reserve`, which subtracts it from the window
        budget — the three-way-competition deduction. ``0.0`` (the default)
        disables it, reproducing the two-way baseline exactly.

        After a decode step the next set is predicted by :meth:`predict_next`
        and reserved via :meth:`ExpertOffloadCache.reserve` (dynamic ``k``).
        """
        routed_set = frozenset(int(i) for i in routed)

        if is_prefill:
            # New sequence: the prefill batch's huge routed set is not a decode
            # step — drop stale decode history and do not look ahead from it.
            self.reset()
            return

        # Decode step: age every previously-seen expert by one, refresh routed to 0.
        for e in list(self._age):
            self._age[e] += 1
        for e in routed_set:
            self._age[e] = 0
        self._history.append(routed_set)

        # predict_next() already filters out resident experts, so the very first
        # decode step (whose just-used set is all resident) naturally predicts
        # nothing; no special-casing needed.
        predicted = self.predict_next()
        if predicted:
            self.cache.reserve(
                predicted,
                forward_ms=forward_ms,
                protect_current=routed,
                inflight_bytes=inflight_bytes,
            )

    # -------------------------------------------------------------- prediction
    def predict_next(self, limit: Optional[int] = None) -> List[int]:
        """Rank non-resident experts by recent activation frequency.

        Score per expert = number of steps in the window it was routed (a simple
        frequency), with a recency bonus so a just-seen expert outranks a stale
        one at the same raw frequency. Returns the top ``limit`` ranked experts
        (or all, if ``limit`` is None) that are *not* already resident in the
        LRU — exactly the ones the lookahead would need to pull in.

        The frequency is over the whole window, not one step, so a one-off
        spurious activation does not dominate.
        """
        freq: Dict[int, int] = {}
        for s in self._history:
            for e in s:
                freq[e] = freq.get(e, 0) + 1
        if not freq:
            return []
        # score = freq (primary) + recency bonus (secondary). Higher = predict first.
        scored = [
            (e, freq[e] * 1000 + (self.window - self._age.get(e, self.window)))
            for e in freq
        ]
        scored.sort(key=lambda p: (-p[1], p[0]))
        out: List[int] = []
        for e, _ in scored:
            if e in self.cache._lru:  # already resident — nothing to prefetch
                continue
            out.append(e)
            if limit is not None and len(out) >= limit:
                break
        return out


@dataclass
class ExpertOffloadCache:
    """Pinned-host expert pool + a small VRAM LRU window.

    Parameters
    ----------
    device:
        VRAM device the LRU lives on (``cuda:0`` on the 1070ti, ``cpu`` for a
        test where "VRAM" is just host memory).
    lru_capacity:
        Max number of expert *slots* (one slot = one expert's w1+w2) kept in
        VRAM. Sized by the engine from the VRAM budget.
    dtype:
        Compute dtype of the resident experts (fp16 storage / fp32 compute is
        handled downstream in ``fused_experts``).
    """

    device: torch.device
    lru_capacity: int
    dtype: torch.dtype = torch.float16
    _host: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list, repr=False)
    _host_buf: Optional[torch.Tensor] = field(default=None, repr=False)
    _pinned: bool = field(default=False, repr=False)
    _lru: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]]" = field(
        default_factory=OrderedDict, repr=False
    )
    stats: OffloadStats = field(default_factory=OffloadStats, repr=False)
    # Experts the lookahead reserved for the *next* decode step (see
    # :meth:`reserve`). Soft-protected by :meth:`prefetch`'s eviction so the
    # current step's load doesn't evict the very experts the next step needs.
    _next_reserve: frozenset = field(default_factory=frozenset, repr=False, compare=False)
    # Offline cosine similarity between this layer's experts: [E, E] float,
    # sim[a, b] = cos(expert_a, expert_b). None -> normal load-on-miss behavior.
    sim_table: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)
    # Rolling routing history that drives the next-step lookahead (see
    # :class:`Prediction`). None until the engine wires one up (opt-in).
    _predictor: Optional[Prediction] = field(default=None, repr=False, compare=False)
    # Measured host->VRAM transfer rate in GB/s (None until measured / no CUDA).
    # Used as the bandwidth cap in :meth:`reserve`.
    _bw_gb_s: Optional[float] = field(default=None, repr=False, compare=False)
    # Fraction of the measured PCIe budget :meth:`reserve` is willing to spend
    # (default 0.9) — the remaining ~10% absorbs bandwidth variance so the
    # lookahead transfer still completes inside the inter-forward window.
    _bw_headroom: float = field(default=0.9, repr=False, compare=False)
    # CPU/GPU parallel expert split (RE_MOE_CPU_SPLIT). When on, the LRU keeps
    # only a bounded "resident" subset and the routed experts it can't hold are
    # computed on the CPU against this int8 bank instead of stalling on a PCIe
    # load. See :meth:`quantize_int8_host` and
    # ``reframework.moe.fused.fused_experts_cpu_split``.
    _cpu_split: bool = field(default=False, repr=False, compare=False)
    # Per-expert symmetric-int8 host bank: parallel to ``_host``.
    # _int8_host[i] = (w1_q [2I,H] int8, w2_q [H,I] int8, s1, s2).
    _int8_host: List[Tuple[torch.Tensor, torch.Tensor, float, float]] = field(
        default_factory=list, repr=False
    )

    # ----------------------------------------------------------------- factory

    @classmethod
    def from_config(cls, cfg, *, device=None, lru_capacity: Optional[int] = None) -> "ExpertOffloadCache":
        """Build from a ``reframework.models.ModelConfig``.

        ``device`` defaults to the config's device; ``lru_capacity`` defaults
        to the env override (``RE_MOE_CACHE_SIZE``) or 0 = auto, which the
        engine resolves against the free-VRAM budget via
        :meth:`ModelConfig.size_moe_lru` before the first step.
        """
        import torch  # local: keep this module importable on meta-init paths
        from reframework import env

        return cls(
            device=torch.device(device or cfg.device),
            lru_capacity=lru_capacity if lru_capacity is not None else env.get_moe_cache_size(),
            dtype=cfg.dtype,
        )

    # ------------------------------------------------------------------ setup

    def load(
        self,
        experts: Sequence[Dict[str, torch.Tensor]],
    ) -> None:
        """Stage per-expert weights into (pinned) host memory.

        ``experts[i]`` maps ``"w1" -> [2I,H]`` and ``"w2" -> [H,I]``. The caller
        (``moe_layer.build_offload_cache``) is responsible for the HF->bank
        naming; here we only pin and convert dtype.
        """
        w1s = [e["w1"].to(dtype=self.dtype).contiguous() for e in experts]
        w2s = [e["w2"].to(dtype=self.dtype).contiguous() for e in experts]
        n1 = sum(w.numel() for w in w1s)
        n2 = sum(w.numel() for w in w2s)

        # One contiguous host pool so the HtoD copy is a single DMA-able region.
        # pin_memory() needs a CUDA build; on a CPU-only dev box it raises, so
        # fall back to page-able memory (correct, just synchronous copies).
        try:
            buf = torch.empty(n1 + n2, dtype=self.dtype).pin_memory()
            self._pinned = True
        except (RuntimeError, AssertionError):
            buf = torch.empty(n1 + n2, dtype=self.dtype)
            self._pinned = False
        self._host_buf = buf

        off = 0
        self._host = []
        for w1, w2 in zip(w1s, w2s):
            v1 = buf[off : off + w1.numel()].view_as(w1)
            v1.copy_(w1)
            off += w1.numel()
            v2 = buf[off : off + w2.numel()].view_as(w2)
            v2.copy_(w2)
            off += w2.numel()
            self._host.append((v1, v2))
        del w1s, w2s
        self._lru = OrderedDict()
        self.stats.reset()
        logger.info(
            "ExpertOffloadCache staged %d experts (%s) in %s host pool; LRU cap=%d",
            len(self._host), self.dtype, "pinned" if self._pinned else "pageable",
            self.lru_capacity,
        )

    @property
    def num_experts(self) -> int:
        return len(self._host)

    @property
    def cpu_split(self) -> bool:
        """True once the int8 host bank is built and the split is armed."""
        return self._cpu_split

    def resident_set(self) -> frozenset:
        """Global expert ids currently resident in the LRU (the "hits")."""
        return frozenset(self._lru.keys())

    def set_sim_table(self, sim: torch.Tensor) -> None:
        """Attach this layer's [E, E] cosine-similarity table (row = query
        expert, col = candidate). When set, a prefetch miss can be served by
        the most-similar *resident* expert instead of a PCIe load."""
        E = len(self._host)
        sim = torch.as_tensor(sim, dtype=torch.float32)
        if tuple(sim.shape) != (E, E):
            raise ValueError(f"sim_table shape {tuple(sim.shape)} != [E,E]=[{E},{E}]")
        self.sim_table = sim

    def attach_predictor(self, window: int = 32) -> None:
        """Wire a :class:`Prediction` history buffer for this layer.

        The engine calls this once per offload layer when lookahead is enabled
        (``RE_MOE_PREDICT``). After that, ``moe_layer.forward`` feeds each step's
        routed set via :meth:`Prediction.record_step` and the predictor pulls the
        predicted next set into the LRU ahead of the step that needs it.
        """
        from reframework import env  # local: keep this module import light

        self._predictor = Prediction(self, window=window)
        self._bw_headroom = max(0.05, min(1.0, env.get_moe_predict_headroom()))
        self.measure_bandwidth()

    def quantize_int8_host(self) -> None:
        """Build the per-expert symmetric-int8 host bank for the CPU/GPU split.

        Called once (after :meth:`load`) when ``RE_MOE_CPU_SPLIT`` is on. Each
        expert's fp16 ``w1``/``w2`` is quantized per-tensor to int8 with a
        symmetric scale, giving a *host* bank ``_int8_host`` the CPU can run
        against without a PCIe round-trip. The int8 codes are ~half the bytes
        of the fp16 host pool, and the CPU dequantizes on the fly (see
        ``reframework.moe.fused.fused_experts_cpu_split``).
        """
        if not self._host:
            return
        self._int8_host = []
        for w1, w2 in self._host:
            w1_q, s1 = _sym_int8_quantize(w1)
            w2_q, s2 = _sym_int8_quantize(w2)
            self._int8_host.append((w1_q, w2_q, s1, s2))
        self._cpu_split = True
        logger.info(
            "CPU/GPU split int8 host bank built for %d experts (sym per-tensor)",
            len(self._int8_host),
        )

    def cpu_banks(self, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int]]:
        """Packed int8 banks for the *CPU* side of the split.

        Returns ``(w1_q [N,2I,H] int8, w2_q [N,H,I] int8, scales1 [N],
        scales2 [N], local_index)`` for the (non-resident) experts ``indices``,
        in ascending order — the exact shapes
        ``reframework.moe.fused.fused_experts_cpu_split`` consumes.
        """
        uniq = sorted(set(int(i) for i in indices))
        # 硬限制：一次最多 pack N 个专家，避免 _pack 堆叠 OOM
        import os as _os
        _max_pack = int(_os.environ.get("RE_MAX_PACK", "4"))
        if len(uniq) > _max_pack:
            uniq = uniq[:_max_pack]
        local_index = {idx: k for k, idx in enumerate(uniq)}
        if not uniq:
            H = self._host[0][0].shape[1]
            I = self._host[0][1].shape[0]
            w1q = torch.empty((0, 2 * I, H), dtype=torch.int8)
            w2q = torch.empty((0, H, I), dtype=torch.int8)
            return w1q, w2q, torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.float32), local_index
        w1q = torch.stack([self._int8_host[i][0] for i in uniq], dim=0)
        w2q = torch.stack([self._int8_host[i][1] for i in uniq], dim=0)
        s1 = torch.tensor([self._int8_host[i][2] for i in uniq], dtype=torch.float32)
        s2 = torch.tensor([self._int8_host[i][3] for i in uniq], dtype=torch.float32)
        return w1q, w2q, s1, s2, local_index

    def measure_bandwidth(self) -> Optional[float]:
        """One-time host->VRAM copy benchmark -> GB/s, stored on ``_bw_gb_s``.

        Returns None (and leaves ``_bw_gb_s`` unset) when there is no CUDA
        (CPU dev box) — in that case :meth:`reserve` caps by LRU room only.
        Measuring once is enough: steady-state PCIe bandwidth is stable, while
        the *window* (per-step inter-forward time) is what varies at runtime and
        is supplied to :meth:`reserve` on every call.
        """
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            return None
        if not str(self.device).startswith("cuda"):
            return None
        try:
            n = 1 << 20  # 1M elements
            src = torch.empty(n, dtype=self.dtype)
            _ = src.to(self.device, non_blocking=self._pinned)  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = 3
            for _ in range(reps):
                _ = src.to(self.device, non_blocking=self._pinned)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if dt > 0:
                self._bw_gb_s = (n * self.dtype.itemsize * reps) / dt / 1e9
            del src
            torch.cuda.empty_cache()
            return self._bw_gb_s
        except Exception as exc:  # pragma: no cover - benchmarking is best-effort
            logger.warning("PCIe bandwidth measure failed: %s", exc)
            return None

    def reset_lru(self) -> None:
        """Drop all resident experts and prediction history (start of a sequence)."""
        self._lru = OrderedDict()
        self._next_reserve = frozenset()
        if self._predictor is not None:
            self._predictor.reset()

    # ------------------------------------------------------------------ access

    def prefetch(self, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
        # 过滤掉哨兵 id（熵感知用 first_id 填充超出的位置）
        indices = [i for i in indices if i >= 0]
        """Bring ``indices`` into the VRAM LRU and return their packed banks.

        Returns ``(w1, w2, local_index)`` where ``w1`` is ``[N,2I,H]``, ``w2``
        is ``[N,H,I]`` (N = len of the unique indices, in ascending order) and
        ``local_index`` maps each *global* expert id to its row ``0..N-1`` in
        the packed banks — exactly what ``fused_experts`` needs to group tokens.
        """
        uniq = sorted(set(int(i) for i in indices))
        _hits = sum(1 for i in uniq if i in self._lru)
        _misses = len(uniq) - _hits
        logger.debug(
            "[offload] prefetch: requested=%d unique=%d hit=%d miss=%d lru_size=%d",
            len(indices), len(uniq), _hits, _misses, len(self._lru),
        )
        for idx in uniq:
            if idx not in self._lru:
                if self._substitute(idx) is None:
                    self._fill(idx)
            else:
                self.stats.hits += 1
            self._lru.move_to_end(idx)  # mark most-recently used
        # Two-tier protection: this step's set is *hard* (never evicted), the
        # next-step lookahead set is *soft* (evicted only after all stale
        # experts are gone) so the current load doesn't blow out the very
        # experts the next step needs.
        self._evict(uniq, self._next_reserve - set(uniq))
        return self._pack(uniq)

    def split_prefetch(
        self, indices: Sequence[int]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, Dict[int, int]], List[int]]:
        """CPU/GPU expert split: dispatch the routed set by LRU residency.

        Returns ``(gpu_banks, miss_ids)`` where ``gpu_banks = (w1, w2,
        local_index)`` is the packed bank of the **resident** ("hit") experts
        already in the LRU (no PCIe load this step — the GPU computes them), and
        ``miss_ids`` is the list of routed experts **not** resident, to be
        computed on the CPU against the int8 host bank (see
        :meth:`cpu_banks` / ``fused_experts_cpu_int8``). This is the dispatch
        the parallel branch needs: "GPU computes its LRU experts, CPU computes
        the misses", so a step never stalls on a PCIe load of a miss.

        The misses are reactively loaded into the LRU (via ``_fill``) *after*
        the split is captured, so any routed expert becomes a GPU hit on the
        next step — the LRU grows to track hot experts, independent of the
        lookahead predictor. Misses are loaded directly (no ``_substitute``),
        because the CPU computes them exactly rather than serving an
        approximate similar expert.
        """
        uniq = sorted(set(int(i) for i in indices))
        hits = [i for i in uniq if i in self._lru]
        misses = [i for i in uniq if i not in self._lru]
        # Recency for the hits (they stay resident); the GPU reads their rows.
        for i in hits:
            self._lru.move_to_end(i)
            self.stats.hits += 1
        # Capture the GPU bank BEFORE mutating residency further.
        if hits:
            gpu_banks = self._pack(hits)
        else:
            w1r, w2r = self._host[0]
            H, I = w1r.shape[1], w2r.shape[0]
            gpu_banks = (
                torch.empty((0, 2 * I, H), dtype=self.dtype, device=self.device),
                torch.empty((0, H, I), dtype=self.dtype, device=self.device),
                {},
            )
        # NOTE: the misses are *not* loaded here. The parallel forward calls
        # :meth:`reactive_load` after the compute/merge, so the PCIe HtoD copy
        # lands in the inter-forward window instead of stalling this step's
        # GPU compute (which is precisely the stall this split removes).
        return gpu_banks, misses

    def reactive_load(self, indices: Sequence[int]) -> None:
        """Bring ``indices`` resident for the *next* step (split path only).

        Called by the CPU/GPU parallel forward after the two halves are
        merged, so the HtoD copy of this step's miss experts overlaps the
        inter-forward gap rather than blocking this step's compute. Experts are
        loaded directly (no :meth:`_substitute` — the CPU computed them exactly
        this step, and a substitute row would corrupt next step's GPU result).
        """
        uniq = sorted(set(int(i) for i in indices))
        for i in uniq:
            if i not in self._lru:
                self._fill(i)  # _fill counts the miss itself
        self._evict(uniq, self._next_reserve - set(uniq))

    def reserve(
        self,
        predicted: Sequence[int],
        *,
        forward_ms: float = 0.0,
        bandwidth_gb_s: Optional[float] = None,
        protect_current: Sequence[int] = (),
        inflight_bytes: float = 0.0,
    ) -> int:
        """Pull the predicted next-step experts into the LRU (the lookahead).

        ``predicted`` is the ranked list from :meth:`Prediction.predict_next`
        (highest activation frequency first, all non-resident). The dynamic
        ``k`` — how many we actually load — is bounded by:

          * **bandwidth** — the number of ranked experts whose combined bytes
            fit in the ``forward_ms`` inter-forward window at
            ``bandwidth_gb_s`` (defaults to the one-time measured rate). We load
            in ranked order so the most-likely experts get the scarce bandwidth
            first, and stop at the first one that no longer fits.
          * **LRU room** — we evict *stale* experts (neither the just-used
            ``protect_current`` set nor the predicted set) to make room, so the
            lookahead keeps working even when the LRU is full. We stop as soon
            as no stale expert remains to evict (we never evict a just-used or
            predicted expert here).

        ``inflight_bytes`` models the *three-way* PCIe competition: token
        hidden-state transfers (traffic A) and SSD prefetches (traffic C) share
        the same link as the lookahead (traffic B). Subtracting the bytes
        already in flight from the window budget is exactly the paper's
        ``(bw - bw_A) * t`` form expressed as a byte budget, so the lookahead
        only reserves the bandwidth the other flows haven't already consumed.
        ``0.0`` (the default) disables the deduction — the two-way baseline.

        Returns the number of experts actually loaded (the effective ``k``).
        """
        if not predicted or self.lru_capacity <= 0:
            return 0
        # Keep both the just-used set and the predicted set resident; evict only
        # experts in neither to make room for the lookahead.
        protect = set(int(i) for i in protect_current) | set(int(i) for i in predicted)
        bw = bandwidth_gb_s if bandwidth_gb_s is not None else self._bw_gb_s
        budget = None
        if forward_ms > 0 and bw and bw > 0:
            # Leave headroom (default ~10%) so the transfer still fits the
            # window despite PCIe bandwidth variance.
            budget = bw * 1e9 * (forward_ms / 1000.0) * self._bw_headroom
            # Three-way competition: subtract concurrent PCIe traffic (token
            # transfers A + SSD prefetch C) from the lookahead's window budget.
            if inflight_bytes and inflight_bytes > 0:
                budget = max(0.0, budget - inflight_bytes)
                self.stats.inflight_bytes += int(inflight_bytes)
        n_loaded = 0
        loaded: List[int] = []
        for idx in (int(i) for i in predicted):  # ranked order = most-likely first
            if idx in self._lru:
                continue  # already resident (e.g. just used) — nothing to load
            w1, w2 = self._host[idx]
            b = (w1.numel() + w2.numel()) * self.dtype.itemsize
            if budget is not None:
                if b > budget and n_loaded > 0:
                    break  # can't fit this one in the window; ranked => stop
                budget -= b
            self._fill(idx)  # appends at MRU end, counts miss + transfer bytes
            self.stats.predicts += 1
            n_loaded += 1
            loaded.append(idx)
            self._evict(protect)  # free stale room so we stay at/below capacity
            if len(self._lru) > self.lru_capacity:
                # No stale expert left to evict; further loads would exceed cap.
                break
        # Remember what we pulled in so prefetch's eviction (next step) soft-
        # protects it — this is what makes the lookahead actually stick.
        self._next_reserve = frozenset(loaded)
        return n_loaded

    def _fill(self, idx: int) -> None:
        w1, w2 = self._host[idx]
        # host -> VRAM; count the *bytes* moved (numel * element_size)
        w1v = w1.to(device=self.device, non_blocking=self._pinned)
        w2v = w2.to(device=self.device, non_blocking=self._pinned)
        self.stats.transfers_bytes += (w1.numel() + w2.numel()) * self.dtype.itemsize
        self.stats.misses += 1
        self._lru[idx] = (w1v, w2v)

    def _evict(
        self, hard_protect: Sequence[int], soft_protect: Sequence[int] = ()
    ) -> None:
        """Evict victims down to ``lru_capacity`` (policy: ``RE_CACHE_POLICY``).

        Two-tier protection (both policies):

        * ``hard_protect`` (this step's routed set) is *never* evicted.
        * ``soft_protect`` (the next-step lookahead set) is evicted only after
          every stale expert is gone — so the current step's load prefers to
          evict stale experts rather than the experts the *next* step needs.
        * everything else is a stale candidate.

        Victim selection within a tier depends on the policy:

        * ``lru`` (default) — stale candidates are evicted LRU-first (oldest
          first); the same for the soft tier. This is the original behavior.
        * ``least_stale`` — the candidate *least likely to be routed again* is
          evicted first: highest routing staleness, i.e. the most decode steps
          since its last activation (``Prediction._age``; an expert never seen
          in the window counts as maximally stale). Ties fall back to LRU
          order. When no predictor is attached there is no routing signal, so
          ``least_stale`` degrades to plain LRU (identical to the default).
        """
        from reframework import env  # local: keep this module import light

        hard = set(int(i) for i in hard_protect)
        soft = set(int(i) for i in soft_protect) - hard
        predictor = self._predictor
        if env.get_moe_cache_policy() == "least_stale" and predictor is not None:
            age = predictor._age
            window = predictor.window

            def victim_of(cands: List[int]) -> Optional[int]:
                # cands are in LRU order (oldest first); pick the most stale,
                # ties -> oldest, by scanning LRU-first with a strict '>' so
                # the first (oldest) max wins.
                best: Optional[int] = None
                best_age = -1
                for i in cands:
                    a = age.get(i, window)  # never routed => maximally stale
                    if a > best_age:
                        best, best_age = i, a
                return best
        else:
            def victim_of(cands: List[int]) -> Optional[int]:
                return cands[0] if cands else None  # LRU-first

        while len(self._lru) > self.lru_capacity:
            stale = [i for i in self._lru if i not in hard and i not in soft]
            victim = victim_of(stale)            # prefer stale
            if victim is None:                   # else give up a soft entry
                victim = victim_of([i for i in self._lru if i in soft])
            if victim is None:
                return  # only hard-protected entries remain; can't evict
            (ow1, ow2) = self._lru.pop(victim)
            del ow1, ow2  # frees the VRAM tensors
            self.stats.evictions += 1

    def _substitute(self, idx: int) -> Optional[int]:
        """If ``idx`` is not resident and a sim table is set, pick the
        most-similar *resident* expert and place its weights into ``idx``'s row.

        Returns the substitute's global id, or None to fall back to a normal
        ``_fill(idx)`` (no resident expert, no table, or only ``idx`` itself).
        """
        if self.sim_table is None or idx in self._lru:
            return None
        residents = [i for i in self._lru if i != idx]
        if not residents:
            return None
        row = self.sim_table[idx].to(self.device)
        # mask self (already excluded) and any non-resident to -inf
        cand = torch.full((self.num_experts,), float("-inf"), dtype=row.dtype, device=row.device)
        cand[residents] = row[residents]
        best = int(cand.argmax().item())
        self.stats.subs += 1
        self._lru[idx] = self._lru[best]  # row now holds the substitute's weights
        return best

    def _pack(
        self, uniq: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
        local_index = {idx: k for k, idx in enumerate(uniq)}
        # stack (not cat): each resident expert is [2I,H] / [H,I]; stacking adds
        # the leading batch dim -> [N,2I,H] / [N,H,I] that fused_experts expects.
        w1 = torch.stack([self._lru[i][0] for i in uniq], dim=0)  # [N,2I,H]
        w2 = torch.stack([self._lru[i][1] for i in uniq], dim=0)  # [N,H,I]
        return w1, w2, local_index

    # ------------------------------------------------------------------ stats

    def report(self) -> str:
        s = self.stats.as_dict()
        return (
            f"offload: hit={s['hits']} miss={s['misses']} sub={s['subs']} "
            f"pred={s['predicts']} evict={s['evictions']} transfer={s['transfer_mb']:.1f}MB"
        )


__all__ = ["ExpertOffloadCache", "OffloadStats", "Prediction"]
