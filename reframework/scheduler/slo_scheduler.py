"""SLO-driven scheduler — an explicit objective function for the engine.

The stock :class:`reframework.scheduler.scheduler.Scheduler` has no objective:
it admits in arrival order, preempts the *oldest* running request, and only
cares that the KV pool does not overflow ("尽量不 OOM"). This scheduler
replaces that with a single, measurable goal:

  **P95 request latency < SLO**  (``slo_ms``, default 500).

Latency is measured per request from arrival (``add_request``) to the forward
pass in which it finishes, and reported through :meth:`SLOScheduler.report`
(the same statistic NovaServe et al. publish; their deadline-aware policy
cuts the violation rate ~2.1x on mixed loads). The policy is built to make
that objective hold under a mixed short/long load:

  1. **EDF admission** — among waiting requests that fit the KV budget, admit
     the one with the *earliest deadline* (arrival + SLO + slack). A late,
     long prompt does not block the short, urgent prompts behind it.
  2. **Short-first decode order** — running requests are ordered by
     remaining tokens (shortest first). Short requests have little KV and
     finish in one or two passes, so they drain before long requests' KV
     accumulates and squeezes the pool.
  3. **KV offload for long requests** — when the free KV pool is below
     ``kv_pressure_ratio * total_slots``, the *longest* running requests' KV
     is stashed to host memory (``on_kv_offload``) and their slots freed,
     making room for short requests to run on the GPU. Restored
     (``on_kv_restore``) when pressure clears. This is the direct
     "短请求优先在 GPU 上跑（KV 小、算得快），长请求的 KV 卸载到 CPU（省显存给短请求）"
     policy.

The scheduler stays pure logic: all KV side effects flow through callbacks
(``on_admit`` / ``on_preempt`` / ``on_finish`` / ``on_kv_offload`` /
``on_kv_restore``), exactly like the stock scheduler, so it is unit-testable
without a model or a GPU. ``record_forward_pass`` closes the loop: the
engine reports each pass's real wall time and the pass's requests, and the
scheduler books per-request latency + SLO violations + P95 statistics.

Hook contract (engine side):
  * ``on_kv_offload(req)`` — copy ``req``'s live KV to host memory, free its
    slots, and set ``req.table_idx = -1``. The req stays *running* in the
    scheduler but is excluded from forward passes while offloaded.
  * ``on_kv_restore(req)`` — allocate fresh slots and copy the KV back.
  * ``on_preempt(req)`` — recompute preemption (stock semantics). The
    scheduler never picks an offloaded req as a victim (it holds no GPU
    slots, so evicting it frees nothing).

Known limitation: on a single stream (one request at a time) the policy
cannot beat FCFS — there is no contention to arbitrate. The win shows up
under a real mixed load, which is what :meth:`report` exists to measure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from reframework.core import Req
from reframework.utils.logging import init_logger

# Reuse the stock scheduler's public vocabulary so the engine's step loop is
# unchanged (it only reads ``action`` / ``reqs`` / ``is_prefill``).
from reframework.scheduler.scheduler import Action, ScheduleDecision

logger = init_logger(__name__)

__all__ = ["SLOScheduler", "SLOStats"]


@dataclass
class SLOStats:
    """Rolling SLO accounting. All latencies in milliseconds."""

    completed: int = 0
    violations: int = 0
    _latencies_ms: List[float] = field(default_factory=list, repr=False)

    def record_finish(self, latency_ms: float, slo_ms: float) -> bool:
        """Book a finished request. Returns True if it violated its SLO."""
        self.completed += 1
        violated = latency_ms > slo_ms
        if violated:
            self.violations += 1
        if len(self._latencies_ms) < 100_000:
            self._latencies_ms.append(latency_ms)
        return violated

    def percentile(self, pct: float) -> Optional[float]:
        if not self._latencies_ms:
            return None
        s = sorted(self._latencies_ms)
        k = max(0, min(len(s) - 1, int(math.ceil(pct / 100.0 * len(s))) - 1))
        return s[k]

    @property
    def p95_ms(self) -> Optional[float]:
        return self.percentile(95)

    @property
    def p50_ms(self) -> Optional[float]:
        return self.percentile(50)

    @property
    def violation_rate(self) -> float:
        return self.violations / self.completed if self.completed else 0.0

    def as_dict(self) -> dict:
        return {
            "completed": self.completed,
            "violations": self.violations,
            "violation_rate": round(self.violation_rate, 4),
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
        }


class SLOScheduler:
    """Deadline-driven continuous-batching scheduler (objective: P95 < SLO).

    Parameters
    ----------
    slo_ms:
        Per-request latency budget (arrival -> finish). The objective is
        P95 < ``slo_ms``.
    slack_ms:
        Deadline slack on top of ``slo_ms``. The *deadline* used for ordering
        is ``arrival + slo_ms + slack_ms``: a request that already blew its
        SLO is not the one we keep evicting others for.
    max_num_seqs / available_tokens / on_admit / on_preempt / on_finish:
        Same semantics as the stock :class:`Scheduler`.
    on_kv_offload / on_kv_restore:
        Engine hooks for the long-request KV policy (see module docstring).
        ``None`` (default) disables the policy — pure EDF + short-first.
    kv_pressure_ratio:
        Offload triggers when free KV < this fraction of total pool slots;
        restore triggers when free KV >= 60% of that threshold (hysteresis).
    long_request_tokens:
        Only running requests whose prompt is at least this long are
        offload candidates (short requests never leave the GPU).
    """

    def __init__(
        self,
        *,
        slo_ms: float = 500.0,
        slack_ms: float = 250.0,
        max_num_seqs: int = 32,
        available_tokens: Optional[Callable[[], int]] = None,
        total_tokens: Optional[Callable[[], int]] = None,
        on_admit: Optional[Callable[[Req], None]] = None,
        on_preempt: Optional[Callable[[Req], None]] = None,
        on_finish: Optional[Callable[[Req], None]] = None,
        on_kv_offload: Optional[Callable[[Req], None]] = None,
        on_kv_restore: Optional[Callable[[Req], None]] = None,
        kv_pressure_ratio: float = 0.5,
        long_request_tokens: int = 256,
    ) -> None:
        self.slo_ms = float(slo_ms)
        self.slack_ms = float(slack_ms)
        self.max_num_seqs = max_num_seqs
        self._available_tokens = available_tokens or (lambda: float("inf"))
        self._total_tokens = total_tokens or (lambda: 0)
        self._on_admit = on_admit or (lambda r: None)
        self._on_preempt = on_preempt or (lambda r: None)
        self._on_finish = on_finish or (lambda r: None)
        self._on_kv_offload = on_kv_offload
        self._on_kv_restore = on_kv_restore
        self.kv_pressure_ratio = float(kv_pressure_ratio)
        self.long_request_tokens = int(long_request_tokens)

        self.waiting: List[Req] = []
        self.running: List[Req] = []
        self._arrival_ms: dict[int, float] = {}   # id(req) -> arrival clock
        self._offloaded: set = set()               # id(req) of KV-stashed running reqs
        self.stats = SLOStats()

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _now_ms() -> float:
        return time.perf_counter() * 1000.0

    def _deadline(self, req: Req) -> float:
        base = self._arrival_ms.get(id(req), self._now_ms())
        return base + self.slo_ms + self.slack_ms

    def _remaining(self, req: Req) -> int:
        return max(0, req.sampling_params.max_new_tokens - len(req.output_ids))

    def _is_long(self, req: Req) -> bool:
        return req.len >= self.long_request_tokens

    def _resident(self, req: Req) -> bool:
        return id(req) not in self._offloaded

    def _kv_pressure(self) -> bool:
        """True when free KV is tight enough to want to reclaim slots."""
        total = self._total_tokens()
        if not total:
            return False
        return self.available_tokens() < self.kv_pressure_ratio * total

    def _kv_relaxed(self) -> bool:
        total = self._total_tokens()
        if not total:
            return True
        return self.available_tokens() >= 0.6 * self.kv_pressure_ratio * total

    # ------------------------------------------------------------- state
    def add_request(self, req: Req, arrival_ms: Optional[float] = None) -> None:
        """Enqueue ``req``. ``arrival_ms`` overrides the arrival clock (tests)."""
        self._arrival_ms[id(req)] = self._now_ms() if arrival_ms is None else float(arrival_ms)
        self.waiting.append(req)

    def available_tokens(self) -> int:
        return int(self._available_tokens())

    def num_running(self) -> int:
        return len(self.running)

    def num_waiting(self) -> int:
        return len(self.waiting)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _fits(self, req: Req) -> bool:
        # prompt tokens + 1 for the first decode token (stock-scheduler parity)
        return self.available_tokens() >= req.len + 1

    # ------------------------------------------------------- KV offload policy
    def _maybe_offload_kv(self) -> None:
        """Stash the *longest* running requests' KV until pressure clears.

        The longest hold the most KV and are the ones that can be paused on
        the host with the least SLO damage: short requests must never leave
        the GPU (they are the SLO population), and among the long ones the
        ones furthest along consume the most slots.
        """
        if self._on_kv_offload is None or not self._kv_pressure():
            return
        cands = [r for r in self.running
                 if self._resident(r) and self._is_long(r)]
        cands.sort(key=lambda r: (-r.num_computed_tokens, -r.len))
        for req in cands:
            if not self._kv_pressure():
                break
            self.running.remove(req)
            self._offloaded.add(id(req))
            self._on_kv_offload(req)
            logger.info(
                "kv offload: stashed %s (computed=%d, prompt=%d) to host",
                req.rid, req.num_computed_tokens, req.len,
            )

    def _maybe_restore_kv(self) -> None:
        """Bring offloaded requests back once pressure clears (hysteresis).

        Restores only as many as fit (shortest remaining first) so a restore
        never re-triggers the offload it is escaping.
        """
        if self._on_kv_restore is None or not self._offloaded:
            return
        if not self._kv_relaxed():
            return
        cands = [r for r in self.running if id(r) in self._offloaded and not r.finished]
        cands.sort(key=lambda r: (self._remaining(r), -r.num_computed_tokens))
        restored = 0
        for req in cands:
            if not (self.available_tokens() >= req.num_computed_tokens + 1):
                continue
            self._offloaded.discard(id(req))
            self._on_kv_restore(req)
            restored += 1
        if restored:
            logger.info("kv restore: re-admitted %d request(s) from host", restored)

    # ------------------------------------------------------------- step
    def _sweep_finished(self) -> None:
        for r in list(self.running):
            # length-based completion is pure Req state, so the scheduler owns
            # it; the engine additionally sets finished for EOS.
            if not r.finished and r.remaining_tokens() <= 0:
                r.finished = True
                r.finished_reason = "length"
            if r.finished:
                self.running.remove(r)
                self._offloaded.discard(id(r))
                self._on_finish(r)
        # Finished reqs still parked in waiting (e.g. engine aborted) drop out.
        self.waiting = [r for r in self.waiting if not r.finished]

    def _try_admit(self) -> Optional[Req]:
        """EDF: admit the waiting request with the earliest deadline.

        Falls back to recompute-preemption (victim = resident running request
        with the *latest* deadline — the one whose SLO hurts least) when the
        chosen candidate does not fit but evicting would make room.
        """
        if len(self.running) >= self.max_num_seqs or not self.waiting:
            return None
        req = min(self.waiting, key=lambda r: (self._deadline(r), -r.len))
        if not self._fits(req) and self.running:
            victims = [r for r in self.running if self._resident(r)]
            if not victims:
                return None  # everything resident is offloaded; nothing to evict
            victim = max(victims, key=lambda r: self._deadline(r))
            freed = max(1, victim.num_computed_tokens)
            if self.available_tokens() + freed < req.len + 1:
                return None  # even preemption won't make room
            self.running.remove(victim)
            self._offloaded.discard(id(victim))
            self._on_preempt(victim)
            self.waiting.append(victim)
            logger.info("preempted %s (latest deadline) to admit %s", victim.rid, req.rid)
        if not self._fits(req):
            return None
        self.waiting.remove(req)
        self.running.append(req)
        self._on_admit(req)
        req.extend_len = req.len
        return req

    def schedule(self) -> ScheduleDecision:
        self._sweep_finished()
        self._maybe_offload_kv()
        admitted = self._try_admit()
        if admitted is not None:
            self._maybe_restore_kv()
            return ScheduleDecision(Action.PREFILL, [admitted])
        resident = [r for r in self.running if self._resident(r)]
        if not resident:
            # Everything running is offloaded: restore the lightest one so the
            # GPU is not left idle while work exists.
            self._maybe_restore_kv()
            resident = [r for r in self.running if self._resident(r)]
            if not resident:
                return ScheduleDecision(Action.IDLE, [])
        if resident:
            # Short-first: drain the shortest remaining work before the long
            # requests' KV accumulates.
            resident.sort(key=lambda r: (self._remaining(r), self._deadline(r)))
            for r in resident:
                r.extend_len = 1
            return ScheduleDecision(Action.DECODE, resident)
        return ScheduleDecision(Action.IDLE, [])

    # ------------------------------------------------- feedback / reporting
    def record_forward_pass(self, reqs: List[Req], wall_ms: float, is_prefill: bool) -> None:
        """Engine callback after each ``_forward_pass``: real wall time + which
        requests were in the pass. Books latency for any request that finished
        inside the pass (completion is attributed to this pass). ``wall_ms`` is
        kept for future per-step backpressure (pass time vs remaining budget).
        """
        for r in reqs:
            if r.finished:
                arrived = self._arrival_ms.get(id(r), self._now_ms())
                self.stats.record_finish(self._now_ms() - arrived, self.slo_ms)
                self._arrival_ms.pop(id(r), None)

    def report(self) -> str:
        d = self.stats.as_dict()
        return (
            f"slo: completed={d['completed']} violations={d['violations']} "
            f"rate={d['violation_rate']:.3f} p50={_fmt(d['p50_ms'])} "
            f"p95={_fmt(d['p95_ms'])} target<{self.slo_ms:.0f}ms"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SLOScheduler(slo={self.slo_ms:.0f}ms, waiting={len(self.waiting)}, "
            f"running={len(self.running)}, offloaded={len(self._offloaded)}, "
            f"free_kv={self.available_tokens()})"
        )


def _fmt(v: Optional[float]) -> str:
    return f"{v:.1f}ms" if v is not None else "n/a"
