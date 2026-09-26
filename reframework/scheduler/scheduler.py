"""Minimal continuous-batching scheduler.

Owns the *waiting* / *running* split and the per-step scheduling decision,
keeping the engine's KV side effects (row alloc, slot alloc, free) behind
small callbacks so the scheduler is pure logic and can be unit-tested without
a model or a GPU.

Decision per ``schedule()`` call (minimal, whole-prompt prefill):

  1. Sweep ``running`` for finished requests (the engine frees their KV).
  2. If a waiting request fits the KV budget (and we are under
     ``max_num_seqs``), admit it and return a PREFILL step for it. If it does
     not fit and preemption would make room, evict the oldest running request
     (its KV is freed via ``on_preempt``) and retry.
  3. Otherwise run a DECODE step over every running request (each extends by
     one token).
  4. If there is nothing to do, return IDLE.

The engine consumes a :class:`ScheduleDecision` by running its ``reqs``
(whose ``extend_len`` is already set) through ``_forward_pass``.

Known limitation (minimal by design): preemption has no anti-starvation /
aging. Under heavy KV pressure a long waiting request can repeatedly evict
the same running request. A real engine adds request priority + an aging
counter; that is deliberately out of scope here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from reframework.core import Req
from reframework.utils.logging import init_logger

logger = init_logger(__name__)

__all__ = ["Action", "ScheduleDecision", "Scheduler"]


class Action(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    IDLE = "idle"


@dataclass
class ScheduleDecision:
    action: Action
    reqs: List[Req] = field(default_factory=list)

    @property
    def is_prefill(self) -> bool:
        return self.action is Action.PREFILL


class Scheduler:
    """KV-budget-aware continuous-batching scheduler.

    Parameters
    ----------
    max_num_seqs:
        Upper bound on concurrently running requests.
    available_tokens:
        Zero-arg callable returning the number of free KV tokens. Defaults to
        unbounded (useful for tests); the engine wires
        ``cache_manager.available_tokens``.
    on_admit / on_preempt / on_finish:
        Engine-side hooks invoked when a request enters the running set, is
        evicted (recompute preemption), or finishes. The engine uses these to
        allocate/free KV. ``None`` by default (pure-logic mode).
    """

    def __init__(
        self,
        *,
        max_num_seqs: int = 32,
        available_tokens: Optional[Callable[[], int]] = None,
        on_admit: Optional[Callable[[Req], None]] = None,
        on_preempt: Optional[Callable[[Req], None]] = None,
        on_finish: Optional[Callable[[Req], None]] = None,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self._available_tokens = available_tokens or (lambda: float("inf"))
        self._on_admit = on_admit or (lambda r: None)
        self._on_preempt = on_preempt or (lambda r: None)
        self._on_finish = on_finish or (lambda r: None)
        self.waiting: deque[Req] = deque()
        self.running: List[Req] = []

    # ------------------------------------------------------------- state
    def add_request(self, req: Req) -> None:
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
        # prompt tokens + 1 for the first decode token
        return self.available_tokens() >= req.len + 1

    def _sweep_finished(self) -> None:
        for r in list(self.running):
            # length-based completion is pure Req state, so the scheduler owns
            # it; the engine additionally sets finished for EOS.
            if not r.finished and r.remaining_tokens() <= 0:
                r.finished = True
                r.finished_reason = "length"
            if r.finished:
                self.running.remove(r)
                self._on_finish(r)

    def _try_admit(self) -> Optional[Req]:
        if len(self.running) >= self.max_num_seqs or not self.waiting:
            return None
        req = self.waiting[0]
        if not self._fits(req) and self.running:
            # Not enough room: if evicting the oldest running request frees
            # enough KV, do a recompute-preemption and retry.
            victim = self.running[0]
            freed = max(1, victim.num_computed_tokens)
            if self.available_tokens() + freed < req.len + 1:
                return None  # even preemption won't make room
            self.running.remove(victim)
            self._on_preempt(victim)
            self.waiting.appendleft(victim)
            logger.info("preempted %s to admit %s", victim.rid, req.rid)
        if not self._fits(req):
            return None
        self.waiting.remove(req)
        self.running.append(req)
        self._on_admit(req)
        req.extend_len = req.len
        return req

    # ------------------------------------------------------------- step
    def schedule(self) -> ScheduleDecision:
        self._sweep_finished()
        admitted = self._try_admit()
        if admitted is not None:
            return ScheduleDecision(Action.PREFILL, [admitted])
        if self.running:
            for r in self.running:
                r.extend_len = 1
            return ScheduleDecision(Action.DECODE, list(self.running))
        return ScheduleDecision(Action.IDLE, [])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Scheduler(waiting={len(self.waiting)}, running={len(self.running)}, "
            f"free_kv={self.available_tokens()}, max_seqs={self.max_num_seqs})"
        )
