"""Per-step and aggregate metrics: TTFT, tok/s, cache hit-rate, transfer, and
the activation log (per-layer per-step routed expert sets -> JSONL)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepStat:
    step: int
    ttft_ms: Optional[float] = None     # first-token latency (prefill) ms
    step_ms: float = 0.0                # decode step ms
    hit_rate: float = 0.0               # cache hit-rate up to this step
    evictions: int = 0
    transfer_bytes: int = 0


@dataclass
class RunReport:
    """Aggregate result of one ``Executor.run`` (returned to the CLI)."""
    prompt_tokens: int = 0
    new_tokens: int = 0
    ttft_ms: float = 0.0                # first-token (prefill) latency
    total_ms: float = 0.0               # prefill + all decode steps
    decode_ms: float = 0.0              # decode-phase time only
    tok_per_s: float = 0.0              # new tokens / decode time
    hit_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    transfer_bytes: int = 0
    output_text: str = ""
    output_ids: List[int] = field(default_factory=list)
    activations_file: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"prompt_tokens   : {self.prompt_tokens}",
            f"new_tokens      : {self.new_tokens}",
            f"TTFT            : {self.ttft_ms:8.2f} ms",
            f"decode time     : {self.decode_ms:8.2f} ms",
            f"total time      : {self.total_ms:8.2f} ms",
            f"throughput      : {self.tok_per_s:8.2f} tok/s",
            f"cache hit-rate  : {self.hit_rate:8.4f}",
            f"cache hits/miss : {self.cache_hits}/{self.cache_misses}",
            f"evictions       : {self.evictions}",
            f"transferred     : {self.transfer_bytes/1048576:8.3f} MiB",
        ]
        if self.activations_file:
            lines.append(f"activations     : {self.activations_file}")
        lines.append("-" * 40)
        lines.append(f"output: {self.output_text!r}")
        return "\n".join(lines)


class Profiler:
    """Times the run and accumulates the metrics that feed a :class:`RunReport`."""

    def __init__(self, cache=None, predict_window: int = 32,
                 log_activations: bool = True,
                 activations_file: str = "./litemoe_activations.jsonl"):
        self.cache = cache
        self.predict_window = predict_window
        self.log_activations = log_activations
        self.activations_file = activations_file
        self._fh = None
        self._t_start = 0.0
        self._t_prefill = 0.0
        self._t_decode = 0.0
        self.n_new = 0
        # 接线：LRU 的 set_step / enable_log（其它策略可能没有）
        _enable = getattr(cache, "enable_log", None)
        if log_activations and _enable is not None:
            _enable()

    # -- timing ---------------------------------------------------------
    def begin(self):
        self._t_start = time.perf_counter()

    def prefill_done(self, t0: float) -> float:
        """Call once after the prefill forward returns. Returns TTFT in ms."""
        ttft = (time.perf_counter() - t0) * 1000.0
        self._t_prefill = ttft / 1000.0
        return ttft

    def mark_decode_start(self):
        self._t_decode_anchor = time.perf_counter()

    def decode_step_ms(self) -> float:
        """Time one decode step (call right after each decode forward)."""
        t = (time.perf_counter() - self._t_decode_anchor) * 1000.0
        self._t_decode_anchor = time.perf_counter()
        self._t_decode += t / 1000.0
        self.n_new += 1
        return t

    # -- activation log --------------------------------------------------
    def open_log(self):
        if self.log_activations:
            self._fh = open(self.activations_file, "w", encoding="utf-8")

    def log_activation(self, step: int, layer: int, eids: List[int]):
        # 同步 LRU 的当前 step（激活日志的 hit_mask 依赖它）
        _set = getattr(self.cache, "set_step", None)
        if _set is not None:
            _set(step)
        if self._fh is None:
            return
        self._fh.write(json.dumps(
            {"step": step, "layer": layer, "experts": [int(e) for e in eids]}) + "\n")

    def flush_log(self):
        if self._fh is not None:
            self._fh.flush()

    def close_log(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # -- report ----------------------------------------------------------
    def report(self, prompt_tokens: int, output_ids, output_text: str) -> RunReport:
        total_ms = (time.perf_counter() - self._t_start) * 1000.0
        tok_per_s = (self.n_new / self._t_decode) if self._t_decode > 0 else 0.0
        stats = self.cache.stats() if self.cache is not None else None
        hits = stats.hits if stats else 0
        misses = stats.misses if stats else 0
        total = hits + misses
        return RunReport(
            prompt_tokens=prompt_tokens,
            new_tokens=self.n_new,
            ttft_ms=self._t_prefill * 1000.0,
            total_ms=total_ms,
            decode_ms=self._t_decode * 1000.0,
            tok_per_s=tok_per_s,
            hit_rate=(hits / total) if total else 0.0,
            cache_hits=hits, cache_misses=misses,
            evictions=(stats.evictions if stats else 0),
            transfer_bytes=(stats.transfer_bytes if stats else 0),
            output_text=output_text,
            output_ids=[int(t) for t in output_ids],
            activations_file=self.activations_file if self.log_activations else None,
        )
