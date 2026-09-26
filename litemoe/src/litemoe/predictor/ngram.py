"""N-gram predictor: token-history prediction of each layer's expert set.

The :class:`Predictor` contract is *step-indexed* (``observe(step, layer, eids)``
/ ``predict(step, layer, horizon)``) and carries no token context, so an n-gram
predictor is fed the token stream separately by the executor via
:meth:`note` (called once per generated token with its id). It then learns,
per layer, a frequency table::

    context -> expert_set -> count

where ``context`` is the tuple of the last ``window - 1`` token ids *before* the
step being predicted (a 1-gram for ``window=2``).

* :meth:`note`      — record token id at position ``step`` (fills context).
* :meth:`observe`   — record the actual expert set that fired at ``step``/``layer``.
* :meth:`predict`   — the most-frequent expert set for the context preceding
  ``step + horizon``; ``[]`` when the context is unseen (no prediction).

Because the context for predicting step ``p`` only needs tokens ``< p``, a
prediction for the *current* decode step is valid as soon as the previous
token is known (``horizon=0``, ``window=2``).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from litemoe.interface import Predictor


class NgramPredictor(Predictor):
    """Token n-gram router-choice predictor (per layer)."""

    def __init__(self, n_layers: int, window: int = 2):
        self.n_layers = n_layers
        self.window = max(2, window)          # context length = window - 1
        self._ctx_len = self.window - 1
        self._tokens: Dict[int, int] = {}     # step -> token id
        # per-layer: context tuple -> Counter(expert_set)
        self._tables: Dict[int, Dict[Tuple[int, ...], Counter]] = {L: {} for L in range(n_layers)}
        self._max_observed_step = -1

    # -- token stream feed (used by the executor) ---------------------------
    def note(self, step: int, token_id: int) -> None:
        self._tokens[step] = int(token_id)
        self._max_observed_step = max(self._max_observed_step, step)

    def _context(self, target: int) -> Optional[Tuple[int, ...]]:
        """The last ``ctx_len`` token ids immediately before ``target``.

        Returns ``None`` if any context token is not yet known (i.e. >= the
        highest noted position), so the caller treats it as "no prediction".
        """
        need = [target - i for i in range(1, self._ctx_len + 1)]
        ctx: List[int] = []
        for s in need:
            if s not in self._tokens:
                return None
            ctx.append(self._tokens[s])
        return tuple(ctx)

    # -- Predictor ABC -------------------------------------------------------
    def observe(self, step: int, layer: int, eids: Iterable[int]) -> None:
        ctx = self._context(step)
        if ctx is None or layer >= self.n_layers:
            return
        es = frozenset(int(e) for e in eids)
        self._tables[layer].setdefault(ctx, Counter())[es] += 1

    def predict(self, step: int, layer: int, horizon: int = 1) -> List[int]:
        target = step + horizon
        ctx = self._context(target)
        if ctx is None:
            return []
        table = self._tables.get(layer, {})
        if ctx not in table:
            return []
        best = max(sorted(table[ctx].items(), key=lambda kv: kv[0]), key=lambda kv: kv[1])
        return list(best[0])

    def next_use(self, layer: int, eid: int, ref_step: int = -1) -> Optional[float]:
        """Predicted steps until ``eid`` next fires (smaller = sooner).

        Scans forward over available context; returns ``None`` (unknown -> the
        cache falls back to LRU recency) when no horizon yields a prediction.
        """
        if ref_step < 0:
            ref_step = self._max_observed_step
        for h in range(1, 8):
            pred = self.predict(ref_step, layer, horizon=h)
            if pred and int(eid) in pred:
                return float(h)
        return None

    def ready(self) -> bool:
        return self._max_observed_step + 1 >= self._ctx_len

    def reset(self) -> None:
        self._tokens = {}
        self._tables = {L: {} for L in range(self.n_layers)}
        self._max_observed_step = -1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        n_ctx = sum(len(t) for t in self._tables.values())
        return (f"NgramPredictor(n_layers={self.n_layers}, window={self.window}, "
                f"contexts={n_ctx}, tokens={len(self._tokens)})")
