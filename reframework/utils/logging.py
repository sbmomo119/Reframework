"""A tiny rank-aware logger (FreeToken's logger exposes ``info_rank0``; Re
mirrors that surface because the engine calls it the same way)."""

from __future__ import annotations

import logging
import os

from reframework import env as _env


def init_logger(name: str) -> "_RankLogger":
    return _RankLogger(logging.getLogger(name))


class _RankLogger:
    def __init__(self, inner: logging.Logger) -> None:
        self._inner = inner
        level = _env.get_log_level()
        if inner.level == logging.NOTSET:
            inner.setLevel(level)
            if not inner.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
                )
                inner.addHandler(handler)
        # single-process framework: rank is always 0
        self._rank = 0

    def _emit(self, method: str, rank0_only: bool, msg: str, *args, **kwargs) -> None:
        if rank0_only and self._rank != 0:
            return
        getattr(self._inner, method)(msg, *args, **kwargs)

    def debug_rank0(self, msg: str, *a, **kw) -> None:
        self._emit("debug", True, msg, *a, **kw)

    def info_rank0(self, msg: str, *a, **kw) -> None:
        self._emit("info", True, msg, *a, **kw)

    def warning_rank0(self, msg: str, *a, **kw) -> None:
        self._emit("warning", True, msg, *a, **kw)

    def error_rank0(self, msg: str, *a, **kw) -> None:
        self._emit("error", True, msg, *a, **kw)

    def debug(self, msg: str, *a, **kw) -> None:
        self._emit("debug", False, msg, *a, **kw)

    def info(self, msg: str, *a, **kw) -> None:
        self._emit("info", False, msg, *a, **kw)

    def warning(self, msg: str, *a, **kw) -> None:
        self._emit("warning", False, msg, *a, **kw)

    def error(self, msg: str, *a, **kw) -> None:
        self._emit("error", False, msg, *a, **kw)
