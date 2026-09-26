"""Command-line entry points (chat REPL)."""

_LAZY = {"main", "run_repl", "build_chat_session"}


def __getattr__(name):
    if name in _LAZY:
        from reframework.cli import chat as _chat
        return getattr(_chat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main", "run_repl", "build_chat_session"]
