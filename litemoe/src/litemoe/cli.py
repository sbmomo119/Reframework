"""litemoe CLI — ``python -m litemoe.cli run --config xxx.yaml``.

Uses argparse (no click dependency).
"""

from __future__ import annotations

import argparse
import os
import sys


def _cmd_run(args: argparse.Namespace) -> int:
    from litemoe.config import LitemoeConfig
    from litemoe.runtime.executor import Executor

    cfg = LitemoeConfig.from_file(args.config)

    # CLI overrides take precedence over config file
    if args.prompt is not None:
        cfg.prompt = args.prompt
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens

    # Ensure the activations log directory exists
    act_file = cfg.metrics.activations_file
    if act_file:
        d = os.path.dirname(act_file)
        if d:
            os.makedirs(d, exist_ok=True)

    executor = Executor(cfg)
    report = executor.run(
        prompt=cfg.prompt,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print()
    print(report.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="litemoe",
        description="Lite MoE expert-offload runtime",
    )
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run a generation and print metrics")
    run_p.add_argument("--config", required=True,
                       help="Path to YAML/JSON config file")
    run_p.add_argument("--prompt", default=None,
                       help="Override prompt (config file value used if omitted)")
    run_p.add_argument("--max-new-tokens", type=int, default=None,
                       help="Override max_new_tokens")
    run_p.add_argument("--do-sample", action="store_true", default=False,
                       help="Sample instead of greedy argmax")
    run_p.add_argument("--temperature", type=float, default=1.0)
    run_p.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
