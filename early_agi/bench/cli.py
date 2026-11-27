"""CLI stress runner for the offline/live A/B repair benchmark."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import DEFAULT_BASE_URL
from .routers import OpenAICompatiblePatchRouter
from .runner import run_ab_benchmark

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="early-agi-bench",
        description="Run local multi-file repair A/B curves and a repeat-task reflex stress test.",
    )
    parser.add_argument("--tasks", type=int, default=10, help="sequential tasks (default: 10)")
    parser.add_argument("--k", type=int, default=4, help="maximum patch attempts per task (default: 4)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="local pytest and optional live-router timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--router",
        choices=("offline", "openai"),
        default="offline",
        help="patch router (default: offline; no network)",
    )
    parser.add_argument("--allow-live", action="store_true", help="explicitly permit OpenAI-compatible calls")
    parser.add_argument(
        "--base-url",
        default=os.getenv("EARLY_AGI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL for --router openai",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EARLY_AGI_FAST_MODEL", "fast-executor"),
        help="model name for --router openai",
    )
    parser.add_argument(
        "--api-key-env",
        default="EARLY_AGI_API_KEY",
        help="environment variable holding an optional API key (never logged)",
    )
    parser.add_argument("--output", "-o", help="write JSON to this path instead of stdout")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    return parser


def _dump(payload: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def _write(text: str, output: str | None) -> None:
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.router == "openai" and not args.allow_live:
        parser.error("--router openai requires --allow-live")

    router = None
    if args.router == "openai":
        router = OpenAICompatiblePatchRouter(
            base_url=args.base_url,
            model=args.model,
            api_key=os.getenv(args.api_key_env, ""),
            timeout_s=args.timeout,
            enabled=True,
        )
    try:
        report = run_ab_benchmark(
            router=router,
            k=args.k,
            sequential_tasks=args.tasks,
            timeout_s=args.timeout,
        )
        status = 0 if report["summary"]["ok"] else 1
    except Exception as exc:
        report = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        status = 1
    _write(_dump(report, compact=bool(args.compact)), args.output)
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
