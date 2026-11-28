"""CLI stress runner for the offline/live A/B repair benchmark."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import DEFAULT_BASE_URL
from .race_benchmark import run_race_ab_benchmark
from .routers import OpenAICompatiblePatchRouter
from .runner import run_ab_benchmark

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="early-agi-bench",
        description="Run local multi-file repair A/B curves and a repeat-task reflex stress test.",
    )
    parser.add_argument("--live-eval", action="store_true", help="run the explicit eight-task live pack")
    parser.add_argument("--deep-model", default=os.getenv("EARLY_AGI_DEEP_MODEL", ""))
    parser.add_argument("--verify-models", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--rpm-limit", type=float, default=40)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--token-budget", type=int, default=200000)
    parser.add_argument("--smoke", action="store_true", help="one task, one pass, baseline only, k=1")
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--passes", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["baseline", "netron_plastic", "netron_race"],
        default=["baseline", "netron_plastic", "netron_race"],
    )
    parser.add_argument("--request-retries", type=int, default=0)
    parser.add_argument("--transport-failure-limit", type=int, default=2)
    parser.add_argument("--tasks", type=int, default=10, help="sequential tasks (default: 10)")
    parser.add_argument("--k", type=int, default=4, help="maximum patch attempts per task (default: 4)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="request/grader timeout in seconds (live-eval: 180; offline: 60)",
    )
    parser.add_argument(
        "--router",
        choices=("offline", "openai"),
        default="offline",
        help="patch router (default: offline; no network)",
    )
    parser.add_argument("--allow-live", action="store_true", help="explicitly permit OpenAI-compatible calls")
    parser.add_argument(
        "--race",
        action="store_true",
        help="add the strictly offline netron_race WAL branch-racing arm",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("EARLY_AGI_RACE_ENDPOINT", ""),
        help="offline-only endpoint label for the netron_race mocked transport",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="deterministic branch-race seed (default: 0)",
    )
    parser.add_argument(
        "--max-tokens-per-branch",
        type=int,
        default=4096,
        help="hard token cap for each netron_race branch (default: 4096)",
    )
    parser.add_argument(
        "--race-workers",
        type=int,
        default=4,
        help="maximum netron_race worker processes (default: 4)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("EARLY_AGI_BASE_URL", os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)),
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
    if args.timeout is None:
        args.timeout = 180.0 if args.live_eval else 60.0
    if args.live_eval:
        if not args.allow_live or not args.deep_model or not args.output:
            parser.error("--live-eval requires --allow-live, --deep-model and --output")
        from .live_eval import run_live_evaluation
        from .live_monitor import stderr_progress

        key = os.getenv(args.api_key_env, "") or os.getenv("NVIDIA_API_KEY", "")
        if not key:
            parser.error("set the API key environment variable locally")
        report = run_live_evaluation(
            endpoint=args.endpoint or args.base_url,
            api_key=key,
            model=args.model,
            deep_model=args.deep_model,
            k=1 if args.smoke else args.k,
            max_tokens=args.max_tokens,
            token_budget=args.token_budget,
            rpm_limit=args.rpm_limit,
            max_workers=args.max_workers,
            verify_models=args.verify_models,
            timeout=args.timeout,
            task_limit=1 if args.smoke else args.task_limit,
            passes=1 if args.smoke else args.passes,
            arms=["baseline"] if args.smoke else args.arms,
            request_retries=args.request_retries,
            transport_failure_limit=args.transport_failure_limit,
            progress=stderr_progress,
            checkpoint_path=args.output,
        )
        _write(_dump(report, compact=args.compact), args.output)
        return 0 if report["status"] == "completed" else 130 if report["status"] == "interrupted" else 1
    if args.verify_models:
        parser.error("--verify-models requires --live-eval")
    if args.router == "openai" and not args.allow_live:
        parser.error("--router openai requires --allow-live")
    if args.race and args.router == "openai":
        parser.error("--race is strictly offline and cannot be combined with --router openai")

    router = None
    if args.router == "openai":
        from ..llm.pacing import RequestPacer

        router = OpenAICompatiblePatchRouter(
            base_url=args.endpoint or args.base_url,
            model=args.model,
            api_key=os.getenv(args.api_key_env, "") or os.getenv("NVIDIA_API_KEY", ""),
            max_tokens=args.max_tokens,
            pacer=RequestPacer(args.rpm_limit),
            timeout_s=args.timeout,
            enabled=True,
        )
    try:
        if args.race:
            report = run_race_ab_benchmark(
                k=args.k,
                sequential_tasks=args.tasks,
                timeout_s=args.timeout,
                endpoint=args.endpoint,
                model=args.model,
                seed=args.seed,
                max_tokens_per_branch=args.max_tokens_per_branch,
                max_workers=args.race_workers,
            )
        else:
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
