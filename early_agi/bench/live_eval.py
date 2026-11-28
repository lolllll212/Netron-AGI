"""Opt-in, budgeted external-grader evaluation. Never run implicitly by pytest.

Usage: python -m early_agi.bench.live_eval --help
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.client import HTTPException
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..core.governor import CognitiveStepCost
from ..llm.client import LLMError, LLMUnavailable, OpenAICompatibleClient
from ..llm.pacing import RequestPacer
from .escalation import StallTracker
from .fixtures import builtin_fixtures, hard_stall_fixtures, tangled_fixtures
from .gates import apply_edits
from .live_monitor import LiveMonitor, checkpoint, stderr_progress
from .records import ArmState, AttemptExecution
from .routers import OpenAICompatiblePatchRouter
from .runner import ABBenchmark


def evaluation_pack():
    """Eight isolated multi-file tasks; candidates/hints never enter live prompts."""
    return tuple(
        {f.id: f for f in (*builtin_fixtures(), *tangled_fixtures(), *hard_stall_fixtures())}.values()
    )


class BudgetRouter:
    """Coordinator-shared token/turn reservations, including parallel requests.

    UTF-8 bytes conservatively bound input tokens for byte-based model tokenizers.
    Reserve the full bound until usage arrives; retain it on transport errors.
    This is a local safety bound, not a dollar/billing guarantee.
    """

    offline = False
    name = "budgeted_openai_compatible"

    def __init__(self, fast, deep, *, token_budget: int, k: int, max_tokens: int, monitor=None):
        self.fast, self.deep = fast, deep
        self.monitor = monitor or LiveMonitor(api_key="")
        self.budget_exhausted = False
        self.token_budget, self.k, self.max_tokens = token_budget, k, max_tokens
        self.reserved = 0
        self.spent = 0
        self.turns = 0
        self._lock = threading.Lock()

    def begin_task(self):
        self.turns = 0

    def propose(self, request):
        # No fixture identity, canned hints, hidden tests, or candidate labels.
        clean = replace(
            request,
            fixture_id="task",
            files={p: text for p, text in request.files.items() if p in request.allowed_paths},
            active_constraints=(),
            orchestrator="repair",
        )
        bound = sum(len(m["content"].encode("utf-8")) for m in clean.messages()) + self.max_tokens + 512
        # Include worst-case retry reservations: failures may have consumed usage
        # even if an upstream 5xx obscures the successful provider response.
        bound *= 1 + max(self.fast.max_retries, self.deep.max_retries)
        with self._lock:
            if self.monitor.stopped:
                raise RuntimeError("transport circuit open; no further requests")
            if self.turns >= self.k:
                raise RuntimeError("task turn budget exhausted")
            if self.reserved + bound > self.token_budget:
                self.budget_exhausted = True
                self.monitor.emit("token_budget_exhausted", reserved=self.reserved, next_reservation=bound)
                raise RuntimeError("arm token reservation budget exhausted")
            self.turns += 1
            self.reserved += bound
        client = self.deep if request.orchestrator == "reframe" else self.fast
        try:
            with self.monitor.waiting(model=client.model, timeout=client.timeout_s):
                reply = client.propose(clean)
        except Exception as exc:
            transport = isinstance(exc, (LLMUnavailable, OSError, HTTPException, urllib.error.URLError)) or (
                isinstance(exc, LLMError) and "returned HTTP" in str(exc)
            )
            self.monitor.failure(exc, transport=transport)
            raise
        self.monitor.success()
        with self._lock:
            self.spent += reply.tokens
            self.reserved += max(0, reply.tokens) - bound
        self.monitor.emit("grading_start", tokens=reply.tokens)
        return reply

    def telemetry(self):
        return {
            "tokens": self.spent,
            "token_reservations": self.reserved,
            "unresolved_token_reservations": max(0, self.reserved - self.spent),
            "rate_limit_retries": self.fast.rate_limit_retries + self.deep.rate_limit_retries,
            "requests_sent": self.fast.requests_sent + self.deep.requests_sent,
        }


class LiveArm(ABBenchmark):
    """Existing Gauge/reflex path, with optional stalled sandbox speculation."""

    def __init__(self, *, mode, max_workers=2, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.max_workers = min(2, max(1, max_workers))
        self.races = 0
        self.rollback_events = 0
        self.recall_streak = 0
        self.tracker = StallTracker()

    def _run_task(self, **kwargs):
        budget = self.k
        try:
            return super()._run_task(**kwargs)
        finally:
            self.k = budget

    def _request(self, **kwargs):
        request = super()._request(**kwargs)
        return replace(
            request,
            orchestrator=(
                "reframe" if self.mode != "baseline" and self.recall_streak else request.orchestrator
            ),
            active_constraints=(),
            files={p: text for p, text in request.files.items() if p in request.allowed_paths},
        )

    def _run_router_attempt(self, **kwargs):
        result = super()._run_router_attempt(**kwargs)
        record = result.record
        if self.router.monitor.stopped or self.router.budget_exhausted:
            self.k = kwargs["request"].attempt
        self.router.monitor.emit("grading_complete", passed=record.passed, error=record.error or "none")
        self.rollback_events += int(record.rollback)
        self.tracker.observe(record)
        self.recall_streak = self.tracker.recall_streak
        remaining = self.k - self.router.turns
        if self.mode != "netron_race" or self.recall_streak < 1 or remaining < 2:
            return result
        self.races += 1
        request, fixture, root = kwargs["request"], kwargs["fixture"], kwargs["root"]
        with TemporaryDirectory(prefix="netron-live-race-") as tmp:

            def candidate(index):
                branch = Path(tmp) / str(index)
                shutil.copytree(root, branch, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
                worker = ABBenchmark(
                    router=self.router, fixtures=[fixture], k=self.k, timeout_s=self.timeout_s
                )
                try:
                    # Reframer and executor see identical source and external
                    # failure feedback; no golden candidates or hidden tests.
                    branch_request = replace(
                        request,
                        attempt=request.attempt + index + 1,
                        orchestrator="reframe" if index else "repair",
                    )
                    return worker._run_router_attempt(
                        request=branch_request,
                        arm_name="netron",
                        task_index=kwargs["task_index"],
                        fixture=fixture,
                        root=branch,
                    )
                finally:
                    worker.close()

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                outcomes = list(pool.map(candidate, range(min(remaining, 2))))
            green = [outcome for outcome in outcomes if outcome.record.passed]
            winner = max(green, key=lambda o: (o.record.eta_cog, -o.record.tokens)) if green else result
            if green:
                apply_edits(root, winner.edits, allowed_paths=fixture.editable_paths)
            self.rollback_events += sum(int(o.record.rollback) for o in outcomes)
            all_records = [record, *(o.record for o in outcomes)]
            cost = CognitiveStepCost(
                discarded_tokens=sum(r.tokens for r in all_records),
                tool_cost=sum(float(r.cost.get("tool_cost", 0)) for r in all_records),
                pruned_branches=len(outcomes) - int(bool(green)),
                latency_ms=sum(r.latency_ms for r in all_records),
            )
            combined = replace(
                winner.record,
                attempt=self.router.turns,
                cost=cost.as_dict(),
                eta_cog=winner.record.verified_drop_bits / max(1e-9, cost.weighted()) if green else 0.0,
                tokens=record.tokens + sum(o.record.tokens for o in outcomes),
                prompt_tokens=record.prompt_tokens + sum(o.record.prompt_tokens for o in outcomes),
                completion_tokens=record.completion_tokens
                + sum(o.record.completion_tokens for o in outcomes),
                rollback=record.rollback or any(o.record.rollback for o in outcomes),
            )
            return AttemptExecution(combined, winner.edits, winner.before)


def run_live_evaluation(
    *,
    endpoint,
    api_key,
    model,
    deep_model,
    k=4,
    max_tokens=2048,
    token_budget=200000,
    rpm_limit=40,
    max_workers=2,
    verify_models=False,
    timeout=180,
    fixtures=None,
    task_limit=None,
    passes=2,
    arms=("baseline", "netron_plastic", "netron_race"),
    request_retries=0,
    transport_failure_limit=2,
    progress=None,
    checkpoint_path=None,
) -> dict[str, Any]:
    if min(k, max_tokens, token_budget, max_workers, timeout) < 1 or request_retries < 0:
        raise ValueError("budgets/timeouts must be positive and retries nonnegative")
    if passes not in (1, 2) or (task_limit is not None and task_limit < 1):
        raise ValueError("passes must be 1 or 2 and task_limit positive")
    if not arms or len(set(arms)) != len(arms) or set(arms) - {"baseline", "netron_plastic", "netron_race"}:
        raise ValueError("select distinct valid evaluation arms")
    monitor = LiveMonitor(api_key=api_key, progress=progress, failure_limit=transport_failure_limit)
    pacer = RequestPacer(rpm_limit)
    pack = tuple(fixtures or evaluation_pack())[:task_limit]
    report = {
        "schema_version": 2,
        "offline": False,
        "status": "running",
        "valid_comparison": False,
        "endpoint": endpoint,
        "models": {"executor": model, "reframer": deep_model},
        "budgets": {
            "k": k,
            "max_tokens": max_tokens,
            "tokens_per_arm": token_budget,
            "rpm_limit": rpm_limit,
            "max_workers": min(2, max_workers),
            "timeout_s": timeout,
            "request_retries": request_retries,
        },
        "plan": {"tasks": [f.id for f in pack], "passes": passes, "arms": list(arms)},
        "arms": {},
        "errors": monitor.errors,
    }
    active = None
    router = benchmark = None
    records = []

    def snapshot(status):
        if active is None:
            return
        first, second = records[: len(pack)], records[len(pack) :]
        report["arms"][active] = {
            **router.telemetry(),
            "status": status,
            "tasks": list(records),
            "completed_tasks": len(records),
            "planned_tasks": len(pack) * passes,
            "races": benchmark.races,
            "solve_rate": sum(r["success"] for r in first) / max(1, len(first)),
            "mean_eta_cog": sum(r["eta_cog"] for r in first) / max(1, len(first)),
            "rollbacks": benchmark.rollback_events,
            "second_pass_reflex_hits": sum(r["reflex_hit"] for r in second),
            "second_pass_token_speedup": (
                sum(r["tokens"] for r in first) / max(1, sum(r["tokens"] for r in second))
            )
            if second
            else None,
        }

    try:
        checkpoint(checkpoint_path, report)
        if verify_models:
            monitor.emit("model_preflight_start", timeout_s=timeout)
            with monitor.waiting(model="model-discovery", timeout=timeout):
                OpenAICompatibleClient(
                    model, base_url=endpoint, api_key=api_key, pacer=pacer, timeout=timeout
                ).verify_models([model, deep_model])
            monitor.emit("model_preflight_complete")
        for mode in arms:
            active, records = mode, []
            monitor.emit("arm_start", arm=mode)

            def transport(name):
                return OpenAICompatiblePatchRouter(
                    base_url=endpoint,
                    model=name,
                    api_key=api_key,
                    timeout_s=timeout,
                    enabled=True,
                    max_tokens=max_tokens,
                    pacer=pacer,
                    max_retries=request_retries,
                    progress=monitor.emit,
                )

            router = BudgetRouter(
                transport(model),
                transport(deep_model),
                token_budget=token_budget,
                k=k,
                max_tokens=max_tokens,
                monitor=monitor,
            )
            benchmark = LiveArm(
                mode=mode,
                max_workers=max_workers,
                router=router,
                fixtures=pack,
                sequential_tasks=len(pack) * passes,
                k=k,
                timeout_s=timeout,
            )
            state = ArmState("plain_retry" if mode == "baseline" else "netron")
            try:
                with TemporaryDirectory(prefix="netron-live-eval-") as tmp:
                    for index in range(len(pack) * passes):
                        fixture = pack[index % len(pack)]
                        root = Path(tmp) / str(index)
                        from .gates import write_workspace

                        write_workspace(fixture, root)
                        router.begin_task()
                        benchmark.recall_streak, benchmark.tracker = 0, StallTracker()
                        report["current_task"] = {"arm": mode, "index": index + 1, "fixture": fixture.id}
                        snapshot("running")
                        checkpoint(checkpoint_path, report)
                        monitor.emit(
                            "task_start",
                            arm=mode,
                            task=index + 1,
                            total=len(pack) * passes,
                            fixture=fixture.id,
                            pass_number=index // len(pack) + 1,
                        )
                        task = benchmark._run_task(
                            arm_name=state.name,
                            state=state,
                            task_index=index + 1,
                            fixture=replace(fixture, probe_hint=""),
                            root=root,
                            first_seen=index < len(pack),
                        )
                        records.append(task.as_dict())
                        monitor.emit(
                            "task_complete", success=task.success, tokens=task.tokens, reflex=task.reflex_hit
                        )
                        status = (
                            "transport_failed"
                            if monitor.stopped
                            else "budget_exhausted"
                            if router.budget_exhausted
                            else "running"
                        )
                        snapshot(status)
                        report.pop("current_task", None)
                        checkpoint(checkpoint_path, report)
                        if monitor.stopped or router.budget_exhausted:
                            break
                if monitor.stopped:
                    report["status"] = "transport_failed"
                    break  # Do not repeat the same outage in the remaining arms.
                snapshot("budget_exhausted" if router.budget_exhausted else "completed")
            finally:
                benchmark.close()
            checkpoint(checkpoint_path, report)
        if report["status"] == "running":
            report["status"] = (
                "completed"
                if all(a["status"] == "completed" for a in report["arms"].values())
                else "budget_exhausted"
            )
    except KeyboardInterrupt:
        snapshot("interrupted")
        report["status"] = "interrupted"
        monitor.emit("interrupted", message="completed tasks preserved; in-flight usage may be unknown")
    except Exception as exc:
        monitor.failure(exc)
        snapshot("failed")
        report["status"] = "failed"
    finally:
        if benchmark is not None:
            benchmark.close()
        report["valid_comparison"] = report["status"] == "completed" and len(arms) == 3 and not monitor.errors
        baseline = report["arms"].get("baseline")
        for arm in report["arms"].values():
            arm["rollbacks_avoided"] = baseline["rollbacks"] - arm["rollbacks"] if baseline else None
        checkpoint(checkpoint_path, report)
        monitor.emit("evaluation_end", status=report["status"], checkpoint=checkpoint_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true", required=True)
    parser.add_argument(
        "--endpoint", default=os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--deep-model", required=True)
    parser.add_argument("--verify-models", action="store_true")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--token-budget", type=int, default=200000)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--rpm-limit", type=float, default=40)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--smoke", action="store_true")
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    key = os.getenv(args.api_key_env, "")
    if not key:
        parser.error(f"set {args.api_key_env} locally; never put credentials in reports")
    report = run_live_evaluation(
        endpoint=args.endpoint,
        api_key=key,
        model=args.model,
        deep_model=args.deep_model,
        k=1 if args.smoke else args.k,
        max_tokens=args.max_tokens,
        token_budget=args.token_budget,
        max_workers=args.max_workers,
        rpm_limit=args.rpm_limit,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "completed" else 130 if report["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
