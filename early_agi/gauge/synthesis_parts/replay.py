"""Stratified replay, sandbox benchmarking, gates, and promotion archives."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

from early_agi.core.adaptation.episodes import stratified_episodes
from early_agi.core.kernel import ShiftArtifact
from early_agi.operators.registry import OperatorSpec

from .models import _SAFE_BUILTINS, _SANDBOX_WORKER, ReplayCase

__all__ = ["_ReplayMixin"]


class _ReplayMixin:
    """Replay and gated-promotion methods."""

    def _benchmark(
        self,
        source: str,
        function_name: str,
        metric: str,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        cases = self._replay_cases(context)
        payload = {
            "source": source,
            "name": function_name,
            "metric": metric,
            "cases": [{"state": case.state, "baseline_eta": case.baseline_eta} for case in cases],
        }
        started = time.perf_counter()
        empty = {
            "ok": False,
            "sandbox": "python-I-subprocess",
            "cases": 0,
            "episode_source": cases[0].origin.split(":", 1)[0] if cases else "none",
            "baseline_eta": 0.0,
            "candidate_eta": 0.0,
            "improvement": 0.0,
            "average_eta_gain": 0.0,
            "new_task_eta_gain": 0.0,
            "new_task_eta_improved": False,
            "strictly_improved": False,
            "gates_green": False,
            "historical_gate_pass_rate": 1.0,
            "candidate_gate_pass_rate": 1.0,
            "gate_rate_retained": True,
            "stratified": any(case.role == "historical" for case in cases),
            "strata": sorted(
                {
                    str(case.state.get("task") or case.state.get("id") or "unlabelled")
                    for case in cases
                    if case.role == "historical"
                }
            ),
            "dark_gates": self._gate_names(),
            "detail": "",
            "episodes": [case.as_dict() for case in cases],
            "elapsed_ms": 0.0,
        }
        try:
            encoded = json.dumps(payload, default=str, allow_nan=False)
        except (TypeError, ValueError) as exc:
            empty["detail"] = f"replay state is not JSON-safe: {type(exc).__name__}: {exc}"
            return empty
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _SANDBOX_WORKER],
                input=encoded,
                capture_output=True,
                text=True,
                timeout=self.sandbox_timeout,
                check=False,
            )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            worker = json.loads(lines[-1]) if lines else {"ok": False, "error": "empty sandbox output"}
        except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
            empty["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            empty["detail"] = f"sandbox failed: {type(exc).__name__}: {exc}"
            return empty
        empty["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        if completed.returncode != 0 or not worker.get("ok"):
            empty["detail"] = (
                f"sandbox rejected operator: {worker.get('error', 'unknown error')}"
                f"{'; ' + completed.stderr.strip()[:500] if completed.stderr.strip() else ''}"
            )
            return empty

        results = worker.get("cases") or []
        if len(results) != len(cases) or not results:
            empty["detail"] = "sandbox returned an incomplete replay batch"
            return empty
        mutations = [item for item in results if item.get("mutated")]
        if mutations:
            empty["detail"] = "synthesized operator mutated a replayed state"
            return empty
        improvements = [float(item.get("improvement", 0.0)) for item in results]
        strictly_improved = all(math.isfinite(value) and value > 0.0 for value in improvements)
        candidate_states = [dict(item.get("state", {})) for item in results]
        gates_green, gate_detail = self._gates_green(candidate_states)
        baselines = [float(item.get("baseline_eta", 0.0)) for item in results]
        candidates = [float(item.get("candidate_eta", 0.0)) for item in results]
        historical = [index for index, case in enumerate(cases) if case.role == "historical"]
        new_task = [index for index, case in enumerate(cases) if case.role == "new_task"]
        baseline_rate = self.gate_pass_rate([cases[index].state for index in historical])
        candidate_rate = self.gate_pass_rate([candidate_states[index] for index in historical])
        gate_rate_retained = bool(not historical or candidate_rate + 1e-9 >= baseline_rate)
        average_eta_gain = sum(improvements) / len(improvements) if improvements else 0.0
        new_task_eta_gain = (
            sum(improvements[index] for index in new_task) / len(new_task) if new_task else 0.0
        )
        new_task_eta_improved = bool(new_task and new_task_eta_gain > 0.0)
        details: list[str] = []
        if not strictly_improved:
            details.append("no strictly positive eta_cog improvement on every replay")
        if not new_task_eta_improved:
            details.append("average eta_cog did not rise on the new task")
        if not gate_rate_retained:
            details.append("historical DarkStateGate pass rate degraded")
        if not gates_green:
            details.append(gate_detail)
        ok = strictly_improved and new_task_eta_improved and gate_rate_retained and gates_green
        return {
            "ok": ok,
            "sandbox": "python-I-subprocess",
            "cases": len(cases),
            "episode_source": cases[0].origin.split(":", 1)[0] if cases else "none",
            "baseline_eta": round(sum(baselines) / len(baselines), 6),
            "candidate_eta": round(sum(candidates) / len(candidates), 6),
            "improvement": round(min(improvements), 6) if improvements else 0.0,
            "average_eta_gain": round(average_eta_gain, 6),
            "new_task_eta_gain": round(new_task_eta_gain, 6),
            "new_task_eta_improved": new_task_eta_improved,
            "strictly_improved": strictly_improved,
            "gates_green": gates_green,
            "historical_gate_pass_rate": round(baseline_rate, 6),
            "candidate_gate_pass_rate": round(candidate_rate, 6),
            "gate_rate_retained": gate_rate_retained,
            "stratified": bool(historical),
            "strata": sorted(
                {
                    str(cases[index].state.get("task") or cases[index].state.get("id") or "unlabelled")
                    for index in historical
                }
            ),
            "dark_gates": self._gate_names(),
            "detail": "; ".join(details),
            "episodes": [
                {
                    **case.as_dict(),
                    "candidate_eta": round(candidates[index], 6),
                    "improvement": round(improvements[index], 6),
                }
                for index, case in enumerate(cases)
            ],
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    def _replay_cases(self, context: Mapping[str, Any]) -> list[ReplayCase]:
        """Stratified historical episodes plus the new task as a held-out case."""
        historical: list[ReplayCase] = []
        if self.sink is not None:
            records = self.sink.records(
                kinds=("uv_state", "gauge_collapse", "pruned_branch"),
                limit=max(self.replay_limit * 8, 8),
                oldest_first=False,
            )
            eligible: list[Any] = []
            for record in records:
                payload = record.payload if isinstance(record.payload, Mapping) else {}
                state_payload = payload.get("state") if isinstance(payload.get("state"), Mapping) else payload
                eta_value = self._episode_eta(record.eta_cog, payload, state_payload)
                baseline = self._baseline_eta(payload, state_payload, eta_value)
                if self._high_drag(record.eta_cog, baseline, payload, state_payload):
                    eligible.append(record)
            sampled = stratified_episodes(eligible, per_stratum=2)
            seen: set[str] = set()
            for record in [*sampled, *eligible]:
                key = str(getattr(record, "hash", repr(record)))
                if key in seen:
                    continue
                seen.add(key)
                payload = record.payload if isinstance(record.payload, Mapping) else {}
                state_payload = payload.get("state") if isinstance(payload.get("state"), Mapping) else payload
                eta_value = self._episode_eta(record.eta_cog, payload, state_payload)
                baseline = self._baseline_eta(payload, state_payload, eta_value)
                historical.append(
                    ReplayCase(
                        state=self._json_state(state_payload),
                        baseline_eta=baseline,
                        origin=f"cold_sink:{record.short}",
                        role="historical",
                    )
                )
                if len(historical) >= max(1, self.replay_limit - 1):
                    break
        baseline = float(context.get("eta", context.get("eta_cog", 0.0)) or 0.0)
        new_task = ReplayCase(
            state=self._json_state(context),
            baseline_eta=baseline,
            origin="new_task" if historical else "bootstrap:new_task",
            bootstrap=not historical,
            role="new_task",
        )
        return [*historical, new_task]

    @staticmethod
    def _json_state(state: Any) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            state = {"value": state}
        try:
            return json.loads(json.dumps(dict(state), default=str, allow_nan=False))
        except (TypeError, ValueError):
            return {"state_repr": repr(dict(state))[:2000]}

    @staticmethod
    def _episode_eta(
        record_eta: float | None,
        payload: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> float:
        for value in (record_eta, payload.get("eta_cog"), payload.get("eta"), state.get("eta")):
            if value is not None:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                return parsed if math.isfinite(parsed) else 0.0
        return 0.0

    @staticmethod
    def _baseline_eta(
        payload: Mapping[str, Any],
        state: Mapping[str, Any],
        fallback: float,
    ) -> float:
        cost = payload.get("cost") or state.get("cost")
        verified = payload.get("verified_drop_bits", state.get("verified_drop_bits"))
        if isinstance(cost, Mapping) and verified is not None:
            weighted = (
                0.001 * float(cost.get("discarded_tokens", 0) or 0)
                + float(cost.get("tool_cost", 0) or 0)
                + 2.0 * float(cost.get("pruned_branches", 0) or 0)
                + 0.0001 * float(cost.get("latency_ms", 0) or 0)
            )
            if weighted > 0:
                try:
                    value = float(verified) / (weighted + 1e-6)
                    return value if math.isfinite(value) else float(fallback)
                except (TypeError, ValueError):
                    pass
        return float(fallback) if math.isfinite(float(fallback)) else 0.0

    def _high_drag(
        self,
        record_eta: float | None,
        baseline: float,
        payload: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> bool:
        if baseline <= self.eta_min or (record_eta is not None and float(record_eta) <= self.eta_min):
            return True
        if payload.get("drag") is True or state.get("drag") is True:
            return True
        directive = str(payload.get("directive") or state.get("directive") or "").upper()
        return directive in {"SHIFT_COORDINATE", "RECALL_SINK"}

    def _gate_names(self) -> list[str]:
        return [str(getattr(gate, "name", type(gate).__name__)) for gate in self.gates]

    def gate_pass_rate(self, states: Sequence[Mapping[str, Any]]) -> float:
        """Fraction of DarkStateGate evaluations that pass on given states."""
        if not states or not self.gates:
            return 0.0
        passed = 0
        total = 0
        for state in states:
            for gate in self.gates:
                total += 1
                try:
                    result = gate.evaluate(state)
                    verdict = getattr(result, "passed", result)
                    if verdict is True:
                        passed += 1
                except Exception:
                    continue
        return passed / total if total else 0.0

    def _gates_green(self, states: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
        if not self.gates:
            return False, "no dark-state gates configured"
        for state in states:
            for gate in self.gates:
                name = str(getattr(gate, "name", type(gate).__name__))
                try:
                    result = gate.evaluate(state)
                    passed = getattr(result, "passed", result)
                except Exception as exc:
                    return False, f"dark-state gate {name!r} raised {type(exc).__name__}: {exc}"
                if passed is not True:
                    verdict = getattr(result, "verdict", str(passed))
                    return False, f"dark-state gate {name!r} is {verdict}"
        return True, ""

    def _register_operator(self, name: str, source: str, function_name: str, metric: str) -> None:
        namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        exec(compile(source, f"<gauge:{name}>", "exec"), namespace, namespace)
        fn = namespace[function_name]
        if not callable(fn):
            raise ValueError(f"synthesized operator {function_name!r} is not callable")

        def registered_fn(**kwargs: Any) -> dict[str, Any]:
            payload = copy.deepcopy(dict(kwargs.get("state", {})))
            try:
                produced = fn(payload, metric=kwargs.get("metric", metric))
            except TypeError:
                try:
                    produced = fn(payload)
                except TypeError:
                    produced = fn()
            return dict(produced) if isinstance(produced, Mapping) else payload

        assert self.registry is not None
        self.registry.replace(
            OperatorSpec(
                name=name,
                fn=registered_fn,
                kind="READ",
                reads=frozenset({"coordinate/*"}),
                cost=0.0,
                description=f"synthesized replay-gated Gauge operator improving {metric}",
            )
        )

    def _archive_synthesis(
        self,
        name: str,
        metric: str,
        artifact: ShiftArtifact,
        replay: Mapping[str, Any],
        promoted: bool,
        origin: str,
        detail: str,
        *,
        scale: int,
    ) -> None:
        if self.sink is None:
            return
        payload = self._json_state(
            {
                "name": name,
                "metric": metric,
                "accepted": promoted,
                "origin": origin,
                "detail": detail,
                "replay": dict(replay),
                "artifact": artifact.as_dict(),
            }
        )
        candidate_eta = float(replay.get("candidate_eta", 0.0) or 0.0)
        self.sink.put(
            "gauge_synthesis" if promoted else "gauge_synthesis_rejected",
            payload,
            scale=max(0, int(scale)),
            eta_cog=candidate_eta if math.isfinite(candidate_eta) else None,
        )
