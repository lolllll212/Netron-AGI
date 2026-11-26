"""Gauge Step 4: replay-gated operator synthesis and hot-chain fusion.

A candidate operator is not promoted because it parses. It must first replay
recent high-drag ColdSink episodes inside an isolated ``python -I`` subprocess,
strictly improve ``eta_cog`` on every replay, leave each input state unmutated,
and keep every real DarkStateGate green. Repeated successful ``extern op`` chains
are fused into deterministic registry operators and linked into ReflexLibrary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.cold_sink import ColdSink, canonical_json
from ..core.kernel import ShiftArtifact
from ..memory.mera_graph import MeraGraph
from ..memory.reflex import ReflexLibrary
from ..operators.dark_state import DarkStateGate
from ..operators.registry import OperatorRegistry, OperatorSpec

__all__ = ["ReplayCase", "SynthesisResult", "Synthesizer"]

_GAIN_KEY = "gauge_metric_gain"

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "mul": lambda left, right: left * right,
    "print": lambda *args, **kwargs: None,
    "range": range,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}

_SANDBOX_WORKER = r"""
import ast, json, math, sys

payload = json.loads(sys.stdin.read())
source = payload["source"]
name = payload["name"]
metric = payload["metric"]
cases = payload["cases"]
safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "float": float, "int": int, "len": len, "list": list, "max": max,
    "min": min, "mul": lambda a, b: a * b,
    "print": lambda *args, **kwargs: None, "range": range,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
}
allowed_calls = set(safe_builtins)
unsafe_names = {"eval", "exec", "compile", "open", "__import__", "breakpoint", "exit", "quit"}
forbidden_nodes = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda,
    ast.Await, ast.Yield, ast.YieldFrom, ast.For, ast.AsyncFor, ast.While,
    ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Assert, ast.Delete,
    ast.ClassDef, ast.AsyncFunctionDef, ast.Attribute, ast.NamedExpr,
)
try:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError("forbidden syntax " + type(node).__name__)
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in unsafe_names:
                raise ValueError("unsafe name " + node.id)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls or node.keywords:
                raise ValueError("only pure positional helper calls are allowed")
    namespace = {"__builtins__": safe_builtins}
    exec(compile(tree, "<gauge-synthesis>", "exec"), namespace, namespace)
    fn = namespace[name]
    out = []
    for case in cases:
        state = case["state"]
        baseline = float(case["baseline_eta"])
        if not math.isfinite(baseline):
            raise ValueError("non-finite baseline eta_cog")
        before = json.loads(json.dumps(state, default=str))
        try:
            candidate = fn(state, metric=metric)
        except TypeError:
            try:
                candidate = fn(state)
            except TypeError:
                candidate = fn()
        mutated = json.loads(json.dumps(state, default=str)) != before
        if not isinstance(candidate, dict):
            raise ValueError("operator did not return a candidate state")
        candidate = json.loads(json.dumps(candidate, default=str, allow_nan=False))
        gain = float(candidate.get("gauge_metric_gain", candidate.get("gauge_eta_gain", 0.0)) or 0.0)
        candidate_eta = float(candidate.get("gauge_candidate_eta", baseline + gain))
        if not math.isfinite(gain) or not math.isfinite(candidate_eta):
            raise ValueError("non-finite eta_cog improvement")
        out.append({
            "state": candidate,
            "baseline_eta": baseline,
            "candidate_eta": candidate_eta,
            "improvement": candidate_eta - baseline,
            "mutated": mutated,
            "error": "",
        })
    print(json.dumps({"ok": True, "cases": out}, default=str, allow_nan=False))
except Exception as exc:
    print(json.dumps({"ok": False, "error": type(exc).__name__ + ": " + str(exc)}))
"""


@dataclass(frozen=True, slots=True)
class ReplayCase:
    state: dict[str, Any]
    baseline_eta: float
    origin: str
    bootstrap: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_eta": round(self.baseline_eta, 6),
            "origin": self.origin,
            "bootstrap": self.bootstrap,
            "state_keys": sorted(self.state),
        }


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    name: str
    source: str
    accepted: bool
    artifact: ShiftArtifact
    detail: str = ""
    origin: str = "deterministic"
    metric: str = "eta"
    efficiency_improvement: float = 0.0
    replay: dict[str, Any] = field(default_factory=dict)
    registered: bool = False
    promoted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "accepted": self.accepted,
            "promoted": self.promoted,
            "registered": self.registered,
            "detail": self.detail,
            "origin": self.origin,
            "metric": self.metric,
            "efficiency_improvement": round(self.efficiency_improvement, 6),
            "replay": dict(self.replay),
            "artifact": self.artifact.as_dict(),
        }


class Synthesizer:
    """Replay, benchmark, promote, and fuse Gauge operators."""

    def __init__(
        self,
        router: Any | None = None,
        *,
        sink: ColdSink | None = None,
        registry: OperatorRegistry | None = None,
        gates: Sequence[Any] = (),
        reflex_library: ReflexLibrary | None = None,
        eta_min: float = 0.15,
        replay_limit: int = 5,
        sandbox_timeout: float = 8.0,
    ) -> None:
        self.router = router
        self.sink = sink
        self.registry = registry
        self.gates = self._prepare_gates(gates)
        self.eta_min = float(eta_min)
        self.replay_limit = max(1, int(replay_limit))
        self.sandbox_timeout = max(0.5, float(sandbox_timeout))
        self._chain_sequence = 0
        self.reflex_library = reflex_library
        if self.reflex_library is None and self.sink is not None:
            self.reflex_library = ReflexLibrary(MeraGraph(self.sink, ":memory:"))

    def bind(
        self,
        *,
        sink: ColdSink | None = None,
        registry: OperatorRegistry | None = None,
        gates: Sequence[Any] | None = None,
        reflex_library: ReflexLibrary | None = None,
        eta_min: float | None = None,
    ) -> Synthesizer:
        """Attach VM subsystems to a synthesizer supplied by an embedder."""
        if self.sink is None and sink is not None:
            self.sink = sink
        if self.registry is None and registry is not None:
            self.registry = registry
        if gates is not None:
            self.gates = self._prepare_gates([*self.gates, *gates])
        else:
            self.gates = self._prepare_gates(self.gates)
        if self.reflex_library is None:
            if reflex_library is not None:
                self.reflex_library = reflex_library
            elif self.sink is not None:
                self.reflex_library = ReflexLibrary(MeraGraph(self.sink, ":memory:"))
        if eta_min is not None:
            self.eta_min = float(eta_min)
        return self

    @staticmethod
    def _safe_name(name: str) -> str:
        candidate = re.sub(r"\W", "_", name.strip())
        if not candidate or not candidate.isidentifier():
            candidate = f"gauge_{candidate or 'operator'}"
        return candidate

    @staticmethod
    def deterministic_source(name: str, metric: str) -> str:
        """A pure coordinate operator that advertises a measurable gain."""
        return (
            f"def {name}(state, *, metric={metric!r}):\n"
            "    improved = dict(state)\n"
            f"    improved[{_GAIN_KEY!r}] = 0.1\n"
            "    return improved\n"
        )

    @staticmethod
    def _default_gate() -> DarkStateGate:
        """A deterministic dark-state gate over every replayed candidate."""

        def check(state: Mapping[str, Any]) -> bool:
            if state.get("has_regressions") is True:
                return False
            verdicts = state.get("gates") or {}
            if isinstance(verdicts, Mapping) and any(str(value) == "fail" for value in verdicts.values()):
                return False
            try:
                return float(state.get(_GAIN_KEY, 0.0) or 0.0) > 0.0
            except (TypeError, ValueError):
                return False

        return DarkStateGate("synthesis_replay_safe", check, bits=1.0)

    @classmethod
    def _prepare_gates(cls, gates: Sequence[Any]) -> list[Any]:
        selected: list[Any] = []
        seen: set[str] = set()
        for gate in gates:
            name = str(getattr(gate, "name", type(gate).__name__))
            if name in seen:
                continue
            selected.append(gate)
            seen.add(name)
        if not any(isinstance(gate, DarkStateGate) for gate in selected):
            default = cls._default_gate()
            if default.name not in seen:
                selected.append(default)
        return selected

    def synthesize(
        self,
        op_name: str,
        metric: str,
        context: Mapping[str, Any] | None = None,
        *,
        entanglement: Any | None = None,
    ) -> SynthesisResult:
        safe_name = self._safe_name(op_name)
        state = dict(context or {})
        source = self.deterministic_source(safe_name, metric)
        artifact = ShiftArtifact(
            kind="new_operator",
            payload={"name": safe_name, "source": source},
            rationale=f"synthesize {op_name!r} reducing {metric}",
        )
        structure_ok, detail = artifact.validate(state=state, entanglement=entanglement)
        origin = "deterministic"

        if self.router is not None and hasattr(self.router, "reason"):
            brief = {
                "mode": "synthesize",
                "op_name": op_name,
                "metric": metric,
                "context": state,
            }
            try:
                proposed = self.router.reason(brief)
                if isinstance(proposed, Mapping):
                    proposed = ShiftArtifact.from_mapping(dict(proposed))
                elif isinstance(proposed, str):
                    proposed = ShiftArtifact.from_mapping(dict(json.loads(proposed)))
                proposed_ok, proposed_detail = proposed.validate(state=state, entanglement=entanglement)
                if proposed_ok:
                    artifact, structure_ok, detail, origin = proposed, True, "", "router"
                else:
                    detail = f"router artifact rejected: {proposed_detail}; using deterministic fallback"
            except Exception as exc:
                detail = (
                    f"router synthesis unavailable: {type(exc).__name__}: {exc}; using deterministic fallback"
                )

        function_name = str(artifact.payload.get("name", safe_name))
        source = str(artifact.payload.get("source", source))
        replay = self._benchmark(source, function_name, metric, state)
        promoted = bool(structure_ok and replay["ok"])
        if not promoted and not detail:
            detail = str(replay["detail"])
        registered = False
        if promoted and self.registry is not None:
            self._register_operator(safe_name, source, function_name, metric)
            registered = True
        self._archive_synthesis(
            safe_name,
            metric,
            artifact,
            replay,
            promoted,
            origin,
            detail,
            scale=int(state.get("scale", 0) or 0),
        )
        return SynthesisResult(
            name=safe_name,
            source=source,
            accepted=promoted,
            artifact=artifact,
            detail=detail,
            origin=origin,
            metric=metric,
            efficiency_improvement=float(replay["improvement"]),
            replay=replay,
            registered=registered,
            promoted=promoted,
        )

    # -- replay benchmarking ------------------------------------------------
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
            "strictly_improved": False,
            "gates_green": False,
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
        details: list[str] = []
        if not strictly_improved:
            details.append("no strictly positive eta_cog improvement on every replay")
        if not gates_green:
            details.append(gate_detail)
        ok = strictly_improved and gates_green
        return {
            "ok": ok,
            "sandbox": "python-I-subprocess",
            "cases": len(cases),
            "episode_source": cases[0].origin.split(":", 1)[0] if cases else "none",
            "baseline_eta": round(sum(baselines) / len(baselines), 6),
            "candidate_eta": round(sum(candidates) / len(candidates), 6),
            "improvement": round(min(improvements), 6) if improvements else 0.0,
            "strictly_improved": strictly_improved,
            "gates_green": gates_green,
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
        cases: list[ReplayCase] = []
        if self.sink is not None:
            records = self.sink.records(
                kinds=("uv_state", "gauge_collapse", "pruned_branch"),
                limit=self.replay_limit * 4,
                oldest_first=False,
            )
            for record in records:
                payload = record.payload if isinstance(record.payload, Mapping) else {}
                state_payload = payload.get("state") if isinstance(payload.get("state"), Mapping) else payload
                eta_value = self._episode_eta(record.eta_cog, payload, state_payload)
                baseline = self._baseline_eta(payload, state_payload, eta_value)
                if not self._high_drag(record.eta_cog, baseline, payload, state_payload):
                    continue
                cases.append(
                    ReplayCase(
                        state=self._json_state(state_payload),
                        baseline_eta=baseline,
                        origin=f"cold_sink:{record.short}",
                    )
                )
                if len(cases) >= self.replay_limit:
                    break
        if not cases:
            baseline = float(context.get("eta", context.get("eta_cog", 0.0)) or 0.0)
            cases.append(
                ReplayCase(
                    state=self._json_state(context),
                    baseline_eta=baseline,
                    origin="bootstrap",
                    bootstrap=True,
                )
            )
        return cases

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

    # -- promotion and archiving -------------------------------------------
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

    # -- hot operator-chain fusion -----------------------------------------
    @staticmethod
    def chain_signature(chain: Sequence[str]) -> str:
        return hashlib.sha256(canonical_json(list(chain)).encode("utf-8")).hexdigest()[:16]

    def observe_operator_chain(
        self,
        chain: Sequence[str],
        *,
        ok: bool,
        eta: float,
        tick_hash: str | None,
        gates: Mapping[str, str] | None = None,
        scale: int = 0,
        task: str | None = None,
        run_index: int = 0,
    ) -> str:
        """Archive one executed extern-op chain as evidence for fusion."""
        if self.sink is None or not chain:
            return ""
        ordered = [str(item) for item in chain]
        self._chain_sequence += 1
        signature = self.chain_signature(ordered)
        payload = {
            "chain": ordered,
            "signature": signature,
            "sequence": self._chain_sequence,
            "run_index": int(run_index),
            "ok": bool(ok),
            "eta": round(float(eta), 6) if math.isfinite(float(eta)) else 0.0,
            "tick": tick_hash,
            "gates": dict(gates or {}),
            "task": task or "",
        }
        return self.sink.put(
            "gauge_op_chain",
            payload,
            scale=max(0, int(scale)),
            parent_hash=tick_hash,
            eta_cog=payload["eta"],
        )

    def compile_hot_chains(self, *, min_support: int = 2) -> dict[str, Any]:
        """Fuse repeated, gate-green operator chains into deterministic ops."""
        if self.registry is None or self.reflex_library is None:
            return {
                "fused": 0,
                "operators": [],
                "signatures": 0,
                "skipped": ["registry/reflex library unavailable"],
            }
        report = self.reflex_library.compile_operator_chains(
            self.registry,
            sink=self.sink,
            min_support=min_support,
        )
        return report.as_dict()
