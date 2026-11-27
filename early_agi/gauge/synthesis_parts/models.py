"""Replay cases, synthesis results, and the isolated sandbox worker source."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from early_agi.core.kernel import ShiftArtifact

__all__ = [
    "_GAIN_KEY",
    "_SAFE_BUILTINS",
    "_SANDBOX_WORKER",
    "ReplayCase",
    "SynthesisResult",
]


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
    role: str = "historical"

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_eta": round(self.baseline_eta, 6),
            "origin": self.origin,
            "bootstrap": self.bootstrap,
            "role": self.role,
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
