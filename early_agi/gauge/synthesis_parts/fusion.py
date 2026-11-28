"""Hot extern-op chain observation and replay-gated fusion."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from early_agi.core.cold_sink import canonical_json

__all__ = ["_FusionMixin"]


class _FusionMixin:
    """Hot-chain fusion methods."""

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
        name = "reflex_promotion"
        replay = self._benchmark(
            self.deterministic_source(name, "eta"),
            name,
            "eta",
            {"eta": 0.0, "task": "gauge:reflex-promotion", "has_regressions": False},
        )
        if not replay["ok"]:
            return {
                "fused": 0,
                "operators": [],
                "signatures": 0,
                "skipped": [f"sandbox replay gate rejected reflex promotion: {replay['detail']}"],
                "replay": replay,
            }
        report = self.reflex_library.compile_operator_chains(
            self.registry,
            sink=self.sink,
            min_support=min_support,
        )
        result = report.as_dict()
        result["replay"] = replay
        return result
