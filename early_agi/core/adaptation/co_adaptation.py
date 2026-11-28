"""Implicit-feedback policy tracking for autonomy and communication style."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "AutonomyProfile",
    "UserCoAdapter",
]


@dataclass(frozen=True)
class AutonomyProfile:
    """Live communication/autonomy preferences inferred from implicit feedback."""

    patch_granularity: str  # "surgical" | "balanced" | "bold"
    probe_before_edit: bool
    rationale_style: str  # "terse" | "detailed"
    accept_rate: float
    epistemic_probe: bool = False
    multi_file_crystallization: bool = False
    manual_overrides: int = 0
    avg_prompt_words: float = 18.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "patch_granularity": self.patch_granularity,
            "probe_before_edit": self.probe_before_edit,
            "rationale_style": self.rationale_style,
            "accept_rate": self.accept_rate,
            "epistemic_probe": self.epistemic_probe,
            "multi_file_crystallization": self.multi_file_crystallization,
            "manual_overrides": self.manual_overrides,
            "avg_prompt_words": round(self.avg_prompt_words, 3),
        }


class UserCoAdapter:
    """Adapts agent autonomy, patch size, and explanation style to a user."""

    def __init__(self) -> None:
        self.accepted_turns: int = 0
        self.rejected_or_reverted_turns: int = 0
        self.manual_overrides: int = 0
        self._prompt_lengths: list[int] = []

    def record_turn(
        self,
        *,
        prompt_text: str,
        edits_accepted: bool,
        was_reverted: bool = False,
        user_overrode: bool = False,
        rejected_edits: int = 0,
    ) -> None:
        words = len(str(prompt_text).split())
        if words > 0:
            self._prompt_lengths.append(words)
            if len(self._prompt_lengths) > 64:
                del self._prompt_lengths[:-64]

        if user_overrode:
            self.manual_overrides += 1
        if edits_accepted and not was_reverted and rejected_edits <= 0:
            self.accepted_turns += 1
        elif was_reverted or user_overrode or rejected_edits > 0:
            self.rejected_or_reverted_turns += 1

    def profile(self) -> AutonomyProfile:
        total = self.accepted_turns + self.rejected_or_reverted_turns
        accept_rate = (self.accepted_turns / total) if total > 0 else 0.75
        avg_words = float(np.mean(self._prompt_lengths)) if self._prompt_lengths else 18.0

        frequent_reverts = (
            self.rejected_or_reverted_turns >= 2 or self.manual_overrides >= 2 or accept_rate < 0.55
        )
        sustained_acceptance = accept_rate > 0.85 and total >= 3 and self.rejected_or_reverted_turns == 0
        if frequent_reverts:
            granularity = "surgical"
            probe_first = True
            epistemic_probe = True
            multi_file = False
        elif sustained_acceptance:
            granularity = "bold"
            probe_first = False
            epistemic_probe = False
            multi_file = True
        else:
            granularity = "balanced"
            probe_first = accept_rate < 0.70
            epistemic_probe = False
            multi_file = False

        return AutonomyProfile(
            patch_granularity=granularity,
            probe_before_edit=probe_first,
            rationale_style="terse" if avg_words < 14.0 else "detailed",
            accept_rate=round(accept_rate, 3),
            epistemic_probe=epistemic_probe,
            multi_file_crystallization=multi_file,
            manual_overrides=self.manual_overrides,
            avg_prompt_words=avg_words,
        )

    def replay(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Rebuild a profile from append-only ``user_feedback`` sink records."""
        for record in records:
            payload: Mapping[str, Any] = (
                record if isinstance(record, Mapping) else getattr(record, "payload", {})
            )
            self.record_turn(
                prompt_text=str(payload.get("prompt_text", "")),
                edits_accepted=bool(payload.get("edits_accepted", False)),
                was_reverted=bool(payload.get("was_reverted", False)),
                user_overrode=bool(payload.get("user_overrode", False)),
                rejected_edits=int(payload.get("rejected_edits", 0) or 0),
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "accepted_turns": self.accepted_turns,
            "rejected_or_reverted_turns": self.rejected_or_reverted_turns,
            "manual_overrides": self.manual_overrides,
            "prompt_lengths": list(self._prompt_lengths),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.accepted_turns = int(snapshot.get("accepted_turns", 0) or 0)
        self.rejected_or_reverted_turns = int(snapshot.get("rejected_or_reverted_turns", 0) or 0)
        self.manual_overrides = int(snapshot.get("manual_overrides", 0) or 0)
        self._prompt_lengths = [int(item) for item in snapshot.get("prompt_lengths", ())]
