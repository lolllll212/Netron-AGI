"""Locally evolved adaptation directives (SQLite, offline, deterministic)."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

__all__ = [
    "AdaptationDirective",
    "DirectivePool",
    "evaluate_directive",
]


@dataclass(frozen=True)
class AdaptationDirective:
    """A parameterized adaptation policy rather than a static system prompt."""

    directive_id: str
    patch_granularity: str
    probe_depth: int
    reframe: str
    exploration_patience: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def system_directive(self) -> str:
        return (
            "Live adaptation directive: use "
            f"{self.patch_granularity} patches, probe depth {self.probe_depth}, "
            f"reframe via {self.reframe}, and allow {self.exploration_patience} "
            "exploration steps before crystallizing. Verify with local gates."
        )


class DirectivePool:
    """Small SQLite-backed, locally evolved pool of adaptation directives."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS directives (
                directive_id TEXT PRIMARY KEY,
                patch_granularity TEXT NOT NULL,
                probe_depth INTEGER NOT NULL,
                reframe TEXT NOT NULL,
                exploration_patience INTEGER NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                held_out_score REAL NOT NULL DEFAULT 0,
                trials INTEGER NOT NULL DEFAULT 0,
                generation INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1
            )"""
        )
        self.conn.commit()
        if not self.conn.execute("SELECT 1 FROM directives LIMIT 1").fetchone():
            for directive in self._seed():
                self._insert(directive)

    @staticmethod
    def _seed() -> tuple[AdaptationDirective, ...]:
        return (
            AdaptationDirective("balanced", "surgical", 1, "lock tests", 1),
            AdaptationDirective("explorer", "surgical", 3, "invert invariant", 3),
            AdaptationDirective("coarse", "module", 1, "module cut", 2),
        )

    def _insert(self, directive: AdaptationDirective, *, generation: int = 0, score: float = 0.0) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO directives
               (directive_id, patch_granularity, probe_depth, reframe,
                exploration_patience, score, generation)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                directive.directive_id,
                directive.patch_granularity,
                directive.probe_depth,
                directive.reframe,
                directive.exploration_patience,
                float(score),
                int(generation),
            ),
        )
        self.conn.commit()

    def active(self) -> AdaptationDirective:
        row = self.conn.execute(
            """SELECT * FROM directives WHERE active = 1
               ORDER BY held_out_score DESC, score DESC, trials DESC, directive_id ASC LIMIT 1"""
        ).fetchone()
        if row is None:
            row = self.conn.execute(
                "SELECT * FROM directives ORDER BY score DESC, directive_id ASC LIMIT 1"
            ).fetchone()
        return self._from_row(row)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> AdaptationDirective:
        return AdaptationDirective(
            row["directive_id"],
            row["patch_granularity"],
            int(row["probe_depth"]),
            row["reframe"],
            int(row["exploration_patience"]),
        )

    def directives(self, *, active_only: bool = True) -> list[AdaptationDirective]:
        sql = "SELECT * FROM directives"
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY directive_id"
        return [self._from_row(row) for row in self.conn.execute(sql).fetchall()]

    def record(self, directive_id: str, eta_cog: float, *, held_out: bool = False) -> None:
        """Blend a post-adaptation score into the directive's running mean."""
        row = self.conn.execute(
            "SELECT trials, score, held_out_score FROM directives WHERE directive_id = ?",
            (directive_id,),
        ).fetchone()
        if row is None:
            return
        trials = int(row["trials"])
        score = float(row["score"])
        held_out_score = float(row["held_out_score"])
        score = (score * trials + float(eta_cog)) / (trials + 1)
        if held_out:
            held_out_score = (held_out_score * trials + float(eta_cog)) / (trials + 1)
        self.conn.execute(
            """UPDATE directives SET score = ?, held_out_score = ?, trials = ?
               WHERE directive_id = ?""",
            (score, held_out_score, trials + 1, directive_id),
        )
        self.conn.commit()

    def evolve(self, *, retained: int = 3) -> dict[str, Any]:
        """Mutate leading directives and retain the best held-out variants."""
        keep = max(1, int(retained))
        leaders = list(
            self.conn.execute(
                """SELECT * FROM directives WHERE active = 1
                   ORDER BY score DESC, held_out_score DESC, directive_id ASC LIMIT ?""",
                (keep,),
            ).fetchall()
        )
        generation = int(
            self.conn.execute("SELECT COALESCE(MAX(generation), 0) FROM directives").fetchone()[0]
        )
        stamp = f"{int(time.time() * 1000)}"
        mutations: list[tuple[AdaptationDirective, float]] = []
        for parent in leaders:
            base = self._from_row(parent)
            variants = (
                replace(
                    base, directive_id=f"{base.directive_id}-deep-{stamp}", probe_depth=base.probe_depth + 1
                ),
                replace(
                    base,
                    directive_id=f"{base.directive_id}-patient-{stamp}",
                    exploration_patience=base.exploration_patience + 1,
                ),
                replace(
                    base,
                    directive_id=f"{base.directive_id}-cut-{stamp}",
                    patch_granularity="module" if base.patch_granularity != "module" else "surgical",
                    reframe="module cut" if base.reframe != "module cut" else "lock tests",
                ),
            )
            for variant in variants:
                score = evaluate_directive(variant)
                self._insert(variant, generation=generation + 1, score=score)
                self.record(variant.directive_id, score, held_out=True)
                mutations.append((variant, score))
        ranked = list(
            self.conn.execute(
                """SELECT directive_id FROM directives
                   ORDER BY held_out_score DESC, score DESC, directive_id ASC LIMIT ?""",
                (keep,),
            ).fetchall()
        )
        active_ids = [str(row["directive_id"]) for row in ranked]
        self.conn.execute("UPDATE directives SET active = 0")
        self.conn.executemany(
            "UPDATE directives SET active = 1 WHERE directive_id = ?", [(item,) for item in active_ids]
        )
        self.conn.commit()
        best = max(mutations, key=lambda item: item[1], default=None)
        return {
            "generation": generation + 1,
            "mutations": len(mutations),
            "retained": len(active_ids),
            "active_ids": active_ids,
            "best": best[0].as_dict() if best is not None else None,
        }

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> DirectivePool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def evaluate_directive(directive: AdaptationDirective) -> float:
    """Deterministic held-out post-adaptation eta_cog for offline evolution."""
    probe_gain = min(directive.probe_depth, 4) * 0.08
    patience_gain = min(directive.exploration_patience, 4) * 0.06
    granularity_gain = 0.18 if directive.patch_granularity == "surgical" else 0.10
    reframe_gain = 0.12 if directive.reframe in {"lock tests", "module cut"} else 0.04
    penalty = max(0, directive.probe_depth - 4) * 0.05 + max(0, directive.exploration_patience - 5) * 0.04
    return round(0.30 + probe_gain + patience_gain + granularity_gain + reframe_gain - penalty, 6)
