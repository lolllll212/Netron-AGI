"""Phase-4 local adaptability benchmark: exactly three offline curves."""

from __future__ import annotations

import socket

from early_agi.core.adaptability import local_benchmark


def test_three_local_curves_without_external_apis(monkeypatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("adaptability benchmark must not call external services or subprocesses")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("early_agi.gauge.synthesizer.subprocess.run", forbidden)

    report = local_benchmark()

    assert set(report) == {
        "discovery@k",
        "zero_regression_retention",
        "cold_start_eta_cog",
    }

    discovery = report["discovery@k"]
    assert discovery["shift_at"] == 5
    assert discovery["k"] == 8
    assert len(discovery["static_discovery_ticks"]) == len(discovery["adaptive_discovery_ticks"])
    assert discovery["adaptive_mean"] < discovery["static_mean"]
    assert discovery["speedup"] > 1.0
    assert any(score > 0.65 for score in discovery["novelty_scores"])

    retention = report["zero_regression_retention"]
    assert len(retention["tasks"]) == 10
    assert retention["stratified_episodes"] == 10
    assert retention["zero_regression"] is True
    assert retention["regressions"] == 0
    assert all(
        candidate >= historical
        for historical, candidate in zip(
            retention["historical_gate_pass_rate"],
            retention["candidate_gate_pass_rate"],
            strict=True,
        )
    )

    transfer = report["cold_start_eta_cog"]
    assert len(transfer["unseen_topologies"]) == 6
    assert all(similarity >= 0.75 for similarity in transfer["cosine_similarity"])
    assert all(
        recalled > baseline
        for baseline, recalled in zip(transfer["baseline_eta_cog"], transfer["recalled_eta_cog"], strict=True)
    )
    assert transfer["cold_start_improvement"] > 0.0
