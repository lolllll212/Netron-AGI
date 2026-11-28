"""Gauge VM integration: replay-gated synthesis, trap escalation, and recall."""

from __future__ import annotations

from pathlib import Path

from _gauge_fixtures import SOURCE, VolumeBackend
from early_agi.config import Settings
from early_agi.gauge import run_gauge
from early_agi.gauge.vm import GaugeVM


def _settings() -> Settings:
    return Settings.from_env({}, load_dotenv=False)


def test_vm_runs_example_against_real_subsystems(tmp_path: Path) -> None:
    result = run_gauge(
        SOURCE,
        tmp_path,
        settings=_settings(),
        name="netron_solve.gauge",
    )
    assert result.status == "halted"
    assert result.ok
    assert not result.violations
    assert "patch_ensemble" in result.collapses
    assert result.manifolds["patch_ensemble"]["collapsed"] is True
    assert "EVOLVE" in {event["op"] for event in result.trace}
    assert result.tick["hash"]


def test_drag_trap_synthesis_is_replay_gated(tmp_path: Path) -> None:
    vm = GaugeVM(tmp_path, settings=_settings(), backend=VolumeBackend())
    result = vm.run(SOURCE, name="volume-law.gauge")

    assert result.ok
    assert {"volume_law", "drag"} <= set(result.traps)
    assert len(result.synthesis) == 1
    synthesis = result.synthesis[0]
    assert synthesis["accepted"] is True
    assert synthesis["registered"] is True
    assert synthesis["efficiency_improvement"] > 0.0
    assert synthesis["origin"] == "deterministic"
    assert synthesis["replay"]["cases"] >= 1
    assert synthesis["replay"]["episode_source"] in {"cold_sink", "bootstrap"}
    assert synthesis["replay"]["strictly_improved"] is True
    assert synthesis["replay"]["gates_green"] is True
    assert synthesis["replay"]["dark_gates"]
    assert vm.registry.operator("disentangle_cut").description.startswith("synthesized")


def test_multi_tick_trap_escalation_replays_collapse_sink(tmp_path: Path) -> None:
    vm = GaugeVM(tmp_path, settings=_settings(), backend=VolumeBackend())
    # The test isolates trap escalation from the coordinate-shift write lock;
    # the lock's own regression coverage lives in tests/test_session_safety.py.
    vm.kernel.lock_writes_on_shift = False

    first = vm.run(SOURCE, name="volume-law-first.gauge")
    assert first.ok
    assert {"volume_law", "drag"} <= set(first.traps)
    assert not first.recalled
    assert first.manifolds["patch_ensemble"]["basis"] == "dependency_cut"
    assert first.collapses

    vm.registry.release_writes()
    second = vm.run(SOURCE, name="volume-law-second.gauge")
    assert second.ok
    assert {"volume_law", "drag", "frustration"} <= set(second.traps)
    assert second.governor["directive"] == "RECALL_SINK"
    assert any(record["kind"] == "gauge_collapse" for record in second.recalled)
    assert second.manifolds["patch_ensemble"]["basis"] == "identity"
    assert not vm.inverse_stack
    assert second.reflexes["fused"] >= 1
    assert second.reflexes["operators"]
    assert all(name in vm.registry.names() for name in second.reflexes["operators"])
    reflex = vm.reflex_library.match_operator_chain(["capture_baseline", "apply_candidate"])
    assert reflex is not None
    assert reflex.support >= 2
