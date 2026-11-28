"""Entrypoint wiring for the selectable relaxation backends."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from early_agi.backends import available_backends, get_backend
from early_agi.config import ConfigError, Settings
from early_agi.demo import backend_showcase
from early_agi.demo import main as demo_main
from early_agi.gauge.cli import main as gauge_main
from early_agi.gauge.vm import GaugeVM

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = (REPO_ROOT / "examples" / "netron_solve.gauge").read_text(encoding="utf-8")


def _clear_early_agi_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("EARLY_AGI_"):
            monkeypatch.delenv(key, raising=False)


def test_settings_backend_selector_validates_and_builds_runtime(tmp_path: Path) -> None:
    default = Settings.from_env({}, load_dotenv=False)
    assert default.backend == "classical_svd"
    assert default.backend in available_backends()

    with pytest.raises(ConfigError, match="EARLY_AGI_BACKEND"):
        Settings.from_env({"EARLY_AGI_BACKEND": "not-a-backend"}, load_dotenv=False)

    selected = Settings.from_env({"EARLY_AGI_BACKEND": "bounded_chi"}, load_dotenv=False)
    bundle = selected.build_runtime(tmp_path)
    try:
        assert bundle.settings.backend == "bounded_chi"
        assert bundle.kernel.backend.name == "bounded_chi"
        assert bundle.graph.backend is bundle.kernel.backend
        assert bundle.backend is bundle.kernel.backend
    finally:
        bundle.sink.close()

    thrml = Settings.from_env({"EARLY_AGI_BACKEND": "thrml"}, load_dotenv=False)
    alias_bundle = thrml.build_runtime(tmp_path)
    try:
        assert alias_bundle.settings.backend == "thrml"
        assert alias_bundle.kernel.backend.name == "thrml_informed"
    finally:
        alias_bundle.sink.close()

    explicit = get_backend("classical_svd")
    override_bundle = selected.build_runtime(tmp_path, backend=explicit)
    try:
        assert override_bundle.backend is explicit
        assert override_bundle.kernel.backend is explicit
    finally:
        override_bundle.sink.close()


@pytest.mark.parametrize(
    ("selector", "implementation"),
    [("bounded_chi", "bounded_chi"), ("thrml", "thrml_informed")],
)
def test_gauge_vm_result_reports_selected_backend(tmp_path: Path, selector: str, implementation: str) -> None:
    settings = Settings.from_env({"EARLY_AGI_BACKEND": selector}, load_dotenv=False)
    vm = GaugeVM(tmp_path, settings=settings)
    try:
        payload = vm.result.as_dict()
        assert payload["backend"] == selector
        assert payload["backend_impl"] == implementation
        assert vm.requested_backend == selector
        assert vm.backend.name == implementation
        assert vm.kernel.backend is vm.backend
        assert vm.graph.backend is vm.backend
    finally:
        vm.sink.close()


def test_gauge_cli_backend_flag_runs_and_reports_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_early_agi_env(monkeypatch)
    program = tmp_path / "netron_solve.gauge"
    program.write_text(EXAMPLE, encoding="utf-8")
    root = tmp_path / "workspace"
    root.mkdir()
    output = tmp_path / "run.json"

    status = gauge_main(
        [
            "run",
            str(program),
            "--root",
            str(root),
            "--backend",
            "bounded_chi",
            "--compact",
            "--no-dotenv",
            "--output",
            str(output),
        ]
    )

    assert status == 0, output.read_text(encoding="utf-8") if output.exists() else "no output"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["backend"] == "bounded_chi"
    assert payload["backend_impl"] == "bounded_chi"


def test_demo_showcase_uses_environment_selector_and_cli_exposes_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clear_early_agi_env(monkeypatch)
    monkeypatch.setenv("EARLY_AGI_BACKEND", "bounded_chi")
    settings = Settings.from_env(load_dotenv=False)

    backend_showcase(settings)

    captured = capsys.readouterr().out
    assert "Backend seams" in captured
    assert '"settings": "bounded_chi"' in captured
    assert "bounded-chi factorization" in captured
    assert "block-Gibbs sample" in captured

    with pytest.raises(SystemExit) as exc:
        demo_main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--backend" in help_text
    assert "bounded_chi" in help_text
    assert "--no-dotenv" in help_text


def test_optional_thrml_extra_is_declared_without_hard_dependency() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dependencies = ["numpy>=1.24"]' in pyproject
    assert 'thrml = ["thrml>=0.1.4"]' in pyproject


@pytest.mark.parametrize("selector", ["default", "cli", "environment"])
def test_full_demo_with_windows_redirected_output_encoding(tmp_path: Path, selector: str) -> None:
    """Exercise real stdout encoding, not pytest's Unicode capture stream."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("EARLY_AGI_")}
    env["PYTHONIOENCODING"] = "cp1252:strict"
    # Running from a throwaway cwd avoids local .env/database state. Explicit
    # paths also support source checkouts without an editable installation.
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), *sys.path])
    command = [sys.executable, "-m", "early_agi.demo", "--no-dotenv"]
    if selector == "cli":
        command.extend(["--backend", "bounded_chi"])
    elif selector == "environment":
        env["EARLY_AGI_BACKEND"] = "bounded_chi"
    completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, timeout=120)
    stdout = completed.stdout.decode("cp1252")
    stderr = completed.stderr.decode("cp1252", errors="replace")
    assert completed.returncode == 0, stderr
    assert "7. Backend seams: bounded-chi" in stdout
    assert "bounded-chi factorization" in stdout
    assert "bounded-chi entanglement" in stdout
    assert "Done. Try --backend bounded_chi" in stdout
    backend = "classical_svd" if selector == "default" else "bounded_chi"
    assert f'"settings": "{backend}"' in stdout
