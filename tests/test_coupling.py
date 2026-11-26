from __future__ import annotations

from pathlib import Path

from early_agi.core.coupling import Constraint, RepoScanner


def build_modular(root: Path) -> None:
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (pkg / "b.py").write_text(
        "from pkg.a import alpha\n\n\ndef beta():\n    return alpha()\n", encoding="utf-8"
    )
    (pkg / "c.py").write_text(
        "from pkg.b import beta\n\n\ndef gamma():\n    return beta()\n", encoding="utf-8"
    )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text(
        "from pkg.a import alpha\n\n\ndef test_alpha():\n    assert alpha() == 1\n",
        encoding="utf-8",
    )


def build_tangled(root: Path) -> None:
    """All-to-all imports, with *heterogeneous* call multiplicity.

    Uniform all-to-all coupling is rank-1 (one symmetric mode), which the sensor
    correctly reports as low entropy. Real spaghetti has uneven weights, so each
    module calls its neighbours a different number of times.
    """
    names = ["one", "two", "three", "four", "five"]
    for position, name in enumerate(names, start=1):
        others = [other for other in names if other != name]
        imports = "\n".join(f"from {other} import work_{other}" for other in others)
        calls = " + ".join(
            f"work_{other}(n - {repetition})"
            for other in others
            for repetition in range(1, position + 1)
        )
        (root / f"{name}.py").write_text(
            f"{imports}\n\n\ndef work_{name}(n):\n    return (n or 0) + ({calls})\n",
            encoding="utf-8",
        )


MODULAR_CONSTRAINTS = [
    Constraint(id="gate:syntax_ok", paths=(".",)),
    Constraint(id="gate:tests_pass", paths=("tests",)),
    Constraint(id="hypothesis:a", paths=("pkg/a.py",)),
    Constraint(id="hypothesis:b", paths=("pkg/b.py",)),
]

TANGLED_CONSTRAINTS = [
    Constraint(id="gate:syntax_ok", paths=(".",)),
    Constraint(id="hypothesis:one", paths=("one.py",)),
    Constraint(id="hypothesis:two", paths=("two.py",)),
    Constraint(id="hypothesis:three", paths=("three.py",)),
]


def test_global_coupling_collapses_to_one_collective_mode(tmp_path: Path) -> None:
    """All-to-all coupling is a single global mode; disjoint blocks are not.

    Note the direction: *uniform* coupling has low S_SVD (one symmetric mode),
    which is exactly the dangerous case for local patching — every entity moves
    together, so the mode cannot be decomposed. A chain of disjoint blocks keeps
    several independent modes. Repository shape alone does not produce a
    volume law; that classification is covered by synthetic spectra in
    ``test_entropy.py``.
    """
    modular_root = tmp_path / "modular"
    tangled_root = tmp_path / "tangled"
    modular_root.mkdir()
    tangled_root.mkdir()
    build_modular(modular_root)
    build_tangled(tangled_root)

    modular = RepoScanner(modular_root).scan().coupling(MODULAR_CONSTRAINTS)
    tangled = RepoScanner(tangled_root).scan().coupling(TANGLED_CONSTRAINTS)

    assert modular.matrix.shape == (len(modular.rows), 4)
    assert tangled.matrix.shape == (len(tangled.rows), 4)
    assert modular.density() < tangled.density()

    tangled_report = tangled.report(chi_max=2)
    modular_report = modular.report(chi_max=2)
    assert tangled_report.top_weight > 0.99  # one mode carries almost all weight
    assert tangled_report.effective_rank <= 2
    assert modular_report.effective_rank >= 3  # the blocks stay independent
    assert modular_report.top_weight < 0.95


def test_scan_and_matrix_are_deterministic(tmp_path: Path) -> None:
    build_modular(tmp_path)
    constraints = [Constraint(id="all", paths=(".",))]

    first = RepoScanner(tmp_path).scan().coupling(constraints)
    second = RepoScanner(tmp_path).scan().coupling(constraints)

    assert first.rows == second.rows
    assert first.fingerprint() == second.fingerprint()


def test_entities_and_constraints_are_named_not_embedded(tmp_path: Path) -> None:
    build_modular(tmp_path)
    coupling = RepoScanner(tmp_path).scan().coupling(MODULAR_CONSTRAINTS)

    assert "pkg/b.py" in coupling.rows
    assert "pkg/b.py::beta" in coupling.rows
    assert "gate:tests_pass" in coupling.cols
    assert coupling.meta["files"] >= 5
    assert coupling.top_rows(k=1)
