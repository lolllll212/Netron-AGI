from __future__ import annotations

import numpy as np
import pytest

from early_agi.core.entropy_sensor import (
    entanglement_report,
    matrix_fingerprint,
    schmidt_weights,
)


def test_empty_and_zero_matrices_are_not_volume_law() -> None:
    for matrix in (np.zeros((0, 0)), np.zeros((4, 3)), np.zeros(0)):
        report = entanglement_report(matrix, chi_max=2)
        assert report.effective_rank == 0
        assert report.is_volume_law is False
        assert report.spectral_entropy_bits == 0.0


def test_rank_one_coupling_is_area_law() -> None:
    matrix = np.outer([1.0, 0.8, 0.2, 0.1], [1.0, 0.5, 0.2, 0.1])
    report = entanglement_report(matrix, chi_max=1, tau=0.15)
    assert report.spectral_entropy_bits < 1e-9
    assert report.truncation_error < 1e-9
    assert report.top_weight == pytest.approx(1.0)
    assert report.is_volume_law is False


def test_dense_coupling_is_volume_law() -> None:
    rng = np.random.default_rng(42)
    report = entanglement_report(rng.standard_normal((8, 8)), chi_max=2, tau=0.15)
    assert report.truncation_error > 0.15
    assert report.is_volume_law is True
    assert report.effective_rank > 2


def test_weights_are_normalised_and_fingerprints_are_stable() -> None:
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    assert schmidt_weights(matrix).sum() == pytest.approx(1.0)
    assert matrix_fingerprint(matrix) == matrix_fingerprint(matrix.copy())
    assert matrix_fingerprint(matrix) != matrix_fingerprint(matrix + 1.0)


def test_truncation_costs_the_discarded_weight() -> None:
    matrix = np.diag([3.0, 2.0, 1.0])
    report = entanglement_report(matrix, chi_max=1)
    assert report.truncation_error == pytest.approx(5.0 / 14.0)
