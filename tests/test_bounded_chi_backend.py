"""Pure-Python bounded-χ and thermodynamic backend seams."""

from __future__ import annotations

import math

import numpy as np
import pytest

from early_agi.backends import available_backends, available_thrml_adapter, get_backend
from early_agi.backends.base import RelaxationBackend
from early_agi.backends.bounded_chi import (
    BlockGibbsSchedule,
    BoundedChiBackend,
    ThrmlUnavailableError,
)
from early_agi.backends.thrml_backend import ThrmlBackend


def test_bounded_backends_are_registered_and_satisfy_the_protocol() -> None:
    names = available_backends()
    assert {"bounded_chi", "classical_svd", "thrml", "thrml_informed", "thrml_pbit"} <= set(names)
    backend = get_backend("bounded_chi")
    assert isinstance(backend, (BoundedChiBackend, RelaxationBackend))
    assert isinstance(get_backend("thrml_informed"), ThrmlBackend)
    assert get_backend("thrml").name == "thrml_informed"


def test_randomized_truncation_is_capped_by_chi_max() -> None:
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(12, 9))
    backend = get_backend("bounded_chi")
    factor = backend.factorize(
        matrix,
        power_iterations=2,
        oversampling=4,
        options={"chi_max": 3},
    )
    assert factor.chi <= 3
    assert factor.U.shape == (12, factor.chi)
    assert factor.S.shape == (factor.chi,)
    assert factor.Vh.shape == (factor.chi, 9)

    report = backend.measure_entanglement(matrix, chi_max=3, tau=0.05)
    assert report.effective_rank <= 3
    assert 0.0 <= report.truncation_error <= 1.0
    assert math.isfinite(report.spectral_entropy_bits)
    assert backend.truncate(matrix, chi_max=3).shape == matrix.shape
    assert backend.left_basis(matrix, k=2).shape[1] <= 2


def test_streaming_mpdo_and_block_gibbs_are_deterministic_and_bounded() -> None:
    backend = get_backend("bounded_chi")
    rng = np.random.default_rng(23)
    blocks = [rng.normal(size=(4, 6)), rng.normal(size=(3, 6))]
    stream = backend.streaming_factorize(blocks, chi_max=3)
    assert stream.chi <= 3
    assert stream.U.shape == (stream.chi, stream.chi)
    assert stream.Vh.shape == (stream.chi, 6)

    couplings = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, -0.3], [0.5, -0.3, 0.0]])
    sample = backend.block_gibbs(
        couplings,
        biases=[0.2, 0.0, -0.2],
        beta=0.8,
        schedule=BlockGibbsSchedule(n_warmup=3, n_samples=4, thinning=2, seed=5),
        adapter="auto",
    )
    assert sample.states.shape == (4, 3)
    assert set(np.unique(sample.states)) <= {-1.0, 1.0}
    assert sample.energies.shape == (4,)
    assert np.all(np.isfinite(sample.energies))
    assert sample.as_dict()["backend"] == "bounded_chi.block_gibbs"
    assert sample.adapter in {"pure_python", "thrml"}


def test_thrml_adapter_is_optional_but_has_a_stable_seam() -> None:
    backend = get_backend("thrml_informed")
    adapter = backend.thrml_adapter()
    payload = adapter.as_dict()
    assert "IsingEBM" in payload["expected_api"]
    assert "sample_states" in payload["expected_api"]
    assert available_thrml_adapter().as_dict()["module"] == "thrml"

    couplings = np.zeros((2, 2))
    if adapter.available:
        pytest.skip("THRML is installed; the required-adapter error is not expected")
    with pytest.raises(ThrmlUnavailableError):
        backend.block_gibbs(couplings, adapter="require")
