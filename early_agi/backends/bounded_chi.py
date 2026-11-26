"""Bounded-χ MPDO backend and THRML-informed block-Gibbs sampler.

The implementation uses only Python, NumPy, and the standard library. It never
materialises an unbounded singular spectrum: randomized range finding and a
deterministic power-iteration refinement are capped at ``chi_max``. Its
thermodynamic seam implements two-colour block Gibbs updates over an Ising-style
energy model and preserves the shape of THRML's block-sampling API for an
environment that installs ``thrml``/JAX separately.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.entropy_sensor import EntanglementReport

__all__ = [
    "BlockGibbsSchedule",
    "BoundedChiBackend",
    "FactorizationResult",
    "GibbsSample",
    "THRMLAdapter",
    "ThrmlUnavailableError",
    "available_thrml_adapter",
]

_RANK_KEY = "chi_max"


class ThrmlUnavailableError(RuntimeError):
    """Raised only when the optional THRML adapter is explicitly required."""


@dataclass(frozen=True, slots=True)
class FactorizationResult:
    """A rank-capped factorization returned by the bounded-χ backend."""

    U: np.ndarray
    S: np.ndarray
    Vh: np.ndarray
    chi: int
    truncation_error: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chi": self.chi,
            "truncation_error": round(self.truncation_error, 8),
            "method": self.method,
            "singular_values": [round(float(value), 8) for value in self.S],
            "metadata": dict(self.metadata),
        }


def _rank_from(options: Mapping[str, Any], width: int) -> int:
    if width <= 0:
        return 0
    for key in (_RANK_KEY, "chi", "max_bond", "rank"):
        raw = options.get(key)
        if raw is not None:
            try:
                return max(1, min(width, int(raw)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {key}: {raw!r}") from exc
    return max(1, min(width, 64))


def _matrix(array: Any) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix contains non-finite values")
    return values


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(42 if seed is None else seed)


def _truncate(values: np.ndarray, rank: int) -> np.ndarray:
    return np.asarray(values[:rank], dtype=np.float64)


def _prepared_couplings(couplings: Any) -> np.ndarray:
    matrix = _matrix(couplings)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("couplings must be a square N×N matrix")
    symmetric = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(symmetric, 0.0)
    return symmetric


@dataclass(frozen=True, slots=True)
class BlockGibbsSchedule:
    """Deterministic two-colour block sampling schedule."""

    n_warmup: int = 20
    n_samples: int = 32
    thinning: int = 1
    seed: int | None = 17

    def __post_init__(self) -> None:
        if any(int(value) < 0 for value in (self.n_warmup, self.n_samples)):
            raise ValueError("Gibbs schedule counts must be non-negative")
        if int(self.thinning) < 1:
            raise ValueError("Gibbs thinning must be at least 1")

    @property
    def n_steps(self) -> int:
        return self.n_warmup + self.n_samples * self.thinning


@dataclass(frozen=True, slots=True)
class GibbsSample:
    """Block-Gibbs samples, their Ising energy per frame, and adapter metadata."""

    states: np.ndarray
    energies: np.ndarray
    beta: float
    backend: str = "bounded_chi.block_gibbs"
    adapter: str = "pure_python"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_energy(self) -> float:
        return float(np.mean(self.energies)) if self.energies.size else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "adapter": self.adapter,
            "beta": self.beta,
            "frames": int(self.states.shape[0]),
            "nodes": int(self.states.shape[1]),
            "mean_energy": round(self.mean_energy, 8),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class THRMLAdapter:
    """Descriptor for an optional, import-lazy THRML backend.

    The core backend has no JAX dependency. This seam records the public THRML
    API shape (``Block``, ``SamplingSchedule``, ``IsingEBM``, ``sample_states``)
    for a dedicated environment that installs THRML separately.
    """

    available: bool = False
    version: str | None = None
    module: str = "thrml"
    notes: str = "optional adapter; install thrml/JAX explicitly to delegate sampling"

    @staticmethod
    def expected_api() -> tuple[str, ...]:
        return (
            "SpinNode",
            "Block",
            "SamplingSchedule",
            "sample_states",
            "IsingEBM",
            "IsingSamplingProgram",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "version": self.version,
            "module": self.module,
            "notes": self.notes,
            "expected_api": list(self.expected_api()),
        }


def available_thrml_adapter() -> THRMLAdapter:
    """Report optional THRML availability without importing JAX eagerly."""
    try:
        found = importlib.util.find_spec("thrml") is not None
    except (ImportError, ValueError):
        found = False
    version: str | None = None
    if found:
        try:
            version = importlib.metadata.version("thrml")
        except importlib.metadata.PackageNotFoundError:
            version = "optional"
    return THRMLAdapter(available=found, version=version)


class BoundedChiBackend:
    """Randomized/iterative truncation with an explicit bond-dimension cap."""

    name = "bounded_chi"
    supports_streaming = True
    supports_sparse = False

    def factorize(
        self,
        matrix: Any,
        *,
        power_iterations: int = 1,
        oversampling: int = 8,
        seed: int | None = 17,
        options: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> FactorizationResult:
        values = _matrix(matrix)
        merged = dict(options or {})
        merged.update(kwargs)
        rows, cols = values.shape
        smaller = min(rows, cols)
        rank = _rank_from(merged, smaller)
        if power_iterations < 0 or oversampling < 0:
            raise ValueError("power_iterations and oversampling must be non-negative")
        if rank == 0 or values.size == 0:
            return FactorizationResult(
                U=np.zeros((rows, 0), dtype=np.float64),
                S=np.zeros(0, dtype=np.float64),
                Vh=np.zeros((0, cols), dtype=np.float64),
                chi=0,
                truncation_error=0.0,
                method=f"{self.name}/empty",
                metadata={"backend": self.name, "chi_max": rank, "shape": [rows, cols]},
            )
        if rank > smaller:
            raise ValueError("chi_max exceeds the matrix rank")

        sketch_width = min(smaller, rank + oversampling)
        generator = _rng(seed)
        if rows <= cols:
            gram = values @ values.T
            sketch = generator.standard_normal((gram.shape[1], sketch_width))
            basis = gram @ sketch
            for _ in range(power_iterations):
                basis = gram @ basis
            q, _ = np.linalg.qr(basis, mode="reduced")
            q = q[:, : min(q.shape[1], sketch_width)]
            projected = q.T @ values
            u_projected, singular_values, vt = np.linalg.svd(projected, full_matrices=False)
            left = q @ u_projected
        else:
            gram = values.T @ values
            sketch = generator.standard_normal((gram.shape[1], sketch_width))
            basis = gram @ sketch
            for _ in range(power_iterations):
                basis = gram @ basis
            q, _ = np.linalg.qr(basis, mode="reduced")
            q = q[:, : min(q.shape[1], sketch_width)]
            projected = values @ q
            left, singular_values, vt_projected = np.linalg.svd(projected, full_matrices=False)
            vt = vt_projected @ q.T

        singular_values = _truncate(singular_values, rank)
        left = np.asarray(left[:, : singular_values.size], dtype=np.float64)
        vt = np.asarray(vt[: singular_values.size, :], dtype=np.float64)
        energy = float(np.sum(singular_values**2))
        total = float(np.sum(values**2))
        retained = min(1.0, energy / total) if total else 1.0
        residual = max(0.0, 1.0 - retained)
        metadata = {
            "backend": self.name,
            "chi": singular_values.size,
            "chi_max": rank,
            "randomized": True,
            "power_iterations": int(power_iterations),
            "oversampling": int(oversampling),
            "seed": seed,
            "retained_energy": retained,
            "shape": [rows, cols],
        }
        return FactorizationResult(
            U=left,
            S=singular_values,
            Vh=vt,
            chi=singular_values.size,
            truncation_error=residual,
            method=f"{self.name}/randomized-svd",
            metadata=metadata,
        )

    def measure_entanglement(
        self,
        coupling_matrix: Any,
        chi_max: int = 4,
        tau: float = 0.15,
    ) -> EntanglementReport:
        """Measure a rank-capped Schmidt spectrum and its truncation cost."""
        if chi_max < 1:
            raise ValueError("chi_max must be >= 1")
        values = _matrix(coupling_matrix)
        partial = self.factorize(values, options={_RANK_KEY: int(chi_max)}, seed=17)
        weights = partial.S**2
        total = float(np.sum(weights))
        weights = weights / total if total else np.zeros(partial.S.size, dtype=np.float64)
        tail = float(np.sum(weights[max(1, int(chi_max)) :]))
        truncation = min(1.0, max(partial.truncation_error, tail))
        effective_rank = int(np.count_nonzero(weights > 1e-3))
        max_entropy = float(np.log2(weights.size)) if weights.size > 1 else 0.0
        return EntanglementReport(
            spectral_entropy_bits=float(
                max(0.0, -np.sum(weights * np.log2(np.maximum(weights, np.finfo(float).tiny))))
            ),
            truncation_error=truncation,
            effective_rank=effective_rank,
            is_volume_law=bool(truncation > tau),
            max_entropy_bits=max_entropy,
            top_weight=float(weights[0]) if weights.size else 0.0,
        )

    def truncate(self, coupling_matrix: Any, chi_max: int = 4) -> np.ndarray:
        """Best rank-``chi_max`` approximation from the bounded factorization."""
        values = _matrix(coupling_matrix)
        rank = min(max(1, int(chi_max)), min(values.shape)) if values.size else 0
        if rank == 0:
            return values.copy()
        partial = self.factorize(values, options={_RANK_KEY: rank}, seed=17)
        return np.asarray((partial.U * partial.S) @ partial.Vh, dtype=np.float64)

    def left_basis(self, coupling_matrix: Any, k: int = 4) -> np.ndarray:
        """First ``k`` left singular vectors from the bounded factorization."""
        values = _matrix(coupling_matrix)
        rank = min(max(1, int(k)), min(values.shape)) if values.size else 0
        if rank == 0:
            return np.zeros((values.shape[0], 0), dtype=np.float64)
        partial = self.factorize(values, options={_RANK_KEY: rank}, seed=17)
        return np.asarray(partial.U[:, :rank], dtype=np.float64)

    # -- THRML-informed thermodynamic sampling -----------------------------
    @staticmethod
    def energy(state: Any, couplings: Any, biases: Any | None = None) -> float:
        spins = np.asarray(state, dtype=np.float64).reshape(-1)
        matrix = _prepared_couplings(couplings)
        field = np.zeros(matrix.shape[0], dtype=np.float64)
        if biases is not None:
            field = np.asarray(biases, dtype=np.float64).reshape(-1)
            if field.size != matrix.shape[0]:
                raise ValueError("biases must have one value per node")
        if spins.size != matrix.shape[0]:
            raise ValueError("state has the wrong dimension")
        interaction = -0.5 * float(spins @ matrix @ spins) - float(field @ spins)
        return interaction

    def block_gibbs(
        self,
        couplings: Any,
        biases: Any | None = None,
        *,
        beta: float = 1.0,
        initial_state: Any | None = None,
        schedule: BlockGibbsSchedule | Mapping[str, Any] | None = None,
        adapter: THRMLAdapter | str | None = None,
    ) -> GibbsSample:
        """Sample bipartite blocks with an Ising EBM; optionally inspect THRML."""
        matrix = _prepared_couplings(couplings)
        size = matrix.shape[0]
        beta = float(beta)
        if beta < 0:
            raise ValueError("beta must be non-negative")
        field = np.zeros(size, dtype=np.float64)
        if biases is not None:
            field = np.asarray(biases, dtype=np.float64).reshape(-1)
            if field.size != size:
                raise ValueError("biases must have one value per node")

        options = (
            BlockGibbsSchedule()
            if schedule is None
            else schedule
            if isinstance(schedule, BlockGibbsSchedule)
            else BlockGibbsSchedule(**dict(schedule))
        )
        chosen_adapter = self._resolve_adapter(adapter)
        generator = _rng(options.seed)
        if initial_state is None:
            state = generator.choice(np.array([-1.0, 1.0]), size=size)
        else:
            state = np.where(np.asarray(initial_state).reshape(-1) >= 0, 1.0, -1.0)
            if state.size != size:
                raise ValueError("initial_state has the wrong dimension")

        started = time.perf_counter()
        samples: list[np.ndarray] = []
        energies: list[float] = []
        for step in range(options.n_steps):
            for parity in (0, 1):
                block = np.arange(parity, size, 2)
                local_field = matrix[np.ix_(block, np.arange(size))] @ state + field[block]
                probability = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))
                state[block] = np.where(generator.random(block.size) < probability, 1.0, -1.0)
            if step >= options.n_warmup and (step - options.n_warmup) % options.thinning == 0:
                samples.append(state.copy())
                energies.append(self.energy(state, matrix, field))
            if len(samples) >= options.n_samples:
                break
        if options.n_samples == 0:
            samples, energies = [], []
        return GibbsSample(
            states=np.asarray(samples, dtype=np.float64).reshape(-1, size),
            energies=np.asarray(energies, dtype=np.float64),
            beta=beta,
            backend="bounded_chi.block_gibbs",
            adapter=chosen_adapter.module if chosen_adapter.available else "pure_python",
            metadata={"elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3)},
        )

    def thrml_adapter(self) -> THRMLAdapter:
        return available_thrml_adapter()

    @staticmethod
    def _resolve_adapter(adapter: THRMLAdapter | str | None) -> THRMLAdapter:
        if adapter is None:
            return THRMLAdapter()
        if isinstance(adapter, THRMLAdapter):
            selected = adapter
        elif adapter in {"thrml", "auto", "optional"}:
            selected = available_thrml_adapter()
        elif adapter == "require":
            selected = available_thrml_adapter()
            if not selected.available:
                raise ThrmlUnavailableError(
                    "THRML is not installed; use the pure-Python Gibbs backend "
                    "or install thrml/JAX in a dedicated environment"
                )
        else:
            raise ValueError(f"unknown thermodynamic adapter {adapter!r}")
        return selected

    # -- streaming MPDO-style seam -----------------------------------------
    def streaming_factorize(
        self,
        blocks: Sequence[Any],
        *,
        chi_max: int = 64,
        power_iterations: int = 1,
        seed: int | None = 17,
    ) -> FactorizationResult:
        """Incrementally renormalise row blocks to a bounded bond dimension."""
        rank = max(1, int(chi_max))
        right_basis = np.zeros((0, 0), dtype=np.float64)
        singular: list[float] = []
        for block in blocks:
            values = _matrix(block)
            if right_basis.size == 0:
                stacked = values
            else:
                if right_basis.shape[1] != values.shape[1]:
                    raise ValueError("streaming blocks must share a feature dimension")
                stacked = np.vstack([np.diag(singular) @ right_basis, values])
            partial = self.factorize(
                stacked,
                power_iterations=power_iterations,
                seed=seed,
                options={_RANK_KEY: rank},
            )
            right_basis = np.asarray(partial.Vh, dtype=np.float64)
            singular = [float(item) for item in partial.S]
        energy = float(sum(value * value for value in singular))
        residual = float(singular[-1] ** 2 / energy) if energy > 0 and singular else 0.0
        return FactorizationResult(
            U=np.eye(len(singular), dtype=np.float64),
            S=np.asarray(singular, dtype=np.float64),
            Vh=right_basis,
            chi=len(singular),
            truncation_error=residual,
            method=f"{self.name}/streaming-mpdo",
            metadata={"backend": self.name, "chi_max": rank, "streaming": True},
        )
