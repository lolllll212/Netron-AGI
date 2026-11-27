# Low-level boundaries preserved by the structural cleanup

The benchmark-gated cleanup is a source-layout change only. Public import paths,
class identities, deterministic arithmetic, gate semantics, sink records, and
offline A/B outputs are unchanged. In particular, the following three
foundations were **documented, not rewritten**.

## `core/cold_sink.py` — append-only evidence, not entity-scoped storage

`ColdSink` is a content-addressed SQLite archive. A record's hash is computed
from its content only; first-write metadata lives in the `state_commits` side
table, and `states_no_update` / `states_no_delete` triggers enforce append-only
behaviour. `commits()` exposes metadata history without mutating a state row.

The sink deliberately has **no entity or tenant scoping**. Benchmark fixture
identity, task labels, orchestrator names, and run indices live in payloads and
sink metadata inside the existing schema. No migration, foreign key, or new
scoping column was introduced by the A/B harness or this cleanup. Reflex
pre-state hashes and exact-gate expectations continue to be the replay boundary;
only gate-verified edits become `applied_edit` evidence.

## `core/coupling.py` — pure-Python AST incidence, not a semantic model

`RepoScanner`, `RepoGraph`, `Constraint`, and `CouplingMatrix` remain a stdlib
`ast` scanner producing a deterministic bipartite incidence matrix from imports
and qualified calls. It is intentionally not a type checker, dependency
resolver, or learned model; unresolved names and truncation are reported rather
than guessed. Gauge, `AgentSession`, and the offline benchmark all consume the
same matrix and fingerprint semantics. The cleanup did not replace this scanner
with a framework, external parser, or network-backed analysis.

## `backends/thrml_backend.py` — thin optional seam, not a hard dependency

`thrml_backend.py` is an 847-byte adapter boundary. THRML/JAX is optional and is
not imported on the default path; `classical_svd`, `bounded_chi`, and the other
pure-Python/NumPy backends work without it. The offline A/B benchmark uses
`bounded_chi` explicitly and never requires THRML. This cleanup did not add a
THRML import, GPU requirement, or live solver call.

## Compatibility façades and the single adaptation boundary

`core/plasticity.py` and `core/adaptability.py` remain public compatibility
façades. Their implementation now lives in `core/adaptation/`: one surprise EMA
helper, one scale-invariant signature encoder, one directive pool, one stratified
episode sampler, and one three-curve offline benchmark. The historical
`motif_ratios` runtime view and `motifs` adaptation view are thin data shapes,
not duplicate novelty trackers. Internal imports use the new boundary; external
imports and re-exports are unchanged.

`session.py`, `gauge/instructions.py`, `gauge/synthesizer.py`, and
`core/kernel.py` likewise remain public façades over single-responsibility
modules in `session_parts/`, `gauge/instruction_parts/`,
`gauge/synthesis_parts/`, and `core/kernel_parts/`. `GaugeVM` still imports
`GaugeOpcodeRunner` and `Synthesizer` from their historical modules, and
`early_agi.gauge.synthesizer.subprocess.run` remains a live monkeypatch seam
shared by the replay sandbox.

## Regression lock

`benchmarks/structural-cleanup/` stores the pre-refactor offline A/B JSON
report, its summary, the identical post-refactor report, an explicit parity
record, and a stdlib-only checker. See
[the baseline README](../benchmarks/structural-cleanup/README.md) and
[docs/benchmark.md](benchmark.md#structural-cleanup-parity) for commands and
locked `eta_cog`, `success@k`, rollback, and repeat-task reflex speedups.
