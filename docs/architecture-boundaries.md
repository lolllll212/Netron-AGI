# Low-level boundaries preserved by the structural cleanup

The benchmark-gated cleanup is a source-layout change only. Public import paths,
class identities, deterministic arithmetic, gate semantics, sink records, and
offline A/B outputs are unchanged. In particular, the following three
foundations were initially documented, not rewritten. Upgrade 2 extends only
the sink boundary as described below.

## `core/cold_sink.py` — append-only evidence with scoped WAL commits

Upgrade 2 adds an additive v2-to-v3 migration with user, agent, run and branch
scope on states and commit observations. Content hashes retain the v2 algorithm;
identical content deduplicates while each scoped commit remains in history.
Append-only state triggers remain enforced. Scoped recall selects observations
within the requested scope. Scope is a retrieval filter, not an authorization boundary.

File-backed sinks use SQLite WAL, busy timeouts and `BEGIN IMMEDIATE` write
transactions to avoid deferred read-to-write snapshot upgrades. Optimistic branch
leases expire or become PRUNED; a partial unique index allows only one COMMITTED
winner per run and parent hash. MeraGraph records append-only SUPERSEDED/MERGED
lineage; latest-only retrieval excludes retired race candidates.

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
