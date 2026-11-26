# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `EARLY_AGI_BACKEND` / `Settings.backend` and `--backend` selectors for `classical_svd`, `bounded_chi`, and the `thrml` adapter seams.
- Wired the selected backend through `build_runtime()`, `GaugeVM`, Gauge results, and the offline demo; added an optional `[thrml]` extra and CI smoke coverage.
- **Gauge**, a custom cognitive and thermodynamic orchestration language: span-tagged lexer/AST/parser, static algebra and reversibility checker, safe tri-state expression evaluator, compact bytecode IR/compiler, hybrid VM, CLI (`python -m early_agi.gauge`), docs, and `examples/netron_solve.gauge`.
- Gauge Step-4 replay-gated operator synthesis: isolated `python -I` benchmarking against recent high-drag cold-sink episodes, strict `eta_cog` improvement, green `DarkStateGate` checks, promotion records, and optional router-authored candidates.
- Gauge VM decomposition: `builtins.py`, `state.py`, `traps.py`, `instructions.py`, `results.py`, and `errors.py` leave `vm.py` as the bytecode dispatcher.
- Pure-Python `bounded_chi` relaxation backend with randomized/iterative rank-capped truncation, streaming MPDO-style updates, deterministic two-colour Ising block-Gibbs sampling, and optional THRML adapter seams (`thrml`, `thrml_informed`).
- Deterministic operator-chain reflexes in `memory/reflex.py`: repeated gate-green extern-op chains fuse into registry operators and MERA nodes linked to source ticks.
- Split Gauge test modules for parser precedence/spans/malformed syntax, checker algebra/no-leak/reversibility, compiler IR lowering, VM trap escalation/collapse replay, replay-gated synthesis, and bounded-χ backends.
- `Settings.build_runtime()`, strict environment validation, `.env` loading, explicit `EARLY_AGI_LLM_ENABLED` opt-in, and complete wiring of physics/storage/scan settings into the demo.
- Real Z3 SMT-LIB2 solving with `unsat`/`sat`/`unknown` mapped to true tri-state verifier results.
- `docs/gauge.md`, `CONTRIBUTING.md`, `SECURITY.md`, and an MIT `LICENSE`.
- Regression tests for Gauge parsing/checking/compiling/running, transactional rollback, allowlists, task-aware reflexes, config, JSON extraction, backend use, cold-sink metadata, wildcard commutator audits, semantic shift artifacts, and deep-tier cost accounting.

### Changed

- `GaugeRunResult` now reports `run_index`, fused-reflex summaries, and rich synthesis replay metadata.
- Gauge expression built-ins, workspace/manifold state, trap evaluation, inverse-stack rollback, and opcode handlers are no longer embedded in the VM dispatcher.
- `docs/gauge.md`, `README.md`, `SECURITY.md`, and this changelog now describe the replay gate, bounded-χ backend, and optional THRML seam.

### Fixed

- Gauge `recall sink` now includes `gauge_collapse` records regardless of the infrared collapse scale, so `collapse -> sink` can be replayed on later ticks.
- Enforced the brief file allowlist in both live-router parsing and the session write path.
- Made edit batches transactional: partial write failures restore workspace bytes, kernel checkpoints, gate state, and registry bookkeeping.
- Reconciled gate state after recall/revert so a rolled-back branch cannot award the same verified bits twice.
- Enforced the coordinate-shift write lock at the actual `write_file` registry boundary; `run_shell` is now disabled by default.
- Preserved tri-state verifier outcomes and fixed wildcard write-race auditing without `KeyError`.
- Made reflex matching task-aware while retaining exact pre-state hashes and expected gates.
- Added semantic verification of shift artifacts (false invariants, unsafe sources, and mutating operators are rejected) and charged deep-tier tokens/latency to `eta_cog`.
- Fixed noisy JSON array extraction and provider usage totals.
- Cold-sink identity now retains all metadata commits, persists compression format, and losslessly encodes bytes in canonical JSON.
- `MeraGraph.coarse_grain()` now uses the backend `truncate()`/`left_basis()` contract instead of a hard-coded SVD call.
- Improved coupling scanner resolution for qualified imports/calls and explicit scan truncation reporting.
- Handled non-UTF-8 source files without crashing brief construction; aligned CI and dev-tooling versions.

## [0.3.0] — 2026-11-26

Reflex compilation: memory that makes the loop cheaper without making it less
verified.

### Added

- `early_agi/memory/reflex.py`: compiles a repeated, gate-passing edit batch into
  a replayable procedure (`min_support=2` by default), pins it to the exact
  pre-state hashes it was learned on, and retires it after two misses.
- `AgentSession(reflexes=...)`: a matching reflex replaces the router call, so a
  replayed turn spends zero tokens; its edits still face the same gates, and a
  replay that cannot reproduce its expected gates is restored before the turn
  ends (`status="reflex_reverted"`).
- `AgentSession.compile_reflexes()` for off-hot-path consolidation.
- `applied_edit` traces, archived only for edits that survived the gates, giving
  the compiler its evidence base and each reflex lineage back to its ticks.
- `MeraGraph.sources()` for inbound `compiled_from` edges.
- Demo section 6: two learning turns compile into a reflex that then fires with
  `ScriptedRouter(strict=True)` and an empty call list.

### Fixed

- The cold sink is content-addressed and idempotent, so two identical edits on
  identical pre-states collapsed into one record and could never reach
  `min_support`. Records now include their tick in the payload: repeating a fix
  is a second observation, not a duplicate.

## [0.2.0] — 2026-11-25

The deterministic agent loop, end to end.

### Added

- Cold sink with append-only triggers, re-hashed reads, exact undo and lineage.
- SVD spectral entropy sensor with a volume-law detector and truncation error.
- `eta_cog` governor over verified bits per unit cost (tokens, tools, branches).
- `RepoScanner`/`CouplingMatrix`: an explicit bipartite entity × constraint
  incidence matrix from imports and call counts (no random embeddings).
- Operator registry with a commutator matrix and write isolation.
- Dark-state verifiers: AST syntax, ruff, pytest, and an optional Z3 gate with
  tri-state results (`pass` / `fail` / unverifiable ⇒ neutral).
- MERA store: scale-indexed SQLite renormalisation plus deterministic L0→L1
  consolidation.
- Stdlib-only OpenAI-compatible chat client and a dual-tier fast/deep router.
- `AgentSession`: scan → brief → execute → snapshot → write → tick → revert.
- Offline demo and 52 tests.

### Fixed

- The governor sketch compared `eta <= eta_high`, which made `CRYSTALLIZE`
  unreachable; the ladder now uses `eta < eta_high` with a two-tick stall window.

## [0.1.0] — 2026-11-25

### Added

- Initial kernel sketch: cold sink, entropy sensor, governor, tick loop, and the
  swappable `RelaxationBackend` interface with a numpy SVD implementation.
