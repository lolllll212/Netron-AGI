# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Offline demo output now uses ASCII `bounded-chi` display labels so redirected
  Windows cp1252 stdout does not raise `UnicodeEncodeError`. Added full demo
  subprocess regression coverage for default, CLI and environment backends
  under strict cp1252 encoding; no UTF-8 environment workaround is required.

### Added

- Upgrade 2: additive scoped ColdSink v3 WAL storage, expiring branch leases and
  atomic single-winner commits; immutable MeraGraph SUPERSEDED/MERGED history.
- Offline parallel branch racer with isolated worktrees/copies, gate-green eta
  ranking, mocked transport and an opt-in third `--race` benchmark arm.
- Gauge `race` primitive, `examples/netron_race.gauge`, and typed
  `GaugeRunResult.races` outcomes (omitted from legacy no-race JSON).
- Hardening tests for spawned four-process WAL contention, raw/compressed v2
  migration, persisted lineage, real git-worktree teardown, all-fail source
  immutability, mocked worker pickling and the three-arm benchmark. Explicit
  Ubuntu/Windows CI coverage; default offline benchmark parity remains locked.

- Added `benchmarks/structural-cleanup/`: the pre-refactor offline A/B JSON report and summary, the identical post-refactor report, explicit locked metrics in `parity.json`, and a stdlib-only `check_parity.py` verifier. Added `docs/architecture-boundaries.md` documenting the unchanged `ColdSink`, pure-Python AST coupling, and optional THRML seam contracts.
- Added a byte-level equality assertion in `tests/test_ab_benchmark.py` so the packaged `early_agi/bench/netron_plastic.gauge` and `examples/netron_plastic.gauge` cannot silently drift.
- Added `early_agi/bench/`: a focused offline/live A/B repair benchmark with real multi-file Python bug fixtures, a pluggable `PatchRouter` seam, deterministic offline and opt-in OpenAI-compatible routers, transactional workspace gates, `netron_plastic.gauge` + `bounded_chi` orchestration, `ReflexLibrary` replay, and a `python -m early_agi.bench` stress CLI. It reports discovery@k, sequential zero-forgetting retention, repeat-task reflex speedup, real `eta_cog`, tokens, rollbacks, and gate constraints.
- Added `tests/test_ab_benchmark.py` covering the three A/B curves, offline/no-network execution, real fixture failure-to-fix gates, the live router prompt seam, and the packaged Gauge program.
- Added `docs/benchmark.md` and the `early-agi-bench` console script; packaged `early_agi/bench/netron_plastic.gauge` as wheel data and documented the CLI in README, Gauge docs, and SECURITY.
- Added `early_agi/core/adaptability.py`: minimal pytest/Z3 failure-to-constraint distillation, stratified ColdSink episode sampling, scale-invariant topology signatures, a SQLite-backed `DirectivePool`, and exactly three offline benchmark curves.
- Added `tests/test_adaptability_bench.py` and `tests/test_adaptability.py` covering discovery@k under mid-task shift, zero-regression retention across 10 sequential tasks, cold-start `eta_cog` transfer on unseen topologies, and the three integration pillars.
- Added `early_agi/core/plasticity.py`: NumPy-only `AdaptiveThresholds` / `HomeostaticRegulator`, structural `AnalogicalMatcher`, `AutonomyProfile` / `UserCoAdapter`, and `SynapticDecay` with fast weights.
- Added compatibility alias `KernelDirective` and `Directive.EPISTEMIC_PROBE`; high-novelty first ticks now request read-only epistemic probing before crystallisation.
- Added `OperatorRegistry.epistemic_probe()` and a read-only `inspect_deps` operator; probe calls skip mutating specs and isolate writes.
- Added Gauge `plastic govern`, `on high_novelty`, `probe ... with [...]`, and `recall analogy where similarity >= ...` syntax, plus `examples/netron_plastic.gauge`.
- Added structural signatures to MERA L1/L2 promotion, persisted `analogy` nodes, `MeraGraph.best_analogy()`, and `ReflexLibrary.analogical_hint()`.
- Added implicit user-feedback records in `ColdSink`, live profile injection into `DualTierRouter` / `ScriptedRouter`, and adaptive session risk/probe handling.
- Added `tests/test_plasticity.py` and `tests/test_gauge_plasticity.py`, documenting the five upgrades and Gauge syntax end to end.
- Added `EARLY_AGI_DIRECTIVE_DB`, parameterized router directives, ColdSink post-mortem mutation/evolution, and held-out `eta_cog` retention for adaptation policies.
- Added Gauge manifold `constraints` / `active_constraints`, `GATE_PROJECT` constraint distillation, and `EPISTEMIC_PROBE` preflight before mutating `OP_APPLY` instructions.
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

- **Benchmark-gated structural cleanup**: consolidated `core/plasticity.py` and `core/adaptability.py` into one `core/adaptation/` boundary (shared surprise EMA, one scale-invariant signature encoder, directive pool, constraints, episodes, and three-curve benchmark), and decomposed `session.py`, `gauge/instructions.py`, `gauge/synthesizer.py`, and `core/kernel.py` into single-responsibility modules under `session_parts/`, `gauge/instruction_parts/`, `gauge/synthesis_parts/`, and `core/kernel_parts/`. The original modules remain public compatibility façades with unchanged re-exports, class identities, module-level monkeypatch seams, and behaviour.
- The offline A/B CLI was run before and after the cleanup with `--tasks 10 --k 4 --timeout 60 --compact`; the full post-refactor JSON report is byte-for-report identical to the baseline. `eta_cog` curves, `success@k`, rollback counts, and repeat-task reflex speedups are unchanged (`repeat_token_speedup=7980.0`, `reflex_hits=7`, `network_calls=0`).
- The A/B benchmark keeps architectural boundaries explicit: no duplicate surprise/novelty tracker, no ColdSink schema migration, no AST-coupling rewrite, and no THRML import; benchmark scoping lives in temporary payloads/sink metadata and the shared `core/adaptation/` + `core/governor.py` state is the single source of truth.
- CI now smokes `python -m early_agi.bench --help`, and the wheel check verifies that the packaged `netron_plastic.gauge` program is included.
- `ThermodynamicGovernor` keeps fixed thresholds by default and, when plasticity is enabled, derives live `eta_min`, `eta_high`, `tau`, `chi_max`, novelty, exploration patience, crystallization bias, and probe policy from surprise, analogy similarity, and risk.
- `EarlyAGIKernel.tick()` accepts `predicted_drop`, `analogy_similarity`, `risk_level`, and `probe_completed`, persists plasticity checkpoints, and releases probe locks after read-only measurements; `engage_probe_lock()` enforces READ/VERIFY-only registry access.
- `GaugeVM` emits a preflight `EPISTEMIC_PROBE` after setup and before any mutating body op; `GATE_PROJECT` distills failed gates into active manifold constraints for the next tick.
- `MeraGraph.put_node()` stamps every L1/L2 node with a structural signature and decomposition strategy; `coarse_grain()` emits path-independent guidance nodes and `recall_decomposition_strategy()` recalls at cosine ≥ 0.75.
- `gauge/synthesizer.py` uses stratified historical replay plus a held-out new-task case, promotes only on average new-task `eta_cog` gain with non-degrading historical `DarkStateGate` pass rates, and gates fused/reflex promotion behind the same replay.
- `ReflexLibrary` replaces binary active/retired behavior with continuous confidence weighted by recency, surprise, and fast-weight drift.
- `CompiledReflex` carries confidence; `ReflexLibrary` skips faded habits and time/surprise-weighted fast weights archive outcomes in `reflex_outcome`.
- `AgentSession.turn()` accepts `user_overrode`, archives rejected edits/reverts/overrides/prompt brevity, and feeds the live profile into router system briefs.
- `GaugeRunResult` now reports `run_index`, fused-reflex summaries, rich synthesis replay metadata, and a `plasticity` section.
- Gauge expression built-ins, workspace/manifold state, trap evaluation, inverse-stack rollback, and opcode handlers are no longer embedded in the VM dispatcher.
- `docs/gauge.md`, `README.md`, `SECURITY.md`, and this changelog now describe the replay gate, bounded-χ backend, and optional THRML seam.

### Fixed

- `examples/netron_plastic.gauge` now evaluates `ast_check(p.files)`, `pytest_run(p.targets)`, and `p.has_regressions` against the candidate view, so the plastic example verifies real workspace files instead of returning unverifiable gates on the manifold object.
- The A/B benchmark removes stale `__pycache__` before local verifier runs, preventing full-file fixture edits from being shadowed by bytecode compiled before the patch.
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

### Path A development

- Added stall-gated offline racing with an independent sequential/reflex archive,
  explicit failed-progress evidence, and retained failed-attempt token costs.
- Split the offline racer behind its compatible facade into race models, sandbox
  lifecycle and coordinator modules; added multi-module invariant stress fixtures.
- Added NIM-compatible environment aliases, strict text roles, Retry-After/jitter
  retries, shared pacing and an opt-in eight-task live evaluation command. Live
  tests use mocked HTTP only. No live model performance claims or results.

### Live evaluation reliability

- Added timestamped task/request/grading progress, 15-second waiting heartbeats,
  redacted transport error details, and atomic per-task/interrupt checkpoints.
- Added a shared transport-failure circuit, explicit retry controls (live default:
  zero retries), live timeout default of 180 seconds, and incomplete-run exit codes.
- Added `--smoke` (one task/pass/arm/candidate), task/pass/arm selectors, and schema
  v2 completion/validity metadata so connection failures are not published as
  model-performance comparisons. Checkpoints preserve results but do not resume.

### Live scoring and telemetry correction

- Added schema v3 resource-normalized external task scoring shared by every live
  arm, with fixed end-to-end gate-completion credit and cumulative failure/branch
  costs. Report first-pass and overall means, including reflex replay; preserve
  original Gauge governance scores/directives separately. Offline parity is unchanged.
- Removed blank live constraints and stale task-preflight flags; closed passing
  attempts expose terminal CRYSTALLIZE. Zero-token reflex replay no longer yields
  an invented finite token-speedup ratio.
- Added an offline re-scoring command preserving source evidence and experiment
  validity flags, without overwriting the original report or making API calls.
