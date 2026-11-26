# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
