# early-agi

[![ci](https://github.com/lolll212/Netron-AGI/actions/workflows/ci.yml/badge.svg)](https://github.com/lolll212/Netron-AGI/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![changelog](https://img.shields.io/badge/changelog-keep%20a%20changelog-informational)](CHANGELOG.md)

A binary-orchestrated, physics-backend hybrid cognitive kernel, now with
**Gauge**: a span-tagged cognitive and thermodynamic orchestration language that
compiles `.gauge` scripts into the existing `early_agi` VM. The deterministic
compiler/VM is the outer loop; an LLM is an optional tool invoked only by an
explicit operator or synthesis trap.

Runtime dependencies: **numpy** and **sqlite3**. The LLM layer speaks plain
OpenAI-compatible HTTP with `urllib`, so any local server (Ollama, llama.cpp,
vLLM, LM Studio) or hosted endpoint works with zero code changes. Live routing
is opt-in.

## What is built

| Pillar | Component | Status |
| --- | --- | --- |
| — | Cold sink: append-only, hash-addressed, exact undo, metadata commit history | working (`core/cold_sink.py`) |
| 2, 4 | SVD spectral entropy + volume-law detector | working (`core/entropy_sensor.py`) |
| — | `eta_cog` governor with rolling drag, stall, recall, and synthesis trigger | working (`core/governor.py`) |
| 1 | Transactional edit loop, brief allowlist, write-boundary lock, exact rollback | working (`session.py`) |
| 2 | Non-commuting registry, tri-state verifier logging, wildcard race audit | working (`operators/registry.py`) |
| 3 | Dark-state verifiers: AST, Ruff, pytest, real Z3 solver semantics | working (`operators/dark_state.py`) |
| 3 | Qualified call/import coupling scanner with explicit truncation reporting | working (`core/coupling.py`) |
| 3 | MERA L0→L1 store using the backend `truncate`/`left_basis` contract | working (`memory/mera_graph.py`) |
| 3 | Task-aware reflex compilation: recurring verified edit → zero-LLM replay | working (`memory/reflex.py`) |
| 5 | Homeostatic thresholds, structural analogy, active probes, graceful unlearning, user co-adaptation | working (`core/plasticity.py` façade, `core/adaptation/`) |
| 1–3 | Three-pillar adaptability: TTA surprise/patience, continual stratified replay, meta-transfer directive pool | working (`core/adaptability.py` façade, `core/adaptation/`, `gauge/synthesizer.py`, `llm/router.py`, `session.py`) |
| 4 | Three offline adaptation curves: discovery@k, zero-regression retention, cold-start eta_cog | working (`tests/test_adaptability_bench.py`) |
| A/B | Real multi-file repair benchmark: offline/live router seam, discovery@k, zero-forgetting retention, repeat-task reflex speedup | working (`bench/`, `docs/benchmark.md`) |
| 4 | **Gauge frontend**: lexer, spans, AST, parser, static checker, IR, compiler | working (`gauge/`, `examples/`, `docs/gauge.md`) |
| 4 | **Hybrid Thermodynamic VM**: real sink/backend/governor/gates/registry | working (`gauge/vm.py`) |
| 4 | **Validated operator synthesis**: offline default, stratified replay, positive `eta_cog` gain, no historical gate regression | working (`gauge/synthesizer.py` façade, `gauge/synthesis_parts/`) |

## Layout

```text
early_agi/
├── core/
│   ├── adaptation/         # one surprise/signature/policy/benchmark boundary
│   │   ├── signatures.py   # canonical SVD encoder + compatibility data views
│   │   ├── surprise.py     # shared surprise EMA, thresholds, and decay
│   │   ├── co_adaptation.py# implicit-feedback autonomy profile
│   │   ├── constraints.py  # compact gate-failure distillation
│   │   ├── episodes.py     # stratified ColdSink episode sampling
│   │   ├── directives.py   # SQLite-backed directive pool
│   │   └── benchmarks.py   # exactly three offline adaptation curves
│   ├── kernel_parts/       # shift validation + tick loop behind kernel.py
│   ├── cold_sink.py        # hash-addressed SQLite reversible archive
│   ├── entropy_sensor.py   # SVD spectral entropy & volume-law detector
│   ├── governor.py         # adaptive eta_cog thermodynamic controller
│   ├── plasticity.py       # public compatibility façade
│   ├── adaptability.py     # public compatibility façade
│   ├── coupling.py         # bipartite incidence matrix from imports + calls
│   └── kernel.py           # public compatibility façade + tick result types
├── gauge/
│   ├── span.py             # exact source locations
│   ├── ast.py              # span-tagged Gauge AST
│   ├── lexer.py            # tokens with 1-based, exclusive end columns
│   ├── parser.py           # declarations, expressions, traps, collapse, uncall
│   ├── checker.py          # algebra, reversibility, after(), and no-leak checks
│   ├── expressions.py      # safe tri-state evaluator (no eval)
│   ├── ir.py               # compact bytecode IR
│   ├── compiler.py         # checked AST -> deterministic VM instructions
│   ├── synthesizer.py      # public synthesis façade + monkeypatch seam
│   ├── synthesis_parts/    # models, promotion, replay, and hot-chain fusion
│   ├── runtime.py          # manifolds, candidate views, inverse frames
│   ├── errors.py           # shared runtime error
│   ├── builtins.py         # safe expression functions and GaugeGate
│   ├── state.py            # workspace/manifold views and inverse snapshots
│   ├── traps.py            # trap aliasing, uncall, and rollback
│   ├── instructions.py     # public composed opcode-runner façade
│   ├── instruction_parts/  # setup, governance, plasticity, expression handlers
│   ├── results.py          # structured run report
│   ├── vm.py               # thin bytecode dispatcher / public API
│   └── cli.py              # python -m early_agi.gauge / gauge console script
├── operators/
│   ├── registry.py         # non-commuting tools + write isolation
│   └── dark_state.py       # AST / Ruff / pytest / Z3 verifiers
├── memory/
│   ├── mera_graph.py       # scale-indexed SQLite renormalisation store
│   └── reflex.py           # gated edit + operator-chain replay reflexes
├── backends/
│   ├── base.py             # RelaxationBackend protocol
│   ├── classical_svd.py    # dense numpy SVD baseline
│   ├── bounded_chi.py      # randomized truncation, streaming MPDO, block Gibbs
│   └── thrml_backend.py    # optional THRML/JAX adapter seam
├── llm/
│   ├── client.py           # stdlib OpenAI-compatible chat client
│   └── router.py           # fast executor / deep reasoner + scripted stand-in
├── bench/
│   ├── contracts.py        # PatchRequest / PatchResponse router seam
│   ├── fixtures.py         # real multi-file Python bug repositories
│   ├── routers.py          # deterministic offline + opt-in live routers
│   ├── gates.py            # transactional IO, pytest gates, Gauge orchestration
│   ├── records.py          # immutable attempt/task/arm report records
│   ├── metrics.py          # discovery, retention, and reflex curves
│   ├── runner.py           # task engine, ColdSink reflexes, retention replay
│   ├── cli.py              # python -m early_agi.bench stress runner
│   └── netron_plastic.gauge # packaged copy of the plastic Gauge program
├── config.py               # strict env + .env wiring and runtime builder
├── session.py              # public AgentSession façade
└── session_parts/          # turn loop, learning, archives, and workspace helpers

benchmarks/structural-cleanup/ # pre/post-refactor A/B parity lock
examples/netron_solve.gauge  # canonical Gauge program
docs/architecture-boundaries.md # ColdSink / coupling / THRML contracts
docs/gauge.md                # language, VM, safety, and mapping documentation
docs/benchmark.md            # offline/live A/B repair benchmark guide
```

## Install

Requires Python ≥ 3.10. Nothing here needs a GPU, a model, or a network.

```bash
python -m pip install -e .             # runtime: numpy only
python -m pip install -e ".[dev]"      # + pytest and pinned ruff
python -m pip install -e ".[verify]"   # + z3-solver for the SMT gate
python -m pip install -e ".[thrml]"    # optional JAX/THRML adapter environment
```

## Run Gauge

```bash
python -m early_agi.gauge check examples/netron_solve.gauge
python -m early_agi.gauge compile examples/netron_solve.gauge --output ir.json
python -m early_agi.gauge run examples/netron_solve.gauge --root . --output run.json
python -m early_agi.gauge run examples/netron_solve.gauge --root . --backend bounded_chi --output bounded-run.json
python -m early_agi.gauge check examples/netron_plastic.gauge
python -m early_agi.gauge run examples/netron_plastic.gauge --root . --output plastic-run.json
```

`check` reports static algebra/reversibility errors. `compile` shows the exact
bytecode IR. `run` dispatches the IR against the SQLite cold sink, configurable
relaxation backend, governor, dark-state predicates, operator registry, and MERA
graph. Select the backend with `--backend`, `EARLY_AGI_BACKEND`, or `.env`;
`classical_svd`, `bounded_chi`, `thrml`, `thrml_informed`, and `thrml_pbit` are
registered. The `thrml` aliases use the bundled pure-Python block-Gibbs backend;
the optional extra only installs a JAX/THRML adapter environment. Operator
synthesis is replay-gated, and successful extern-op chains can fuse into
deterministic reflexes. All commands are offline; `run` uses an in-memory sink
by default. See [docs/gauge.md](docs/gauge.md) for the language specification,
VM module boundaries, and subsystem mapping.

## Run the kernel and tests

```bash
python -m early_agi.demo     # offline: all seven sections, no API key
python -m early_agi.demo --backend bounded_chi --no-dotenv
python -m pytest -q          # focused unit, integration, Gauge, and regression tests
ruff check .
ruff format --check .
```

The demo prints the three physical regimes, the coupling matrix of this package,
the commutator table with a write lock being enforced, MERA consolidation, a
transactional edit → verify → `eta_cog` → directive loop, task-aware reflex
compilation replaying a learned fix with zero router calls, and a bounded-χ
factorization/block-Gibbs backend exercise. `--backend` or `EARLY_AGI_BACKEND`
selects the relaxation backend used by the kernel, MERA graph, and Gauge runtime.

## Run the A/B repair benchmark

```bash
python -m early_agi.bench --tasks 10 --k 4 --timeout 60 --output bench-ab.json
python -m early_agi.bench --tasks 5 --k 3 --compact      # faster smoke
```

This is the empirical stress companion to the fast, exactly-three-curve
`tests/test_adaptability_bench.py` smoke check. It writes real multi-file Python
bug fixtures, runs local pytest gates, and compares a plain retry-loop baseline
with `netron_plastic.gauge` + `bounded_chi` + `ReflexLibrary`. The JSON report
logs cumulative `discovery@k`, sequential zero-forgetting retention, repeat-task
reflex speedups, real `eta_cog`, tokens, rollbacks, and gate constraints. The
default router is deterministic and offline; no network or external model call
is made. See [docs/benchmark.md](docs/benchmark.md) for the router seam, live
opt-in command, fixture safety model, and architectural boundaries.

### Live models

Live dual-tier routing is disabled by default. Copy `.env.example` to `.env` or
export these variables:

```bash
export EARLY_AGI_LLM_ENABLED=1
export EARLY_AGI_BASE_URL=http://localhost:11434/v1
export EARLY_AGI_FAST_MODEL=qwen2.5-coder:7b
export EARLY_AGI_DEEP_MODEL=qwen2.5-coder:32b
# export EARLY_AGI_API_KEY=...            # only for hosted endpoints
```

Invalid numeric configuration fails fast. `EARLY_AGI_BACKEND`,
`EARLY_AGI_CHI_MAX`, `EARLY_AGI_TAU`, `EARLY_AGI_ETA_HIGH`,
`EARLY_AGI_ETA_MIN`, `EARLY_AGI_SINK`, `EARLY_AGI_MERA`,
`EARLY_AGI_DIRECTIVE_DB`, and `EARLY_AGI_MAX_FILES` are wired into
`Settings.build_runtime()`, the directive pool, and the demo. The A/B benchmark
reuses the same client but keeps live routing behind both
`--router openai` and `--allow-live`; its default path remains fully offline.

```python
from early_agi.config import Settings
from early_agi.core.coupling import Constraint
from early_agi.llm.router import DualTierRouter, ScriptedRouter
from early_agi.operators.dark_state import build_standard_gates
from early_agi.session import AgentSession

settings = Settings.from_env()
runtime = settings.build_runtime("path/to/repo", gates=build_standard_gates())
router = DualTierRouter.from_settings(settings) if settings.llm_configured else ScriptedRouter()
session = AgentSession(
    "path/to/repo",
    runtime.kernel,
    router,
    scanner=runtime.scanner,
    constraints=[Constraint(id="gate:tests_pass", paths=("tests",))],
)
for _ in range(8):
    print(session.turn("fix the failing parser test").as_dict())
```

## The three invariants

1. **Nothing is deleted.** The sink rejects `UPDATE`/`DELETE` with SQLite
   triggers, verifies reads by content hash, persists compression format, and
   records every metadata commit even when payloads deduplicate. Pruned branches,
   file snapshots, rejected artifacts, and rejected shift proposals all live
   there, so `RECALL_SINK` is an exact restore rather than a re-prompt.
2. **Only verified collapse pays.** A gate awards `bits` the first time it
   passes and charges them back if a later edit breaks it. Gate state is
   reconciled after a revert, so the same gate cannot be awarded twice for a
   branch that was rolled back. Unverifiable gates (missing Z3, no tests
   collected, timeout) are tri-state `None` and neutral.
3. **A coordinate shift physically blocks edits.** On `SHIFT_COORDINATE` the
   registry isolates mutating operators at the actual `write_file` boundary. The
   next tick unlocks only after an artifact passes structural *and* semantic
   validation: a distinct decomposition, a safe invariant that is not false for
   the candidate state, or a restricted `new_operator` that runs without
   mutating its input. Another patch holds the lock and is archived as
   `shift_rejected`.

## Design notes

**`eta_cog` ladder.** `efficiency = verified_bits / (w_tok·tokens + w_tool·tools + w_branch·branches + w_latency·latency)`.
Priority: two consecutive non-positive drops → `RECALL_SINK`; volume-law or
`eta < eta_high` or rolling drag → `SHIFT_COORDINATE`; otherwise `CRYSTALLIZE`.
The sketch's `eta <= eta_high` comparison was inverted; the rolling window and
deep-tier token/latency accounting are covered by regression tests.

**Where the coupling matrix comes from.** `RepoScanner` parses the repo, resolves
relative and qualified imports/calls, builds `M[entity, constraint]`, and reports
when a file cap truncates the scan. No random embeddings; identical trees produce
identical fingerprints.

**Scale memory.** `MeraGraph.coarse_grain()` builds a gate/directive feature
matrix, calls the configured backend's `truncate()` and `left_basis()`, and
partitions records by orthant in the Schmidt basis, emitting `rule` nodes at
`scale+1` with edges back to members. `get_backend("bounded_chi")` provides a
randomized rank-capped alternative plus deterministic Ising block-Gibbs sampling;
THRML/JAX remains an optional adapter, never a required dependency.

**Reflexes.** `memory/reflex.py` compiles an edit batch into a replayable
procedure once it has won `min_support` times and every win passed a gate. It
also fuses repeated gate-green Gauge operator chains into deterministic
`fused_*` registry operators linked to their source ticks. A file reflex fires
only when the task signature, target pre-state hashes, and expected gates all
match. Failed replays are restored and repeated misses retire it.

**Gauge.** The language keeps policy and constraint graphs out of hardcoded
Python loops. `.gauge` scripts compile to deterministic IR; `vm.py` is a thin
dispatcher with focused modules for instructions, state, built-ins, traps, and
results. `instructions.py` and `synthesizer.py` are compatibility façades over
`instruction_parts/` and `synthesis_parts/`. See
[docs/gauge.md](docs/gauge.md).

**Plasticity and adaptability.** `core/plasticity.py` and
`core/adaptability.py` are public compatibility façades over one
`core/adaptation/` boundary: a NumPy-only homeostatic regulator, shared surprise
EMA, one scale-invariant signature encoder, synaptic decay, implicit user
co-adaptation, compact failure constraints, stratified ColdSink sampling, a
SQLite-backed directive pool, and exactly three local benchmark curves. The
runtime still sees `motif_ratios`; the adaptation view still sees `motifs`; both
share the same encoder and no duplicate novelty tracker. `ThermodynamicGovernor`
adapts `eta_min`, `eta_high`, `tau`, `chi_max`, novelty, exploration patience,
and `should_probe_first` from predicted/verified surprise, analogy similarity,
and risk. High surprise relaxes thresholds and grants exploration patience; low
surprise tightens the crystallization envelope. `MeraGraph` stamps every L1/L2
node with a structural signature and recalls `module_cut` / test-locking order
at cosine ≥ 0.75;
`ReflexLibrary` uses continuous confidence instead of a binary retired flag.
`GaugeVM` emits `EPISTEMIC_PROBE` before mutating setup ops, and
`OperatorRegistry` allows only READ/VERIFY operators under that directive.
`gauge/synthesizer.py` gates fused operators and reflex promotion behind
stratified sandbox replay with new-task `eta_cog` gain and non-degrading
historical `DarkStateGate` pass rates. `llm/router.py` replaces sole reliance on
static prompts with evolved directive briefs; `AgentSession` archives accept
rate, manual overrides, prompt length, and autonomy policy in `user_feedback`.
`tests/test_adaptability_bench.py` measures exactly three local curves without
external APIs. `examples/netron_plastic.gauge` demonstrates the Gauge syntax.

**Empirical A/B harness.** `early_agi/bench/` is split into focused contract,
fixture, router, gate, record, metric, runner, and CLI modules, keeping every
source file below the old `vm.py` monolith threshold. It adds no second surprise
or novelty state: the Netron arm consumes the existing homeostatic governor and
Gauge plastic program, while the baseline uses the same `eta_cog` cost formula
with fixed thresholds. ColdSink schema, AST coupling, and THRML seams stay
unchanged; benchmark scoping lives in temporary workspace payloads and sink
metadata. The fast three-curve pytest smoke remains a synthetic unit check, and
the CLI stress runner is the path for real multi-file discovery@k, sequential
zero-forgetting, and repeat-task reflex measurements. It is also the regression
lock for the structural cleanup: `benchmarks/structural-cleanup/` stores the
pre-refactor report, the identical post-refactor report, and a checker for
`eta_cog`, `success@k`, rollbacks, and repeat-task reflex speedups. See
[docs/benchmark.md](docs/benchmark.md) and
[docs/architecture-boundaries.md](docs/architecture-boundaries.md).

## Limitations

- `run_shell` is disabled by default. When explicitly enabled, it executes in
  the repository with a timeout; that is process containment, **not** a security
  sandbox. `pytest`/`ruff` subprocess verifiers have the same trust boundary.
- Token cost is taken from the provider's `usage` field; there is no currency
  budgeting yet. The offline A/B router uses deterministic character-based
  estimates; a live local-model run is required to claim model-quality gains.
- Semantic shift validation is deliberately conservative: an invariant that
  references unavailable values is accepted structurally, never treated as a
  verified pass.
- Reflexes replay recorded full-file contents and are pinned to seen task
  signatures and pre-states; they do not generalise to renamed symbols.
- Gauge's default synthesizer is deterministic and offline. An optional router
  can propose an operator, but promotion still requires isolated replay,
  strict `eta_cog` improvement, and green dark-state gates. THRML is an
  optional adapter; the bundled sampler is pure Python/NumPy.

## Project policies

- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [License](LICENSE)
