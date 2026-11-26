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
| 4 | **Gauge frontend**: lexer, spans, AST, parser, static checker, IR, compiler | working (`gauge/`, `examples/`, `docs/gauge.md`) |
| 4 | **Hybrid Thermodynamic VM**: real sink/backend/governor/gates/registry | working (`gauge/vm.py`) |
| 4 | **Validated operator synthesis**: offline default, positive efficiency gain required | working (`gauge/synthesizer.py`) |

## Layout

```text
early_agi/
├── core/
│   ├── cold_sink.py        # hash-addressed SQLite reversible archive
│   ├── entropy_sensor.py   # SVD spectral entropy & volume-law detector
│   ├── governor.py         # rolling eta_cog thermodynamic controller
│   ├── coupling.py         # bipartite incidence matrix from imports + calls
│   └── kernel.py           # tick loop, semantic shift verification, checkpoints
├── gauge/
│   ├── span.py             # exact source locations
│   ├── ast.py              # span-tagged Gauge AST
│   ├── lexer.py            # tokens with 1-based, exclusive end columns
│   ├── parser.py           # declarations, expressions, traps, collapse, uncall
│   ├── checker.py          # algebra, reversibility, after(), and no-leak checks
│   ├── expressions.py      # safe tri-state evaluator (no eval)
│   ├── ir.py               # compact bytecode IR
│   ├── compiler.py         # checked AST -> deterministic VM instructions
│   ├── synthesizer.py      # replay-gated promotion + hot-chain fusion
│   ├── runtime.py          # manifolds, candidate views, inverse frames
│   ├── errors.py           # shared runtime error
│   ├── builtins.py         # safe expression functions and GaugeGate
│   ├── state.py            # workspace/manifold views and inverse snapshots
│   ├── traps.py            # trap aliasing, uncall, and rollback
│   ├── instructions.py     # opcode handlers
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
├── config.py               # strict env + .env wiring and runtime builder
└── session.py              # per-turn transactional loop over a real repository

examples/netron_solve.gauge  # canonical Gauge program
docs/gauge.md                # language, VM, safety, and mapping documentation
```

## Install

Requires Python ≥ 3.10. Nothing here needs a GPU, a model, or a network.

```bash
python -m pip install -e .             # runtime: numpy only
python -m pip install -e ".[dev]"      # + pytest and pinned ruff
python -m pip install -e ".[verify]"   # + z3-solver for the SMT gate
```

## Run Gauge

```bash
python -m early_agi.gauge check examples/netron_solve.gauge
python -m early_agi.gauge compile examples/netron_solve.gauge --output ir.json
python -m early_agi.gauge run examples/netron_solve.gauge --root . --output run.json
```

`check` reports static algebra/reversibility errors. `compile` shows the exact
bytecode IR. `run` dispatches the IR against the SQLite cold sink, configurable
relaxation backend, governor, dark-state predicates, operator registry, and MERA
graph. Operator synthesis is replay-gated, and successful extern-op chains can
fuse into deterministic reflexes. All commands are offline; `run` uses an
in-memory sink by default. See [docs/gauge.md](docs/gauge.md) for the language
specification, VM module boundaries, and subsystem mapping.

## Run the kernel and tests

```bash
python -m early_agi.demo     # offline: all six sections, no API key
python -m pytest -q          # focused unit, integration, Gauge, and regression tests
ruff check .
ruff format --check .
```

The demo prints the three physical regimes, the coupling matrix of this package,
the commutator table with a write lock being enforced, MERA consolidation, a
transactional edit → verify → `eta_cog` → directive loop, and task-aware reflex
compilation replaying a learned fix with zero router calls.

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

Invalid numeric configuration fails fast. `EARLY_AGI_CHI_MAX`, `EARLY_AGI_TAU`,
`EARLY_AGI_ETA_HIGH`, `EARLY_AGI_ETA_MIN`, `EARLY_AGI_SINK`, `EARLY_AGI_MERA`,
and `EARLY_AGI_MAX_FILES` are wired into `Settings.build_runtime()` and the demo.

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
results. See [docs/gauge.md](docs/gauge.md).

## Limitations

- `run_shell` is disabled by default. When explicitly enabled, it executes in
  the repository with a timeout; that is process containment, **not** a security
  sandbox. `pytest`/`ruff` subprocess verifiers have the same trust boundary.
- Token cost is taken from the provider's `usage` field; there is no currency
  budgeting yet.
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
