# Gauge

Gauge is a domain-specific **cognitive and thermodynamic orchestration language**
for `early_agi`. It is not ThoughtWorks Gauge, and it is not a classical
imperative scripting language. A `.gauge` file describes scales, uncollapsed
candidate manifolds, dark-state gates, non-commuting operators, reversible
coordinate transforms, thermodynamic governance, collapse into the cold sink,
recall, and operator synthesis.

The compiler and hybrid VM are the sovereign control layer. External tools and
LLMs are called only when a manifold evolves, a coordinate shift is required, or
a `synthesize op` trap is triggered. The model is never the outer loop.

## Pipeline

1. **Frontend** — `lexer.py` produces exact span-tagged tokens; `parser.py`
   builds a span-tagged AST. Source locations survive into checker errors,
   bytecode, and runtime traces.
2. **Static algebra** — `checker.py` validates `scale`, `gate`, `extern op`,
   `after(...)` ordering, reversible transform bodies, `govern` configuration,
   trap names, recall conditions, and the rule that an uncollapsed manifold must
   not leak into a sink. Irreversible calls such as `write_file`, `run_shell`,
   `collapse`, `drop`, and `synthesize` are rejected inside a reversible body.
3. **Bytecode IR and VM** — `compiler.py` lowers a checked AST into compact
   instructions. `vm.py` is strictly the instruction dispatcher: `instructions.py`
   is the public opcode façade over `instruction_parts/` (setup, governance,
   plasticity, and expression handlers), `state.py` holds workspace/manifold
   views, `builtins.py` holds the expression vocabulary, `traps.py` owns trap
   evaluation and inverse recovery, and `results.py` holds the run report.
   Together they call the real kernel, SQLite cold sink, relaxation backend,
   dark-state verifiers, operator registry, governor, and MERA graph.
4. **Replay-gated synthesis** — `synthesizer.py` is the public façade over
   `synthesis_parts/` (models, construction/promotion, replay benchmarking, and
   hot-chain fusion). It pulls the last N high-drag `ColdSink` episodes,
   benchmarks a candidate in an isolated `python -I` subprocess against replayed
   states, requires strictly positive `eta_cog` improvement on every case and
   green `DarkStateGate` results, then registers the operator. Repeated gate-green
   `extern op` chains are fused into deterministic reflexes through
   `memory/reflex.py`.

## Constructs

| Construct | Meaning | Runtime mapping |
| --- | --- | --- |
| `scale L0 -> L2` | ultraviolet and infrared scale indices | `MeraGraph`, `ColdSink.scale` |
| `manifold p : PatchSpace { ... }` | uncollapsed candidate ensemble | `GaugeState`, `ManifoldState`, `GaugeExecutionState` |
| `gate name(p) : predicate` | tri-state dark-state predicate | `GaugeGate` / `GateResult` |
| `extern op f(...) deforms(...) commutes(...) after(...)` | ordered operator with side-effect footprint | `OperatorRegistry` / `OperatorSpec` |
| `reversible transform t(...) -> ... { ... }` | invertible coordinate re-encoding | `ReversibleTransform` + inverse stack |
| `uncall t` | replay the inverse of the latest `t` | `TrapRunner` / inverse stack snapshot restore |
| `govern eta_min = ..., chi_max = ... { ... }` | thermodynamic policy block | `ThermodynamicGovernor`, `EarlyAGIKernel` |
| `plastic govern eta_min = ..., chi_max = ..., adaptation_rate = ... { ... }` | homeostatic policy block | `HomeostaticRegulator`, live `eta_min` / `eta_high` / `tau` / `chi_max` |
| `evolve p under [g1, g2]` | measure, project, govern, tick | `ENT_MEASURE`, `GATE_PROJECT`, `GOVERN_CHECK`, `EVOLVE` |
| `on volume_law(ent) { ... }` | trap for volume-law coupling | `TrapPolicy` / `TRAP_VOLUME_LAW` |
| `on drag(eta) { ... }` | trap for rolling efficiency drag | `TrapPolicy` / `TRAP_DRAG` |
| `on frustration { ... }` | trap for recall/stall | `TrapPolicy` / `TRAP_FRUSTRATION` |
| `on high_novelty { ... }` | trap for surprise/unfamiliar structure | `TrapPolicy` / `TRAP_HIGH_NOVELTY` |
| `probe p with [read_file, inspect_deps]` | read-only active-curiosity measurements | `OperatorRegistry.epistemic_probe` |
| `recall analogy where similarity >= 0.72` | structural strategy recall | `MeraGraph` / `AnalogicalMatcher` |
| `synthesize op name reducing eta` | replay-gated self-invention hook | `Synthesizer`, `DarkStateGate`, `OperatorRegistry` |
| `collapse(p) -> sink` | explicit irreversible projection | `ColdSink.put("gauge_collapse", ...)` |
| `recall sink where scale == L0` | restore pruned/latent branch | `ColdSink.records`, kernel checkpoint, gate reconciliation |
| `x = transform(x)` | reversible assignment | `TRANSFORM_APPLY` with inverse frame |

Expressions support literals, identifiers, members, calls with keyword
arguments, lists, arithmetic, comparisons, `in`, `and`, `or`, and `not`. They
are evaluated by a safe tri-state interpreter; Python `eval()` is not used.

## Plastic governance, curiosity, and analogy

`plastic govern` enables the NumPy-only `HomeostaticRegulator`. It turns the gap
between `predicted_drop` and `verified_drop`, the best analogical similarity,
and the current risk level into live `eta_min`, `eta_high`, `tau`, `chi_max`,
novelty, and `should_probe_first` values. `adaptation_rate` sets the surprise
EMA coefficient. A high-novelty first tick emits `EPISTEMIC_PROBE` before
`CRYSTALLIZE`; the VM also emits a preflight probe after setup but before any
body `OP_APPLY`, so even compiled mutating setup operators observe the directive
first. The kernel isolates writes and the read-only `probe` statement runs only
`extern op` declarations with a `deforms(none)` footprint through
`OperatorRegistry.epistemic_probe`. Under `EPISTEMIC_PROBE` the registry allows
READ operators and read-only VERIFY baseline tests only; file writes are blocked
at `apply()`. Mutating probe operators are rejected by the checker and skipped
defensively at runtime. A failed pytest/Z3 gate is distilled into one minimal
assertion or counterexample and appended to `ManifoldState.constraints` for the
next tick, never pasted as a raw stack trace.

`recall analogy where similarity >= 0.72` encodes the workspace with a
normalised singular/Schmidt spectrum plus AST motif ratios, then queries MERA
L1/L2 `analogy` nodes. `coarse_grain()` stamps every L1/L2 rule, analogy, and
guidance node with the same signature and persists path-independent
`module_cut` / test-locking strategy metadata. `recall_decomposition_strategy()`
recalls that strategy at cosine ≥ 0.75 across unrelated file and function
names. The best match feeds back into the homeostatic thresholds and into
`ReflexLibrary.analogical_hint()`.

`examples/netron_plastic.gauge` exercises all of these constructs alongside the
existing `volume_law`, `drag`, `frustration`, `recall sink`, synthesis, and
`collapse(fix) -> sink` traps. `examples/netron_solve.gauge` remains unchanged.

## VM module boundaries

| Module | Responsibility |
| --- | --- |
| `gauge/vm.py` | public API, run orchestration, and bytecode dispatch only |
| `gauge/instructions.py` | compatibility façade and composed `GaugeOpcodeRunner` |
| `gauge/instruction_parts/` | setup, governance, plasticity, expression, and helper opcode groups |
| `gauge/synthesizer.py` | compatibility façade, composed `Synthesizer`, and monkeypatch seam |
| `gauge/synthesis_parts/` | replay cases/results, promotion, sandbox replay, and hot-chain fusion |
| `gauge/state.py` | workspace scan, manifold views, variables, and inverse snapshots |
| `gauge/builtins.py` | safe expression functions and the `GaugeGate` bridge |
| `gauge/traps.py` | trap aliasing, `uncall`, recall, and LIFO frame rollback |
| `gauge/results.py` | `GaugeRunResult` and JSON report shape |
| `gauge/runtime.py` | manifold/candidate dataclasses and reversible transform objects |
| `gauge/errors.py` | shared `GaugeRuntimeError` |

This split keeps reversibility and state ownership auditable without turning the
dispatcher into a hidden interpreter for workspace state.

## Bytecode IR

The most important opcodes are:

| Opcode | Subsystem |
| --- | --- |
| `SCALE_INIT`, `MANIFOLD_INIT` | scale hierarchy and candidate state |
| `GATE_DEF`, `GATE_PROJECT` | dark-state verifiers |
| `OP_DEF`, `OP_APPLY` | operator registry and commutator log |
| `TRANSFORM_REV`, `TRANSFORM_APPLY`, `UNCALL_TRANSFORM` | reversible transforms |
| `ENT_MEASURE` | `RelaxationBackend.measure_entanglement` |
| `GOVERN_BEGIN`, `GOVERN_CHECK`, `EVOLVE` | `ThermodynamicGovernor` and `EarlyAGIKernel.tick` |
| `TRAP_VOLUME_LAW`, `TRAP_DRAG`, `TRAP_FRUSTRATION` | `TrapPolicy` and `TrapRunner` |
| `SYNTHESIZE_OP` | replay-gated operator invention |
| `COLLAPSE_SINK`, `RECALL_SINK` | SQLite cold sink and checkpoint restore |
| `HALT` | final no-leak check |

`ProgramIR.as_dict()` is deterministic and can be archived, diffed, or inspected
with the CLI.

## Mapping to `early_agi`

The user-facing `netron_solve.gauge` example maps to these repository modules:

| Gauge concept | Existing subsystem |
| --- | --- |
| `scale` | `early_agi/memory/mera_graph.py` |
| `manifold`, `govern`, `evolve` | `core/kernel.py`, `core/governor.py`, `core/entropy_sensor.py`, `backends/` |
| `gate` | `operators/dark_state.py`, `gauge/builtins.py` |
| `extern op` | `operators/registry.py`, `core/coupling.py` |
| `collapse`, `recall sink` | `core/cold_sink.py`, `gauge/traps.py`, `memory/reflex.py` |
| `synthesize op` | `gauge/synthesizer.py`, `core/kernel.py` shift validation |
| `uncall`, inverse frames | `gauge/state.py`, `gauge/traps.py`, `gauge/runtime.py` |

`examples/netron_solve.gauge` is the canonical program. It defines two gates,
two ordered extern ops, one reversible `module_cut`, all three traps, a
`synthesize op disentangle_cut reducing eta` statement, and an explicit
`collapse(...) -> sink`.

## Bounded-χ and thermodynamic backends

`backends/classical_svd.py` remains the dense NumPy baseline. Two no-new-dependency
seams live beside it:

| Name | Implementation | Notes |
| --- | --- | --- |
| `bounded_chi` | `backends/bounded_chi.py` | randomized/iterative SVD capped by `chi_max`, streaming MPDO-style block updates, two-colour Ising block-Gibbs sampling |
| `thrml` / `thrml_informed` / `thrml_pbit` | `backends/thrml_backend.py` | optional adapter facade over the same pure-Python backend; records expected THRML APIs but does not import JAX/THRML |

```python
from early_agi.backends import get_backend

backend = get_backend("bounded_chi")
report = backend.measure_entanglement(matrix, chi_max=4, tau=0.15)
sample = backend.block_gibbs(couplings, biases, beta=0.8)
print(report.as_dict(), sample.as_dict(), backend.thrml_adapter().as_dict())
```

`get_backend("bounded_chi")` satisfies the `RelaxationBackend` protocol and can
be passed directly to `GaugeVM`/`run_gauge`. `adapter="require"` raises
`ThrmlUnavailableError` unless THRML is installed in that environment; the
default sampler remains pure Python/NumPy.

The selector is wired through the entrypoints:

```bash
export EARLY_AGI_BACKEND=bounded_chi       # Settings.from_env / .env
python -m early_agi.demo --backend bounded_chi --no-dotenv
python -m early_agi.gauge run examples/netron_solve.gauge --backend thrml --root .
python -m pip install -e ".[thrml]"        # optional JAX/THRML environment
```

`Settings.backend` defaults to `classical_svd`, is validated against
`available_backends()`, and is used by `Settings.build_runtime()` and `GaugeVM`.
`GaugeRunResult.as_dict()` reports both the requested `backend` selector and the
concrete `backend_impl` name. The optional `[thrml]` extra does not add a hard
runtime dependency; bundled aliases continue to use the pure-Python adapter seam.

## CLI

```bash
python -m early_agi.gauge parse examples/netron_solve.gauge
python -m early_agi.gauge check examples/netron_solve.gauge
python -m early_agi.gauge compile examples/netron_solve.gauge --output ir.json
python -m early_agi.gauge run examples/netron_solve.gauge --root . --output run.json
python -m early_agi.gauge run examples/netron_solve.gauge --root . --backend bounded_chi --output bounded-run.json
```

After installation, the same commands are available as `gauge` and
`early-agi-gauge`. Use `--compact` for one-line JSON, `--no-dotenv` to ignore a
local `.env`, `--backend` to select the relaxation backend for `run`, and
`--root` to choose the workspace that gates scan. `parse`, `check`, and
`compile` are offline and side-effect free. `run` writes only to the
configured cold sink (in-memory by default) and reports synthesis replay,
recall, and fused-reflex results.

## Empirical A/B harness

`early_agi/bench/` runs `netron_plastic.gauge` against real multi-file Python
bug fixtures with the `bounded_chi` backend and a shared `ReflexLibrary`. The
plastic example's gates use the candidate view (`p.files`, `p.targets`, and
`p.has_regressions`) so `pytest_run` and `ast_check` verify the actual temporary
workspace instead of the manifold object. The benchmark compares that arm with
a plain retry-loop baseline using the same router seam and reports discovery@k,
zero-forgetting retention, repeat-task reflex speedup, real `eta_cog`, tokens,
and rollbacks. See [docs/benchmark.md](benchmark.md) for commands, the offline
guarantee, and the opt-in live OpenAI-compatible adapter.

## Python API

```python
from pathlib import Path
from early_agi.config import Settings
from early_agi.gauge import check_program, compile_gauge, parse_gauge, run_gauge

source = Path("examples/netron_solve.gauge").read_text(encoding="utf-8")
program = parse_gauge(source)
report = check_program(source)
ir = compile_gauge(source)
result = run_gauge(source, ".", settings=Settings.from_env({}, load_dotenv=False))
print(report.ok, len(ir.instructions()), result.status, result.collapses)
print(result.synthesis, result.reflexes)
```

Pass `backend=` or `sink=` to `GaugeVM`/`run_gauge` to embed Gauge in another
runtime. Pass `router=` to allow model-assisted synthesis; without it, synthesis
is deterministic, replay-gated, and offline.

## Synthesis contract

A synthesized `new_operator` must:

1. parse and define the requested identifier;
2. contain no imports, attribute/dunder access, `eval`, `exec`, `open`, arbitrary
   calls, loops, or exception/try smuggling;
3. run inside an isolated `python -I` subprocess with a minimal builtins
   namespace;
4. replay the last N high-drag `ColdSink` episodes (or one bootstrap state);
5. leave every replayed input state unmutated;
6. return candidate states with strictly positive `eta_cog` improvement on
   **every** replay;
7. keep every configured `DarkStateGate` green, including the default
   `synthesis_replay_safe` gate;
8. be archived as `gauge_synthesis` only after promotion.

A rejected candidate is archived as `gauge_synthesis_rejected` with its reason.
The default deterministic operator advertises a 0.1 gain; a router-authored
artifact that cannot demonstrate the gain is not registered.

## Reflex fusion

Successful `extern op` chains are archived as `gauge_op_chain` records with their
tick, gate verdicts, task, scale, and `eta_cog`. After the same chain wins in at
least `min_support=2` distinct ticks with no failing gate,
`ReflexLibrary.compile_operator_chains()` installs a deterministic `fused_*`
`OperatorSpec` and writes an `operator_chain_reflex` MERA node linked to every
source tick. `ReflexLibrary.match_operator_chain()` exposes the compiled reflex
without re-running a model.

## Safety model

- Gauge expressions are interpreted, not evaluated as Python.
- Synthesized and fused operators are AST-restricted and benchmarked in a
  separate `python -I` process against a stratified sample of past ColdSink
  episodes plus the new task. Promotion requires average new-task `eta_cog`
  gain, green dark-state gates, and no degradation of the historical gate pass
  rate. Reflex promotion uses the same gate.
- Reversible transforms snapshot manifold state before mutation. `uncall` and
  `recall sink` restore those snapshots in `traps.py` and reconcile gate state
  without paying duplicate verified bits.
- Recall queries include `gauge_collapse` records so an explicit
  `collapse -> sink` can be replayed on later ticks.
- Mutating operators go through `OperatorRegistry.apply`, which enforces the
  coordinate-shift write lock at the actual write boundary.
- `EPISTEMIC_PROBE` locks writes before crystallisation. `probe ... with [...]`
  accepts only `deforms(none)` extern operators, and `epistemic_probe` isolates
  writes and skips any mutating spec at runtime.
- `run_shell` is disabled by default and has a wildcard commutator footprint.
- The cold sink is append-only; even deduplicated payloads retain all metadata
  commits in `state_commits`.
- Live LLM routing is disabled unless `EARLY_AGI_LLM_ENABLED` and endpoint/model
  variables are explicitly configured.
- Z3 and THRML are optional. Missing or unknown verifiers are tri-state
  `unverifiable`, never a fake pass; missing THRML never becomes a hard import.

See `SECURITY.md` for process containment limits.

## Counterfactual branch racing

Inside a `volume_law` or `frustration` handler:

```gauge
race [module_cut, dependency_cut] under [syntax_valid, tests_passing] -> winner
```

Branches must be 2–4 distinct declared reversible transforms or read-only extern
ops; mutating operations are rejected. Gates must be declared and the winner
variable fresh. Compilation emits `RACE_BRANCHES`. Runtime requires a due trap
and an explicit offline `GaugeVM(race_config=RaceConfig(...))`; without fixture
configuration it records a skip. The winner variable receives the archived winner
hash. Branch names select offline fixture strategies, not arbitrary extern-op
execution. Only a gate-green highest-eta candidate crystallises; losing evidence
and its SUPERSEDED/MERGED lineage remain append-only. The packaged default
benchmark program does not opt into racing.
