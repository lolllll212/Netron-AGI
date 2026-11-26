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
   holds opcode handlers, `state.py` holds workspace/manifold views, `builtins.py`
   holds the expression vocabulary, `traps.py` owns trap evaluation and inverse
   recovery, and `results.py` holds the run report. Together they call the real
   kernel, SQLite cold sink, relaxation backend, dark-state verifiers, operator
   registry, governor, and MERA graph.
4. **Replay-gated synthesis** — `synthesizer.py` pulls the last N high-drag
   `ColdSink` episodes, benchmarks a candidate in an isolated `python -I`
   subprocess against replayed states, requires strictly positive `eta_cog`
   improvement on every case and green `DarkStateGate` results, then registers
   the operator. Repeated gate-green `extern op` chains are fused into
   deterministic reflexes through `memory/reflex.py`.

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
| `evolve p under [g1, g2]` | measure, project, govern, tick | `ENT_MEASURE`, `GATE_PROJECT`, `GOVERN_CHECK`, `EVOLVE` |
| `on volume_law(ent) { ... }` | trap for volume-law coupling | `TrapPolicy` / `TRAP_VOLUME_LAW` |
| `on drag(eta) { ... }` | trap for rolling efficiency drag | `TrapPolicy` / `TRAP_DRAG` |
| `on frustration { ... }` | trap for recall/stall | `TrapPolicy` / `TRAP_FRUSTRATION` |
| `synthesize op name reducing eta` | replay-gated self-invention hook | `Synthesizer`, `DarkStateGate`, `OperatorRegistry` |
| `collapse(p) -> sink` | explicit irreversible projection | `ColdSink.put("gauge_collapse", ...)` |
| `recall sink where scale == L0` | restore pruned/latent branch | `ColdSink.records`, kernel checkpoint, gate reconciliation |
| `x = transform(x)` | reversible assignment | `TRANSFORM_APPLY` with inverse frame |

Expressions support literals, identifiers, members, calls with keyword
arguments, lists, arithmetic, comparisons, `in`, `and`, `or`, and `not`. They
are evaluated by a safe tri-state interpreter; Python `eval()` is not used.

## VM module boundaries

| Module | Responsibility |
| --- | --- |
| `gauge/vm.py` | public API, run orchestration, and bytecode dispatch only |
| `gauge/instructions.py` | opcode handlers and subsystem calls |
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

## CLI

```bash
python -m early_agi.gauge parse examples/netron_solve.gauge
python -m early_agi.gauge check examples/netron_solve.gauge
python -m early_agi.gauge compile examples/netron_solve.gauge --output ir.json
python -m early_agi.gauge run examples/netron_solve.gauge --root . --output run.json
```

After installation, the same commands are available as `gauge` and
`early-agi-gauge`. Use `--compact` for one-line JSON, `--no-dotenv` to ignore a
local `.env`, and `--root` to choose the workspace that gates scan. `parse`,
`check`, and `compile` are offline and side-effect free. `run` writes only to the
configured cold sink (in-memory by default) and reports synthesis replay,
recall, and fused-reflex results.

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
- Synthesized operators are AST-restricted and benchmarked in a separate
  `python -I` process; promotion never happens from source text alone.
- Reversible transforms snapshot manifold state before mutation. `uncall` and
  `recall sink` restore those snapshots in `traps.py` and reconcile gate state
  without paying duplicate verified bits.
- Recall queries include `gauge_collapse` records so an explicit
  `collapse -> sink` can be replayed on later ticks.
- Mutating operators go through `OperatorRegistry.apply`, which enforces the
  coordinate-shift write lock at the actual write boundary.
- `run_shell` is disabled by default and has a wildcard commutator footprint.
- The cold sink is append-only; even deduplicated payloads retain all metadata
  commits in `state_commits`.
- Live LLM routing is disabled unless `EARLY_AGI_LLM_ENABLED` and endpoint/model
  variables are explicitly configured.
- Z3 and THRML are optional. Missing or unknown verifiers are tri-state
  `unverifiable`, never a fake pass; missing THRML never becomes a hard import.

See `SECURITY.md` for process containment limits.
