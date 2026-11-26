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
   instructions. `vm.py` executes them against the real kernel, SQLite cold
   sink, SVD/tensor backend contract, dark-state verifiers, operator registry,
   governor, and MERA graph.
4. **Synthesis** — `synthesizer.py` produces an offline deterministic operator
   by default, or asks an optional router for a `new_operator` shift artifact.
   The VM executes the source in a restricted namespace and registers it only
   after a strictly positive efficiency improvement is measured.

## Constructs

| Construct | Meaning | Runtime mapping |
| --- | --- | --- |
| `scale L0 -> L2` | ultraviolet and infrared scale indices | `MeraGraph`, `ColdSink.scale` |
| `manifold p : PatchSpace { ... }` | uncollapsed candidate ensemble | `GaugeState`, `ManifoldState` |
| `gate name(p) : predicate` | tri-state dark-state predicate | `GaugeGate` / `GateResult` |
| `extern op f(...) deforms(...) commutes(...) after(...)` | ordered operator with side-effect footprint | `OperatorRegistry` / `OperatorSpec` |
| `reversible transform t(...) -> ... { ... }` | invertible coordinate re-encoding | `ReversibleTransform` + inverse stack |
| `uncall t` | replay the inverse of the latest `t` | inverse stack snapshot restore |
| `govern eta_min = ..., chi_max = ... { ... }` | thermodynamic policy block | `ThermodynamicGovernor`, `EarlyAGIKernel` |
| `evolve p under [g1, g2]` | measure, project, govern, tick | `ENT_MEASURE`, `GATE_PROJECT`, `GOVERN_CHECK`, `EVOLVE` |
| `on volume_law(ent) { ... }` | trap for volume-law coupling | `TRAP_VOLUME_LAW` |
| `on drag(eta) { ... }` | trap for rolling efficiency drag | `TRAP_DRAG` |
| `on frustration { ... }` | trap for recall/stall | `TRAP_FRUSTRATION` |
| `synthesize op name reducing eta` | self-invention hook | `Synthesizer`, efficiency check |
| `collapse(p) -> sink` | explicit irreversible projection | `ColdSink.put("gauge_collapse", ...)` |
| `recall sink where scale == L0` | restore pruned/latent branch | `ColdSink.records`, kernel checkpoint, gate reconciliation |
| `x = transform(x)` | reversible assignment | `TRANSFORM_APPLY` with inverse frame |

Expressions support literals, identifiers, members, calls with keyword
arguments, lists, arithmetic, comparisons, `in`, `and`, `or`, and `not`. They
are evaluated by a safe tri-state interpreter; Python `eval()` is not used.

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
| `TRAP_VOLUME_LAW`, `TRAP_DRAG`, `TRAP_FRUSTRATION` | thermodynamic control flow |
| `SYNTHESIZE_OP` | validated operator invention |
| `COLLAPSE_SINK`, `RECALL_SINK` | SQLite cold sink and checkpoint restore |
| `HALT` | final no-leak check |

`ProgramIR.as_dict()` is deterministic and can be archived, diffed, or inspected
with the CLI.

## Mapping to `early_agi`

The user-facing `netron_solve.gauge` example maps to these repository modules:

| Gauge concept | Existing subsystem |
| --- | --- |
| `scale` | `early_agi/memory/mera_graph.py` |
| `manifold`, `govern`, `evolve` | `early_agi/core/kernel.py`, `core/governor.py`, `core/entropy_sensor.py`, `backends/classical_svd.py` |
| `gate` | `early_agi/operators/dark_state.py`, `early_agi/gauge/vm.py` |
| `extern op` | `early_agi/operators/registry.py`, `early_agi/core/coupling.py` |
| `collapse`, `recall sink` | `early_agi/core/cold_sink.py`, `early_agi/memory/reflex.py` |
| `synthesize op` | `early_agi/gauge/synthesizer.py`, `core/kernel.py` shift validation |

`examples/netron_solve.gauge` is the canonical program. It defines two gates,
two ordered extern ops, one reversible `module_cut`, all three traps, a
`synthesize op disentangle_cut reducing eta` statement, and an explicit
`collapse(...) -> sink`.

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
configured cold sink (in-memory by default).

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
```

Pass `backend=` or `sink=` to `GaugeVM`/`run_gauge` to embed Gauge in another
runtime. Pass `router=` to allow model-assisted synthesis; without it, synthesis
is deterministic and offline.

## Synthesis contract

A synthesized `new_operator` must:

1. parse and define the requested identifier;
2. contain no imports, dunder access, `eval`, `exec`, `open`, or arbitrary calls;
3. run in a restricted builtins namespace;
4. leave its input state unmutated;
5. return a candidate state with a strictly positive `gauge_metric_gain`;
6. be archived as `gauge_synthesis` before registration.

A rejected candidate is archived as `gauge_synthesis_rejected` with its reason.
The default deterministic operator returns a 0.1 gain; a router-authored
artifact that cannot demonstrate a gain is not registered.

## Safety model

- Gauge expressions are interpreted, not evaluated as Python.
- Reversible transforms snapshot manifold state before mutation. `uncall` and
  `recall sink` restore those snapshots and reconcile gate state without paying
  duplicate verified bits.
- Mutating operators go through `OperatorRegistry.apply`, which enforces the
  coordinate-shift write lock at the actual write boundary.
- `run_shell` is disabled by default and has a wildcard commutator footprint.
- The cold sink is append-only; even deduplicated payloads retain all metadata
  commits in `state_commits`.
- Live LLM routing is disabled unless `EARLY_AGI_LLM_ENABLED` and endpoint/model
  variables are explicitly configured.
- Z3 is optional. Missing or unknown verifiers are tri-state `unverifiable`,
  never a fake pass.

See `SECURITY.md` for process containment limits.
