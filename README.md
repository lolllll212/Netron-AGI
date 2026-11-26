# early-agi

A binary-orchestrated, physics-backend hybrid cognitive kernel. The agent loop is
deterministic Python + SQLite; the "physics" lives behind a swappable
`RelaxationBackend` interface, so numpy SVD can be replaced by tensor networks,
a p-bit simulator (`thrml`), or an accelerator without touching the loop.

Runtime dependencies: **numpy** and **sqlite3**. The LLM layer speaks plain
OpenAI-compatible HTTP with `urllib`, so any local server (Ollama, llama.cpp,
vLLM, LM Studio) or hosted endpoint works with zero code changes.

## What is built

| Pillar | Component | Status |
| --- | --- | --- |
| — | Cold sink: append-only, hash-addressed, exact undo | working (`core/cold_sink.py`) |
| 2, 4 | SVD spectral entropy + volume-law detector | working (`core/entropy_sensor.py`) |
| — | `eta_cog` governor (CRYSTALLIZE / SHIFT_COORDINATE / RECALL_SINK) | working (`core/governor.py`) |
| 1 | AST edit + **live dual-tier router** over a real repo | working (`core/coupling.py`, `llm/`, `session.py`) |
| 2 | Non-commuting registry, commutator matrix, write isolation | working (`operators/registry.py`) |
| 3 | Dark-state verifiers: AST, Ruff, pytest, Z3 | working (`operators/dark_state.py`) |
| 3 | MERA L0→L1 store + deterministic consolidation | working (`memory/mera_graph.py`) |
| 3 | Reflex compilation: recurring verified edit → zero-LLM replay | working (`memory/reflex.py`) |
| 4 | Operator synthesis on `SHIFT` streak ≥ 2 | **not built** — `governor.synthesis_due` is the trigger |

## Layout

```
early_agi/
├── core/
│   ├── cold_sink.py        # hash-addressed SQLite reversible archive
│   ├── entropy_sensor.py   # SVD spectral entropy & volume-law detector
│   ├── governor.py         # rolling eta_cog thermodynamic controller
│   ├── coupling.py         # bipartite incidence matrix from imports + calls
│   └── kernel.py           # tick loop, directive ladder, shift enforcement
├── operators/
│   ├── registry.py         # non-commuting tools + commutator matrix
│   └── dark_state.py       # AST / Ruff / pytest / Z3 verifiers
├── memory/
│   ├── mera_graph.py       # scale-indexed SQLite renormalisation store
│   └── reflex.py           # recurring gated edits → zero-LLM replays
├── backends/
│   ├── base.py             # RelaxationBackend protocol
│   └── classical_svd.py    # O(n·χ³) numpy backend
├── llm/
│   ├── client.py           # stdlib OpenAI-compatible chat client
│   └── router.py           # fast executor / deep reasoner + scripted stand-in
└── session.py              # per-turn loop over a real repository
```

## Run it

```bash
python -m early_agi.demo     # offline: all six sections, no API key
python -m pytest -q          # 59 focused tests
```

The demo prints the three physical regimes, the coupling matrix of this package,
the commutator table with a write lock being enforced, MERA consolidation, a full
edit → verify → `eta_cog` → directive loop against a throwaway repo, and reflex
compilation replaying a learned fix with zero router calls.

### Live models

```bash
export EARLY_AGI_BASE_URL=http://localhost:11434/v1   # any OpenAI-compatible server
export EARLY_AGI_FAST_MODEL=qwen2.5-coder:7b
export EARLY_AGI_DEEP_MODEL=qwen2.5-coder:32b
# export EARLY_AGI_API_KEY=...            # only for hosted endpoints
```

```python
from early_agi.core.cold_sink import ColdSink
from early_agi.core.coupling import Constraint
from early_agi.core.kernel import EarlyAGIKernel
from early_agi.llm.router import DualTierRouter
from early_agi.operators.dark_state import build_standard_gates
from early_agi.session import AgentSession

kernel = EarlyAGIKernel(ColdSink("cold_sink.db"), gates=build_standard_gates())
session = AgentSession(
    "path/to/repo",
    kernel,
    DualTierRouter.from_settings(),
    constraints=[Constraint(id="gate:tests_pass", paths=("tests",))],
)
for _ in range(8):
    print(session.turn("fix the failing parser test").as_dict())
```

## The three invariants

1. **Nothing is deleted.** The sink rejects `UPDATE`/`DELETE` with SQLite
   triggers, and every read re-hashes the payload. Pruned branches, file
   snapshots, rejected artefacts and rejected shift proposals all live there, so
   `RECALL_SINK` is an exact restore rather than a re-prompt.
2. **Only verified collapse pays.** A gate awards `bits` the first time it
   passes and charges them back if a later edit breaks it. Unverifiable gates
   (missing `z3`, no tests collected, timeout) return `None` and are neutral —
   a missing tool can neither fake efficiency nor fake a regression.
3. **A coordinate shift physically blocks edits.** On `SHIFT_COORDINATE` the
   registry isolates mutating operators; the next tick only unlocks if it carries
   a structurally valid artefact (`decomposition` with ≥2 steps, a compiling
   `invariant` check, or a `new_operator` source that defines its name). Another
   patch holds the lock and is archived as `shift_rejected`.

## Design notes

**`eta_cog` ladder.** `efficiency = verified_bits / (w_tok·tokens + w_tool·tools + w_branch·branches)`.
Priority: two consecutive non-positive drops → `RECALL_SINK`; volume-law or
`eta < eta_high` → `SHIFT_COORDINATE`; otherwise `CRYSTALLIZE`. The sketch's
`eta <= eta_high` comparison was inverted — it routed high-yield ticks to
`SHIFT_COORDINATE` and made `CRYSTALLIZE` unreachable. Fixed here, with the
ladder covered by `tests/test_governor.py`.

**Where the coupling matrix comes from.** `RepoScanner` parses the repo, resolves
imports and calls, and builds `M[entity, constraint] = 1[in scope] + deps into
scope + deps out of scope`. Constraints are gate scopes (`gate:tests_pass` over
`tests/`) plus caller-supplied hypotheses over paths or entity ids. No random
embeddings; identical trees produce identical fingerprints.

**Scale memory.** `MeraGraph.coarse_grain()` reads high-`eta_cog` traces, builds a
gate/directive feature matrix, truncates it with the same backend the kernel uses,
and partitions records by orthant in the Schmidt basis, emitting one `rule` node
at `scale+1` with edges back to its members. Coarse nodes go through the sink, so
they inherit the append-only guarantee.

**Reflexes.** `memory/reflex.py` compiles an edit batch into a replayable
procedure once it has won `min_support` times and every win passed a gate. Firing
is gated three ways: the target file must hash to a pre-state it was learned on,
the same gates must still pass after the replay, and two misses retire it. A
replay that misses is restored before the turn ends, so memory can make the loop
cheaper but never less verified.

## Limitations

- `run_shell`/pytest execute in the repo with a timeout; that is process
  containment, **not** a security sandbox.
- Token cost is taken from the provider's `usage` field; there is no currency
  budgeting yet.
- The deep reasoner's artefacts are structurally validated and archived but not
  yet replayed against the cold sink before registration — that is Phase 4.
- Reflexes replay recorded full-file contents, so they are exact-patch memories:
they fire only on a pre-state they have literally seen and do not generalise to a
new path or a renamed symbol.
