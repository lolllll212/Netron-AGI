# Offline A/B repair benchmark

The fast `tests/test_adaptability_bench.py` suite remains a deterministic smoke
check for exactly three synthetic adaptation curves. The A/B stress runner in
`early_agi/bench/` is the empirical complement: it creates real multi-file Python
bug repositories, runs real local pytest gates, and compares a plain retry-loop
baseline with the Netron-AGI loop.

## Run it

```bash
# 10 sequential tasks, 4 attempts per task, offline deterministic router
python -m early_agi.bench --tasks 10 --k 4 --timeout 60 --output bench-ab.json

# installed console script
early-agi-bench --tasks 10 --k 4 --compact --output bench-ab.json
```

The default router is `OfflineFixtureRouter`: no sockets, no hosted APIs, and no
external model calls. The runner writes a JSON report with:

- `discovery@k`: cumulative first-encounter success, attempts, tokens, and real
  `eta_cog` for both arms;
- `zero_forgetting_retention`: sequential replays of every solved workspace's
  AST + pytest gates after each task, with a non-degrading pass-rate curve;
- `repeat_task_reflex_speedup`: repeat-task attempts, tokens, rollbacks,
  `eta_cog`, and `ReflexLibrary` hits.

Every arm and attempt also records prompt/completion tokens, deterministic
latency charges, gate verdicts, distilled constraints, directives, and rollback
counts. The CLI exits non-zero when a task fails, a historical workspace
regresses, or no reflex speedup is observed.

## What is compared

| Arm | Controller | Verification | Memory |
| --- | --- | --- | --- |
| `plain_retry` | retry loop over the shared router | `ast_syntax_gate` + `pytest_gate` | none |
| `netron` | `netron_plastic.gauge` with `bounded_chi` | Gauge `ast_check(p.files)` + `pytest_run(p.targets)` gates | append-only `ColdSink`, `MeraGraph`, `ReflexLibrary` |

Both arms receive the same `PatchRequest` schema and the same fixture sources.
The baseline sees failed-test feedback only. The Netron arm first runs the
plastic Gauge program as a read-only preflight probe; failed gates are distilled
into active manifold constraints and passed to the router. The offline router
uses those constraints as a deterministic stand-in for a model that can exploit
probe feedback. A successful first-encounter edit is archived as `uv_state` and
`applied_edit` sink records, compiled into an exact-prestate reflex, and on a
repeat task replayed with zero router tokens. The replay still faces the same
Gauge gates and is restored if expected gates do not pass.

## Router seam

`PatchRouter.propose(PatchRequest) -> PatchResponse` is the only model-facing
interface. `PatchRequest.messages()` returns OpenAI-compatible chat messages
with the task, source files, test targets, editable allowlist, active manifold
constraints, and prior failure constraints. `PatchResponse` carries full-file
edits plus token and latency accounting. The offline router estimates tokens
from prompt/completion character counts; the live router uses provider usage.

To point the same fixtures and metrics at a local OpenAI-compatible endpoint
later:

```bash
export EARLY_AGI_BASE_URL=http://localhost:11434/v1
export EARLY_AGI_FAST_MODEL=qwen2.5-coder:7b
python -m early_agi.bench --router openai --allow-live --tasks 10 --k 4 \
  --output bench-live.json
```

The `--allow-live` flag is required. Without it, the live adapter raises
`LiveRouterDisabled` before any network call. API keys are read from an
environment variable named by `--api-key-env` and are never included in reports.

```python
from early_agi.bench import OpenAICompatiblePatchRouter, run_ab_benchmark

router = OpenAICompatiblePatchRouter(
    base_url="http://localhost:11434/v1",
    model="qwen2.5-coder:7b",
    enabled=True,
)
report = run_ab_benchmark(router=router, sequential_tasks=10, k=4)
```

## Fixtures and safety

`builtin_fixtures()` provides three deterministic repositories. Each bug spans
two Python modules; each candidate patch is a full-file edit. Test files and
`conftest.py` are excluded from the editable allowlist, so a router cannot make
a failing test pass by editing the verifier. Workspaces live in temporary
directories, full-file edits are transactional, stale `__pycache__` is cleared
before verifier runs, and failed patches are restored from byte snapshots.

Archived reflex evidence is append-only and verified: only edits that survived
Gauge gates become `applied_edit` records. `ReflexLibrary` still requires the
task signature, exact pre-state hashes, and expected gates to match.

## Architectural boundaries

This benchmark intentionally does not add another surprise/novelty controller.
Surprise EMA, adaptive thresholds, structural signatures, and probe policy now
share one implementation boundary in `core/adaptation/`; `core/plasticity.py`
and `core/adaptability.py` are backwards-compatible façades. `ThermodynamicGovernor`
still consumes the same live state, `netron_plastic.gauge` and `GaugeVM` still own
orchestration, and the benchmark only records their outputs into A/B curves.
Upgrade 2 adds scoped WAL commits and branch leases to `ColdSink`, while keeping
its content hashes and default offline A/B output stable. The pure-Python AST
coupling scanner and THRML adapter remain unchanged.
See [architecture-boundaries.md](architecture-boundaries.md)
for those three low-level contracts. The package is split into small modules
(`contracts.py`, `fixtures.py`, `routers.py`, `gates.py`, `records.py`,
`metrics.py`, `runner.py`, `cli.py`) to avoid another monolithic control file;
the largest module remains well below the old 40 KB `vm.py` threshold.

## Structural-cleanup parity

The structural cleanup used this CLI as a zero-behaviour-change regression lock.
A 10-task, `k=4`, offline baseline was captured before any refactor and is saved
in `benchmarks/structural-cleanup/baseline-report.json`; the compact summary is
`baseline-summary.json`. After decomposing the session, Gauge instruction,
Gauge synthesis, kernel, and adaptation modules, the same command produced
`after-refactor-report.json`, which is byte-for-report identical to the baseline.
Both reports have `offline=true`, `network_calls=0`, and `summary.ok=true`.

```bash
python -m early_agi.bench --tasks 10 --k 4 --timeout 60 --compact \
  --output /tmp/structural-cleanup-ab-after.json
python benchmarks/structural-cleanup/check_parity.py \
  /tmp/structural-cleanup-ab-after.json
```

Locked values include Netron/plain `eta_cog` arm means `0.3561` / `0.907019`,
discovery `eta_cog` `0.337667` / `0.906125`, cumulative `success@k` `[1,1,1,1]`
/ `[0,1,1,1]`, arm rollbacks `0` / `10`, repeat-task rollbacks `[0]*7` / `[1]*7`,
and repeat-token speedup `7980.0` with `7` reflex hits. `parity.json` records
these values explicitly. The checker rejects any full-report drift, renamed
curve, changed `eta_cog`, changed `success@k`, changed rollback count, or changed
repeat-task reflex speedup.

## Limitations

The default offline router is a deterministic model stand-in, not a language
model. It proves the harness, gate integration, transactional rollback, reflex
lineage, and metric wiring without external APIs. A live local-model run is
required to claim model-quality gains. Token counts are deterministic estimates
offline and provider-reported on live runs; there is still no currency budget.
`pytest` subprocesses execute fixture code inside a temporary workspace and
carry the same trust boundary described in `SECURITY.md`.

## Offline counterfactual race arm (Upgrade 2)

```sh
python -m early_agi.bench --race --tasks 4 --k 4 --timeout 30 --compact
```

The opt-in third arm races 2–4 isolated sandboxes, gates every candidate and
commits the green candidate with highest eta_cog. Token count, latency and branch
order break ties. Losers remain immutable pruned evidence with MeraGraph lineage.
The default two-arm program is unchanged. `--seed`, `--max-tokens-per-branch`
and `--race-workers` configure the race. File-backed runs use spawned processes;
in-memory sinks use threads. Workers receive picklable fixture/config data.

`--endpoint mock://local --model race-fast` exercises a mocked OpenAI-compatible
seam only: all race runs remain offline with zero network calls. Live OpenAI
routing cannot be combined with `--race`. Reports include winner eta, branch
tokens, pruned counts, lineage, executor and sandbox modes. Parallel racing can
spend more total tokens than serial baseline; a green report is not a claim of
token savings. Use at least four tasks to exercise repeat/reflex report thresholds.

### Path A update: escalation and live evaluation

The offline race arm no longer forks unconditionally. It starts with an isolated
copy of the same sequential Gauge/reflex controller. Only failed progress with a
recall/shift directive or repeated identical failed gates permits an escalation,
and branches consume the remaining candidate-turn budget. Previous successful
race repairs are available for exact-prestate reflex replay. The race curves list
only escalated tasks; `branch_race_tokens` excludes ordinary sequential tokens.
The default two-arm JSON parity contract remains unchanged.

The explicit `--live-eval --allow-live` path is separate from the offline fixture
racer. It uses a shared pacer across threaded candidate generators, external
pytest/AST grading, independent per-arm archives and identical token/turn caps.
It currently does not emit live-branch WAL lineage. See
[benchmarks/live-eval/README.md](../benchmarks/live-eval/README.md).
