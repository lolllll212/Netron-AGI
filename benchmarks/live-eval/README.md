# Path A: explicit live multi-file evaluation

No live results are checked in. The suite is **not SWE-bench** and makes no
claims about provider model availability, pricing, or real-model solve rates.
It contains eight distinct task IDs: three original repair fixtures, two
additional tangled fixtures, and three reservation-invariant variants. The
last three share a topology; this small, correlated suite is a smoke evaluation,
not eight independent real-world bugs.

## Offline checks

```sh
python -m pytest
python -m early_agi.bench --race --tasks 4 --compact --output /tmp/race.json
python -m early_agi.bench --compact --output /tmp/default.json
python benchmarks/structural-cleanup/check_parity.py /tmp/default.json
```

Normal `--race` is still **offline**, even if `--endpoint` is specified. It uses
exactly the existing sequential Gauge/reflex implementation in an independent
archive and forks only after a failed governance shift/recall or repeated
identical failed gate verdicts (task-local frustration). Provider errors alone
never trigger racing. The failed sequential attempt remains in the ledger;
branches must fit the remaining `k` candidate budget. A single ordinary test
failure is not enough unless accompanied by an explicit governance stall.

The greedy-stall router in tests intentionally repeats a one-sided repair; all
candidate patches remain accessible to every offline arm. The hard fixtures
have real external failure checks, but their scripted recovery is **policy
coverage**, not proof that a live model will stall or benefit from parallelism.
Easy tasks must not fork and retain positive sequential/reflex token savings.

## Configure credentials locally

Use `NVIDIA_API_KEY` and `OPENAI_BASE_URL` environment variables. Existing
`EARLY_AGI_API_KEY` and `EARLY_AGI_BASE_URL` take precedence in `Settings`.
Never commit `.env`, API keys, or raw Authorization headers. Do not put a key
in a command-line argument. Install with `python -m pip install -e '.[dev]'`.

Select exact account-visible model slugs yourself. They are not hardcoded.
`--verify-models` explicitly performs an authenticated `GET <endpoint>/models`
and refuses to continue if either chosen ID is absent. No discovery happens
on import or during normal offline use.

```sh
python -m early_agi.bench --live-eval --allow-live \
  --endpoint https://integrate.api.nvidia.com/v1 \
  --api-key-env NVIDIA_API_KEY \
  --model YOUR_EXECUTOR_ID --deep-model YOUR_REFRAMER_ID --verify-models \
  --k 4 --max-tokens 2048 --token-budget 200000 \
  --max-workers 2 --rpm-limit 40 --timeout 90 \
  --output benchmarks/live-eval/local-summary.json
```

On PowerShell put this command on one line (or use backticks, not backslashes).
A standalone `python -m early_agi.bench.live_eval` entry point is also available.

## Comparability and safety

- Three arms: baseline, sequential `netron_plastic`, and gated `netron_race`.
  Each has its own Gauge/reflex archive. Each task starts from fresh files;
  each race candidate has a separate copy and verifier process. An exact second
  pass measures reflex replay. Only successful gate-verified patches transfer.
- Grading uses subprocess pytest and AST gates. Tests/conftest cannot be edited
  and are excluded from model-visible source. Task text, source, and verifier
  failure feedback are shared; canned patches and probe hints are withheld.
- `k` counts provider candidate turns per task, including race branches and deep
  repair candidates. `max_tokens` limits completion length. `token-budget` is
  the same **per-arm, across both passes** token reservation cap. Input UTF-8
  bytes plus output allowance and overhead are reserved before dispatch, with
  allowance for three transport attempts. Errors retain their reservation;
  successful usage reconciles it. This conservative guard assumes byte-based
  tokenization; provider billing remains authoritative. It is **not a dollar
  guarantee**. Use provider-side spending limits too.
- All clients in one evaluation share one request-start pacer, including retries.
  Live concurrency is capped at two even if a larger worker count is requested.
  This protects one runner only; multiple machines need external coordination.
- Retries honor numeric/date Retry-After and add bounded exponential backoff
  with jitter for 429/5xx. `rate_limit_retries` counts actual scheduled 429 retries.
- The live sandbox coordinator reuses the benchmark's transactional Gauge gates;
  it is separate from the offline WAL fixture racer. It does **not yet publish
  WAL/MeraGraph lineage for live branches**. No claim of native live Gauge
  `RACE_BRANCHES` support is made.
- Reports include observed usage, conservative outstanding reservations, requests,
  429 retries, solve rate, mean eta, rollbacks/avoided rollbacks, races, and second
  pass token speedup/reflex hits. Retry failures may obscure provider usage;
  retained reservations expose that uncertainty. Review generated reports before
  publishing them; no fabricated live summaries are supplied here.

## Reliability and a small first run

Update the branch before running again. Live evaluation now defaults to a
**180-second per-request/grader timeout** and **zero automatic request retries**.
The offline benchmark keeps its previous 60-second timeout default. An earlier
chat explanation said the old live CLI default was 30 seconds; the code default
was actually 60. A slow successful diagnostic does not by itself prove that all
previous failures were timeouts; new reports preserve the actual error text.

First run just one task, one pass, one arm and one candidate request:

```powershell
python -m early_agi.bench --live-eval --allow-live --smoke --endpoint https://integrate.api.nvidia.com/v1 --api-key-env NVIDIA_API_KEY --model "moonshotai/kimi-k3.1-pro" --deep-model "moonshotai/kimi-k3.1-pro" --timeout 180 --max-tokens 2048 --token-budget 20000 --output benchmarks/live-eval/smoke.json
```

This is **one paid/live completion attempt**, not a connectivity-only test.
`--smoke` forces task-limit=1, passes=1, baseline only and k=1; it does not make a
three-arm comparison or claim reflex speedup. `--verify-models` adds a separate
models-list request if desired. A first repaired task still requires real external
grading, and 2048 output tokens may not suffice for every reasoning model.

For larger controlled runs, use `--task-limit N`, `--passes 1|2`, and
`--arms baseline netron_plastic netron_race`. Only increase the workload once the
smoke JSON shows a successful task, not merely a successful HTTP response.

- Timestamped progress goes to stderr immediately: task/request starts, request
  heartbeats every 15 seconds, retry waits, grading and completion. A heartbeat
  means the client is waiting, not that the provider is generating tokens.
- Output JSON is atomically checkpointed before/after each task and at exit.
  A checkpoint is **not an automatic resume mechanism**; starting the same command
  again makes a new run and can overwrite that file. Use unique output filenames.
- Two consecutive exhausted transport failures open a shared circuit and stop
  the evaluation, without trying the remaining arms. `--transport-failure-limit`
  configures this threshold. `--request-retries N` enables explicit retry attempts;
  reservations account for them. Already in-flight parallel calls cannot be undone.
- Schema v2 records `status`, `valid_comparison`, selected plan, completed/planned
  task counts and secret-redacted error details. Unattempted tasks are omitted,
  not silently counted as model failures. A budget-exhausted, interrupted or
  transport-failed report is **not a valid model-performance comparison**.
- Ctrl+C preserves completed tasks and marks the current task interrupted. It
  does not guarantee immediate cancellation of in-flight HTTP calls or final
  billing visibility. The CLI returns 130 on interruption and 1 on other failed
  or incomplete runs. A completed smoke run returns 0 but valid_comparison=false.
- Progress/JSON omit credentials and prompt bodies. Still review artifacts before
  publishing; unresolved token reservations are not confirmed billed usage.

## Schema v3: common external scoring and terminal telemetry

Live headlines no longer compare Gauge's stateful marginal progress with the
baseline's stateless gate credit. All arms receive **one fixed six-unit task
completion credit** only when a passing repair closes both `syntax_valid` and
`tests_passing`. Partial/failed tasks get zero. This is a scoring convention,
not physical entropy measured from a task's initial state.

The common resource-normalized score is:

```
eta = verified_task_bits / (0.001 * total_tokens + 0.0001 * latency_ms + tool_cost)
```

Costs accumulate across every failed attempt and every speculative candidate,
including losing branches. Tools use the same base attempt cost model for all
arms; latency uses provider latency plus the standardized external grading
charge. Read-only Gauge preflight/internal control penalties do not change this
external numerator or denominator. Therefore this is **not an all-in wall-clock
performance measure**, nor a dollar-saving claim. Internal governance efficiency
is preserved separately under each task/attempt's `governance` object and in
`overall_mean_governance_eta_cog`; the controller itself is not rewritten.

Reports expose `first_pass_mean_eta_cog`, `second_pass_mean_eta_cog`, and
`overall_mean_eta_cog`, including failed tasks as zeros and repeat/reflex tasks.
`mean_eta_cog` now aliases the overall external mean. Attempt scores and curves
are external too; the task score credits completion once over cumulative cost.
Normalized constraints omit blanks. `preflight_probe` denotes the task's actual
preflight (not an inner VM's probe); exact replay with no preflight reports false.
A gate-complete passing attempt reports terminal `CRYSTALLIZE`, while the raw
internal directive remains in `governance.directive`.

A zero-token successful reflex pass is reported as
`second_pass_zero_token_replay: true` and `second_pass_token_speedup: null`, not
a finite ratio made by replacing a zero denominator with one.

Re-score an existing report **offline**, without overwriting the original:

```powershell
python -m early_agi.bench.rescore_live benchmarks/live-eval/nemotron-small.json --output benchmarks/live-eval/nemotron-small-rescored.json
```

Reanalysis preserves task outcomes, token usage, validity flags and raw controller
scores. It does not rerun external tests or contact NIM, and is labeled with its
source schema and provenance. Do not present it as a new experiment.

For the full live pack use `--task-limit 8` (or omit the limit), not `--tasks 8`:
`--tasks` configures the older offline benchmark. Use `--deep-model` for the
reframer ID. Verify both account-visible model IDs and choose the executor before
launching a paid run. The eight-task pack includes three correlated invariant
variants. A stronger model may solve all of them without a stall; never force a
worse baseline or grant hidden fixes just to obtain `races > 0`. Real escalation
and reframing benefit remain unverified until recorded in a live ledger.
