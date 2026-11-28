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
