# Structural-cleanup A/B baseline

These JSON files are the regression lock for the zero-behaviour-change
structural cleanup. The baseline was captured on `arena/987aad9f-netron-agi`
before any refactor, from the pushed offline-A/B commit `f05bc71`, with a clean
worktree:

```bash
python -m early_agi.bench --tasks 10 --k 4 --timeout 60 --compact \
  --output benchmarks/structural-cleanup/baseline-report.json
```

The summary below was derived from that report. After the refactor, the exact
same command was run again and the full post-refactor report compared
byte-for-report with the baseline. No live router, network endpoint, or external
model was used: `offline=true` and `network_calls=0` in both reports.

## Files

| File | Purpose |
| --- | --- |
| `baseline-report.json` | full pre-refactor offline A/B JSON report |
| `baseline-summary.json` | compact pre-refactor curves, arms, and summary |
| `after-refactor-report.json` | full post-refactor report, identical to the baseline |
| `parity.json` | explicit locked values for `eta_cog`, `success@k`, rollback counts, and repeat-task reflex speedups |
| `check_parity.py` | stdlib-only verifier for a freshly generated report |

## Reproduce the parity check

```bash
python -m early_agi.bench --tasks 10 --k 4 --timeout 60 --compact \
  --output /tmp/structural-cleanup-ab-after.json
python benchmarks/structural-cleanup/check_parity.py \
  /tmp/structural-cleanup-ab-after.json
```

Expected output:

```text
offline A/B parity OK: eta_cog curves identical, success@k identical, rollbacks identical, repeat-token speedup 7980.0, reflex hits 7, network calls 0
```

## Locked results

- `summary.ok=true`, `baseline_solved=10`, `netron_solved=10`, `zero_regression=true`.
- `eta_cog`: Netron arm mean `0.3561` vs plain retry `0.907019`; discovery `0.337667` vs `0.906125`; repeat Netron `[0.364] * 7` vs plain `[0.91506, 0.881925, 0.92139, 0.91506, 0.881925, 0.92139, 0.91506]`.
- `success@k`: attempt grid `[1, 2, 3, 4]`; Netron cumulative success `[1, 1, 1, 1]` vs plain `[0, 1, 1, 1]`; attempt speedup `2.0`.
- Rollbacks: Netron `0` vs plain `10`; repeat Netron `[0] * 7` vs plain `[1] * 7`; `rollbacks_avoided=7` in the repeat curve and `10` in the summary.
- Repeat-task reflex speedups: attempt speedup `2.0`, token speedup `7980.0`, reflex hits `7`, Netron repeat tokens `[0] * 7`.

See [docs/benchmark.md](../../docs/benchmark.md) for the harness and
[docs/architecture-boundaries.md](../../docs/architecture-boundaries.md) for the
low-level boundaries that this cleanup intentionally did not alter.
