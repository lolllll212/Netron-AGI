# Security policy

## Supported versions

The current `main` branch and active feature branches receive security fixes.
Released versions are listed in `CHANGELOG.md`.

## Trust model

`early_agi` intentionally executes verifiers (`pytest`, `ruff`, optional Z3) and
can edit files in a repository. It is **not** a sandbox for untrusted code.

The following controls are enabled by default:

- Live LLM routing is disabled until `EARLY_AGI_LLM_ENABLED=1` and explicit
  endpoint/model variables are configured.
- `run_shell` is registered with a wildcard commutator footprint but raises
  `PermissionError` unless `build_default_registry(..., allow_shell=True)` is
  used. Enable it only inside a disposable container or VM.
- Session edits are restricted to paths declared in the turn brief and are
  applied through `OperatorRegistry.apply()` with an expected-content guard.
- A failed edit batch is restored as a transaction: workspace bytes, kernel
  checkpoints, gate state, and registry bookkeeping are rolled back together.
- Coordinate shifts isolate mutating operators. The lock is checked at the
  registry write boundary, not merely in the planner.
- `EPISTEMIC_PROBE` isolates writes before crystallisation and before the first
  mutating Gauge body op. Gauge `probe` lists are statically restricted to
  `deforms(none)` extern ops; `OperatorRegistry.apply()` allows only READ and
  read-only VERIFY operators under the directive and blocks file writes.
  `epistemic_probe()` skips any mutating spec at runtime. Failed pytest/Z3 gates
  are distilled to one assertion/counterexample on the active manifold, not a
  raw stack trace in prompt history. The SQLite directive pool, stratified
  replay, and three-curve benchmark are local: no network or external model is
  used.
- The offline A/B repair benchmark writes multi-file Python fixtures into
  temporary workspaces, excludes tests and `conftest.py` from the edit
  allowlist, clears stale bytecode before verifier runs, and restores byte
  snapshots after failed gates. Its default router is local and deterministic;
  live OpenAI-compatible routing requires both `--router openai` and
  `--allow-live`. API keys are read from an environment variable and are never
  written to benchmark reports. ColdSink reflex evidence remains append-only
  and gated.
- Gauge expressions are interpreted by a small tri-state evaluator. Python
  `eval()` is not used.
- Synthesized operators are AST-restricted, benchmarked in an isolated
  `python -I` subprocess with a minimal builtins namespace, replayed against
  high-drag cold-sink states, checked for input mutation, and registered only
  after strictly positive `eta_cog` improvement and green dark-state gates.
- Cold-sink SQLite triggers reject `UPDATE` and `DELETE`; reads can be re-hashed.
  Even deduplicated payloads retain all metadata commits.
- Z3 and pytest failures/timeouts are tri-state `unverifiable`, not a fake pass.

## Secrets and configuration

`.env` is ignored by Git. `Settings.to_dict()` masks the API key. Invalid numeric
configuration fails fast rather than silently falling back. Do not paste
credentials into Gauge source files, sink records, or issue reports.

## Reporting a vulnerability

Please use GitHub's private security advisory flow for this repository, or open
a minimal issue that says a security issue exists without exploit details. Include
affected paths, reproduction steps, and whether `run_shell` or live routing was
enabled. Do not include real API keys or private repository contents.
