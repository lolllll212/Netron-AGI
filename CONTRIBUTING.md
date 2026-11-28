# Contributing

Thanks for working on `early_agi` and Gauge. The project values deterministic,
verifiable behavior over cleverness.

## Setup

```bash
python -m pip install -e ".[dev,verify]"
cp .env.example .env  # optional; live LLM routing is disabled by default
```

The offline test suite and demo require no API key, network, or model server.

## Before opening a change

```bash
ruff check .
ruff format --check .
python -m pytest -q
python -m early_agi.demo
python -m early_agi.gauge check examples/netron_solve.gauge
python -m early_agi.gauge run examples/netron_solve.gauge --root . --compact
```

CI runs the same checks on Linux and Windows, on the oldest and newest supported
Python versions, plus a packaging build and an optional Z3 verification job.

## Adding to Gauge

1. Add syntax to `early_agi/gauge/lexer.py`, `ast.py`, and `parser.py` with exact
   spans.
2. Extend `checker.py` before adding runtime behavior. Reversibility, `after()`,
   and no-leak rules must be static where possible.
3. Emit deterministic opcodes in `compiler.py`.
4. Implement the opcode in `vm.py` using an existing subsystem rather than a new
   side channel.
5. Add a parser/checker/VM regression test and update `docs/gauge.md`.

Do not use Python `eval()` for Gauge expressions, and do not let an LLM decide
the outer control flow. External models belong behind explicit operators or the
synthesis hook.

## Runtime safety

- Enforce path allowlists at the write boundary, not only in prompt parsing.
- Keep write locks in `OperatorRegistry.apply()`.
- Preserve tri-state verifier results: `None` is neither pass nor fail.
- Restore workspace files, kernel checkpoints, gate state, and registry
  bookkeeping together when an edit batch fails.
- Treat `run_shell` as unconstrained; it is disabled by default.

See `SECURITY.md` before changing verifier subprocesses or synthesis execution.
