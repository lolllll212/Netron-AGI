"""Command-line frontend for Gauge programs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import Settings
from . import (
    GaugeSemanticError,
    GaugeSyntaxError,
    check_program,
    compile_gauge,
    node_to_dict,
    parse_gauge,
    validate_program,
)
from .vm import GaugeVM

__all__ = ["main"]


def _source(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _dump(payload: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def _write(text: str, output: str | None) -> None:
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gauge",
        description="Parse, check, compile, and run Gauge cognitive orchestration programs.",
    )
    parser.add_argument(
        "command",
        choices=("parse", "check", "compile", "ir", "run"),
        help="pipeline stage to execute",
    )
    parser.add_argument("file", nargs="?", default="-", help="Gauge source file, or - for stdin")
    parser.add_argument("--root", default=".", help="workspace root for run (default: current directory)")
    parser.add_argument("--output", "-o", help="write JSON to this path instead of stdout")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument("--no-dotenv", action="store_true", help="do not load a local .env file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    compact = bool(args.compact)
    try:
        source = _source(args.file)
        if args.command == "parse":
            payload: Any = node_to_dict(parse_gauge(source))
            status = 0
        elif args.command == "check":
            report = check_program(source)
            checked = validate_program(source) if report.ok else None
            payload = {
                **report.as_dict(),
                "op_order": list(checked.op_order) if checked else [],
                "scales": list(checked.scales) if checked else [],
                "declarations": len(parse_gauge(source).declarations),
            }
            status = 0 if report.ok else 2
        elif args.command in {"compile", "ir"}:
            payload = compile_gauge(source).as_dict()
            status = 0
        else:
            settings = Settings.from_env(load_dotenv=not args.no_dotenv)
            result = GaugeVM(args.root, settings=settings).run(source, name=args.file)
            payload = result.as_dict()
            status = 0 if result.ok else 2
        _write(_dump(payload, compact=compact), args.output)
        return status
    except (GaugeSyntaxError, GaugeSemanticError) as exc:
        _write(
            _dump({"ok": False, "error": str(exc), "kind": type(exc).__name__}, compact=compact), args.output
        )
        return 2
    except Exception as exc:
        _write(
            _dump(
                {"ok": False, "error": str(exc), "kind": type(exc).__name__},
                compact=compact,
            ),
            args.output,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
