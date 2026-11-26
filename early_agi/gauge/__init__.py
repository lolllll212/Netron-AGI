"""Gauge: a cognitive and thermodynamic orchestration language for early_agi.

The frontend is span-tagged and deterministic. A ``.gauge`` program is lexed,
parsed, statically checked for algebraic ordering and reversibility, compiled to
bytecode IR, then dispatched by a thin VM. Gauge is the control layer; external
tools and LLMs are invoked only by explicit operators or traps.
"""

from __future__ import annotations

from .ast import Program, node_to_dict
from .builtins import GaugeBuiltins, GaugeGate, SymbolResolver
from .checker import (
    CheckedProgram,
    CheckIssue,
    CheckReport,
    check_program,
    validate_program,
)
from .compiler import GaugeCompiler, compile_gauge, compile_program
from .errors import GaugeRuntimeError
from .expressions import EvaluationContext, evaluate
from .instructions import GaugeOpcodeRunner
from .ir import Instruction, ProgramIR
from .lexer import Token, tokenize
from .parser import GaugeParser, parse_gauge, parse_program
from .results import GaugeRunResult
from .runtime import CandidateView, GaugeState, ManifoldState, ReversibleTransform
from .span import GaugeSemanticError, GaugeSyntaxError, Span
from .state import GaugeExecutionState, WorkspaceView
from .synthesizer import SynthesisResult, Synthesizer
from .traps import TrapPolicy, TrapRunner
from .vm import GaugeVM, run_gauge, run_program

__version__ = "0.2.0"

__all__ = [
    "CandidateView",
    "CheckIssue",
    "CheckReport",
    "CheckedProgram",
    "EvaluationContext",
    "GaugeBuiltins",
    "GaugeCompiler",
    "GaugeExecutionState",
    "GaugeGate",
    "GaugeOpcodeRunner",
    "GaugeParser",
    "GaugeRunResult",
    "GaugeRuntimeError",
    "GaugeSemanticError",
    "GaugeState",
    "GaugeSyntaxError",
    "GaugeVM",
    "Instruction",
    "ManifoldState",
    "Program",
    "ProgramIR",
    "ReversibleTransform",
    "Span",
    "SymbolResolver",
    "SynthesisResult",
    "Synthesizer",
    "Token",
    "TrapPolicy",
    "TrapRunner",
    "WorkspaceView",
    "__version__",
    "check_program",
    "compile_gauge",
    "compile_program",
    "evaluate",
    "node_to_dict",
    "parse_gauge",
    "parse_program",
    "run_gauge",
    "run_program",
    "tokenize",
    "validate_program",
]
