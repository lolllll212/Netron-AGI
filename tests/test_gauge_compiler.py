"""Gauge bytecode lowering: setup, body, trap bodies, and deterministic IR."""

from __future__ import annotations

from _gauge_fixtures import SOURCE, all_opcodes
from early_agi.gauge import compile_gauge, compile_program, validate_program


def test_compiler_emits_required_hybrid_vm_opcodes() -> None:
    ir = compile_gauge(SOURCE)
    opcodes = all_opcodes(ir)
    required = {
        "SCALE_INIT",
        "MANIFOLD_INIT",
        "TRANSFORM_REV",
        "GATE_PROJECT",
        "OP_APPLY",
        "ENT_MEASURE",
        "GOVERN_CHECK",
        "EVOLVE",
        "COLLAPSE_SINK",
        "UNCALL_TRANSFORM",
        "RECALL_SINK",
        "SYNTHESIZE_OP",
        "HALT",
    }
    assert required <= opcodes
    assert ir.metadata["op_order"] == ["capture_baseline", "apply_candidate"]
    assert ir.metadata["scales"] == ["L0", "L2"]
    assert [instruction.op for instruction in ir.setup] == [
        "SCALE_INIT",
        "MANIFOLD_INIT",
        "GATE_DEF",
        "GATE_DEF",
        "OP_DEF",
        "OP_DEF",
        "TRANSFORM_REV",
        "GOVERN_BEGIN",
    ]
    assert [instruction.op for instruction in ir.body][:6] == [
        "OP_APPLY",
        "OP_APPLY",
        "ENT_MEASURE",
        "GATE_PROJECT",
        "GOVERN_CHECK",
        "EVOLVE",
    ]
    assert ir.body[-1].op == "HALT"


def test_trap_bodies_lower_to_reversible_synthesis_and_recall() -> None:
    ir = compile_gauge(SOURCE)
    traps = {instruction.op: instruction for instruction in ir.body if instruction.op.startswith("TRAP_")}
    assert set(traps) == {"TRAP_VOLUME_LAW", "TRAP_DRAG", "TRAP_FRUSTRATION"}
    assert [instruction.op for instruction in traps["TRAP_VOLUME_LAW"].body] == ["TRANSFORM_APPLY"]
    assert [instruction.op for instruction in traps["TRAP_DRAG"].body] == ["SYNTHESIZE_OP"]
    assert [instruction.op for instruction in traps["TRAP_FRUSTRATION"].body] == [
        "UNCALL_TRANSFORM",
        "RECALL_SINK",
    ]
    recall = traps["TRAP_FRUSTRATION"].body[1]
    assert recall.args["condition"] is not None


def test_ir_serialization_is_deterministic_for_checked_programs() -> None:
    ir = compile_gauge(SOURCE)
    recompiled = compile_program(validate_program(SOURCE))
    assert ir.as_dict() == recompiled.as_dict()
    assert ir.span.as_dict() == recompiled.span.as_dict()
