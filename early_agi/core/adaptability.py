"""Public compatibility façade for offline adaptability primitives.

Directive evolution, distilled constraints, stratified replay, topology
signatures, and the three benchmark curves are implemented once in
:mod:`early_agi.core.adaptation`. The historical import path is preserved.
"""

from __future__ import annotations

from .adaptation.benchmarks import local_benchmark
from .adaptation.constraints import distilled_constraint
from .adaptation.directives import AdaptationDirective, DirectivePool, evaluate_directive
from .adaptation.episodes import stratified_episodes
from .adaptation.signatures import (
    AdaptationSignature,
    Topology,
    encode_topology,
    recall_strategy,
)

StructuralSignature = AdaptationSignature
StructuralSignature.__name__ = "StructuralSignature"
StructuralSignature.__qualname__ = "StructuralSignature"

__all__ = [
    "AdaptationDirective",
    "DirectivePool",
    "StructuralSignature",
    "Topology",
    "distilled_constraint",
    "encode_topology",
    "evaluate_directive",
    "local_benchmark",
    "recall_strategy",
    "stratified_episodes",
]

for _public_item in (
    AdaptationDirective,
    DirectivePool,
    StructuralSignature,
    Topology,
    distilled_constraint,
    encode_topology,
    evaluate_directive,
    local_benchmark,
    recall_strategy,
    stratified_episodes,
):
    _public_item.__module__ = __name__
