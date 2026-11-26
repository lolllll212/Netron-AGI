"""early_agi — a binary-orchestrated, physics-backend hybrid cognitive kernel.

The package is deliberately split so the deterministic agent loop never depends
on how entanglement is measured:

    core/       deterministic execution (cold sink, sensor, governor, kernel)
    operators/  non-commuting tools + dark-state verifiers
    backends/   swappable relaxation backends (classical SVD, bounded-χ
                randomized truncation, and block-Gibbs/THRML seams)
    memory/     scale-indexed MERA store over the cold sink
    llm/        OpenAI-compatible dual-tier client + router

Sub-packages are intentionally *not* re-exported here; importing
``early_agi.core.kernel`` should not drag in the LLM layer.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
