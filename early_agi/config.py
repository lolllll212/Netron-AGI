"""Environment-driven configuration.

Everything is optional: with no environment at all the kernel still runs
deterministically against an in-memory sink, and the LLM layer points at a
local OpenAI-compatible server (Ollama, llama.cpp, vLLM, LM Studio).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_BASE_URL = "http://localhost:11434/v1"


def _str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime knobs for the kernel and the LLM layer."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    fast_model: str = "fast-executor"
    deep_model: str = "deep-reasoner"
    timeout_s: float = 90.0
    max_retries: int = 2
    temperature: float = 0.0
    chi_max: int = 4
    tau: float = 0.15
    eta_high: float = 1.2
    eta_min: float = 0.15
    sink_path: str = "cold_sink.db"
    mera_path: str = "mera.db"
    max_files: int = 200
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            base_url=_str("EARLY_AGI_BASE_URL", DEFAULT_BASE_URL),
            api_key=_str("EARLY_AGI_API_KEY", ""),
            fast_model=_str("EARLY_AGI_FAST_MODEL", "fast-executor"),
            deep_model=_str("EARLY_AGI_DEEP_MODEL", "deep-reasoner"),
            timeout_s=_float("EARLY_AGI_TIMEOUT", 90.0),
            max_retries=_int("EARLY_AGI_MAX_RETRIES", 2),
            temperature=_float("EARLY_AGI_TEMPERATURE", 0.0),
            chi_max=_int("EARLY_AGI_CHI_MAX", 4),
            tau=_float("EARLY_AGI_TAU", 0.15),
            eta_high=_float("EARLY_AGI_ETA_HIGH", 1.2),
            eta_min=_float("EARLY_AGI_ETA_MIN", 0.15),
            sink_path=_str("EARLY_AGI_SINK", "cold_sink.db"),
            mera_path=_str("EARLY_AGI_MERA", "mera.db"),
            max_files=_int("EARLY_AGI_MAX_FILES", 200),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("extra", None)
        if out.get("api_key"):
            out["api_key"] = "***"
        return out

    @property
    def llm_configured(self) -> bool:
        """A base URL always exists (local default); report key presence only."""
        return bool(self.base_url)


__all__ = ["DEFAULT_BASE_URL", "Settings"]
