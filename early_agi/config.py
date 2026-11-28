"""Environment-driven configuration with strict validation and dotenv support."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .backends import available_backends, get_backend

DEFAULT_BASE_URL = "http://localhost:11434/v1"
_ENV_PREFIX = "EARLY_AGI_"

__all__ = ["DEFAULT_BASE_URL", "ConfigError", "RuntimeBundle", "Settings"]


class ConfigError(ValueError):
    """An environment variable is present but invalid."""


def _dotenv_values(path: str | Path) -> dict[str, str]:
    file = Path(path)
    if not file.exists() or not file.is_file():
        return {}
    values: dict[str, str] = {}
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        if key.isidentifier():
            values[key] = value.replace("\\n", "\n")
    return values


def _lookup(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return value if value is not None and value.strip() else None


def _str(env: Mapping[str, str], name: str, default: str) -> str:
    value = _lookup(env, name)
    return default if value is None else value.strip()


def _number(env: Mapping[str, str], name: str, default: float, kind: type[int] | type[float]) -> Any:
    raw = _lookup(env, name)
    if raw is None:
        return default
    try:
        value = kind(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a valid {kind.__name__}, got {raw!r}") from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{name} must be finite, got {raw!r}")
    return value


def _float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = float(_number(env, name, default, float))
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def _int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = int(_number(env, name, default, int))
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _lookup(env, name)
    if raw is None:
        return default
    lowered = raw.strip().casefold()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True)
class RuntimeBundle:
    """Objects built from one validated configuration."""

    settings: Settings
    sink: Any
    kernel: Any
    graph: Any
    scanner: Any
    backend: Any = None


@dataclass(frozen=True)
class Settings:
    """Runtime knobs for the kernel, memory, scanner, backend, and the LLM layer."""

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
    sink_path: str = ":memory:"
    mera_path: str = ":memory:"
    max_files: int = 200
    llm_enabled: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    # Appended so existing positional Settings(...) constructions keep their meaning.
    backend: str = "classical_svd"
    directive_path: str = ":memory:"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        load_dotenv: bool = True,
        dotenv_path: str | Path = ".env",
    ) -> Settings:
        source = _dotenv_values(dotenv_path) if load_dotenv else {}
        # Explicit mappings and the real environment override .env defaults.
        source.update(dict(os.environ if env is None else env))

        explicit_llm = any(
            key in source
            for key in (
                "EARLY_AGI_BASE_URL",
                "EARLY_AGI_FAST_MODEL",
                "EARLY_AGI_DEEP_MODEL",
                "EARLY_AGI_API_KEY",
                "OPENAI_BASE_URL",
                "NVIDIA_API_KEY",
            )
        )
        settings = cls(
            base_url=_str(source, "EARLY_AGI_BASE_URL", _str(source, "OPENAI_BASE_URL", DEFAULT_BASE_URL)),
            api_key=_str(source, "EARLY_AGI_API_KEY", _str(source, "NVIDIA_API_KEY", "")),
            fast_model=_str(source, "EARLY_AGI_FAST_MODEL", "fast-executor"),
            deep_model=_str(source, "EARLY_AGI_DEEP_MODEL", "deep-reasoner"),
            timeout_s=_float(source, "EARLY_AGI_TIMEOUT", 90.0, minimum=0.1),
            max_retries=_int(source, "EARLY_AGI_MAX_RETRIES", 2, minimum=0, maximum=10),
            temperature=_float(source, "EARLY_AGI_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0),
            chi_max=_int(source, "EARLY_AGI_CHI_MAX", 4, minimum=1, maximum=256),
            tau=_float(source, "EARLY_AGI_TAU", 0.15, minimum=1e-9, maximum=1.0),
            backend=_str(source, "EARLY_AGI_BACKEND", "classical_svd"),
            directive_path=_str(source, "EARLY_AGI_DIRECTIVE_DB", ":memory:"),
            eta_high=_float(source, "EARLY_AGI_ETA_HIGH", 1.2, minimum=0.0),
            eta_min=_float(source, "EARLY_AGI_ETA_MIN", 0.15, minimum=0.0),
            sink_path=_str(source, "EARLY_AGI_SINK", ":memory:"),
            mera_path=_str(source, "EARLY_AGI_MERA", ":memory:"),
            max_files=_int(source, "EARLY_AGI_MAX_FILES", 200, minimum=1, maximum=100_000),
            llm_enabled=_bool(source, "EARLY_AGI_LLM_ENABLED", explicit_llm),
            extra={
                key: value
                for key, value in source.items()
                if key.startswith(_ENV_PREFIX) and key not in _KNOWN_ENV
            },
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.eta_min > self.eta_high:
            raise ConfigError("EARLY_AGI_ETA_MIN cannot exceed EARLY_AGI_ETA_HIGH")
        if self.backend not in available_backends():
            raise ConfigError(
                f"EARLY_AGI_BACKEND must be one of {', '.join(available_backends())}; got {self.backend!r}"
            )
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ConfigError("EARLY_AGI_BASE_URL must use http:// or https://")
        if self.llm_enabled and (not self.fast_model or not self.deep_model):
            raise ConfigError("live routing requires both model names")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("extra", None)
        out["llm_configured"] = self.llm_configured
        if out.get("api_key"):
            out["api_key"] = "***"
        return out

    @property
    def llm_configured(self) -> bool:
        """True only after live routing is explicitly enabled in the environment."""
        return bool(self.llm_enabled and self.base_url and self.fast_model and self.deep_model)

    def build_runtime(
        self,
        root: str | Path,
        *,
        gates: Sequence[Any] = (),
        registry: Any | None = None,
        backend: Any | None = None,
    ) -> RuntimeBundle:
        """Construct a wired sink, kernel, MERA graph, and scanner."""
        self.validate()
        from .core.cold_sink import ColdSink
        from .core.coupling import RepoScanner
        from .core.governor import ThermodynamicGovernor
        from .core.kernel import EarlyAGIKernel
        from .memory.mera_graph import MeraGraph

        sink = ColdSink(self.sink_path)
        selected = backend if backend is not None else get_backend(self.backend)
        kernel = EarlyAGIKernel(
            sink,
            backend=selected,
            governor=ThermodynamicGovernor(eta_high=self.eta_high, eta_min=self.eta_min),
            gates=list(gates),
            registry=registry,
            chi_max=self.chi_max,
            tau=self.tau,
        )
        graph = MeraGraph(sink, self.mera_path, backend=kernel.backend)
        scanner = RepoScanner(root, max_files=self.max_files)
        return RuntimeBundle(
            settings=self,
            sink=sink,
            kernel=kernel,
            graph=graph,
            scanner=scanner,
            backend=kernel.backend,
        )


_KNOWN_ENV = frozenset(
    {
        "EARLY_AGI_BASE_URL",
        "EARLY_AGI_API_KEY",
        "EARLY_AGI_FAST_MODEL",
        "EARLY_AGI_DEEP_MODEL",
        "EARLY_AGI_TIMEOUT",
        "EARLY_AGI_MAX_RETRIES",
        "EARLY_AGI_TEMPERATURE",
        "EARLY_AGI_CHI_MAX",
        "EARLY_AGI_TAU",
        "EARLY_AGI_BACKEND",
        "EARLY_AGI_DIRECTIVE_DB",
        "EARLY_AGI_ETA_HIGH",
        "EARLY_AGI_ETA_MIN",
        "EARLY_AGI_SINK",
        "EARLY_AGI_MERA",
        "EARLY_AGI_MAX_FILES",
        "EARLY_AGI_LLM_ENABLED",
    }
)
