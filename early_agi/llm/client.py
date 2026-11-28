"""Minimal OpenAI-compatible chat client (stdlib HTTP, no vendor SDK)."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import DEFAULT_BASE_URL, Settings
from .pacing import RequestPacer, retry_after

__all__ = [
    "LLMError",
    "LLMReply",
    "LLMUnavailable",
    "OpenAICompatibleClient",
    "extract_json",
]


class LLMError(RuntimeError):
    """Transport or protocol failure."""


class LLMUnavailable(LLMError):
    """No endpoint configured / reachable."""


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class LLMReply:
    text: str
    model: str
    usage: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        total = self.usage.get("total_tokens")
        if total is not None:
            return _coerce_int(total)
        return _coerce_int(self.usage.get("prompt_tokens", 0)) + _coerce_int(
            self.usage.get("completion_tokens", 0)
        )


def extract_json(text: str) -> Any:
    """Pull the outermost JSON object/array out of a chat completion."""
    stripped = text.strip()
    if not stripped:
        raise LLMError("empty completion")

    candidates = [stripped]
    if "```" in stripped:
        for block in stripped.split("```")[1::2]:
            body = block.strip()
            if body.startswith(("json", "JSON")):
                body = body[4:].strip()
            candidates.append(body)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    found: list[tuple[int, Any]] = []
    for start, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        found.append((end, value))
    if found:
        found.sort(key=lambda item: item[0], reverse=True)
        return found[0][1]
    raise LLMError(f"no JSON found in completion: {stripped[:200]!r}")


def sanitize_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Map developer instructions to system; reject unsupported tool semantics."""
    clean = []
    for message in messages:
        role = message.get("role", "")
        role = "system" if role == "developer" else role
        if role not in {"system", "user", "assistant"}:
            raise LLMError(f"unsupported chat role: {role!r}")
        if not isinstance(message.get("content"), str):
            raise LLMError("chat content must be text")
        if message.get("tool_calls") or message.get("function_call"):
            raise LLMError("tool/function calls require an explicit text adapter")
        clean.append({"role": role, "content": message["content"]})
    return clean


class OpenAICompatibleClient:
    """Synchronous chat-completions client with bounded retries."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        timeout: float = 90.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        pacer: RequestPacer | None = None,
        progress: Any = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.temperature = float(temperature)
        self.pacer = pacer
        self.progress = progress
        self.rate_limit_retries = 0
        self.requests_sent = 0

    @classmethod
    def from_settings(cls, tier: str = "fast", settings: Settings | None = None) -> OpenAICompatibleClient:
        settings = settings or Settings.from_env()
        model = settings.fast_model if tier == "fast" else settings.deep_model
        return cls(
            model,
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_s,
            max_retries=settings.max_retries,
            temperature=settings.temperature,
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def __repr__(self) -> str:  # pragma: no cover
        return f"OpenAICompatibleClient(model={self.model!r}, base_url={self.base_url!r})"

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> LLMReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": sanitize_messages(messages),
            "temperature": self.temperature if temperature is None else float(temperature),
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
            delay = 0.0
            try:
                if self.pacer is not None:
                    self.pacer.wait()
                self.requests_sent += 1
                with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "backslashreplace")[:800]
                last_error = LLMError(f"{self.endpoint} returned HTTP {exc.code}: {detail}")
                delay = retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                if exc.code == 429 and attempt < self.max_retries:
                    self.rate_limit_retries += 1
                exc.close()
                if exc.code < 500 and exc.code != 429:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = LLMUnavailable(f"cannot reach {self.endpoint}: {exc}")
            except json.JSONDecodeError as exc:
                last_error = LLMError(f"non-JSON response from {self.endpoint}: {exc}")
            if attempt < self.max_retries:
                wait = max(delay, min(30.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.25))
                if self.progress:
                    self.progress("retry", attempt=attempt + 1, wait_s=round(wait, 2), error=str(last_error))
                time.sleep(wait)
        else:
            assert last_error is not None
            raise last_error

        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            text = raw["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed completion: {json.dumps(raw)[:400]}") from exc

        return LLMReply(
            text=text,
            model=str(raw.get("model", self.model)),
            usage=dict(raw.get("usage") or {}),
            latency_ms=latency_ms,
            raw=raw,
        )

    def verify_models(self, models: Sequence[str]) -> tuple[str, ...]:
        """Explicit authenticated discovery; never invoked during construction."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        request = urllib.request.Request(f"{self.base_url}/models", headers=headers)
        if self.pacer is not None:
            self.pacer.wait()
        self.requests_sent += 1
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        available = tuple(str(item["id"]) for item in data["data"])
        missing = set(models) - set(available)
        if missing:
            raise LLMError(f"models not available: {sorted(missing)}")
        return available

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> tuple[Any, LLMReply]:
        reply = self.chat(messages, json_mode=True, **kwargs)
        return extract_json(reply.text), reply
