"""Minimal OpenAI-compatible chat client (stdlib HTTP, no vendor SDK).

Works against Ollama (``/v1``), llama.cpp server, vLLM, LM Studio, or any hosted
OpenAI-compatible endpoint. Swapping providers is an environment variable, not a
code change, which is the "zero cloud lock-in" requirement.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import DEFAULT_BASE_URL, Settings

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


@dataclass(frozen=True)
class LLMReply:
    text: str
    model: str
    usage: Mapping[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        total = self.usage.get("total_tokens")
        if total is not None:
            return int(total)
        return int(self.usage.get("prompt_tokens", 0)) + int(self.usage.get("completion_tokens", 0))


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a chat completion."""
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

    # Last resort: scan for a balanced object/array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : index + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"no JSON found in completion: {stripped[:200]!r}")


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
    ) -> None:
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.temperature = float(temperature)

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

    def __repr__(self) -> str:  # pragma: no cover - cosmetics
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
            "messages": [dict(message) for message in messages],
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
            try:
                with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "backslashreplace")[:800]
                last_error = LLMError(f"{self.endpoint} returned HTTP {exc.code}: {detail}")
                if exc.code < 500 and exc.code != 429:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = LLMUnavailable(f"cannot reach {self.endpoint}: {exc}")
            except json.JSONDecodeError as exc:
                last_error = LLMError(f"non-JSON response from {self.endpoint}: {exc}")
            if attempt < self.max_retries:
                time.sleep(min(2.0, 0.25 * (2**attempt)))
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

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> tuple[Any, LLMReply]:
        reply = self.chat(messages, json_mode=True, **kwargs)
        return extract_json(reply.text), reply
