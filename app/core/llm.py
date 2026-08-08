"""The one shared async LLM client every node calls through.

Two axes of failover, kept deliberately separate:

- **Pool failover (D-01, sticky).** Each provider is a pool of keys. The client
  stays on the current key for every call until *that key* fails with a
  retryable error (429, 5xx, connection/timeout — never 400), then moves to the
  next key in the same provider's pool. Only once every Groq key has failed
  does it cross to Cerebras. The position is sticky across calls, not reset
  per-request, so a run's trace shows one slot used repeatedly and a visible
  jump on failure rather than calls interleaved unpredictably.
- **Tool-calling fallback (D-02, not a failover).** If native tool calling
  succeeds at the transport level but the model simply doesn't return a tool
  call, that is not a provider failure — the same key answered fine, it just
  ignored ``tools``. The fallback re-asks the same slot for prompted JSON
  instead of advancing the pool.

Both paths are traced (``path`` = ``native_tool_call`` | ``json_fallback`` |
``text``), so which one actually served a given call is visible after the
fact, not just implied by the code.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from app.core.config import settings
from app.core.tracing import TraceEvent, tracer


class AllProvidersExhaustedError(RuntimeError):
    """Every key in every configured pool failed for this request."""


class ToolCallParseError(RuntimeError):
    """Neither native tool calling nor the prompted-JSON fallback parsed."""


@dataclass(frozen=True)
class _Slot:
    provider: str  # "groq" | "cerebras"
    key_index: str  # "groq[0]", "cerebras[1]", ...
    model: str
    client: AsyncOpenAI


def _build_slots() -> list[_Slot]:
    slots: list[_Slot] = []
    for i, key in enumerate(settings.groq_api_keys):
        slots.append(
            _Slot(
                provider="groq",
                key_index=f"groq[{i}]",
                model=settings.groq_model,
                client=AsyncOpenAI(
                    api_key=key,
                    base_url=settings.groq_base_url,
                    timeout=settings.request_timeout_s,
                ),
            )
        )
    for i, key in enumerate(settings.cerebras_api_keys):
        slots.append(
            _Slot(
                provider="cerebras",
                key_index=f"cerebras[{i}]",
                model=settings.cerebras_model,
                client=AsyncOpenAI(
                    api_key=key,
                    base_url=settings.cerebras_base_url,
                    timeout=settings.request_timeout_s,
                ),
            )
        )
    return slots


def _is_retryable(exc: Exception) -> bool:
    """429, 5xx, connection/timeout — never 400 (D-01).

    A 400 would fail identically on every key and every provider; retrying it
    would just mask a bug in our own request construction as a transient one.
    """
    if isinstance(exc, (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


def _reasoning_tokens(usage: Any) -> int | None:
    """`gpt-oss` reports reasoning tokens under `completion_tokens_details`."""
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return None
    return getattr(details, "reasoning_tokens", None)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first balanced ``{...}`` block out of text and parse it.

    `gpt-oss` is known to prepend reasoning prose before the JSON despite being
    told not to (D-02) — this substring-extraction is the same hack the
    previous project needed, ported rather than re-invented.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class LLMPool:
    """Holds every configured key as an ordered slot list and the sticky cursor."""

    def __init__(self) -> None:
        self._slots = _build_slots()
        if not self._slots:
            raise RuntimeError(
                "No API keys configured — set GROQ_API_KEY and/or CEREBRAS_API_KEY in .env"
            )
        self._current = 0
        self._advance_lock = asyncio.Lock()

    @property
    def slot_labels(self) -> list[str]:
        return [s.key_index for s in self._slots]

    def reset(self) -> None:
        """Return the sticky cursor to the first slot.

        Used by the smoke test so the full pool-walk can be exercised
        deterministically from a clean starting position.
        """
        self._current = 0

    async def request(
        self,
        make_request: Callable[[_Slot], Awaitable[Any]],
        *,
        node: str,
        purpose: str,
    ) -> tuple[Any, _Slot, int]:
        """Run ``make_request`` against slots starting from the sticky cursor.

        Advances the cursor forward — never wraps — on a retryable failure.
        Returns ``(response, slot_used, attempt_number)`` on success, or raises
        :class:`AllProvidersExhaustedError` once every remaining slot has
        failed. `attempt_number` counts from 1 within this call's chain, so a
        first-try success still reports ``1`` even if an earlier, unrelated
        call already advanced the cursor.
        """
        start = self._current
        last_exc: Exception | None = None

        for offset in range(len(self._slots) - start):
            idx = start + offset
            slot = self._slots[idx]
            t0 = time.monotonic()
            try:
                response = await make_request(slot)
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                last_exc = exc
                latency_ms = int((time.monotonic() - t0) * 1000)
                await tracer.emit(
                    TraceEvent(
                        kind="llm_failover",
                        node=node,
                        purpose=purpose,
                        provider=slot.provider,
                        key_index=slot.key_index,
                        model=slot.model,
                        latency_ms=latency_ms,
                        decision=f"{type(exc).__name__} — advancing to next key",
                        attempt=offset + 1,
                    )
                )
                async with self._advance_lock:
                    if self._current == idx:
                        self._current = idx + 1
                continue
            else:
                return response, slot, offset + 1

        raise AllProvidersExhaustedError(
            f"All {len(self._slots)} configured keys failed; last error: {last_exc!r}"
        ) from last_exc


_pool: LLMPool | None = None


def get_pool() -> LLMPool:
    """Lazily construct the module-level pool.

    Lazy so importing this module never requires API keys — only calling it
    does. Unit tests elsewhere in the codebase (RRF, citations, state) import
    from ``app`` freely without needing a key pool built at import time.
    """
    global _pool
    if _pool is None:
        _pool = LLMPool()
    return _pool


async def call_llm(
    *,
    node: str,
    purpose: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
) -> str:
    """Plain completion. Returns the assistant's text content."""

    async def make_request(slot: _Slot) -> Any:
        return await slot.client.chat.completions.create(
            model=slot.model,
            messages=messages,
            temperature=settings.decision_temperature if temperature is None else temperature,
            max_tokens=settings.max_tokens,
        )

    response, slot, attempt = await get_pool().request(make_request, node=node, purpose=purpose)
    usage = response.usage
    content = response.choices[0].message.content or ""

    await tracer.emit(
        TraceEvent(
            kind="llm_call",
            node=node,
            purpose=purpose,
            provider=slot.provider,
            key_index=slot.key_index,
            model=slot.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            reasoning_tokens=_reasoning_tokens(usage),
            total_tokens=getattr(usage, "total_tokens", None),
            path="text",
            attempt=attempt,
            decision=content[:200],
        )
    )
    return content


async def call_llm_with_tools(
    *,
    node: str,
    purpose: str,
    messages: list[dict[str, str]],
    tool_name: str,
    tool_schema: dict[str, Any],
    temperature: float | None = None,
) -> dict[str, Any]:
    """Agent decision via native tool calling, prompted JSON as fallback (D-02).

    ``tool_schema`` is ``{"description": str, "parameters": <JSON schema dict>}``.
    Returns the parsed arguments dict either way; which path served the call is
    recorded in the trace's ``path`` field, not just implied by the code.
    """
    tool_def = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_schema.get("description", ""),
            "parameters": tool_schema["parameters"],
        },
    }
    temp = settings.decision_temperature if temperature is None else temperature

    async def make_native_request(slot: _Slot) -> Any:
        return await slot.client.chat.completions.create(
            model=slot.model,
            messages=messages,
            tools=[tool_def],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            temperature=temp,
            max_tokens=settings.max_tokens,
        )

    response, slot, attempt = await get_pool().request(make_native_request, node=node, purpose=purpose)
    usage = response.usage
    tool_calls = response.choices[0].message.tool_calls or []

    if tool_calls:
        try:
            args = json.loads(tool_calls[0].function.arguments)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(
                f"Native tool call on {slot.key_index} returned unparseable arguments: {exc}"
            ) from exc

        await tracer.emit(
            TraceEvent(
                kind="llm_call",
                node=node,
                purpose=purpose,
                provider=slot.provider,
                key_index=slot.key_index,
                model=slot.model,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                reasoning_tokens=_reasoning_tokens(usage),
                total_tokens=getattr(usage, "total_tokens", None),
                path="native_tool_call",
                attempt=attempt,
                decision=json.dumps(args)[:200],
            )
        )
        return args

    # Fallback: same slot, prompted JSON. The provider answered fine — it just
    # didn't honour `tools` — so this is not a pool failover.
    fallback_messages = messages + [
        {
            "role": "user",
            "content": (
                "Respond with a single JSON object matching this schema, and "
                "nothing else — no prose before or after it:\n"
                f"{json.dumps(tool_schema['parameters'])}"
            ),
        }
    ]
    fb_response = await slot.client.chat.completions.create(
        model=slot.model,
        messages=fallback_messages,
        temperature=temp,
        max_tokens=settings.max_tokens,
    )
    fb_usage = fb_response.usage
    content = fb_response.choices[0].message.content or ""
    args = _extract_json_object(content)

    if args is None:
        raise ToolCallParseError(
            f"Neither native tool calling nor the prompted-JSON fallback parsed "
            f"on {slot.key_index}. Raw content: {content[:300]!r}"
        )

    await tracer.emit(
        TraceEvent(
            kind="llm_call",
            node=node,
            purpose=purpose,
            provider=slot.provider,
            key_index=slot.key_index,
            model=slot.model,
            prompt_tokens=getattr(fb_usage, "prompt_tokens", None),
            completion_tokens=getattr(fb_usage, "completion_tokens", None),
            reasoning_tokens=_reasoning_tokens(fb_usage),
            total_tokens=getattr(fb_usage, "total_tokens", None),
            path="json_fallback",
            attempt=attempt,
            decision=json.dumps(args)[:200],
        )
    )
    return args
