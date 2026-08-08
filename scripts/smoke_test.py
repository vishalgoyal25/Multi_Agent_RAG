"""Phase 1 smoke test.

Verifies assumptions A, B, C, and E (PHASES.md Phase 1) before any other phase
is built on top of ``app/core/llm.py``. Assumption D (RAGAS-on-Groq) is a
separate, timeboxed script — ``scripts/ragas_spike.py`` — because RAGAS is a
big enough dependency surface to deserve its own pass/fail report rather than
being folded in here.

Run:
    python -m scripts.smoke_test
"""

from __future__ import annotations

import asyncio
import json

import httpx
from openai import APIConnectionError, AsyncOpenAI

from app.core.config import settings
from app.core.llm import call_llm, call_llm_with_tools, get_pool
from app.core.tracing import tracer

PLAN_TOOL_SCHEMA = {
    "description": "Decide how to research a user's question.",
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["simple", "multi_angle"],
                "description": "simple for a single-fact lookup, multi_angle for a compound question",
            },
            "reason": {"type": "string", "description": "One sentence explaining the choice"},
        },
        "required": ["mode", "reason"],
    },
}


def _header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _read_trace_tail(n: int) -> list[dict]:
    lines = settings.trace_file.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-n:]]


async def check_provider_directly(provider: str) -> None:
    """A + B + C, against the raw client — independent of our own pool logic,
    so a bug in llm.py can't hide (or fake) a genuine provider problem."""
    keys = settings.groq_api_keys if provider == "groq" else settings.cerebras_api_keys
    if not keys:
        print(f"[{provider}] no key configured — skipping")
        return

    base_url = settings.groq_base_url if provider == "groq" else settings.cerebras_base_url
    model = settings.groq_model if provider == "groq" else settings.cerebras_model
    client = AsyncOpenAI(api_key=keys[0], base_url=base_url, timeout=settings.request_timeout_s)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "A user asks: 'What is the trial length?' Decide the research mode."}
        ],
        tools=[{"type": "function", "function": {"name": "plan_research", **PLAN_TOOL_SCHEMA}}],
        tool_choice={"type": "function", "function": {"name": "plan_research"}},
        temperature=0.0,
        max_tokens=256,
    )
    tool_calls = response.choices[0].message.tool_calls or []
    usage = response.usage
    reasoning = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(reasoning, "reasoning_tokens", None) if reasoning else None

    print(f"[{provider}] model={model}")
    print(f"[{provider}] native tool call returned: {bool(tool_calls)}")
    if tool_calls:
        print(f"[{provider}] arguments: {tool_calls[0].function.arguments}")
    else:
        print(f"[{provider}] !! no tool_calls — D-02's JSON fallback path is what carries this provider")
    print(
        f"[{provider}] usage: prompt={usage.prompt_tokens} "
        f"completion={usage.completion_tokens} total={usage.total_tokens} "
        f"reasoning={reasoning_tokens}"
    )


async def check_real_pool_path() -> None:
    """Exercise the actual production path once: get_pool() + call_llm_with_tools()."""
    _header("Production path — call_llm_with_tools() through the real sticky pool")
    get_pool().reset()
    args = await call_llm_with_tools(
        node="smoke_test",
        purpose="production path check",
        messages=[
            {"role": "user", "content": "A user asks: 'What is the trial length?' Decide the research mode."}
        ],
        tool_name="plan_research",
        tool_schema=PLAN_TOOL_SCHEMA,
    )
    print(f"Result via app.core.llm: {args}")
    assert "mode" in args, "production path did not return the expected schema"


async def check_hard_pool_walk() -> None:
    """E — force every slot but the last to fail and verify, from the trace
    file, that the sticky chain walked every one of them in order and the real
    last key actually served the request.

    The forced failures are `APIConnectionError` — the one retryable exception
    that can be raised without constructing a fake HTTP response object. This
    is NOT a literal 429; it exercises the same retryable branch (D-01 groups
    429/5xx/connection/timeout together) via the simplest honest simulation
    available. Labelled SIMULATED throughout, never presented as a real one.
    """
    _header("E — hard pool walk (SIMULATED failures on every key but the last)")

    pool = get_pool()
    pool.reset()
    slots = pool._slots  # noqa: SLF001 - a test harness verifying the pool's
    # own internals deliberately reaches into private state; this is not
    # external code consuming LLMPool as an API.

    if len(slots) < 2:
        print(
            f"Only {len(slots)} key(s) configured — the sticky *chain* needs at "
            f"least 2 to exercise for real. Add a second key (even a second Groq "
            f"key) to .env to run this check. Skipping E for this run."
        )
        return

    print(f"Configured chain: {pool.slot_labels}")
    originals = [slot.client.chat.completions.create for slot in slots]

    async def _simulated_failure(*_args, **_kwargs):
        raise APIConnectionError(
            request=httpx.Request("POST", "https://simulated-failure.invalid/smoke-test")
        )

    for slot in slots[:-1]:
        slot.client.chat.completions.create = _simulated_failure  # type: ignore[method-assign]

    await tracer.clear()
    try:
        result = await call_llm(
            node="smoke_test",
            purpose="hard pool walk",
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        )
    finally:
        for slot, original in zip(slots, originals):
            slot.client.chat.completions.create = original  # type: ignore[method-assign]

    print(f"Final response (real call, served by {slots[-1].key_index}): {result!r}")

    assert pool._current == len(slots) - 1, (  # noqa: SLF001
        f"cursor ended at index {pool._current}, expected {len(slots) - 1} "
        f"(sticky position should now sit on the last slot)"
    )

    events = _read_trace_tail(len(slots))
    failover_events = [e for e in events if e["kind"] == "llm_failover"]
    call_events = [e for e in events if e["kind"] == "llm_call"]

    assert len(failover_events) == len(slots) - 1, (
        f"expected {len(slots) - 1} failover events, trace shows {len(failover_events)}"
    )
    for expected_slot, event in zip(slots[:-1], failover_events):
        assert event["key_index"] == expected_slot.key_index, (
            f"failover order broken: expected {expected_slot.key_index}, got {event['key_index']}"
        )

    assert len(call_events) == 1, f"expected exactly one successful llm_call, got {len(call_events)}"
    assert call_events[0]["key_index"] == slots[-1].key_index, (
        f"success should be attributed to {slots[-1].key_index}, trace says {call_events[0]['key_index']}"
    )

    print(
        f"PASS — trace confirms all {len(slots) - 1} forced failures were logged "
        f"in order, and the real success landed on {slots[-1].key_index}."
    )
    pool.reset()


async def main() -> None:
    missing = settings.missing_keys()
    if missing:
        raise SystemExit(
            f"Missing: {', '.join(missing)}. Copy .env.example to .env and paste real keys in."
        )

    _header("A/B/C — direct provider checks (bypassing our own pool logic)")
    for provider in ("groq", "cerebras"):
        await check_provider_directly(provider)

    await check_real_pool_path()
    await check_hard_pool_walk()

    _header("Summary")
    print("A — each provider reachable under its own model ID: see output above")
    print("B — native tool calling checked directly per provider, and via the real code path")
    print("C — token usage, including reasoning_tokens where the provider reports it: see above")
    print("E — hard sticky pool walk: PASS above, or explicitly skipped if fewer than 2 keys configured")
    print("D — NOT covered here. Run: python -m scripts.ragas_spike")
    print(f"\nFull trace written to: {settings.trace_file}")


if __name__ == "__main__":
    asyncio.run(main())
