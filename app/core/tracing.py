"""Structured tracing.

One trace record per interesting event: every LLM call, every routing decision,
every cache hit. Records go to two places at once —

- **disk**, as JSON Lines in ``logs/trace.jsonl``, so a run can be inspected after
  the fact;
- **live subscribers**, as in-process asyncio queues, which is what the SSE
  endpoint (Phase 5) and the animated graph (Phase 6) consume.

Both sinks are fed from the same ``emit`` call, so the UI can never show something
the log doesn't have. Subscribers are added here in Phase 1, unused until Phase 5,
because retro-fitting a fan-out into a logger later means touching every call site.

``emit`` is async. A question fans out into several concurrent researcher calls
(D-12, Phase 4), each tracing its own LLM call — a *blocking* file write on every
one of those would stall the event loop precisely while the parallelism this
project demonstrates is supposed to be happening. The write still lands on disk
before ``emit`` returns (durability is unchanged); it just runs off-thread via
``asyncio.to_thread`` instead of blocking the loop while it happens.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from app.core.config import settings

EventKind = Literal["llm_call", "llm_failover", "routing", "cache", "node", "error"]


@dataclass
class TraceEvent:
    """One traced event.

    The LLM-specific fields stay ``None`` on non-LLM events rather than being split
    into a second type — one flat record shape keeps the JSONL readable and the
    frontend's job trivial.
    """

    kind: EventKind
    node: str
    purpose: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # What the event concluded, in one short human-readable line. This is the
    # field you actually read when debugging a run.
    decision: str = ""

    # LLM-call fields.
    provider: str | None = None
    # Which slot in the provider's key pool served (or failed on) this call —
    # "groq[0]", "groq[1]", "cerebras[0]". Sticky failover (D-01) means a run's
    # trace should show one slot used repeatedly, then a jump on failure — this
    # field is what makes that chain visible and testable, not just asserted.
    key_index: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None
    # "native_tool_call" | "json_fallback" | "text" — D-02 requires the two
    # decision paths be distinguishable in the trace, not just in the code.
    path: str | None = None
    attempt: int | None = None

    # Anything node-specific. Kept as a bag so adding a field in Phase 3 doesn't
    # mean changing this class.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != {}}


class Tracer:
    """Writes trace events to disk and fans them out to live subscribers."""

    def __init__(self) -> None:
        settings.ensure_dirs()
        self._path = settings.trace_file
        # asyncio.Lock, not threading.Lock: this class is only ever touched from
        # coroutines, and an asyncio.Lock is what lets concurrent researcher
        # tasks (Send fan-out, Phase 4) queue up for the write without blocking
        # each other's event-loop turn while waiting.
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue[TraceEvent]] = []

    # --- live fan-out (consumed from Phase 5) ----------------------------
    def subscribe(self) -> asyncio.Queue[TraceEvent]:
        queue: asyncio.Queue[TraceEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[TraceEvent]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    # --- emit -------------------------------------------------------------
    def _write_line_sync(self, line: str) -> None:
        """The actual blocking file append. Runs off the event loop thread."""
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    async def emit(self, event: TraceEvent) -> TraceEvent:
        """Persist an event and push it to any live subscribers.

        The write still completes before this coroutine returns — the durability
        guarantee (a trace is on disk before the traced call's caller proceeds) is
        unchanged from a synchronous logger. What changes is *how*: the blocking
        syscall runs via ``asyncio.to_thread`` so it never stalls the event loop,
        which matters here because several researchers trace concurrently.
        """
        line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._write_line_sync, line)

        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queues
                # A slow consumer must never stall the agent graph. Drop the event
                # for that subscriber only; it is already durable on disk.
                pass

        return event

    async def clear(self) -> None:
        """Truncate the trace file. Used by scripts that want a clean run."""
        async with self._lock:
            await asyncio.to_thread(self._path.write_text, "", encoding="utf-8")


tracer = Tracer()
