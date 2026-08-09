"""FastAPI service exposing the graph: `/ask`, `/ask/stream` (SSE), `/resume`
(the resume contract Phase 4 fixed, D-14), and `/health`.

The graph and its checkpointer connection are opened ONCE at app startup
(the lifespan below), not per request — `AsyncSqliteSaver` owns a live
`aiosqlite` connection, and opening/closing one per HTTP request would be
wasteful and risks concurrent-access issues on the same file.

`/ask/stream`'s live progress comes from `tracer`'s subscriber queue (built
in Phase 1, unused until now), filtered by `thread_id` — necessary, not
defensive, because `tracer.emit()` broadcasts to every current subscriber
regardless of which request it belongs to; two concurrent `/ask/stream`
calls would otherwise see each other's events interleaved.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    AskRequest,
    CriticVerdictSchema,
    FinalAnswer,
    FindingSchema,
    HealthResponse,
    PendingApproval,
    PlanSchema,
    ResumeRequest,
    SynthesisSchema,
)
from app.core.tracing import current_thread_id, tracer
from app.graph.builder import RECURSION_LIMIT, build_graph, run_to_interrupt_or_end
from app.graph.state import initial_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with build_graph() as graph:
        app.state.graph = graph
        yield


app = FastAPI(title="Multi-Agent RAG Research Platform", lifespan=lifespan)

_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@app.get("/")
async def index() -> FileResponse:
    """Serves the single self-contained live-graph page (D-08 — plain
    HTML/JS/Cytoscape.js, no build toolchain). One route, not a
    `StaticFiles` mount, since there's exactly one file to serve."""
    return FileResponse(_STATIC_DIR / "index.html")


def _config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def _build_pending_approval(thread_id: str, interrupt_payload: dict) -> PendingApproval:
    verdict = interrupt_payload.get("critic_verdict")
    return PendingApproval(
        thread_id=thread_id,
        question=interrupt_payload["question"],
        plan=PlanSchema.from_plan(interrupt_payload["plan"]),
        findings=[FindingSchema.from_finding(f) for f in interrupt_payload["findings"]],
        synthesis=SynthesisSchema.from_synthesis(interrupt_payload["synthesis"]),
        critic_verdict=CriticVerdictSchema.from_verdict(verdict) if verdict else None,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/ask", response_model=PendingApproval)
async def ask(payload: AskRequest, request: Request) -> PendingApproval:
    """Runs the graph to the human-approval interrupt and returns — never a
    final answer directly, since every path funnels through that checkpoint
    (D-14). Blocking: this call takes as long as the full research run does.
    """
    thread_id = str(uuid.uuid4())
    token = current_thread_id.set(thread_id)
    try:
        interrupt_payload, _ = await run_to_interrupt_or_end(
            request.app.state.graph, initial_state(payload.question), _config_for(thread_id)
        )
    finally:
        current_thread_id.reset(token)

    if interrupt_payload is None:
        raise HTTPException(500, "Graph finished without reaching the human-approval interrupt.")

    return _build_pending_approval(thread_id, interrupt_payload)


@app.get("/ask/stream")
async def ask_stream(q: str, request: Request) -> EventSourceResponse:
    """Same underlying run as `POST /ask`, exposed as Server-Sent Events
    instead of one blocking response — this is what Phase 6's live graph
    consumes. Ends with an `awaiting_approval` event carrying the same
    payload `/ask` returns directly.

    Streams two kinds of events, merged:
    - `llm_call`/`llm_failover` (rich: provider, tokens, latency) — from
      `tracer`'s subscriber queue, same mechanism as before.
    - `node` (a plain human-readable line) — from the graph's own
      `trace_events`, which every node writes to regardless of whether it
      made an LLM call. This closes a real gap: `critic_node`'s auto-approve
      path (an abstained answer) short-circuits without calling an LLM at
      all, so it never emits a tracer event — without this second source,
      the frontend would never show the Critic running in that case, even
      though it genuinely did.

    A single loop over `astream(..., stream_mode="values")` correctly
    interleaves both without task/queue juggling: `tracer.emit()` completes
    *inside* a node before that node's state update ever surfaces to
    `astream`, so every LLM-call event for a step is already queued by the
    time that step's state snapshot arrives — draining the queue right
    before checking the new state is always in the right order.
    """
    graph = request.app.state.graph
    thread_id = str(uuid.uuid4())

    async def event_generator():
        token = current_thread_id.set(thread_id)
        queue = tracer.subscribe()
        printed = 0
        try:
            yield {"event": "started", "data": json.dumps({"thread_id": thread_id})}

            async for state in graph.astream(initial_state(q), _config_for(thread_id), stream_mode="values"):
                # Drain first: guaranteed to hold every LLM-call event this
                # step produced, per the ordering argument above. Filtered by
                # thread_id — necessary, not defensive, since this queue also
                # receives OTHER concurrent requests' events (`tracer.emit()`
                # broadcasts to every subscriber, not just this one).
                while not queue.empty():
                    event = queue.get_nowait()
                    if event.thread_id == thread_id:
                        yield {"event": event.kind, "data": json.dumps(event.to_dict())}

                if "__interrupt__" in state:
                    pending = _build_pending_approval(thread_id, state["__interrupt__"][0].value)
                    yield {"event": "awaiting_approval", "data": pending.model_dump_json()}
                    return

                events = state.get("trace_events", [])
                for line in events[printed:]:
                    yield {"event": "node", "data": json.dumps({"message": line})}
                printed = len(events)

            yield {"event": "error", "data": "Graph finished without reaching the human-approval interrupt."}
        finally:
            tracer.unsubscribe(queue)
            current_thread_id.reset(token)

    return EventSourceResponse(event_generator())


@app.post("/resume", response_model=FinalAnswer)
async def resume(payload: ResumeRequest, request: Request) -> FinalAnswer:
    """The resume contract fixed in Phase 4 (D-14), exposed over HTTP: takes
    the thread being resumed and the human's decision, resumes from the
    persisted checkpoint. Works even across a server restart, since
    `AsyncSqliteSaver` is durable on disk, not in-process memory.
    """
    token = current_thread_id.set(payload.thread_id)
    try:
        interrupt_payload, final_state = await run_to_interrupt_or_end(
            request.app.state.graph,
            Command(resume={"approved": payload.approved}),
            _config_for(payload.thread_id),
        )
    finally:
        current_thread_id.reset(token)

    if final_state is None:
        detail = (
            "Resume hit another interrupt instead of completing — unexpected, "
            "the graph has only one human-approval checkpoint."
            if interrupt_payload is not None
            else "Resume did not produce a final state."
        )
        raise HTTPException(500, detail)

    synthesis = final_state["synthesis"]
    return FinalAnswer(
        thread_id=payload.thread_id,
        answer=synthesis.answer,
        citations=synthesis.citations,
        revision_count=final_state["revision_count"],
        escalation_count=final_state["escalation_count"],
    )
