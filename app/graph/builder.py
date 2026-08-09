"""Assembles the seven nodes and four routing functions into a compiled,
checkpointed `StateGraph` (Phase 4).

Checkpointing is `AsyncSqliteSaver`, not `MemorySaver` — durable, on disk,
survives a process restart. `MemorySaver` would be wiped by `--reload`
between an interrupt and a resume, making both the crash-resume claim and the
Phase 5 `curl` demonstration impossible (D-14).

`build_graph()` is an async context manager, mirroring `AsyncSqliteSaver`'s
own intended usage — it owns a live `aiosqlite` connection that must be
closed cleanly, not a wrapper invented here for its own sake:

    async with build_graph() as graph:
        result = await graph.ainvoke(state, config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        })
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.graph.edges import (
    route_after_clear_for_revision,
    route_after_critic,
    route_after_planner,
    route_after_synthesizer,
)
from app.graph.nodes import (
    clear_for_revision_node,
    critic_node,
    human_approval_node,
    planner_escalate_node,
    planner_node,
    researcher_node,
    synthesizer_node,
)
from app.graph.state import CriticVerdict, Finding, Plan, ResearchState, SynthesisResult

# Registers our four custom dataclasses with the checkpoint serializer's
# msgpack allowlist. Without this, every resume prints "Deserializing
# unregistered type ... This will be blocked in a future version" — real
# behavior observed on a live run, not a hypothetical: the default
# (`allowed_msgpack_modules=True`) is permissive-with-a-warning today, but
# LangGraph documents that a future release tightens the default, at which
# point an unregistered custom type would fail to deserialize outright and
# every existing checkpoint using it would become unresumable.
#
# `JsonPlusSerializer()` with no arguments defaults to `allowed_msgpack_modules
# = True` ("allow everything, warn"), and `with_msgpack_allowlist()` has an
# early return — `if base_allowlist is True or base_allowlist is False: return
# self` — that treats adding specific types to an already-allow-everything
# base as a no-op. Calling `.with_msgpack_allowlist(...)` directly on a bare
# `JsonPlusSerializer()` silently does nothing, which is exactly what the
# first version of this fix got wrong. Starting from an explicit empty list
# instead normalizes to strict mode (only built-in safe types, confirmed from
# `_normalize_allowlist`'s source), which `with_msgpack_allowlist()` then
# correctly extends with exactly these four types — nothing more, nothing
# silently permissive.
_CHECKPOINT_SERDE = JsonPlusSerializer(allowed_msgpack_modules=[]).with_msgpack_allowlist(
    [Plan, Finding, SynthesisResult, CriticVerdict]
)

# Longest possible path (one escalation AND one revision, both bounds used):
# planner -> researcher* -> synthesizer -> planner_escalate -> researcher* ->
# synthesizer -> critic -> clear_for_revision -> researcher* -> synthesizer ->
# critic -> human_approval = 12 steps. A concurrent `Send` fan-out counts as
# one step, not N. 25 gives comfortable headroom without being effectively
# unbounded (D-06) — this is LangGraph's own invoke-time recursion guard, a
# separate mechanism from the hardcoded MAX_REVISIONS/MAX_ESCALATIONS caps in
# app/graph/state.py, not a restatement of them.
RECURSION_LIMIT = 25


def _build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("planner_escalate", planner_escalate_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("clear_for_revision", clear_for_revision_node)
    graph.add_node("human_approval", human_approval_node)

    graph.add_edge(START, "planner")

    # Dynamic fan-out (D-12/D-13): the Planner decides the researcher count
    # at runtime, so this is a conditional edge returning `Send` objects, not
    # a static edge. The same routing function serves all three places a
    # fan-out originates (initial plan, escalated re-plan, revision re-fan) —
    # it only ever reads `state["plan"].angles`, which is what changes.
    graph.add_conditional_edges("planner", route_after_planner, ["researcher"])
    graph.add_conditional_edges("planner_escalate", route_after_planner, ["researcher"])
    graph.add_conditional_edges("clear_for_revision", route_after_clear_for_revision, ["researcher"])

    # Fan-in happens structurally: every `Send`-dispatched researcher writes
    # to `findings` (the D-12 reducer), then all converge on this one edge.
    graph.add_edge("researcher", "synthesizer")

    graph.add_conditional_edges(
        "synthesizer",
        route_after_synthesizer,
        {
            "planner_escalate": "planner_escalate",
            "critic": "critic",
            "human_approval": "human_approval",
        },
    )

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "human_approval": "human_approval",
            "clear_for_revision": "clear_for_revision",
        },
    )

    graph.add_edge("human_approval", END)

    return graph


@asynccontextmanager
async def build_graph():
    """Compile the graph with a durable SQLite checkpointer.

    An async context manager because `AsyncSqliteSaver` owns a live
    `aiosqlite` connection that must be closed cleanly on exit — the
    library's own documented pattern, not a wrapper invented here.

    Connects via `aiosqlite` directly rather than
    `AsyncSqliteSaver.from_conn_string(...)` — confirmed from source that
    classmethod takes no `serde` parameter at all, so it cannot carry our
    msgpack allowlist. This does exactly what `from_conn_string` does
    internally (its own source is `async with aiosqlite.connect(conn_string)
    as conn: yield cls(conn)`), just with our serializer passed through.
    """
    settings.ensure_dirs()
    async with aiosqlite.connect(str(settings.checkpoint_db)) as conn:
        checkpointer = AsyncSqliteSaver(conn, serde=_CHECKPOINT_SERDE)
        yield _build_graph().compile(checkpointer=checkpointer)
