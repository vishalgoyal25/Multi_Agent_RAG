"""Phase 4 exit demonstration: a full graph run, including the HITL
interrupt/resume (D-14).

Two required real runs (Phase 4's own exit criterion):

    python -m scripts.run_graph "How does the Growth tier's pricing compare to what the contract guarantees, and what happens if we exceed the data source limit?"
    python -m scripts.run_graph "What is the trial length?"

The first should show parallel researchers and, if the Critic objects, one
revision — never more. The second should visibly take the cheap path and
never reach the Critic. Both pause at the human-approval interrupt and
resume from the checkpoint — proving `AsyncSqliteSaver` persistence, not
just that the graph runs once end to end.

`stream_mode="values"` is used deliberately, not "updates": `updates` mode
explicitly filters out interrupted tasks (`map_output_updates` in
langgraph's own `pregel/_io.py` drops any write tagged `INTERRUPT`), so it
would never surface the pause at all. `values` mode's own type-hint
docstring says it emits state "including interrupts" — confirmed against
source, not assumed. Each snapshot in `values` mode carries the FULL state,
so progress is shown by diffing `trace_events` against what's already been
printed, rather than needing per-node update chunks.

Run:
    python -m scripts.run_graph "<question>"
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from langgraph.types import Command

from app.graph.builder import RECURSION_LIMIT, build_graph


def _initial_state(question: str) -> dict:
    return {
        "question": question,
        "plan": None,
        "findings": [],
        "synthesis": None,
        "critic_verdict": None,
        "revision_count": 0,
        "escalation_count": 0,
        "trace_events": [],
    }


async def _stream_until_interrupt_or_end(
    graph, stream_input, config: dict, already_printed: int
) -> tuple[dict | None, dict | None, int]:
    """Prints only the `trace_events` new since the last snapshot — so a
    concurrent researcher fan-out shows all its findings together in one
    step, not out of order. Returns (interrupt_payload, final_state,
    events_printed_so_far); exactly one of the first two is not None.
    """
    last_state: dict | None = None
    async for state in graph.astream(stream_input, config, stream_mode="values"):
        if "__interrupt__" in state:
            return state["__interrupt__"][0].value, None, already_printed
        events = state.get("trace_events", [])
        for event in events[already_printed:]:
            print(f"  {event}")
        already_printed = len(events)
        last_state = state
    return None, last_state, already_printed


def _print_interrupt_payload(payload: dict) -> None:
    print("\n>>> INTERRUPTED — awaiting human approval <<<")
    print(f"Question: {payload['question']}")
    plan = payload["plan"]
    print(f"Plan: mode={plan.mode}, angles={plan.angles}")
    print(f"Findings ({len(payload['findings'])}):")
    for f in payload["findings"]:
        print(f"  [{f.angle_id}] {f.text}  (chunks: {f.chunk_ids})")
    synthesis = payload["synthesis"]
    print(f"Draft answer (abstained={synthesis.abstained}): {synthesis.answer}")
    print(f"Citations: {synthesis.citations}")
    verdict = payload.get("critic_verdict")
    if verdict:
        print(f"Critic verdict: approved={verdict.approved} — {verdict.feedback}")


async def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m scripts.run_graph "<question>"')
    question = sys.argv[1]

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    print(f"Question: {question}")
    print(f"Thread ID: {thread_id}\n")

    async with build_graph() as graph:
        interrupt_payload, _, printed = await _stream_until_interrupt_or_end(
            graph, _initial_state(question), config, already_printed=0
        )

        if interrupt_payload is None:
            print("\nGraph finished without ever reaching the human-approval interrupt — unexpected.")
            return

        _print_interrupt_payload(interrupt_payload)

        print("\nSimulating human approval (approved=True), resuming from the checkpoint...")
        _, final_state, _ = await _stream_until_interrupt_or_end(
            graph, Command(resume={"approved": True}), config, already_printed=printed
        )

        print("\n=== FINAL ANSWER ===")
        final_synthesis = final_state["synthesis"]
        print(final_synthesis.answer)
        print(f"Citations: {final_synthesis.citations}")
        print(f"Revisions used: {final_state['revision_count']}")
        print(f"Escalations used: {final_state['escalation_count']}")


if __name__ == "__main__":
    asyncio.run(main())
