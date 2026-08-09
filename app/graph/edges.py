"""Conditional routing functions for the graph (Phase 4).

Routing functions are pure: given the current state, they return which
node(s) run next — a node name, `END`, or one-or-more `Send` objects for
dynamic fan-out. **They never update state themselves** — LangGraph only
applies state updates from what a NODE returns, never from a routing
function's return value. Where a transition genuinely needs a state change
first (clearing `findings` before a revision re-fans-out, or re-planning
before an escalated attempt), that change is a small dedicated node in
`app/graph/nodes.py` that runs *before* the routing function dispatches the
next fan-out — never bundled into the routing function, which structurally
cannot carry one.

This file assumes two small node additions to `app/graph/nodes.py` that don't
exist yet: `planner_escalate_node` (Planner re-run, forced to `multi_angle`)
and `clear_for_revision_node` (emits `{"findings": CLEAR}` before the revised
research fans out). Both are flagged as the next required step, not silently
assumed — `builder.py` cannot wire the graph without them.
"""

from __future__ import annotations

from langgraph.graph import END
from langgraph.types import Send

from app.graph.state import MAX_ESCALATIONS, MAX_REVISIONS, ResearchAngle, ResearchState


def route_after_planner(state: ResearchState) -> list[Send]:
    """Dynamic fan-out (D-12/D-13): one `Send` per angle, whether the Planner
    chose `simple` (one angle) or `multi_angle` (2-4) — the researcher count
    is a runtime decision the Planner made, which is exactly what `Send`
    exists to express; a static edge cannot.
    """
    plan = state["plan"]
    return [
        Send("researcher", ResearchAngle(angle_id=f"a{i}", question=q))
        for i, q in enumerate(plan.angles)
    ]


def route_after_synthesizer(state: ResearchState) -> str:
    """Three-way split (D-13):
    - `simple` + a real, supported answer -> human_approval (D-14) — the
      cheap path's whole point, still gated by the single HITL checkpoint.
    - `simple` + abstained/unsupported, escalation not yet used -> escalate once.
    - anything else (`multi_angle`, or escalation already used) -> Critic.

    Bounding escalation is belt-and-suspenders, on purpose: escalating flips
    `plan.mode` to `multi_angle` (via `planner_escalate_node`), which alone
    would stop this function from escalating a second time. The explicit
    `escalation_count` check is a second, independent guarantee of the same
    bound — D-06 treats a cap as unsafe if it depends on only one mechanism
    holding.
    """
    plan = state["plan"]
    synthesis = state["synthesis"]
    needs_escalation = synthesis.abstained or not synthesis.citations

    if plan.mode == "simple":
        if needs_escalation and state["escalation_count"] < MAX_ESCALATIONS:
            return "planner_escalate"
        return "human_approval"

    return "critic"


def route_after_critic(state: ResearchState) -> str:
    """Two-way split: approve -> human_approval (D-14), the single HITL
    checkpoint every successful path funnels through before END. Revise -> a
    real node clears `findings` first (this function cannot update state
    itself), then re-fans-out to the same angles. Bounded to one revision
    (D-06) regardless of the verdict.
    """
    verdict = state["critic_verdict"]
    if verdict.approved:
        return "human_approval"
    if state["revision_count"] >= MAX_REVISIONS:
        # Bound reached — ship the best answer already produced rather than loop again.
        return "human_approval"
    return "clear_for_revision"


def route_after_clear_for_revision(state: ResearchState) -> list[Send]:
    """Re-fans-out to the SAME angles the Planner originally chose, now
    carrying the Critic's feedback so the second attempt is corrective, not a
    verbatim repeat of the first attempt's identical search.
    """
    plan = state["plan"]
    feedback = state["critic_verdict"].feedback
    return [
        Send("researcher", ResearchAngle(angle_id=f"a{i}", question=q, feedback=feedback))
        for i, q in enumerate(plan.angles)
    ]
