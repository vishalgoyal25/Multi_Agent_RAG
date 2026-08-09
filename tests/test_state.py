"""Pure tests for graph state mechanics — the reducer, both bounds, and every
routing function. No LLM calls, no live index; every input is synthetic.
"""

from __future__ import annotations

from langgraph.types import Send

from app.graph.edges import (
    route_after_clear_for_revision,
    route_after_critic,
    route_after_planner,
    route_after_synthesizer,
)
from app.graph.state import (
    CLEAR,
    MAX_ESCALATIONS,
    MAX_REVISIONS,
    CriticVerdict,
    Finding,
    Plan,
    SynthesisResult,
    merge_findings,
)


# --- Bounds are exactly what D-06 requires -----------------------------------


def test_bounds_are_exactly_one():
    """D-06: a single revision, a single escalation — hardcoded, not a
    config value. This test exists so raising either constant is a visible,
    deliberate code change caught here, not a silent drift."""
    assert MAX_REVISIONS == 1
    assert MAX_ESCALATIONS == 1


# --- merge_findings reducer (D-12) -------------------------------------------


def test_merge_findings_appends_on_fan_in():
    f1 = Finding(angle_id="a0", text="x", chunk_ids=("c0",))
    f2 = Finding(angle_id="a1", text="y", chunk_ids=("c1",))

    assert merge_findings([f1], [f2]) == [f1, f2]


def test_merge_findings_resets_on_clear_sentinel():
    f1 = Finding(angle_id="a0", text="x", chunk_ids=("c0",))

    assert merge_findings([f1], CLEAR) == []


def test_merge_findings_clear_then_append_starts_fresh():
    """Mirrors the real revision sequence: `clear_for_revision_node` returns
    CLEAR, then the re-fanned researchers append fresh findings on top of an
    empty list — never mixed with findings the Critic already rejected."""
    stale = [Finding(angle_id="a0", text="stale", chunk_ids=("c0",))]
    cleared = merge_findings(stale, CLEAR)
    fresh = Finding(angle_id="a0", text="fresh", chunk_ids=("c1",))

    assert merge_findings(cleared, [fresh]) == [fresh]


# --- route_after_planner (dynamic fan-out, D-12/D-13) ------------------------


def test_route_after_planner_dispatches_one_send_per_angle():
    state = {"plan": Plan(mode="multi_angle", angles=("angle one", "angle two"), reason="r")}

    sends = route_after_planner(state)

    assert len(sends) == 2
    assert all(isinstance(s, Send) for s in sends)
    assert [s.node for s in sends] == ["researcher", "researcher"]
    assert [s.arg.question for s in sends] == ["angle one", "angle two"]
    assert [s.arg.angle_id for s in sends] == ["a0", "a1"]
    assert all(s.arg.feedback is None for s in sends)


def test_route_after_planner_simple_mode_dispatches_exactly_one():
    state = {"plan": Plan(mode="simple", angles=("only angle",), reason="r")}

    sends = route_after_planner(state)

    assert len(sends) == 1
    assert sends[0].arg.question == "only angle"


# --- route_after_synthesizer (three-way split, D-13) -------------------------


def _synthesis(*, abstained=False, citations=("c0",)):
    return SynthesisResult(answer="a", citations=citations, abstained=abstained)


def test_route_after_synthesizer_simple_success_goes_to_human_approval():
    state = {
        "plan": Plan(mode="simple", angles=("q",), reason="r"),
        "synthesis": _synthesis(abstained=False, citations=("c0",)),
        "escalation_count": 0,
    }

    assert route_after_synthesizer(state) == "human_approval"


def test_route_after_synthesizer_simple_abstain_escalates_once():
    state = {
        "plan": Plan(mode="simple", angles=("q",), reason="r"),
        "synthesis": _synthesis(abstained=True, citations=()),
        "escalation_count": 0,
    }

    assert route_after_synthesizer(state) == "planner_escalate"


def test_route_after_synthesizer_no_citations_without_abstain_also_escalates():
    """`needs_escalation` covers BOTH abstained and 'claimed an answer but
    has no real citations' — the same signal Synthesizer's own override
    checks (nodes.py)."""
    state = {
        "plan": Plan(mode="simple", angles=("q",), reason="r"),
        "synthesis": _synthesis(abstained=False, citations=()),
        "escalation_count": 0,
    }

    assert route_after_synthesizer(state) == "planner_escalate"


def test_route_after_synthesizer_wont_escalate_twice():
    """Belt-and-suspenders (edges.py's own docstring): even if somehow still
    `simple` with escalation_count already at the cap, this function refuses
    to escalate again."""
    state = {
        "plan": Plan(mode="simple", angles=("q",), reason="r"),
        "synthesis": _synthesis(abstained=True, citations=()),
        "escalation_count": MAX_ESCALATIONS,
    }

    assert route_after_synthesizer(state) == "human_approval"


def test_route_after_synthesizer_multi_angle_always_goes_to_critic():
    """Even an abstained multi_angle answer goes to Critic — critic_node
    itself (Phase 3) is what short-circuits an abstain, not this router."""
    state = {
        "plan": Plan(mode="multi_angle", angles=("q1", "q2"), reason="r"),
        "synthesis": _synthesis(abstained=True, citations=()),
        "escalation_count": 0,
    }

    assert route_after_synthesizer(state) == "critic"


# --- route_after_critic (two-way split, revision bound, D-06) ---------------


def test_route_after_critic_approved_goes_to_human_approval():
    state = {
        "critic_verdict": CriticVerdict(approved=True, feedback="ok"),
        "revision_count": 0,
    }

    assert route_after_critic(state) == "human_approval"


def test_route_after_critic_revise_under_cap_clears_for_revision():
    state = {
        "critic_verdict": CriticVerdict(approved=False, feedback="missing X"),
        "revision_count": 0,
    }

    assert route_after_critic(state) == "clear_for_revision"


def test_route_after_critic_wont_revise_twice():
    state = {
        "critic_verdict": CriticVerdict(approved=False, feedback="still missing X"),
        "revision_count": MAX_REVISIONS,
    }

    assert route_after_critic(state) == "human_approval"


# --- route_after_clear_for_revision (re-fan-out carries feedback) ----------


def test_route_after_clear_for_revision_carries_critic_feedback():
    state = {
        "plan": Plan(mode="multi_angle", angles=("angle one", "angle two"), reason="r"),
        "critic_verdict": CriticVerdict(approved=False, feedback="check the pricing doc again"),
    }

    sends = route_after_clear_for_revision(state)

    assert len(sends) == 2
    assert all(s.arg.feedback == "check the pricing doc again" for s in sends)
    assert [s.arg.question for s in sends] == ["angle one", "angle two"]
