"""Phase 3 exit check: each node runs standalone on a fixed input.

Two things this phase must prove, not assume (PHASES.md, Phase 3):
  - The Planner's simple/multi_angle split is real (D-13's cheap path routes off it).
  - The Synthesizer genuinely abstains on a question the corpus doesn't cover (D-11),
    rather than stitching a confident answer out of weak partial findings.

`asyncio.gather` below runs two researchers concurrently purely as a test-harness
convenience — it is NOT the graph's fan-out mechanism. The real graph uses
LangGraph's `Send` (Phase 4); nothing here re-implements or substitutes for that.

Run:
    python -m scripts.test_nodes
"""

from __future__ import annotations

import asyncio

from app.graph.nodes import critic_node, planner_node, researcher_node, synthesizer_node
from app.graph.state import ResearchAngle


def _header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def test_planner() -> None:
    _header("Planner — simple vs multi_angle split")

    simple = await planner_node({"question": "What is the trial length?"})
    print(f"Simple question -> mode={simple['plan'].mode}, angles={simple['plan'].angles}")
    assert simple["plan"].mode == "simple", f"expected simple, got {simple['plan'].mode}"

    compound_question = (
        "How does the Growth tier's pricing compare to what the contract guarantees, "
        "and what happens if we exceed the data source limit?"
    )
    compound = await planner_node({"question": compound_question})
    print(f"Compound question -> mode={compound['plan'].mode}, angles={compound['plan'].angles}")
    assert compound["plan"].mode == "multi_angle", f"expected multi_angle, got {compound['plan'].mode}"
    assert len(compound["plan"].angles) >= 2, "multi_angle plan should have more than one angle"


async def test_covered_question() -> tuple[dict, dict]:
    """Researcher + Synthesizer on a question the corpus genuinely covers —
    the baseline that the abstain test below is contrasted against."""
    _header("Researcher + Synthesizer — a covered question (should NOT abstain)")

    angle = ResearchAngle(angle_id="a0", question="What is the trial length according to pricing documentation?")
    research_result = await researcher_node(angle)
    finding = research_result["findings"][0]
    print(f"Finding: {finding.text}")
    print(f"Supporting chunks: {finding.chunk_ids}")

    state = {"question": "What is the trial length?", "findings": [finding]}
    synth_result = await synthesizer_node(state)
    synthesis = synth_result["synthesis"]
    print(f"Synthesis: abstained={synthesis.abstained}, citations={synthesis.citations}")
    print(f"Answer: {synthesis.answer}")
    for event in synth_result["trace_events"]:
        print(f"  trace: {event}")

    assert not synthesis.abstained, "a covered question should not abstain"
    assert synthesis.citations, "a non-abstained answer should carry real citations"

    return state, synth_result


async def test_uncovered_question() -> dict:
    """The corpus has deliberate coverage gaps on headcount and revenue
    (D-07, D-11) — this is the abstain test, not a synthetic empty-findings
    shortcut. Real researchers run against real angles and come back weak;
    the Synthesizer must recognize that honestly."""
    _header("Planner + Researchers + Synthesizer — an UNCOVERED question (should abstain)")

    question = "How many employees does Northbay have, and what was its revenue last year?"
    plan_result = await planner_node({"question": question})
    plan = plan_result["plan"]
    print(f"Plan: mode={plan.mode}, angles={plan.angles}")

    angles = [ResearchAngle(angle_id=f"a{i}", question=q) for i, q in enumerate(plan.angles)]
    research_results = await asyncio.gather(*(researcher_node(a) for a in angles))
    findings = [r["findings"][0] for r in research_results]
    for f in findings:
        print(f"Finding[{f.angle_id}]: {f.text}  (chunks: {f.chunk_ids})")

    state = {"question": question, "findings": findings}
    synth_result = await synthesizer_node(state)
    synthesis = synth_result["synthesis"]
    print(f"Synthesis: abstained={synthesis.abstained}")
    print(f"Answer: {synthesis.answer}")

    assert synthesis.abstained, "an uncovered question must abstain, not guess from weak findings"

    return state | {"synthesis": synthesis}


async def test_critic(covered_state: dict, uncovered_state: dict) -> None:
    _header("Critic — real judgement on a covered answer, auto-approve on an abstain")

    covered_verdict = await critic_node(covered_state)
    verdict = covered_verdict["critic_verdict"]
    print(f"Covered-answer verdict: approved={verdict.approved} — {verdict.feedback}")

    abstained_verdict = await critic_node(uncovered_state)
    verdict2 = abstained_verdict["critic_verdict"]
    print(f"Abstained-answer verdict: approved={verdict2.approved} — {verdict2.feedback}")
    assert verdict2.approved, "an abstained answer should always be auto-approved"
    assert "nothing to fact-check" in verdict2.feedback, "abstain should short-circuit, not call the LLM"


async def main() -> None:
    await test_planner()

    covered_state, covered_synth = await test_covered_question()
    covered_full_state = covered_state | {"synthesis": covered_synth["synthesis"]}

    uncovered_state = await test_uncovered_question()

    await test_critic(covered_full_state, uncovered_state)

    _header("Summary")
    print("Planner: simple/multi_angle split confirmed on real questions.")
    print("Researcher + Synthesizer: real answer with citations on a covered question.")
    print("Synthesizer: genuine abstain on an uncovered question (D-11) — not asserted, verified.")
    print("Critic: real judgement on a supported answer; auto-approved an abstain without an LLM call.")


if __name__ == "__main__":
    asyncio.run(main())
