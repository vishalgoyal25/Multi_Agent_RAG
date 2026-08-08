"""The four agent nodes: Planner, Researcher, Synthesizer, Critic.

Only Planner and Critic are agentic (D-05) — they make the graph's two real
judgement calls, via native tool calling (D-02). Researcher and Synthesizer
also call an LLM, but make no control-flow decision; their tool-calling use
here is for reliable structured output (parseable chunk IDs), not because
they're agentic — D-02 scopes tool calling to where it's needed, not
exclusively to the two agentic nodes.

Every citation from every node is validated against chunks actually placed in
context before being trusted — never what a model merely claims it cited
(CLAUDE.md, D-11).
"""

from __future__ import annotations

from app.core.config import settings
from app.core.llm import call_llm_with_tools
from app.graph.state import (
    CriticVerdict,
    Finding,
    Plan,
    ResearchAngle,
    ResearchState,
    SynthesisResult,
)
from app.retrieval.hybrid import SearchResult, ahybrid_search


def validate_citations(claimed: list[str], available: set[str]) -> tuple[str, ...]:
    """Keep only citations that point at chunks genuinely placed in context.

    Never trust what a model claims it cited (CLAUDE.md, D-11) — a fabricated
    ID and a real ID from elsewhere in the corpus that simply wasn't part of
    this run's retrieved context are rejected the same way: both fail the
    same membership check against what was actually available here.

    Extracted as a standalone pure function — used by both `researcher_node`
    and `synthesizer_node`, and unit-tested directly in
    tests/test_citations.py without needing an LLM or a live index.
    """
    return tuple(cid for cid in claimed if cid in available)


# --- Planner -----------------------------------------------------------------

_PLANNER_SCHEMA = {
    "description": (
        "Decide how to research the user's question: a single quick lookup, "
        "or split into multiple parallel research angles."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["simple", "multi_angle"],
                "description": "simple for a single-fact lookup; multi_angle for a compound question",
            },
            "angles": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "For 'simple': exactly one item — the question itself, or a "
                    "minimal rephrasing for retrieval. Do NOT decompose a single fact "
                    "into investigative steps (e.g. 'identify the product', 'find the "
                    "policy', 'check exceptions') — that is not what 'multi_angle' is "
                    "for and wastes research calls on one straightforward lookup. "
                    "For 'multi_angle': 2-4 angles, each a genuinely distinct topic "
                    "the question actually asks about."
                ),
            },
            "reason": {"type": "string", "description": "One sentence explaining the mode choice."},
        },
        "required": ["mode", "angles", "reason"],
    },
}


async def planner_node(state: ResearchState) -> dict:
    """Agentic (D-05): decides the shape of the work, not a threshold in code."""
    result = await call_llm_with_tools(
        node="planner",
        purpose="decide research mode and angles",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are planning how to research this question over a company's "
                    "internal documentation:\n\n"
                    f"{state['question']}\n\n"
                    "Choose 'simple' if it asks about ONE fact, term, price, or policy — "
                    "even if answering it means reading one document carefully. Do not "
                    "split a single-fact question into multiple investigative steps just "
                    "to be thorough; that is over-decomposition, not multi_angle.\n\n"
                    "Choose 'multi_angle' only if the question itself names or implies "
                    "two or more genuinely distinct topics that must each be researched "
                    "separately — e.g. it explicitly compares two different things, or "
                    "joins unrelated questions with 'and'.\n\n"
                    "Example — simple: 'What is the SLA response time for critical "
                    "tickets?' (one policy, one fact).\n"
                    "Example — multi_angle: 'What data residency options are available, "
                    "and how do they interact with the governance audit requirements?' "
                    "(two distinct topics: deployment options, and governance/audit)."
                ),
            }
        ],
        tool_name="plan_research",
        tool_schema=_PLANNER_SCHEMA,
    )

    mode = result["mode"]
    angles = tuple(result["angles"])
    if mode not in ("simple", "multi_angle") or not angles:
        raise ValueError(f"Planner returned an invalid plan: {result!r}")

    plan = Plan(mode=mode, angles=angles, reason=result["reason"])
    return {
        "plan": plan,
        "trace_events": [f"Planner: {plan.mode} ({len(plan.angles)} angle(s)) — {plan.reason}"],
    }


# --- Researcher ----------------------------------------------------------------

_RESEARCHER_SCHEMA = {
    "description": (
        "Extract a concise finding for this research angle, grounded only in the "
        "supporting chunks actually provided below."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "relevant_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "IDs of the chunks below that actually support the finding. "
                    "Empty if none of them are relevant."
                ),
            },
            "finding": {
                "type": "string",
                "description": (
                    "A concise statement of what was found. If relevant_chunk_ids "
                    "is empty, say so plainly instead of guessing."
                ),
            },
        },
        "required": ["relevant_chunk_ids", "finding"],
    },
}


def _format_candidates(results: list[SearchResult]) -> str:
    return "\n".join(f"[{r.chunk_id}] {r.text}" for r in results)


async def researcher_node(angle: ResearchAngle) -> dict:
    """Non-agentic (D-05): given one angle, retrieve, select, and extract.

    `Send`-dispatched — receives its own payload, not the full graph state
    (Phase 4). Retrieval + selection + extraction happen as one LLM call
    (structured output over the candidates already fused by RRF), rather than
    a separate reranking pass — the model is asked to select the relevant
    subset and write the finding together, which avoids a second round trip
    for no real benefit at this corpus's scale.
    """
    candidates = await ahybrid_search(angle.question, top_k=settings.retrieval_top_k)
    candidate_ids = {c.chunk_id for c in candidates}

    result = await call_llm_with_tools(
        node="researcher",
        purpose=f"extract finding for angle {angle.angle_id}",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Research angle: {angle.question}\n\n"
                    f"Retrieved candidate chunks:\n{_format_candidates(candidates)}\n\n"
                    "Select only the chunks that genuinely support an answer to this "
                    "angle, and extract a concise finding grounded in them."
                ),
            }
        ],
        tool_name="extract_finding",
        tool_schema=_RESEARCHER_SCHEMA,
    )

    valid_ids = validate_citations(result["relevant_chunk_ids"], candidate_ids)

    finding = Finding(angle_id=angle.angle_id, text=result["finding"], chunk_ids=valid_ids)
    return {
        "findings": [finding],
        "trace_events": [
            f"Researcher[{angle.angle_id}]: {len(valid_ids)} supporting chunk(s) — {finding.text[:80]}"
        ],
    }


# --- Synthesizer -----------------------------------------------------------------

_SYNTHESIZER_SCHEMA = {
    "description": (
        "Compose the final answer from the assembled research findings, or abstain "
        "if the findings do not support a confident answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "abstained": {
                "type": "boolean",
                "description": "True if the findings do not support a confident answer.",
            },
            "answer": {
                "type": "string",
                "description": (
                    "The answer, citing chunk IDs inline like [chunk_id], if not "
                    "abstaining. If abstaining, a brief honest explanation of what's "
                    "missing — never a canned string."
                ),
            },
            "citation_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Chunk IDs actually used to support the answer, e.g. "
                    "'09_pricing_tiers::0'. These come ONLY from the 'Citable chunk "
                    "IDs' lists shown with each finding below — never an angle label "
                    "like 'a0'. Empty if abstained."
                ),
            },
        },
        "required": ["abstained", "answer", "citation_chunk_ids"],
    },
}


def _format_findings(findings: list[Finding]) -> str:
    """Deliberately does NOT bracket the angle ID. Citations are taught
    elsewhere as `[chunk_id]` — an earlier version wrote `[a0] finding text`,
    and the Synthesizer cited the angle label `a0` as if it were a chunk ID,
    because it was the nearest bracketed token to the text. Two different
    identifiers must never share the same visual syntax."""
    if not findings:
        return "(no findings)"
    lines = []
    for f in findings:
        cids = ", ".join(f.chunk_ids) if f.chunk_ids else "(none)"
        lines.append(f"Angle {f.angle_id}: {f.text}\n  Citable chunk IDs for this finding: {cids}")
    return "\n".join(lines)


async def synthesizer_node(state: ResearchState) -> dict:
    """Non-agentic (D-05): merges findings into a cited answer, or abstains
    (D-11). Citation validation here is deterministic code, not a second LLM
    judgement — consistent with D-05's "no control-flow decisions" even
    though the abstain flag does affect Phase 4's escalation edge."""
    findings = state["findings"]
    available_ids = {cid for f in findings for cid in f.chunk_ids}

    result = await call_llm_with_tools(
        node="synthesizer",
        purpose="compose answer or abstain",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    f"Research findings:\n{_format_findings(findings)}\n\n"
                    "Answer using only these findings. If they don't support a "
                    "confident answer, abstain honestly rather than guessing."
                ),
            }
        ],
        tool_name="compose_answer",
        tool_schema=_SYNTHESIZER_SCHEMA,
        temperature=settings.synthesis_temperature,
    )

    abstained = bool(result["abstained"])
    answer = result["answer"]
    claimed_ids = result["citation_chunk_ids"]
    citations = validate_citations(claimed_ids, available_ids)

    trace_events = [f"Synthesizer: abstained={abstained}, {len(citations)} citation(s)"]

    if not abstained and not citations:
        # The model claimed a supported answer but every citation it gave was
        # either fabricated or never actually retrieved. Code-level
        # enforcement overrides the model's own claim (D-11 / CLAUDE.md)
        # rather than trusting it — this is also what Phase 4's escalation
        # edge checks for on the cheap path (D-13).
        #
        # Logged permanently, not just for this debug session: when this
        # override fires, the raw claimed IDs vs. what was actually available
        # is exactly what's needed to tell a real model failure apart from an
        # ID-formatting mismatch in the prompt — without that, every future
        # occurrence would need a fresh trace.jsonl dig to diagnose.
        trace_events.append(
            f"Synthesizer: OVERRIDE to abstain — model claimed {claimed_ids!r}, "
            f"available was {sorted(available_ids)!r}"
        )
        abstained = True
        answer = (
            "I don't have well-supported evidence to answer this confidently — "
            "the draft answer's citations could not be validated against the "
            "retrieved context."
        )
        citations = ()

    synthesis = SynthesisResult(answer=answer, citations=citations, abstained=abstained)
    return {"synthesis": synthesis, "trace_events": trace_events}


# --- Critic ------------------------------------------------------------------

_CRITIC_SCHEMA = {
    "description": (
        "Check whether every claim in the draft answer is genuinely supported by "
        "the cited findings. Approve, or send specific, actionable feedback back."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "feedback": {
                "type": "string",
                "description": (
                    "If not approved: which specific claims are unsupported and what "
                    "evidence is missing. If approved: a brief one-line confirmation."
                ),
            },
        },
        "required": ["approved", "feedback"],
    },
}


async def critic_node(state: ResearchState) -> dict:
    """Agentic (D-05): revise or approve is a real judgement call, not a
    threshold. An abstained answer is auto-approved without an LLM call —
    there is no claim to fact-check, and spending the single allowed
    revision (D-06) re-running research that already came up short would be
    wasted, not corrective."""
    synthesis = state["synthesis"]

    if synthesis.abstained:
        verdict = CriticVerdict(approved=True, feedback="Abstained answer — nothing to fact-check.")
        return {
            "critic_verdict": verdict,
            "trace_events": ["Critic: auto-approved (abstained answer)"],
        }

    result = await call_llm_with_tools(
        node="critic",
        purpose="verify claims against findings",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    f"Research findings:\n{_format_findings(state['findings'])}\n\n"
                    f"Draft answer: {synthesis.answer}\n"
                    f"Cited chunks: {', '.join(synthesis.citations) or '(none)'}\n\n"
                    "Check every claim in the draft answer against the findings above. "
                    "Approve only if every claim is genuinely supported."
                ),
            }
        ],
        tool_name="critique_answer",
        tool_schema=_CRITIC_SCHEMA,
    )

    verdict = CriticVerdict(approved=bool(result["approved"]), feedback=result["feedback"])
    return {
        "critic_verdict": verdict,
        "trace_events": [f"Critic: {'approved' if verdict.approved else 'revise'} — {verdict.feedback}"],
    }
