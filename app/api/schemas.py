"""Pydantic request/response contracts for the API layer.

Deliberately separate from `app/graph/state.py`'s dataclasses — the API's
shape is a public contract; the graph's internal state is free to change
without that becoming a breaking API change. Each response schema below
carries a small `from_*` conversion method so `main.py` stays a thin layer
that calls the graph and maps its output, not a place where request/response
shaping logic accumulates.

**`POST /ask` never returns a final answer by itself.** Every path through
the graph funnels through the `human_approval` interrupt (D-14) before `END`
— there is no path that skips it. So `/ask`'s response is always
`PendingApproval`; only `POST /resume` can return `FinalAnswer`. The schemas
below reflect that honestly rather than modeling `/ask` as a one-shot call
that happens to also need a follow-up sometimes.

Trace events streamed over SSE are NOT modeled here as a Pydantic schema —
they're `app.core.tracing.TraceEvent.to_dict()`, reused as-is. That type was
built in Phase 1 specifically for this (its own docstring: "unused until
Phase 5"); wrapping it in a second, parallel schema would just be duplication
for its own sake.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.graph.state import CriticVerdict, Finding, Plan, SynthesisResult


class AskRequest(BaseModel):
    question: str


class PlanSchema(BaseModel):
    mode: Literal["simple", "multi_angle"]
    angles: tuple[str, ...]
    reason: str

    @classmethod
    def from_plan(cls, plan: Plan) -> "PlanSchema":
        return cls(mode=plan.mode, angles=plan.angles, reason=plan.reason)


class FindingSchema(BaseModel):
    angle_id: str
    text: str
    chunk_ids: tuple[str, ...]

    @classmethod
    def from_finding(cls, finding: Finding) -> "FindingSchema":
        return cls(angle_id=finding.angle_id, text=finding.text, chunk_ids=finding.chunk_ids)


class SynthesisSchema(BaseModel):
    answer: str
    citations: tuple[str, ...]
    abstained: bool

    @classmethod
    def from_synthesis(cls, synthesis: SynthesisResult) -> "SynthesisSchema":
        return cls(answer=synthesis.answer, citations=synthesis.citations, abstained=synthesis.abstained)


class CriticVerdictSchema(BaseModel):
    approved: bool
    feedback: str

    @classmethod
    def from_verdict(cls, verdict: CriticVerdict) -> "CriticVerdictSchema":
        return cls(approved=verdict.approved, feedback=verdict.feedback)


class PendingApproval(BaseModel):
    """What `POST /ask` (and a not-yet-approved `POST /resume`, if ever
    reached again) returns: the graph has paused at the HITL checkpoint and
    is waiting for a decision. This carries exactly what a human needs to
    decide approve/reject — the same evidence the Synthesizer had, not a
    black box (same payload shape `human_approval_node` already builds)."""

    status: Literal["awaiting_approval"] = "awaiting_approval"
    thread_id: str
    question: str
    plan: PlanSchema
    findings: list[FindingSchema]
    synthesis: SynthesisSchema
    critic_verdict: CriticVerdictSchema | None = None


class ResumeRequest(BaseModel):
    """The resume contract fixed in Phase 4 (D-14), now exposed over HTTP:
    the thread being resumed, and the human's decision."""

    thread_id: str
    approved: bool


class FinalAnswer(BaseModel):
    status: Literal["completed"] = "completed"
    thread_id: str
    answer: str
    citations: tuple[str, ...]
    revision_count: int
    escalation_count: int


class CacheHit(BaseModel):
    """What `/ask` or `/ask/stream` return on a semantic cache hit (Phase 8,
    D-09) — never silently merged into `FinalAnswer`'s shape. A visitor (or a
    developer reading a response) can always tell a cached answer from a
    freshly computed one, which is the entire point of D-09: a cache hit
    presented as fresh work would make the trace lie about what happened.
    No `thread_id` — a cache hit never touches the graph, so there is no run
    to resume or approve; it was already approved once, the time it was
    stored (only `/resume` writes to the cache, and only on approval).
    """

    status: Literal["cache_hit"] = "cache_hit"
    answer: str
    citations: tuple[str, ...]
    similarity: float


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
