"""Typed state shared across the graph, plus the data shapes nodes pass
around and the two bounds that keep every cycle finite (D-06).

Two things are deliberately declared here rather than in `app/core/config.py`:

- **`MAX_REVISIONS` / `MAX_ESCALATIONS`.** A design constraint is not a tuning
  knob (D-06) — keeping them as module constants next to the state they bound,
  rather than in a settings object, means raising them is a visible code
  change, not a config edit.
- **`merge_findings`, the D-12 sentinel reducer.** `findings` is written
  concurrently by `Send`-dispatched researchers (Phase 4) and also lives
  across a revision cycle. `operator.add` is correct for the first case and
  silently wrong for the second — see D-12 in PHASES.md for why the sentinel
  exists instead of a plain `[]` reset.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

# --- Bounds (D-06) ----------------------------------------------------------

MAX_REVISIONS = 1
MAX_ESCALATIONS = 1

# --- D-12: sentinel reducer for `findings` ----------------------------------

CLEAR: Literal["__clear__"] = "__clear__"


@dataclass(frozen=True)
class Finding:
    """One researcher's output for one angle: a synthesized finding plus the
    exact chunk IDs it's grounded in, so the Synthesizer can cite them and
    citation validation (D-11) can check they were genuinely retrieved."""

    angle_id: str
    text: str
    chunk_ids: tuple[str, ...]


def merge_findings(
    current: list[Finding], new: list[Finding] | Literal["__clear__"]
) -> list[Finding]:
    """Append on fan-in; reset to empty on the `CLEAR` sentinel.

    `operator.add` cannot express the reset half of this — there is no value
    that clears a list through `+`, since returning `[]` appends nothing. The
    Critic's revise edge returns `{"findings": CLEAR}` so a second research
    pass never inherits findings it already rejected.
    """
    if new == CLEAR:
        return []
    return current + new


# --- Payloads and results ---------------------------------------------------


@dataclass(frozen=True)
class ResearchAngle:
    """Dispatched to one researcher via `Send` (Phase 4).

    A `Send`-dispatched node receives this payload directly, not the full
    graph state — `async def researcher_node(payload: ResearchAngle)`, never
    `researcher_node(state)`. Fixed here so Phase 4's fan-out mechanism
    doesn't force a rewrite of the node signature later.
    """

    angle_id: str
    question: str


@dataclass(frozen=True)
class Plan:
    """The Planner's decision (Phase 3, agentic — D-05)."""

    mode: Literal["simple", "multi_angle"]
    angles: tuple[str, ...]  # one entry for "simple"; N for "multi_angle"
    reason: str  # kept, not discarded — this is what makes the routing decision auditable


@dataclass(frozen=True)
class SynthesisResult:
    """The Synthesizer's output: an answer with validated citations, or a
    genuine abstain (D-11) — never both, never a canned string either way."""

    answer: str
    citations: tuple[str, ...]
    abstained: bool


@dataclass(frozen=True)
class CriticVerdict:
    """The Critic's decision (Phase 3, agentic — D-05): approve, or send
    specific, actionable feedback back to the researchers."""

    approved: bool
    feedback: str


# --- Graph state -------------------------------------------------------------


class ResearchState(TypedDict):
    question: str

    plan: Plan | None
    # Concurrent writes from `Send`-dispatched researchers (fan-in) AND a
    # value that must survive across a revision cycle (fan-out again) — the
    # exact two-requirement combination D-12 exists to explain.
    findings: Annotated[list[Finding], merge_findings]

    synthesis: SynthesisResult | None
    critic_verdict: CriticVerdict | None

    # Plain counters, not reducers: only one node (Critic / the escalation
    # edge) ever writes either key in a given step, so there is no concurrent
    # write to fuse — LangGraph only requires a reducer where fan-in happens.
    revision_count: int
    escalation_count: int

    # Human-readable one-line audit trail, appended by every node — this is
    # what a HITL interrupt (Phase 4) shows a human, and it always grows
    # (never cleared, unlike `findings`): a revision's history is worth
    # keeping even after the revision itself is superseded.
    trace_events: Annotated[list[str], operator.add]
