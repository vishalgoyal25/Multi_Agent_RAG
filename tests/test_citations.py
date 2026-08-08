"""Pure tests for citation validation — no LLM, no live index.

`validate_citations()` is the code-level enforcement behind D-11 / CLAUDE.md's
non-negotiable rule: never trust what a model claims it cited. Extracted from
`app/graph/nodes.py`, used by both `researcher_node` and `synthesizer_node`.
"""

from __future__ import annotations

from app.graph.nodes import validate_citations


def test_valid_ids_pass_through():
    available = {"09_pricing_tiers::0", "09_pricing_tiers::1", "10_contract_trial_terms::0"}
    claimed = ["09_pricing_tiers::0", "10_contract_trial_terms::0"]

    result = validate_citations(claimed, available)

    assert result == ("09_pricing_tiers::0", "10_contract_trial_terms::0")


def test_fabricated_id_is_rejected():
    """An ID the model invented outright — never existed in any chunk set."""
    available = {"09_pricing_tiers::0"}
    claimed = ["09_pricing_tiers::0", "99_does_not_exist::7"]

    result = validate_citations(claimed, available)

    assert result == ("09_pricing_tiers::0",)
    assert "99_does_not_exist::7" not in result


def test_real_id_not_in_this_run_context_is_also_rejected():
    """A real chunk ID from elsewhere in the corpus, but not among the chunks
    actually retrieved/placed in context for this run — rejected the same way
    as a fabricated one. The model citing it is indistinguishable from
    hallucination as far as this run's evidence is concerned.
    """
    available = {"09_pricing_tiers::0"}  # only this chunk was in context this run
    claimed = ["14_onboarding_runbook::0"]  # a real ID, just from a different run's context

    result = validate_citations(claimed, available)

    assert result == ()


def test_empty_claims_produce_empty_result():
    assert validate_citations([], {"09_pricing_tiers::0"}) == ()


def test_order_is_preserved():
    """Order matters for how citations read in the final answer — validation
    should filter, not reorder."""
    available = {"a", "b", "c"}
    claimed = ["c", "a", "b"]

    assert validate_citations(claimed, available) == ("c", "a", "b")
