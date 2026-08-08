"""Pure tests for Reciprocal Rank Fusion — no embeddings, no Chroma, no corpus.

`reciprocal_rank_fusion()` was extracted from `HybridIndex.search()` in
`app/retrieval/hybrid.py` specifically so this fusion math is testable on
synthetic rank inputs alone, independent of any real index (Phase 2 spec).
"""

from __future__ import annotations

import pytest

from app.retrieval.hybrid import reciprocal_rank_fusion


def test_known_rank_inputs_produce_known_fused_order():
    """Two short overlapping lists -> hand-computed expected scores."""
    list_a = ["x", "y", "z"]
    list_b = ["y", "x", "w"]

    scores = reciprocal_rank_fusion([list_a, list_b], k=1)

    # x: rank 0 in a (1/1) + rank 1 in b (1/2) = 1.5
    # y: rank 1 in a (1/2) + rank 0 in b (1/1) = 1.5
    # z: rank 2 in a only = 1/3
    # w: rank 2 in b only = 1/3
    assert scores["x"] == pytest.approx(1.5)
    assert scores["y"] == pytest.approx(1.5)
    assert scores["z"] == pytest.approx(1 / 3)
    assert scores["w"] == pytest.approx(1 / 3)


def test_item_present_in_only_one_list_is_still_scored():
    scores = reciprocal_rank_fusion([["only"], []], k=60)

    assert set(scores) == {"only"}
    assert scores["only"] == pytest.approx(1 / 60)


def test_empty_lists_produce_no_scores():
    assert reciprocal_rank_fusion([], k=60) == {}
    assert reciprocal_rank_fusion([[], []], k=60) == {}


def test_moderate_agreement_across_lists_can_beat_a_single_list_top_rank():
    """Regression test for the real finding recorded in PHASES.md's
    failure-mode table (#1, found in Phase 2): a chunk ranked #1 in exactly
    one list can be outranked by a chunk that appears at only moderate rank
    in BOTH lists. Confirmed here as the function's real, deliberate
    behavior — RRF rewards cross-method agreement over single-method
    conviction — not a bug to fix.
    """
    list_a = ["c0", "c1", "c2", "c3", "c4", "consensus"]  # consensus at rank 5
    list_b = ["single_winner", "d1", "d2", "d3", "d4", "d5", "consensus"]  # rank 0, rank 6

    scores = reciprocal_rank_fusion([list_a, list_b], k=60)

    assert scores["single_winner"] == pytest.approx(1 / 60)
    assert scores["consensus"] == pytest.approx(1 / 65 + 1 / 66)
    assert scores["consensus"] > scores["single_winner"]


def test_generalizes_to_more_than_two_lists():
    """The real system only ever fuses BM25 + vector (two lists), but the
    function's signature promises any number of ranked lists — confirm it
    actually does, rather than assuming it from the two-list case."""
    lists = [["a", "b"], ["b", "a"], ["a"]]

    scores = reciprocal_rank_fusion(lists, k=1)

    # a: rank 0 (1/1) + rank 1 (1/2) + rank 0 (1/1) = 2.5
    # b: rank 1 (1/2) + rank 0 (1/1) = 1.5
    assert scores["a"] == pytest.approx(2.5)
    assert scores["b"] == pytest.approx(1.5)
