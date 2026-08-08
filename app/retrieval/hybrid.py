"""Hybrid retrieval: BM25 (keyword) + vector (semantic), fused with
Reciprocal Rank Fusion.

BM25 cannot persist itself (`rank_bm25` is in-memory only), so it is rebuilt
from `data/chunks.json` — written by `index.py` — every time this module
loads, rather than stored anywhere. Vector search reads the persistent Chroma
collection `index.py` already built.

`search()` is sync (BM25 and Chroma both are, under the hood); `ahybrid_search()`
wraps it in a thread so multiple researchers (Phase 4) can query concurrently
without blocking the event loop or each other's turn — same reasoning as the
async trace writes in `tracing.py`.

Run:
    python -m app.retrieval.hybrid
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.retrieval.index import COLLECTION_NAME


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    source: str
    score: float


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int) -> dict[str, float]:
    """Fuse any number of ranked ID lists via Reciprocal Rank Fusion.

    score(id) = sum, over every list containing it, of 1 / (k + rank_in_that_list).
    Rank-based rather than score-based because BM25 scores and cosine
    similarities live on incomparable scales and can't be combined directly.

    Extracted as a standalone pure function (no embeddings, no index, no I/O)
    specifically so it is testable on synthetic rank inputs — see
    tests/test_rrf.py — independent of any real corpus or model.
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, item_id in enumerate(ranked_ids):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridIndex:
    """Loads both halves of hybrid search once; reused across queries."""

    def __init__(self) -> None:
        if not settings.chunks_file.exists():
            raise FileNotFoundError(
                f"{settings.chunks_file} not found — run `python -m app.retrieval.index` first."
            )
        records = json.loads(settings.chunks_file.read_text(encoding="utf-8"))
        self._ids = [r["id"] for r in records]
        self._texts = {r["id"]: r["text"] for r in records}
        self._sources = {r["id"]: r["source"] for r in records}

        self._bm25 = BM25Okapi([_tokenize(r["text"]) for r in records])

        self._embedder = SentenceTransformer(settings.embedding_model)
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection = client.get_collection(COLLECTION_NAME)

    def bm25_ranked_ids(self, query: str, top_k: int) -> list[str]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._ids[i] for i in ranked[:top_k]]

    def vector_ranked_ids(self, query: str, top_k: int) -> list[str]:
        query_embedding = self._embedder.encode([query]).tolist()
        result = self._collection.query(query_embeddings=query_embedding, n_results=top_k)
        return result["ids"][0]

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """BM25 + vector, fused with Reciprocal Rank Fusion.

        score = sum(1 / (rrf_k + rank)) across whichever ranked list(s) a
        chunk appears in — rank-based, not score-based, because BM25 scores
        and cosine similarities live on incomparable scales and can't be
        combined directly.
        """
        top_k = top_k or settings.retrieval_top_k
        # Each method searches a wider candidate pool than top_k so RRF has
        # more than top_k items per list to actually fuse across.
        candidate_k = max(top_k * 3, top_k)

        bm25_ids = self.bm25_ranked_ids(query, candidate_k)
        vector_ids = self.vector_ranked_ids(query, candidate_k)

        rrf_scores = reciprocal_rank_fusion([bm25_ids, vector_ids], settings.rrf_k)

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        return [
            SearchResult(chunk_id=cid, text=self._texts[cid], source=self._sources[cid], score=rrf_scores[cid])
            for cid in ranked_ids
        ]


_index: HybridIndex | None = None


def _get_index() -> HybridIndex:
    global _index
    if _index is None:
        _index = HybridIndex()
    return _index


async def ahybrid_search(query: str, top_k: int | None = None) -> list[SearchResult]:
    """Async wrapper so concurrent researchers (Phase 4) can search without
    blocking the event loop or each other — BM25 and Chroma are both
    synchronous, CPU/disk-bound work under the hood."""
    return await asyncio.to_thread(_get_index().search, query, top_k)


def _self_test() -> None:
    """BM25 should win on an exact, distinctive token; vector should win on a
    paraphrase sharing none of that vocabulary. Both real corpus queries, not
    invented examples — `14_onboarding_runbook.md` uses "kickoff call"
    verbatim and nowhere else in the corpus."""
    index = _get_index()

    print("Exact-token query: 'kickoff call'")
    print("  BM25 top-1:  ", index.bm25_ranked_ids("kickoff call", top_k=1))
    print("  Vector top-1:", index.vector_ranked_ids("kickoff call", top_k=1))

    # First attempt at this query used the word "launched", which pulled the
    # embedding toward deployment/product-launch semantics (a real, correctly
    # retrieved chunk about a genuinely different topic) rather than customer
    # onboarding — a lesson in query ambiguity, not a retrieval bug, but not a
    # clean demonstration either. Rephrased to stay in the onboarding concept
    # space without reusing "onboarding", "runbook", "kickoff", or "customer".
    paraphrase = "what a company goes through in its first few weeks after signing up"
    print(f"\nParaphrase query: '{paraphrase}' (shares no distinctive words)")
    print("  BM25 top-1:  ", index.bm25_ranked_ids(paraphrase, top_k=1))
    print("  Vector top-1:", index.vector_ranked_ids(paraphrase, top_k=1))

    print("\nFused (RRF) result for the paraphrase query:")
    for r in index.search(paraphrase, top_k=3):
        print(f"  {r.chunk_id}  score={r.score:.4f}")


if __name__ == "__main__":
    _self_test()
