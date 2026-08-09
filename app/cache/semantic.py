"""Semantic cache: embeds the incoming question, compares against cached
question embeddings already stored in Redis, and returns a stored answer
when similarity clears a threshold (D-09).

Plain Redis, not a vector-search module (RediSearch/RedisVL) — nothing
confirms the hosted instance this project points at has one installed, and
at this project's scale (a handful to a few hundred cached questions) a
brute-force cosine comparison over everything in the cache, done client-side
in pure Python, is simple, adds no new dependency, and is fast enough. A
deliberate choice, not an oversight: a system with millions of cached
entries would need a real vector index; this one doesn't have that problem.

**D-09's non-negotiable, and the reason this whole module exists**: a cache
hit is ALWAYS traced, with the matched question and similarity score visible
— there is no code path that returns a cached answer silently. A cached
answer presented as fresh work would make the entire trace/SSE/graph-UI
stack lie about what actually happened.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from dataclasses import dataclass

import redis.asyncio as redis
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.tracing import TraceEvent, tracer

_INDEX_KEY = "semantic_cache:index"
_ENTRY_PREFIX = "semantic_cache:entry:"

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class CacheEntry:
    question: str
    answer: str
    citations: tuple[str, ...]
    abstained: bool
    revision_count: int
    escalation_count: int
    similarity: float


class SemanticCache:
    """One Redis connection, reused across the app's lifetime (built once in
    `main.py`'s lifespan, same pattern as the graph's checkpointer)."""

    def __init__(self) -> None:
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)

    async def _embed(self, text: str) -> list[float]:
        # CPU-bound work (sentence-transformers), off-thread so it never
        # blocks the event loop — same reasoning as hybrid.py's ahybrid_search.
        return await asyncio.to_thread(lambda: _get_embedder().encode(text).tolist())

    async def lookup(self, question: str, *, bypass: bool = False) -> CacheEntry | None:
        """Returns the best cached match if similarity clears the configured
        threshold, else None.

        `bypass=True` skips the cache entirely — required by
        `eval/run_ragas.py` (Phase 7): running the cache in front of an eval
        harness that re-runs near-identical questions would return cached
        answers as if they were fresh runs and quietly corrupt the scores.
        """
        if bypass:
            return None

        query_embedding = await self._embed(question)
        keys = list(await self._redis.smembers(_INDEX_KEY))
        if not keys:
            await tracer.emit(
                TraceEvent(kind="cache", node="semantic_cache", purpose="lookup", decision="miss (cache empty)")
            )
            return None

        raw_entries = await self._redis.mget(keys)
        best: CacheEntry | None = None
        best_similarity = 0.0
        for raw in raw_entries:
            if raw is None:
                continue
            data = json.loads(raw)
            similarity = _cosine_similarity(query_embedding, data["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best = CacheEntry(
                    question=data["question"],
                    answer=data["answer"],
                    citations=tuple(data["citations"]),
                    abstained=data["abstained"],
                    revision_count=data["revision_count"],
                    escalation_count=data["escalation_count"],
                    similarity=similarity,
                )

        if best is None or best_similarity < settings.cache_similarity_threshold:
            await tracer.emit(
                TraceEvent(
                    kind="cache",
                    node="semantic_cache",
                    purpose="lookup",
                    decision=f"miss (best similarity {best_similarity:.3f} < {settings.cache_similarity_threshold})",
                )
            )
            return None

        await tracer.emit(
            TraceEvent(
                kind="cache",
                node="semantic_cache",
                purpose="lookup",
                decision=f"HIT (similarity {best.similarity:.3f}) — matched cached question: {best.question!r}",
            )
        )
        return best

    async def store(
        self,
        question: str,
        *,
        answer: str,
        citations: tuple[str, ...],
        abstained: bool,
        revision_count: int,
        escalation_count: int,
        bypass: bool = False,
    ) -> None:
        """Stores a freshly computed answer for future similarity matches.
        `bypass=True` (eval) also skips writing — an eval run's synthetic
        test questions have no business polluting the cache a real user's
        near-duplicate question might later match against.
        """
        if bypass:
            return
        embedding = await self._embed(question)
        key = f"{_ENTRY_PREFIX}{uuid.uuid4()}"
        payload = json.dumps(
            {
                "question": question,
                "embedding": embedding,
                "answer": answer,
                "citations": list(citations),
                "abstained": abstained,
                "revision_count": revision_count,
                "escalation_count": escalation_count,
            }
        )
        await self._redis.set(key, payload)
        await self._redis.sadd(_INDEX_KEY, key)

    async def close(self) -> None:
        await self._redis.aclose()


_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    """Lazy singleton — same reasoning as `app.core.llm.get_pool()`: importing
    this module should never require a live Redis connection, only calling
    it does."""
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
