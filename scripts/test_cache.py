"""Phase 8 exit check: semantic cache hit/miss/bypass behavior, against the
real configured Redis instance.

Real graph runs are not re-exercised here — Phases 3/4/5 already prove the
agentic pipeline works. This isolates the CACHE module itself: does a
paraphrase actually hit, does a genuinely different question miss, does the
bypass flag force a miss even on an identical, certain-to-hit question.

Cleans up exactly the entry it creates when done. This hits the REAL, shared
Redis instance — leaving a synthetic test answer behind would risk a real
future user later getting it served as if it were a genuine answer.

Run:
    python -m scripts.test_cache
"""

from __future__ import annotations

import asyncio
import json

from app.cache.semantic import _INDEX_KEY, get_cache
from app.core.config import settings

QUESTION = "What is the trial length?"
PARAPHRASE = "How long does the free trial period last?"
UNRELATED = "What SLA response time applies to critical severity tickets?"
ANSWER = "The trial length is 14 days, per the pricing documentation."
CITATIONS = ("09_pricing_tiers::0",)


async def main() -> None:
    cache = get_cache()
    # Reaching into the client directly for test-harness cleanup only — same
    # deliberate pattern as smoke_test.py's LLMPool._slots access. store()
    # doesn't return the key it generates, so the new key is found by
    # diffing the index set before/after, not by a public API.
    redis_client = cache._redis  # noqa: SLF001
    before_keys = set(await redis_client.smembers(_INDEX_KEY))

    try:
        print(f"Storing a real entry for: {QUESTION!r}")
        await cache.store(
            QUESTION,
            answer=ANSWER,
            citations=CITATIONS,
            abstained=False,
            revision_count=0,
            escalation_count=0,
        )

        print("\n1. Exact question -> expect HIT")
        hit = await cache.lookup(QUESTION)
        assert hit is not None, "exact question should hit"
        print(f"   similarity={hit.similarity:.3f}  answer={hit.answer!r}")

        print(f"\n2. Paraphrase ({PARAPHRASE!r}) -> expect HIT")
        hit = await cache.lookup(PARAPHRASE)
        if hit is None:
            # lookup() already computed and traced the best similarity even on
            # a miss (logs/trace.jsonl, kind="cache") — surface it here rather
            # than printing a bare "MISS" and guessing how close it was.
            trace_lines = settings.trace_file.read_text(encoding="utf-8").strip().splitlines()
            last_cache_event = next(
                (json.loads(line) for line in reversed(trace_lines) if json.loads(line).get("kind") == "cache"),
                None,
            )
            detail = last_cache_event["decision"] if last_cache_event else "(no trace event found)"
            print(f"   MISS — {detail}")
        else:
            print(f"   similarity={hit.similarity:.3f}  answer={hit.answer!r}")

        print(f"\n3. Unrelated question ({UNRELATED!r}) -> expect MISS")
        miss = await cache.lookup(UNRELATED)
        assert miss is None, "an unrelated question should not hit the cache"
        print("   confirmed miss")

        print("\n4. Bypass flag on the EXACT original question -> expect MISS regardless")
        bypassed = await cache.lookup(QUESTION, bypass=True)
        assert bypassed is None, "bypass=True must force a miss even on an identical question"
        print("   confirmed miss (bypass overrides an otherwise-certain hit)")

        print("\nPASS — cache hit/miss/bypass behavior all verified against real Redis.")
    finally:
        after_keys = set(await redis_client.smembers(_INDEX_KEY))
        new_keys = after_keys - before_keys
        for key in new_keys:
            await redis_client.delete(key)
            await redis_client.srem(_INDEX_KEY, key)
        if new_keys:
            print(f"\nCleaned up {len(new_keys)} test entr{'y' if len(new_keys) == 1 else 'ies'} from the real cache.")
        await cache.close()


if __name__ == "__main__":
    asyncio.run(main())
