"""Phase 1 — D-15 spike: does RAGAS 0.4 evaluate a single question with a
Groq-backed judge and local embeddings?

Both outcomes are acceptable and get recorded either way:

  PASS -> RAGAS is the Phase 7 evaluator.
  FAIL -> the hand-built evaluator (LLM-judge + deterministic citation check,
          ported from the previous project) is the Phase 7 evaluator instead.

This took three attempts, each one a real, sourced fix rather than a guess —
worth recording so D-15 documents what it actually cost:

  1. ragas==0.4.3's OLD API (`ragas.metrics.Faithfulness`, `LangchainLLMWrapper`)
     eagerly imports `ChatVertexAI` from a `langchain_community` path LangChain
     removed. Confirmed upstream, unfixed as of this writing:
     https://github.com/vibrantlabsai/ragas/issues/2741 and /2745.
     Worked around by `app/core/ragas_compat.py` — a `sys.modules` shim.
  2. That OLD API's LangChain-wrapped LLM call requested `n>1` completions per
     call; Groq caps `n` at 1 and rejected it with a 400 — `evaluate()`
     swallowed the per-job exceptions and returned NaN instead of raising.
     Lesson kept in the PASS check below: **absence of a crash is not proof of
     a real result.**
  3. The NEW factory API's class/method names were guessed wrong twice
     (`ResponseRelevancy` doesn't exist; `single_turn_ascore(sample)` doesn't
     exist on these classes) before reading the actual installed source at
     `.venv/Lib/site-packages/ragas/metrics/collections/`. The real shapes,
     confirmed from that source:
       Faithfulness(llm=llm).ascore(user_input, response, retrieved_contexts)
       AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(user_input, response)
     Both return a `MetricResult` (`.value` is the number, `.reason` the
     judge's explanation), not a bare float.

A fourth fix, found after the first PASS: the judge client was a raw
`AsyncOpenAI(api_key=settings.groq_api_keys[0], ...)` — a single hardcoded
key, completely outside `app/core/llm.py`'s sticky pool. If that one key (or
Groq entirely) failed, the eval judge had zero of D-01's failover protection,
unlike every other LLM call in this project. Fixed by routing the judge
through `get_pool().request()` — the same sticky Groq -> Cerebras chain, the
same trace events — instead of a second, unprotected client.

Genuinely async: `ascore()` is a real coroutine wrapping an AsyncOpenAI-backed
judge, not a wrapped blocking batch call.

Run:
    python -m scripts.ragas_spike
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Callable

from app.core.config import settings

import app.core.ragas_compat  # noqa: F401 - must run before any `import ragas`,
# see that module's docstring: shims a confirmed upstream ragas==0.4.3 bug
# (a dead import of ChatVertexAI from a path LangChain removed). This project
# never uses Vertex AI; the shim exists only so ragas's own broken import
# resolves instead of crashing every ragas user regardless of provider.
from app.core.llm import get_pool

SAMPLE = {
    "user_input": "What is the trial length?",
    "response": "The trial length is 14 days, per the pricing documentation.",
    "retrieved_contexts": [
        "Northbay Commerce AI offers a 14-day free trial on all pricing tiers.",
    ],
}


def _value_or_none(result: object) -> float | None:
    """Pull the numeric score out of a MetricResult, treating NaN as absent."""
    value = getattr(result, "value", result)
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


async def score_with_pool_failover(
    metric_builder: Callable[[Any], Any],
    ascore_kwargs: dict[str, Any],
    *,
    node: str,
    purpose: str,
):
    """Run one RAGAS metric's `ascore()` through the project's own sticky pool.

    Reuses `LLMPool.request()` exactly as `app/core/llm.py`'s own `call_llm()`
    does: `metric_builder` rebuilds the metric (cheap — no network call at
    construction) bound to whichever slot's client is being tried, so a
    judge-call failure advances to the next key, or the next provider, with
    the same trace events (`llm_failover`, `key_index`) as the main agentic
    loop. This is D-01's failover, not a second implementation of it.
    """

    async def make_request(slot):
        from ragas.llms import llm_factory

        llm = llm_factory(slot.model, client=slot.client)
        metric = metric_builder(llm)
        return await metric.ascore(**ascore_kwargs)

    result, slot, attempt = await get_pool().request(make_request, node=node, purpose=purpose)
    return result, slot, attempt


async def main() -> None:
    if not settings.groq_api_keys:
        raise SystemExit(
            "This spike specifically checks a Groq judge — set GROQ_API_KEY in .env first."
        )

    start = time.monotonic()
    print("D-15 spike started (ragas's modern factory API, judge routed through the sticky pool).\n")

    try:
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness

        # Provider-independent — built once, not per slot.
        embeddings = HuggingFaceEmbeddings(model=settings.embedding_model)

        faithfulness_result, f_slot, f_attempt = await score_with_pool_failover(
            lambda llm: Faithfulness(llm=llm),
            {
                "user_input": SAMPLE["user_input"],
                "response": SAMPLE["response"],
                "retrieved_contexts": SAMPLE["retrieved_contexts"],
            },
            node="ragas_spike",
            purpose="faithfulness",
        )
        relevancy_result, r_slot, r_attempt = await score_with_pool_failover(
            lambda llm: AnswerRelevancy(llm=llm, embeddings=embeddings),
            {"user_input": SAMPLE["user_input"], "response": SAMPLE["response"]},
            node="ragas_spike",
            purpose="answer_relevancy",
        )
        elapsed = time.monotonic() - start

        print(
            f"faithfulness     = {faithfulness_result.value}  "
            f"(served by {f_slot.key_index}, attempt {f_attempt}; reason: {faithfulness_result.reason})"
        )
        print(
            f"answer_relevancy = {relevancy_result.value}  "
            f"(served by {r_slot.key_index}, attempt {r_attempt}; reason: {relevancy_result.reason})"
        )

        scores = {
            "faithfulness": _value_or_none(faithfulness_result),
            "answer_relevancy": _value_or_none(relevancy_result),
        }
        if any(v is None for v in scores.values()):
            print(f"\nFAIL after {elapsed:.0f}s: ran without raising, but produced no real "
                  f"score (NaN/None) — that is not a pass. See scores above.")
            print("\nD-15 outcome: fall back to the hand-built evaluator for Phase 7.")
            raise SystemExit(1)

        print(f"\nPASS — RAGAS 0.4.3 (modern factory API) scored a real sample via a "
              f"pool-backed Groq/Cerebras judge in {elapsed:.0f}s.")
        print("D-15 outcome: RAGAS is the Phase 7 evaluator. Record this in PHASES.md.")

    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - the spike's entire job is to
        # surface exactly what broke, not to recover from it.
        elapsed = time.monotonic() - start
        print(f"\nFAIL after {elapsed:.0f}s: {type(exc).__name__}: {exc}")
        print("\nD-15 outcome: fall back to the hand-built evaluator (LLM-judge +")
        print("deterministic citation check) for Phase 7. Record this — and the")
        print("exact exception above — in PHASES.md D-15.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())
