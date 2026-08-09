"""Phase 7: offline batch evaluation of real graph runs against RAGAS metrics.

Never wired into the live app (see `eval/__init__.py`) — this is a script you
run by hand, reading its own output file, not an endpoint.

Design decisions worth being explicit about, not silent:

- **Auto-approves the human-in-the-loop checkpoint for every question.** Eval
  measures the AGENT's output quality, not human review — the interrupt is
  resumed immediately with `approved=True` rather than left pending.
- **Faithfulness, AnswerRelevancy, and ContextPrecision are skipped for
  abstained answers** — same reasoning as `critic_node`'s own auto-approve
  shortcut (nodes.py): there is nothing to fact-check in an honest "I don't
  know." Scoring an abstain as if it were a claim would measure the wrong
  thing. By construction (`synthesizer_node`'s citation-validation override),
  `abstained=False` guarantees real citations exist, so this is safe.
- **ContextRecall runs only when `eval_set.json` provides a `reference`.**
  Confirmed from `ragas`'s installed source (same discipline as D-15) that
  `ContextRecall.ascore()` requires one; fabricating a reference for the
  deliberately ambiguous "conflict" and "uncovered" questions would corrupt
  the metric rather than measure anything real.
- **Each question's judge calls go through the same sticky pool as the main
  agentic loop** (`score_with_pool_failover`, from `scripts/ragas_spike.py`,
  reused rather than duplicated) — a rate-limited judge call gets the same
  D-01 failover as any other LLM call in this project.

Run:
    python -m eval.run_ragas
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from langgraph.types import Command

import app.core.ragas_compat  # noqa: F401 - must precede any `import ragas`
from app.core.config import settings
from app.graph.builder import build_graph, run_to_interrupt_or_end
from app.graph.state import initial_state
from scripts.ragas_spike import score_with_pool_failover

EVAL_SET_PATH = settings.root / "eval" / "eval_set.json"
RESULTS_PATH = settings.root / "eval" / "results.json"
RUN_ID = uuid.uuid4().hex[:8]

_chunk_texts: dict[str, str] | None = None


def _load_questions() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))["questions"]


def _chunk_text_lookup() -> dict[str, str]:
    """Real chunk text by ID, from the same file BM25 rebuilds from (Phase 2)
    — not re-fetched from the live index, just the durable record on disk."""
    global _chunk_texts
    if _chunk_texts is None:
        records = json.loads(settings.chunks_file.read_text(encoding="utf-8"))
        _chunk_texts = {r["id"]: r["text"] for r in records}
    return _chunk_texts


async def _run_one_question(graph, entry: dict) -> dict:
    thread_id = f"eval-{RUN_ID}-{entry['id']}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    interrupt_payload, _ = await run_to_interrupt_or_end(graph, initial_state(entry["question"]), config)
    if interrupt_payload is None:
        raise RuntimeError(f"{entry['id']}: graph finished without reaching the human-approval interrupt")

    _, final_state = await run_to_interrupt_or_end(graph, Command(resume={"approved": True}), config)
    if final_state is None:
        raise RuntimeError(f"{entry['id']}: resume did not produce a final state")

    synthesis = final_state["synthesis"]
    trace_events = final_state["trace_events"]
    chunk_texts = _chunk_text_lookup()

    row: dict = {
        "id": entry["id"],
        "category": entry["category"],
        "question": entry["question"],
        "abstained": synthesis.abstained,
        "answer": synthesis.answer,
        "citations": list(synthesis.citations),
        "citation_check_passed": all(cid in chunk_texts for cid in synthesis.citations),
        "ever_took_simple_path": any(e.startswith("Planner: simple") for e in trace_events),
        "escalated": final_state["escalation_count"] > 0,
        "revised": final_state["revision_count"] > 0,
        "scores": {},
    }

    if not synthesis.abstained:
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
            ContextRecall,
            Faithfulness,
        )

        retrieved_contexts = [chunk_texts[cid] for cid in synthesis.citations if cid in chunk_texts]
        embeddings = HuggingFaceEmbeddings(model=settings.embedding_model)

        faithfulness, _, _ = await score_with_pool_failover(
            lambda llm: Faithfulness(llm=llm),
            {"user_input": entry["question"], "response": synthesis.answer, "retrieved_contexts": retrieved_contexts},
            node="eval", purpose=f"faithfulness[{entry['id']}]",
        )
        relevancy, _, _ = await score_with_pool_failover(
            lambda llm: AnswerRelevancy(llm=llm, embeddings=embeddings),
            {"user_input": entry["question"], "response": synthesis.answer},
            node="eval", purpose=f"answer_relevancy[{entry['id']}]",
        )
        precision, _, _ = await score_with_pool_failover(
            lambda llm: ContextPrecisionWithoutReference(llm=llm),
            {"user_input": entry["question"], "response": synthesis.answer, "retrieved_contexts": retrieved_contexts},
            node="eval", purpose=f"context_precision[{entry['id']}]",
        )
        row["scores"]["faithfulness"] = faithfulness.value
        row["scores"]["answer_relevancy"] = relevancy.value
        row["scores"]["context_precision"] = precision.value

        if entry.get("reference"):
            recall, _, _ = await score_with_pool_failover(
                lambda llm: ContextRecall(llm=llm),
                {"user_input": entry["question"], "retrieved_contexts": retrieved_contexts, "reference": entry["reference"]},
                node="eval", purpose=f"context_recall[{entry['id']}]",
            )
            row["scores"]["context_recall"] = recall.value

    return row


def _print_summary(rows: list[dict]) -> None:
    print(f"\n{'=' * 72}\nSummary\n{'=' * 72}")
    for row in rows:
        if "error" in row:
            print(f"[{row['id']}] ERROR: {row['error']}")
            continue
        tag = "ABSTAINED" if row["abstained"] else "answered"
        scores = ", ".join(f"{k}={v:.2f}" for k, v in row["scores"].items()) or "(no RAGAS scores — abstained)"
        print(f"[{row['id']}] {row['category']} — {tag} — {scores}")

    scored_rows = [r for r in rows if "error" not in r and not r["abstained"]]
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        values = [r["scores"][metric] for r in scored_rows if metric in r["scores"]]
        if values:
            print(f"Average {metric}: {sum(values) / len(values):.3f} (n={len(values)})")

    simple_attempts = [r for r in rows if "error" not in r and r["ever_took_simple_path"]]
    misrouted = [r for r in simple_attempts if r["escalated"]]
    if simple_attempts:
        rate = len(misrouted) / len(simple_attempts)
        print(f"Cheap-path misroute rate: {len(misrouted)}/{len(simple_attempts)} = {rate:.0%}")
    else:
        print("Cheap-path misroute rate: n/a (no question took the simple path)")

    bad_citations = [r for r in rows if "error" not in r and not r["citation_check_passed"]]
    print(f"Citation check: {len(rows) - len(bad_citations) - sum('error' in r for r in rows)}/{len(rows)} passed")


async def main() -> None:
    if not settings.groq_api_keys:
        raise SystemExit("Set GROQ_API_KEY in .env first.")

    questions = _load_questions()
    print(f"Evaluating {len(questions)} questions (run id: {RUN_ID})...\n")

    rows: list[dict] = []
    async with build_graph() as graph:
        for entry in questions:
            start = time.monotonic()
            try:
                row = await _run_one_question(graph, entry)
            except Exception as exc:  # noqa: BLE001 - a bad question must not
                # abort the whole eval run; record it and keep going (D-10:
                # report failures honestly, don't let one hide the rest).
                row = {"id": entry["id"], "category": entry["category"], "error": f"{type(exc).__name__}: {exc}"}
            elapsed = time.monotonic() - start
            print(f"[{entry['id']}] done in {elapsed:.0f}s")
            rows.append(row)

    RESULTS_PATH.write_text(json.dumps({"run_id": RUN_ID, "results": rows}, indent=2), encoding="utf-8")
    print(f"\nFull results written to: {RESULTS_PATH}")

    _print_summary(rows)


if __name__ == "__main__":
    asyncio.run(main())
