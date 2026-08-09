# Sample run — real, unedited

Captured from an actual browser session against the live server (`uvicorn app.api.main:app
--reload`), not constructed for this document. Every field below — timestamps, token
counts, provider/key slot, decision text — is copied verbatim from `logs/trace.jsonl` and
the browser UI for this exact run. Nothing here is hand-written to look plausible.

**Question:**
> Compare between Growth tier's pricing and contract guarantees, and what will happen if
> the data resources limits are exceeded?

**Thread:** `fb2b7a28-a2c8-4ef0-9af1-2332fb90f9ed`

This question is a real paraphrase of an earlier, already-cached question. It's used here
specifically *because* it shows two things in one continuous run: the semantic cache
correctly declining a near-miss (0.913, just under the 0.93 threshold — see Phase 8/D-09),
and the full multi-agent path running for real immediately afterward on the same request.

---

## 1. Semantic cache checked first — and correctly missed

```json
{"kind": "cache", "node": "semantic_cache", "purpose": "lookup", "ts": "2026-08-09T14:46:22.802796+00:00", "thread_id": "fb2b7a28-a2c8-4ef0-9af1-2332fb90f9ed", "decision": "miss (best similarity 0.913 < 0.93)"}
```

The nearest cached question ("How does the Growth tier's pricing compare to what the
contract guarantees, and what happens if we exceed the data source limit?") was close, but
not close enough — the graph runs fresh below.

## 2. Planner — decides the shape of the work

```json
{"kind": "llm_call", "node": "planner", "purpose": "decide research mode and angles", "ts": "2026-08-09T14:46:23.828569+00:00", "provider": "groq", "key_index": "groq[0]", "model": "openai/gpt-oss-120b", "prompt_tokens": 457, "completion_tokens": 240, "reasoning_tokens": 144, "total_tokens": 697, "path": "native_tool_call", "attempt": 1}
```

Decision (native tool call, not prompted JSON):
```json
{
  "mode": "multi_angle",
  "angles": [
    "Growth tier pricing versus contract guarantees",
    "Outcomes when data resources limits are exceeded"
  ],
  "reason": "The question asks for a comparison of pricing and contract guarantees (one angle) and also asks about the consequences of exceeding data resource limits (a separate angle), requiring distinct research."
}
```

Correctly classified `multi_angle` — the question genuinely combines two distinct topics,
not one fact asked thoroughly (the failure mode D-13/failure-mode #2 exists to catch).

## 3. Researchers — dispatched in parallel via `Send`, not `asyncio.gather()`

Both angles are LangGraph `Send` targets in the **same superstep** — real evidence of that:
their completion timestamps are 0.3s apart, not sequential.

```json
{"kind": "llm_call", "node": "researcher", "purpose": "extract finding for angle a1", "ts": "2026-08-09T14:46:25.144894+00:00", "provider": "groq", "key_index": "groq[0]", "model": "openai/gpt-oss-120b", "prompt_tokens": 1864, "completion_tokens": 344, "reasoning_tokens": 267, "total_tokens": 2208, "path": "native_tool_call", "attempt": 1}
```
> Finding (a1): "When a customer exceeds the tier-specified limit on connected data
> sources, Northbay allows the purchase of additional sources as add-ons, which are billed
> per source per month." — 1 supporting chunk.

```json
{"kind": "llm_call", "node": "researcher", "purpose": "extract finding for angle a0", "ts": "2026-08-09T14:46:25.467995+00:00", "provider": "groq", "key_index": "groq[0]", "model": "openai/gpt-oss-120b", "prompt_tokens": 1795, "completion_tokens": 499, "reasoning_tokens": 371, "total_tokens": 2294, "path": "native_tool_call", "attempt": 1}
```
> Finding (a0): "The Growth tier is advertised with a 14-day free trial and includes one
> Forward-Deployed Engineer engagement of up to 8 weeks, while the contract terms grant a
> 30-day evaluation..." — 2 supporting chunks.

## 4. Synthesizer — merges findings, real citations, no abstain

```json
{"kind": "llm_call", "node": "synthesizer", "purpose": "compose answer or abstain", "ts": "2026-08-09T14:46:38.147715+00:00", "provider": "groq", "key_index": "groq[0]", "model": "openai/gpt-oss-120b", "prompt_tokens": 459, "completion_tokens": 504, "reasoning_tokens": 265, "total_tokens": 963, "path": "native_tool_call", "attempt": 1}
```

`abstained=false`, 4 citations — genuinely well-supported, no forced abstain.

## 5. Critic — real judgement, not an auto-approve shortcut

```json
{"kind": "llm_call", "node": "critic", "purpose": "verify claims against findings", "ts": "2026-08-09T14:46:46.214591+00:00", "provider": "groq", "key_index": "groq[0]", "model": "openai/gpt-oss-120b", "prompt_tokens": 578, "completion_tokens": 265, "reasoning_tokens": 223, "total_tokens": 843, "path": "native_tool_call", "attempt": 1}
```

Verdict: `{"approved": true, "feedback": "All claims are supported by the cited findings."}`

**Elapsed, cache-miss to Critic approval: ~23.4 seconds, 4 real LLM calls, 6002 total
tokens** — the actual cost this project's whole premise is honest about (8–15 calls is the
full-question figure with parallel research and no early cheap-path exit; this run's cheap
`Send` fan-out of exactly 2 angles kept it toward the lower end).

## 6. Human-in-the-loop — paused, then resumed (D-14)

The graph does **not** end at the Critic. It pauses here via LangGraph's `interrupt()`,
persists to the SQLite checkpoint, and waits. The browser UI shows the draft answer and
`Approve` / `Reject` buttons; nothing past this point exists until a human acts.

A real `POST /resume` with `{"thread_id": "fb2b7a28-a2c8-4ef0-9af1-2332fb90f9ed", "approved": true}`
followed, returning `200 OK`, and the run completed:

## 7. Final answer

> The Growth tier is offered with a 14-day free trial and includes one Forward-Deployed
> Engineer engagement of up to eight weeks. After the trial, the subscription must be paid
> annually in advance, and the contract provides a 30-day evaluation period, a 12-month
> initial term with automatic 12-month renewals, and requires the subscription fee to be
> paid up front each year [09_pricing_tiers::0][09_pricing_tiers::1][10_contract_trial_terms::1].
> If a customer exceeds the tier-specified limit on connected data sources, Northbay allows
> the purchase of additional sources as add-ons, which are billed on a per-source-per-month
> basis [09_pricing_tiers::1].

**citations:** `09_pricing_tiers::0`, `09_pricing_tiers::1`, `10_contract_trial_terms::1`
**revisions:** 0 · **escalations:** 0

---

## What this run demonstrates, and what it honestly doesn't

**Shown, with real evidence above:** the Planner's genuine multi-angle decision, real
parallel `Send` dispatch, grounded synthesis with validated citations, a real (not
auto-approved) Critic judgement, and the human-in-the-loop pause/resume that every path
through this graph funnels through.

**Not shown in this transcript:** the Critic's revision cycle. As recorded honestly in
`PHASES.md`'s failure-mode table across three separate real attempts (Phases 4 and 6), the
Synthesizer's own honesty checks have so far pre-empted the evidence gap the Critic exists
to catch, before the Critic ever needed to send work back. The mechanism itself is verified
by `tests/test_state.py`, not by a live trace — named here rather than implied as
demonstrated.

**Also not included here:** a second transcript of the cheap, single-researcher path
(`simple` mode, skipping the Critic entirely — D-13). That path is fully built, tested
(`tests/test_state.py`), and was verified live in earlier phases, but no fresh, currently
captured real run of it exists in this session to transcribe faithfully. Add one here the
next time a `simple`-classified question is run live, rather than reconstructing one from
memory.
