# Multi-Agent RAG Research Platform

> **Status: Phases 0–7 of 11 complete.** Core system (LLM client, retrieval, all four
> agents, the graph, the API, the live UI) is built and verified against real runs — and
> it has now been formally scored: real RAGAS evaluation numbers are below. Semantic
> caching, CI, and containerisation/final docs are still ahead. See
> [Progress](#progress) for the exact state of each phase and [Evaluation
> results](#evaluation-results) for the real numbers.

A multi-agent research system that answers **complex, multi-step questions** over a
document corpus — and shows its work live while doing it.

A single-pass RAG pipeline runs one fixed retrieval and answers. This system has agents
that decide for themselves how to break a question apart, research several angles **in
parallel**, merge the findings, and **check their own answer** before releasing it. Every
step streams to the browser as it happens, rendered as an animated node graph.

**[See it running →](Multi-Agent%20RAG-Live%20Research%20Graph.pdf)** — a saved page
capture of a real run: the animated graph mid-escalation, the live trace panel, and the
human-approval step, exactly as it looks in a browser.

---

## Progress

| # | Phase | Status |
|---|---|---|
| 0 | Plan, rules & repo skeleton | ✅ |
| 1 | Core foundation — sticky key-pool LLM client, async tracing | ✅ |
| 2 | Retrieval — hybrid search (BM25 + vector + RRF) | ✅ |
| 3 | Agent nodes — Planner, Researcher, Synthesizer, Critic | ✅ |
| 4 | Graph wiring — `Send` fan-out, bounded cycles, checkpointed HITL | ✅ |
| 5 | API layer — FastAPI, SSE streaming, `/resume` | ✅ |
| 6 | Live frontend — animated Cytoscape.js graph | ✅ |
| 7 | Evaluation — real RAGAS scoring against a 6-question set | ✅ |
| 8 | Semantic cache — Redis | ⬜ next |
| 9 | CI — pytest + GitHub Actions | ⬜ |
| 10 | Sample transcript, containerisation, final README | ⬜ |

Each completed phase was closed only after real terminal/browser output confirmed it
worked — not on assumption. Several real bugs were found and fixed along the way (a
citation/angle-ID collision, a checkpoint-serialization warning, three separate live-UI
bugs caught from actual screenshots), each root-caused from the actual installed source
rather than guessed. That process — and the full decision log — lives in this project's
internal working notes, not published here; what's below is the durable, user-facing
result.

---

## Evaluation results

Real [RAGAS](https://docs.ragas.io) scoring, Groq-backed judge, against 6 real questions
run through the actual live graph end to end — not a demo number.

| Metric | Score |
|---|---|
| Faithfulness | **1.00** |
| Answer relevancy | **0.93** |
| Context precision | **0.83** |
| Context recall | **0.83** |
| Citation validity (deterministic, code-checked) | **6 / 6** |
| Cheap-path misroute rate | **25%** (1 of 4 simple-path attempts needed escalation) |

Scores exclude the 2 questions that correctly abstained (no claims to grade faithfulness
against) — same evaluation set includes a deliberately uncovered question and the corpus's
built-in factual conflict, both of which produced genuine abstains rather than confident
guesses. Full per-question results: [`eval/results.json`](eval/results.json).

---

## The agent graph

```
                    ┌──────────────┐
   Question ───────>│   Planner    │  simple lookup, or N research angles?
                    └──────┬───────┘  (agentic decision — native tool call)
                           │
              ┌────────────┼────────────┐
              v            v            v
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Researcher│ │Researcher│ │Researcher│   PARALLEL (LangGraph Send)
        │    #1    │ │    #2    │ │    #3    │   hybrid retrieve → rerank → extract
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └────────────┼────────────┘
                          v
                   ┌─────────────┐
                   │ Synthesizer │  answer + validated citations, OR abstain
                   └──────┬──────┘
                          v
                   ┌─────────────┐
                   │   Critic    │  unsupported claims? (agentic decision)
                   └──┬───────┬──┘
              revise  │       │  approve
         ┌────────────┘       └──────────> Human approval (HITL) ───> Final answer
         │
         └──> back to Researcher  (bounded: max 1 revision cycle)
```

**Two genuine agentic decision points** — the Planner decides the *shape* of the work
(how many angles, and what each investigates), and the Critic decides whether to send
work back. Both are LLM judgement calls, not threshold checks in code.

The Researcher and Synthesizer are deliberately **not** agentic — a researcher handed a
specific angle has nothing meaningful to decide, and making it autonomous would add cost
and variance for no benefit. Restraint here is the design, not a shortcut.

Every path — whether the cheap single-angle route or the full multi-angle one — pauses at
a **human-in-the-loop checkpoint** before the answer is finalized. The run is paused with
LangGraph's `interrupt()`, persisted to a durable SQLite checkpoint (survives a full
process restart, not just a hot-reload), and resumed by a separate approval call.

---

## What's demonstrated (built, not just planned)

| Capability | Where | Status |
|---|---|---|
| Multi-agent orchestration, typed shared state | LangGraph `StateGraph` | ✅ |
| Dynamic parallel fan-out | LangGraph `Send` API — researcher count decided at runtime | ✅ |
| Cyclic feedback loops, bounded | Critic → Researcher, max 1 revision; cheap-path escalation, max 1 | ✅ |
| LLM-driven routing (real agency) | Planner and Critic, via native tool calling | ✅ |
| Reliability | Sticky multi-key provider failover, checkpointed resume (survives a restart), recursion caps | ✅ |
| Human-in-the-loop | `interrupt()`/resume before every final answer | ✅ |
| Observability | Per-call tracing (provider, tokens, latency) → SSE stream → live graph UI | ✅ |
| Hybrid retrieval | BM25 + vector, fused with Reciprocal Rank Fusion | ✅ |
| Grounded answers | Citations validated in code; genuine model-generated abstain | ✅ |
| Async Python throughout | `AsyncOpenAI`, async retrieval, async tracing | ✅ |
| API service | FastAPI + Server-Sent Events + REST `/resume` | ✅ |
| Live animated frontend | Self-contained Cytoscape.js page, no build step | ✅ |
| Evaluation | RAGAS — faithfulness, answer relevancy, context precision/recall, real scores above | ✅ |
| Semantic caching | Redis, similarity-matched, cache hits visibly traced | ⬜ Phase 8 |
| CI | `pytest` on deterministic units + GitHub Actions | ⬜ Phase 9 (tests exist and pass locally; wiring is what's left) |
| Containerisation | Dockerfile + docker-compose | ⬜ Phase 10 |

---

## Stack

- **Orchestration:** LangGraph (`Send` fan-out, conditional edges, `AsyncSqliteSaver`
  checkpointing, `interrupt()`)
- **LLM:** `gpt-oss-120b` — Groq primary, Cerebras automatic failover, both through the
  `openai` SDK's `AsyncOpenAI` client. Each provider supports a **pool of keys** (sticky
  failover: one key used until it fails, then the next, then the next provider)
- **Vector store:** ChromaDB (local persistent, HNSW indexing internally)
- **Keyword search:** `rank_bm25` (BM25Okapi), rebuilt from a JSON sidecar at startup
- **Embeddings:** `sentence-transformers` / `all-MiniLM-L6-v2` (local, CPU, free)
- **API:** FastAPI + `sse-starlette`
- **Frontend:** Cytoscape.js, single self-contained HTML page, no build toolchain
- **Cache (Phase 8):** Redis
- **Eval:** RAGAS 0.4.3 — real scores in [Evaluation results](#evaluation-results)

---

## Setup

### Prerequisites

- Python 3.11 or newer
- A free [Groq](https://console.groq.com/keys) API key (primary)
- A free [Cerebras](https://cloud.cerebras.ai) API key (failover)
- Redis and Docker are **not required yet** — they're only needed once Phases 8 and 10
  land.

### 1. Clone the repo

```bash
git clone https://github.com/vishalgoyal25/Multi_Agent_RAG.git
cd Multi_Agent_RAG
```

### 2. Create a virtual environment and install dependencies

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`sentence-transformers` pulls in PyTorch — expect a few minutes on first install. If you
want the exact dependency versions this project was built and tested against (rather than
whatever the `>=` floors in `requirements.txt` resolve to today), install from the lock
file instead: `pip install -r requirements.lock.txt`.

### 3. Configure your API keys

**Windows:**
```powershell
copy .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Open `.env` and paste in your keys. Each provider supports **multiple keys**, comma-separated,
with no spaces — this stretches free-tier rate limits before ever needing the failover
provider:

```
GROQ_API_KEY=key1,key2,key3
CEREBRAS_API_KEY=key1,key2
```

A single key per provider works too — just don't add a trailing comma. `.env` is
git-ignored; it never leaves your machine.

### 4. Build the retrieval index

**Windows / macOS / Linux (same command once the venv is active):**
```bash
python -m app.retrieval.index
```

Chunks and embeds the 15 documents in `docs/`, stores vectors in ChromaDB, and writes a
JSON sidecar the keyword index rebuilds from at startup. Re-run this any time `docs/`
changes — it always rebuilds from scratch, so there's never a stale leftover chunk.

### 5. Run the server

```bash
uvicorn app.api.main:app --reload
```

Then open **http://127.0.0.1:8000** in a browser, type a question, and watch the graph
build itself live as the agents work. Try both a simple factual question and a compound
one — they take visibly different paths through the graph.

### 6. (Optional) Run the standalone CLI demo instead

If you'd rather see a full run — including the human-in-the-loop pause and resume — as
plain terminal output instead of the browser UI:

```bash
python -m scripts.run_graph "How does the Growth tier's pricing compare to what the contract guarantees, and what happens if we exceed the data source limit?"
```

### 7. A few other useful commands

```bash
pytest -v                    # run the automated test suite (pure, no API calls)
python -m scripts.smoke_test # verify provider failover and tool calling work
python -m eval.run_ragas     # score the system against RAGAS (real API calls, few min)
```

---

## Corpus

The 15 documents in `docs/` describe **Northbay Commerce AI**, a fictional B2B
retail/consumer AI vendor. Everything in them is invented for this project.

Fictional on purpose: the model has no pretrained knowledge of Northbay, so a correct,
cited answer is only possible if retrieval genuinely worked. The corpus also contains a
deliberate contradiction (one document says a 14-day trial, another says a 30-day
evaluation period) — which reliably exercises the abstain path and the escalation cycle in
real runs, not just in a constructed test case — and several topics that are never covered
at all, so the abstain path has genuine gaps to fail correctly on.

---

## Relationship to the previous project

This builds on a single-pass Advanced RAG system
([DotKonnekt take-home](https://github.com/vishalgoyal25/DotKonnekt)) that answers one
question in 4–6 LLM calls. This one costs 8–15.

**It is not a replacement.** For a simple factual lookup, the single-pass pipeline is
faster, cheaper, and equally correct. This system earns its extra cost only on genuinely
multi-step questions — which is why the Planner routes simple questions down a cheap,
single-researcher path rather than always fanning out, with an escape hatch back to the
full path if that cheap attempt comes up empty.

Knowing when *not* to reach for agents is part of the design.

---

## Honest scope

This is a **prototype built to production shape, not a production system.** It
demonstrates the architecture a scalable system uses — async throughout, service-exposed,
observable, checkpointed — but it is not load-tested, not horizontally scaled, and not
deployed to a cloud provider.

Deliberately excluded, each for a stated reason: cloud deployment, Celery workers (SSE
streaming already solves the blocking-request problem here), and a pgvector migration
(Chroma already uses HNSW indexing internally, so the migration would be operations work,
not conceptual gain).

**Known limitations found so far, real and not hidden:**
- Reciprocal Rank Fusion can, by design, let a chunk that both retrieval methods
  moderately agree on outrank a chunk one method strongly prefers — a real, observed
  property of RRF, not a bug, and not retuned away.
- `gpt-oss-120b`'s reported reasoning-token counts differ meaningfully between Groq and
  Cerebras for near-identical output — "reasoning tokens" isn't a perfectly comparable
  number across providers even when total cost is.
- The Critic's revision cycle (send work back once) is fully implemented and unit-tested,
  but has not yet been observed firing on a live run — in every real attempt so far, the
  Synthesizer's own honesty checks caught the evidence gap first. Recorded as a real
  observation, not claimed as demonstrated.
- The `abstained` flag is binary and can't represent a genuinely mixed response. One eval
  question answered half its question with a real, valid citation while abstaining on the
  other half; because abstained answers are excluded from scoring, that well-grounded half
  was never evaluated. A real gap in the scoring model, named rather than smoothed over.

A final, consolidated **Known limitations** section — with the completed RAGAS run above
already folded in — is assembled in Phase 10, alongside the sample transcript and
container setup.
