# Multi-Agent RAG Research Platform

> **Status: in development (Phase 1 of 11).** Architecture and design decisions are
> settled; implementation is underway. Setup steps below are the intended commands —
> they are verified phase by phase as each is built, and this README is finalised in
> Phase 10 with real evaluation numbers and an honest limitations section. Progress is
> tracked in [`PHASES.md`](PHASES.md).

A multi-agent research system that answers **complex, multi-step questions** over a
document corpus — and shows its work live while doing it.

A single-pass RAG pipeline runs one fixed retrieval and answers. This system has agents
that decide for themselves how to break a question apart, research several angles **in
parallel**, merge the findings, and **check their own answer** before releasing it. Every
step streams to the browser as it happens, rendered as an animated node graph.

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
         ┌────────────┘       └──────────> Final answer
         │
         └──> back to Researcher  (bounded: max 1 revision cycle)
```

**Two genuine agentic decision points** — the Planner decides the *shape* of the work
(how many angles, and what each investigates), and the Critic decides whether to send
work back. Both are LLM judgement calls, not threshold checks in code. That distinction
is what makes this agentic rather than a deterministic pipeline expressed in agentic
tooling.

The Researcher and Synthesizer are deliberately **not** agentic — a researcher handed a
specific angle has nothing meaningful to decide, and making it autonomous would add cost
and variance for no benefit. Restraint here is the design, not a shortcut.

---

## What's demonstrated

| Capability | Where |
|---|---|
| Multi-agent orchestration, typed shared state | LangGraph `StateGraph` |
| Cyclic feedback loops, bounded | Critic → Researcher, max 1 revision |
| LLM-driven routing (real agency) | Planner and Critic, via native tool calling |
| Parallel agent execution | LangGraph `Send` API (dynamic fan-out) |
| Self-critique / reflection | Critic node |
| Reliability | Checkpointing (resume, not restart), recursion caps, provider failover, human-in-the-loop gate |
| Observability | Per-node tracing → SSE stream → live graph UI |
| Hybrid retrieval | BM25 + vector, fused with Reciprocal Rank Fusion |
| Grounded answers | Citations validated in code; genuine model-generated abstain |
| Async Python throughout | `AsyncOpenAI`, async retrieval |
| API service | FastAPI + Server-Sent Events |
| Semantic caching | Redis, similarity-matched, cache hits visibly traced |
| Evaluation | RAGAS — faithfulness, answer relevancy, context precision/recall |
| CI | `pytest` on deterministic units + GitHub Actions |
| Containerisation | Dockerfile + docker-compose (app + Redis) |

---

## Stack

- **Orchestration:** LangGraph
- **LLM:** `gpt-oss-120b` — Groq primary, Cerebras automatic failover on 429/5xx, both
  through the `openai` SDK's `AsyncOpenAI` client
- **Vector store:** ChromaDB (local persistent, HNSW indexing internally)
- **Keyword search:** `rank_bm25` (BM25Okapi)
- **Embeddings:** `sentence-transformers` / `all-MiniLM-L6-v2` (local, CPU, free)
- **API:** FastAPI + `sse-starlette`
- **Frontend:** Cytoscape.js, single self-contained HTML page
- **Cache:** Redis
- **Eval:** RAGAS

Reasoning for each of these — including what was rejected and why — is recorded as
numbered decisions (D-01 onward) in [`PHASES.md`](PHASES.md).

---

## Setup

### Prerequisites

- Python 3.10 or newer
- A free [Groq](https://console.groq.com/keys) API key (primary)
- A free [Cerebras](https://cloud.cerebras.ai) API key (failover)
- Docker (for Redis, and for the containerised run)

### 1. Install

Use `python` or `python3`, whichever your system recognises.

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`sentence-transformers` pulls in PyTorch — expect a few minutes on first install.

### 2. Configure keys

```bash
copy .env.example .env     # Windows
# cp .env.example .env     # macOS/Linux
```

Paste your Groq and Cerebras keys into `.env`. It is git-ignored.

### 3. Build the index

```bash
python -m app.retrieval.index
```

Chunks and embeds `docs/`, stores vectors in ChromaDB and a plain-text copy for BM25.
Re-run whenever `docs/` changes — it rebuilds from scratch, no stale leftovers.

### 4. Start Redis

```bash
docker run -d -p 6379:6379 redis
```

### 5. Run

```bash
uvicorn app.api.main:app --reload
```

Open `http://127.0.0.1:8000` — ask a question and watch the graph execute live.

**Or run everything containerised:**
```bash
docker compose up --build
```

---

## Corpus

The 15 documents in `docs/` describe **Northbay Commerce AI**, a fictional B2B
retail/consumer AI vendor. Everything in them is invented for this project.

Fictional on purpose: the model has no pretrained knowledge of Northbay, so a correct,
cited answer is only possible if retrieval genuinely worked. The corpus also contains a
deliberate contradiction (one document says a 14-day trial, another says a 30-day
evaluation period) — which gives the Critic a real conflict to catch between two
researchers — and several topics that are never covered at all, so the abstain path has
genuine gaps to fail correctly on.

---

## Relationship to the previous project

This builds on a single-pass Advanced RAG system
([DotKonnekt take-home](https://github.com/vishalgoyal25/DotKonnekt)) that answers one
question in 4–6 LLM calls. This one costs 8–15.

**It is not a replacement.** For a simple factual lookup, the single-pass pipeline is
faster, cheaper, and equally correct. This system earns its extra cost only on genuinely
multi-step questions — which is why the Planner is allowed to route simple questions
down a single-researcher path rather than always fanning out.

Knowing when *not* to reach for agents is part of the design.

---

## Honest scope

This is a **prototype built to production shape, not a production system.** It
demonstrates the architecture a scalable system uses — async, service-exposed,
containerised, observable, evaluated, CI-checked — but it is not load-tested or
horizontally scaled, and it is not deployed to a cloud provider.

Deliberately excluded, each with a stated reason in [`PHASES.md`](PHASES.md): cloud
deployment, Celery workers (SSE streaming already solves the blocking problem here), and
a pgvector migration (Chroma already uses HNSW indexing, so the migration would be
operations work rather than conceptual gain).

A full **Known limitations** section — real failure modes found during the build, with
real numbers — is added in Phase 10.
