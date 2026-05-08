# Real-Time Multi-Agent LLM Orchestration & Evaluation System

> A production-grade multi-agent pipeline with dynamic runtime routing, context budget management, self-improving prompt loops, adversarial evaluation, token-level SSE streaming, and full execution trace observability — built from scratch without black-box eval frameworks.

[![Tests](https://github.com/Arjit2716/multiagent-llm-system/actions/workflows/test.yml/badge.svg)](https://github.com/Arjit2716/multiagent-llm-system/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](./docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Architecture](#2-architecture)
3. [Agent Descriptions & Decision Boundaries](#3-agent-descriptions--decision-boundaries)
4. [Self-Improving Prompt Loop](#4-self-improving-prompt-loop)
5. [Public API — 5 Endpoints](#5-public-api--5-endpoints)
6. [Running Tests](#6-running-tests)
7. [Known Limitations & Honest Assessment](#7-known-limitations--honest-assessment)
8. [What We Would Build Next](#8-what-we-would-build-next)

---

## 1. Quick Start

### Prerequisites
- Docker & Docker Compose **or** Python 3.11+ and Node 18+
- At least one LLM API key (Groq free tier recommended)

### Docker (zero manual steps)

```bash
git clone https://github.com/Arjit2716/multiagent-llm-system.git
cd multiagent-llm-system

cp .env.example .env
# Open .env and set:
#   GROQ_API_KEY=gsk_...
#   POSTGRES_PASSWORD=choose-a-strong-password

docker compose up -d
```

| Service | URL | Purpose |
|---|---|---|
| **API** | http://localhost:8000/docs | Interactive Swagger docs |
| **Frontend** | http://localhost:3000 | React dashboard |
| **Logs** | http://localhost:9999 | Dozzle — live log search/tail |

### Local Dev (no Docker)

```bash
# Backend
python -m venv venv
.\venv\Scripts\activate           # Windows
pip install -r backend/requirements.txt

# Set env vars
cp .env.example .env              # fill in GROQ_API_KEY

$env:PYTHONPATH="."
python -m uvicorn backend.api.app:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

### Supported LLM Providers

| Provider | Key var | Recommended model |
|---|---|---|
| **Groq** (free) | `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-20241022` |

---

## 2. Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLIENT  (Browser / curl / SSE)                       │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │  HTTP / SSE / WebSocket
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                     FastAPI Gateway  (port 8000)                             │
│   POST /api/v1/query          →  SSE stream (token-by-token)                │
│   GET  /api/v1/jobs/{id}/trace →  Full execution trace                      │
│   GET  /api/v1/eval/summary    →  Eval breakdown by category & dimension    │
│   POST /api/v1/proposals/{id}/decision → Approve / reject prompt rewrite   │
│   POST /api/v1/eval/rerun      →  Targeted re-eval on failed cases         │
└───────┬───────────────────────────────────────────────────────┬─────────────┘
        │                                                       │
        │  sync (SSE jobs)              async (background jobs) │
        ▼                                                       ▼
┌───────────────────┐                              ┌───────────────────────────┐
│  Dynamic          │                              │  Celery Worker            │
│  Orchestrator     │                              │  (agent_jobs / eval_jobs) │
│                   │                              │                           │
│  LLM routing loop │                              │  run_agent_job()          │
│  ─ decides NEXT   │                              │  run_eval_suite()         │
│    agent at each  │                              └────────────┬──────────────┘
│    step via LLM   │                                           │
│    reasoning      │                              ┌────────────▼──────────────┐
│  ─ allocates      │                              │  Redis  (broker + cache)  │
│    token budget   │                              └───────────────────────────┘
│  ─ emits trace    │
│    events         │
└────────┬──────────┘
         │  invokes (one at a time, never direct agent-to-agent)
         │
   ┌─────┴──────────────────────────────────────────────────────────┐
   │                    Shared Context Object                        │
   │  (single source of truth — append-only, all agents read/write) │
   └──────┬───────────┬────────────┬────────────┬───────────────────┘
          │           │            │             │
   ┌──────▼──┐  ┌─────▼──┐  ┌─────▼──┐  ┌──────▼──────┐
   │Decompo- │  │  RAG   │  │Critique│  │ Synthesis   │
   │ sition  │  │ Agent  │  │ Agent  │  │ Agent       │
   │ Agent   │  │        │  │        │  │             │
   └──────┬──┘  └─────┬──┘  └─────┬──┘  └──────┬──────┘
          │           │            │             │
   ┌──────▼───────────▼────────────▼─────────────▼──────┐
   │                  Document Store / Tool Registry      │
   │  (RAG chunks, web search, calculator, code executor) │
   └─────────────────────────────────────────────────────┘
          │
   ┌──────▼─────────────────────────────────────────────┐
   │           Observability Layer                       │
   │  ExecutionTracer  →  per-job event log (in-memory)  │
   │  PromptManager    →  active prompt registry (DB)    │
   │  ContextBudget    →  per-agent token accounting      │
   │  StructuredLogs   →  correlation_id tagged JSON      │
   └─────────────────────────────────────────────────────┘
          │
   ┌──────▼──────────────┐    ┌──────────────────────────┐
   │   PostgreSQL (DB)   │    │  Dozzle  (port 9999)     │
   │  Tasks, Eval runs,  │    │  Live log search + tail  │
   │  Proposals, Traces  │    │  across all containers   │
   └─────────────────────┘    └──────────────────────────┘
```

### Data Flow for a Single Query

```
User query
   │
   ▼
FastAPI /api/v1/query
   │  mints job_id, creates SharedContext
   │  starts orchestrator in background task
   │  returns SSE stream immediately
   ▼
DynamicOrchestrator.run()
   │
   ├─ Step 1: LLM decides → DecompositionAgent
   │    Breaks query into typed subtasks + dependency DAG
   │    Emits: ROUTING_DECISION, AGENT_START, TOKEN_CHUNK×N, AGENT_COMPLETE
   │
   ├─ Step 2: LLM decides → RAGAgent
   │    Multi-hop retrieval, citation tagging
   │    Emits: AGENT_START, TOOL_CALL, TOOL_RESULT, AGENT_COMPLETE
   │
   ├─ Step 3: LLM decides → CritiqueAgent
   │    Per-claim confidence scoring, span-level flags
   │    Emits: AGENT_START, TOKEN_CHUNK×N, AGENT_COMPLETE
   │
   ├─ Step 4: LLM decides → SynthesisAgent
   │    Merges outputs, resolves contradictions, builds provenance map
   │    Emits: AGENT_START, TOKEN_CHUNK×N, AGENT_COMPLETE
   │
   └─ PIPELINE_DONE emitted → SSE stream closes
```

---

## 3. Agent Descriptions & Decision Boundaries

### Dynamic Orchestrator
**Role:** Runtime routing engine. Decides which agent to invoke next at every step by inspecting the full `SharedContext` state and calling the LLM to reason about the decision.

**Decision boundaries:**
- Invokes `DecompositionAgent` first if no dependency graph exists.
- Invokes `RAGAgent` after decomposition for any factual or reasoning subtask.
- Invokes `CritiqueAgent` after RAG completes.
- Invokes `SynthesisAgent` as the terminal step when all retrieval and critique is done.
- Stops the loop when `stop_pipeline: true` is returned by the routing LLM, or when the session token cap (`20,000` tokens) is exhausted.
- **Does not** allow agents to call each other — all handoffs go through the orchestrator exclusively.

**Token budget:** 2,000 tokens per routing decision call.

---

### Decomposition Agent
**Role:** Breaks a user query into a typed, dependency-aware task graph (DAG). Each subtask is assigned a type (`factual`, `reasoning`, `summarization`, `comparison`, `causal`, `procedural`, `ambiguous`) and explicit dependencies.

**Decision boundaries:**
- Runs exactly once per pipeline execution.
- If the query is ambiguous, subtasks are typed `ambiguous` and flagged — the system does not fabricate specificity.
- Does **not** execute subtasks; it only structures them.
- If the LLM output cannot be parsed as a valid DAG, it logs an error and the orchestrator skips to RAG with the raw query.

**Token budget:** 2,000 tokens.

---

### RAG Agent (Retrieval-Augmented Generation)
**Role:** Multi-hop document retrieval with mandatory citation tracking. Each retrieved chunk is tagged with `source`, `hop_index`, `relevance_score`, and the query that produced it.

**Decision boundaries:**
- Performs up to 3 retrieval hops. Each hop can refine the query based on what the previous hop returned.
- Every factual claim in its output must reference a `chunk_id` from the document store. Claims without citations are flagged.
- If the document store is empty, the agent declares `no_documents_available` rather than hallucinating sources.
- Does **not** perform synthesis or make judgment calls — it retrieves and summarises per-chunk.

**Token budget:** 4,000 tokens.

---

### Critique Agent
**Role:** Span-level quality auditor. Reviews the RAG agent's output claim-by-claim, assigns confidence scores, and flags specific text spans with structured `CritiqueFlag` objects.

**Decision boundaries:**
- Flags are typed: `factual_error`, `unsupported_claim`, `logical_inconsistency`, `citation_missing`, `ambiguity`.
- A confidence score below `0.5` on any claim triggers a flag.
- The agent raises flags but does **not** rewrite output — that is the synthesis agent's job.
- If no issues are found, it returns an empty flags list (not a fabricated "all good" statement).

**Token budget:** 3,000 tokens.

---

### Synthesis Agent
**Role:** Final-stage merging, contradiction resolution, and provenance map construction. Produces the user-facing answer.

**Decision boundaries:**
- Must address every `CritiqueFlag` raised — either fix the issue, qualify the claim, or explicitly note the uncertainty.
- Contradictions between RAG output and critique flags must be **resolved**, not surfaced to the user as "agents disagreed."
- Builds a sentence-level `ProvenanceMap` linking each output sentence to its source chunk(s).
- If a contradiction cannot be resolved with available evidence, the agent outputs a qualified answer with explicit uncertainty notation rather than fabricating resolution.

**Token budget:** 3,500 tokens.

---

### Compression Agent (Context Manager)
**Role:** Intercepts agent context when it exceeds the declared budget. Compresses older context before the LLM call proceeds.

**Decision boundaries:**
- **Lossless** for structured data: tool outputs, citation objects, score fields — these are preserved verbatim.
- **Lossy** for conversational filler: pleasantries, transitional phrases, verbose reasoning — these are summarised.
- If compression still leaves the context over budget, a `POLICY_VIOLATION_CONTEXT_OVERFLOW` event is logged. The agent is **not silently truncated** — the violation is recorded in the execution trace.

---

### Meta-Agent (Self-Improvement)
**Role:** Post-eval analyzer. Reads a completed eval run, identifies the worst-performing dimension, and proposes a structured rewrite of the responsible agent's system prompt.

**Decision boundaries:**
- Only proposes a rewrite if the worst dimension average falls below `0.95`.
- Proposals are **never auto-applied** — a human must explicitly call `POST /api/v1/proposals/{id}/decision`.
- Generates a unified diff of the original vs proposed prompt for human review.
- After approval, re-runs only the previously failed test cases — not the full suite.

---

## 4. Self-Improving Prompt Loop

### What It Does

```
eval_harness.run_full_suite()        ← runs 15 test cases
         │
         ▼
MetaAgent.analyze_eval_run(run_id)
  ├─ Aggregates scores by dimension
  ├─ Finds worst-performing dimension (e.g. citation_accuracy: 0.41)
  ├─ Maps dimension → responsible agent (e.g. rag)
  ├─ Extracts failure justifications from the 3 worst cases
  ├─ Calls LLM to propose a rewrite of that agent's system prompt
  ├─ Generates unified diff (original vs proposed)
  └─ Saves PromptRewriteProposal to DB with status="pending"
         │
         ▼
Human reviews at GET /api/v1/eval/proposals
         │
         ▼
POST /api/v1/proposals/{id}/decision { "decision": "approve" }
  ├─ Activates new prompt in PromptManager (DB + in-memory cache)
  ├─ LLMClient auto-intercepts: any agent call now uses the new prompt
  ├─ Re-runs only the previously failed cases
  └─ Stores performance delta (new_avg - old_avg) on the proposal record
```

### What It Does NOT Do

| Claim | Reality |
|---|---|
| "Automatically improves the system" | ❌ It proposes — humans decide. Auto-apply is intentionally absent. |
| "Guarantees the rewrite is better" | ❌ The delta is measured after approval, not before. It could regress. |
| "Works without a good LLM" | ❌ The meta-agent uses the same LLM as the pipeline. A weak LLM writes weak rewrites. |
| "Handles all dimensions at once" | ❌ Only addresses the single worst dimension per eval run. One cycle = one improvement. |
| "Learns from production traffic" | ❌ Only learns from the 15-case eval suite. Real user queries are not fed back. |
| "Remembers across restarts" | ✅ All proposals and approved prompts persist in PostgreSQL. |

---

## 5. Public API — 5 Endpoints

Full interactive docs at **http://localhost:8000/docs**

| # | Method | Path | Purpose |
|---|---|---|---|
| 1 | `POST` | `/api/v1/query` | Submit query → SSE stream with real-time agent activity |
| 2 | `GET` | `/api/v1/jobs/{job_id}/trace` | Full ordered execution trace for any job |
| 3 | `GET` | `/api/v1/eval/summary` | Eval breakdown by category and scoring dimension |
| 4 | `POST` | `/api/v1/proposals/{id}/decision` | Approve or reject a prompt-rewrite proposal |
| 5 | `POST` | `/api/v1/eval/rerun` | Re-evaluate failed cases with the latest approved prompt |

### Error Schema
All non-2xx responses return:
```json
{
  "error_code": "JOB_NOT_FOUND",
  "message": "No trace found for job 'abc123'.",
  "job_id": "abc123"
}
```

### SSE Event Types (Endpoint 1)
| Event | What it tells you |
|---|---|
| `PIPELINE_START` | Job accepted, query, max_steps |
| `ROUTING_DECISION` | Which agent was picked, reasoning excerpt, budget remaining |
| `AGENT_START` | Agent name, allocated token budget |
| `TOKEN_CHUNK` | Raw token from the active LLM call (real-time) |
| `TOOL_CALL` / `TOOL_RESULT` | Tool name, input/output hash, latency |
| `AGENT_COMPLETE` | Latency ms, tokens used, output hash |
| `POLICY_VIOLATION` | Budget overflow or safety failure with details |
| `PIPELINE_DONE` | Final answer ready, steps taken |

---

## 6. Running Tests

```bash
# Full suite (87 tests)
$env:PYTHONPATH="."
python -m pytest tests/ -v --cov=backend

# Single module
python -m pytest tests/test_orchestration.py -v

# With coverage report
python -m pytest tests/ --cov=backend --cov-report=html
# Open htmlcov/index.html
```

---

## 7. Known Limitations & Honest Assessment

### LLM Dependency
The system's intelligence ceiling is the LLM's. On Groq's free llama-3.1-8b-instant, the decomposition agent frequently produces malformed DAGs and the synthesis agent skips contradiction resolution. **Llama-3.3-70b-versatile is the minimum for acceptable quality.**

### Document Store
The RAG agent uses a naive in-memory TF-IDF vector store. It has no semantic embedding model, no re-ranking, and no hybrid search. On any corpus larger than a few hundred chunks, retrieval quality degrades significantly. This is the biggest accuracy bottleneck in the system.

### Orchestrator Routing
The routing LLM occasionally selects agents out of logical order (e.g., synthesis before critique) or loops on the same agent unnecessarily. The `max_steps=8` guard prevents infinite loops but does not prevent wasted routing calls.

### Context Compression
The compression agent uses a heuristic to identify "structured data" (JSON-looking text) vs "conversational filler." This heuristic fails on structured prose (e.g., numbered lists of natural language) and can misclassify important reasoning as filler.

### Self-Improving Loop
- The eval harness runs 15 static cases. It does not adapt to the distribution of real queries the system receives.
- The meta-agent's prompt rewrite quality is entirely dependent on how clearly the failure justifications explain what went wrong. Vague justifications produce vague rewrites.
- There is no guard against adversarial rewrites (a compromised LLM could propose a prompt that weakens safety constraints).

### Adversarial Testing
The detection logic for prompt injections relies on keyword matching and simple heuristics. Sophisticated, encoded, or multi-turn injection attacks will bypass detection. The current test suite covers well-known patterns only.

### Concurrency
The in-memory session store and execution tracer are not thread-safe beyond the asyncio event loop. Under high load (multiple concurrent pipeline runs), trace events can interleave across jobs. A Redis-backed tracer would be required for production multi-worker deployments.

### Circuit Breaker State
The circuit breaker state is in-process memory. If the API server restarts, all breakers reset. A persistent breaker (Redis-backed) is needed for production.

---

## 8. What We Would Build Next

### 1. Semantic RAG with Embeddings
Replace the TF-IDF store with a proper vector database (Qdrant or pgvector). Add a bi-encoder for retrieval and a cross-encoder for re-ranking. This would be the single highest-impact improvement.

### 2. Streaming Worker Results Back to SSE
Currently, async Celery jobs do not stream token-level events back to the SSE client. The worker would need to publish events to Redis pub/sub and the API would relay them to the SSE stream in real time.

### 3. Multi-Turn Conversation Memory
Add a persistent, cross-session memory layer (episodic + semantic) so the system can refer back to previous conversations, user preferences, and prior retrieved facts without re-running the full pipeline.

### 4. Eval Harness on Real Traffic
Instrument production queries: sample a percentage of live traffic, run the scoring pipeline asynchronously, and feed the results into the meta-agent loop. This closes the gap between static test cases and real-world failure modes.

### 5. Human-in-the-Loop Critique Interface
Build a UI where human reviewers can annotate agent outputs directly — marking claims as correct/wrong, adding citations, flagging hallucinations. These annotations become high-quality training signal for the meta-agent.

### 6. Agent Specialisation via Fine-Tuning
Use the accumulated eval data (query, trace, scores, human annotations) to fine-tune smaller models (Llama 3.1 8B) for specific agent roles. A fine-tuned decomposition agent would be faster and cheaper than prompting a large model.

### 7. Persistent Circuit Breakers + Distributed Tracing
Move circuit breaker state to Redis. Integrate OpenTelemetry traces with Jaeger so every cross-agent call appears as a span in a distributed trace, making latency bottlenecks immediately visible.

### 8. Prompt Rewrite Safety Guard
Before storing a meta-agent proposal, run the proposed prompt through an independent safety classifier to detect any weakening of safety constraints or injection of adversarial instructions. Proposals that fail safety review are auto-rejected.

---

## License
MIT
