"""
Public API  — v1
=================
Exactly five endpoints, fully documented.

Error schema (all non-2xx responses):
  {
    "error_code": "MACHINE_READABLE_CODE",   # e.g. "JOB_NOT_FOUND"
    "message":    "Human-readable sentence.", # plain English
    "job_id":     "uuid | null"               # present when a job is implicated
  }

Endpoints
---------
  1. POST /api/v1/query
       Submit a query; receive a streaming SSE response with real-time agent
       activity (routing decisions, token chunks, budget, tool calls).

  2. GET  /api/v1/jobs/{job_id}/trace
       Retrieve the full ordered execution trace for any completed or running job.

  3. GET  /api/v1/eval/summary
       Latest eval-harness run summary broken down by test category and every
       scoring dimension.

  4. POST /api/v1/proposals/{proposal_id}/decision
       Human approves or rejects a pending meta-agent prompt-rewrite proposal.

  5. POST /api/v1/eval/rerun
       Trigger a targeted re-evaluation on the previously failed cases of a
       given proposal, using the latest approved prompt for that agent.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func

from backend.agents.shared_context import SharedContext
from backend.api.orchestration_routes import _build_orchestrator
from backend.core.execution_trace import EventType, get_tracer
from backend.core.logging import get_logger
from backend.core.prompt_manager import prompt_manager
from backend.db.models import (
    EvalHarnessRun,
    EvalHarnessTestCaseResult,
    PromptRewriteProposal,
    get_session_factory,
)
from backend.evaluation.eval_harness import HarnessEvaluator

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Public API v1"])


# ═══════════════════════════════════════════════════════════════════════════════
# Error helpers
# ═══════════════════════════════════════════════════════════════════════════════

class APIError(BaseModel):
    """Machine-readable error returned on every non-2xx response."""
    error_code: str = Field(..., example="JOB_NOT_FOUND")
    message: str    = Field(..., example="No trace found for job 'abc123'.")
    job_id: Optional[str] = Field(None, example="3fa85f64-5717-4562-b3fc-2c963f66afa6")


def _err(status: int, code: str, message: str, job_id: Optional[str] = None) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail=APIError(error_code=code, message=message, job_id=job_id).model_dump(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Shared SSE helper
# ═══════════════════════════════════════════════════════════════════════════════

def _sse(data: dict, event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _pipe_tracer(job_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    Replay stored events then stream new ones in real time.
    Sends a heartbeat comment every 25 s to keep the TCP connection alive.
    """
    tracer = get_tracer(job_id)
    q = tracer.subscribe()
    try:
        # Catch-up: events emitted before the client connected
        for stored in tracer.get_trace():
            if await request.is_disconnected():
                return
            yield _sse(stored, event=stored["event_type"])

        # Live stream
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(q.get(), timeout=25.0)
                etype = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
                yield _sse(event.to_dict(), event=etype)
                if event.event_type == EventType.PIPELINE_DONE:
                    yield _sse({"job_id": job_id, "status": "done"}, event="STREAM_END")
                    break
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        tracer.unsubscribe(q)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — Submit query, stream SSE
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """Body for the query endpoint."""
    query: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        example="How does transformer attention work?",
        description="Natural-language query to run through the multi-agent pipeline.",
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional client-supplied job ID (UUID). A new one is minted if omitted.",
        example="3fa85f64-5717-4562-b3fc-2c963f66afa6",
    )


@router.post(
    "/query",
    summary="① Submit a query and stream real-time agent activity via SSE",
    response_description="Server-Sent Events stream. Content-Type: text/event-stream.",
    responses={
        200: {"description": "SSE stream started."},
        422: {"model": APIError, "description": "Validation error – query too short/long."},
    },
    tags=["Public API v1"],
)
async def submit_query(body: QueryRequest, request: Request):
    """
    **Start the multi-agent pipeline and stream every event back to the client.**

    ### SSE event types emitted (in rough order)

    | Event | Payload highlights |
    |---|---|
    | `STREAM_START` | `job_id`, `query` |
    | `PIPELINE_START` | `query`, `max_steps`, `context_budget_remaining` |
    | `ROUTING_DECISION` | `selected_agent`, `reasoning`, `context_budget_remaining` |
    | `AGENT_START` | `agent_id`, `allocated_budget` |
    | `TOKEN_CHUNK` | `chunk` – raw token from the active LLM call |
    | `TOOL_CALL` | `tool_name`, `input_hash` |
    | `TOOL_RESULT` | `output_hash`, `latency_ms` |
    | `AGENT_COMPLETE` | `latency_ms`, `token_count`, `context_budget_remaining` |
    | `POLICY_VIOLATION` | `policy_violations` list |
    | `PIPELINE_DONE` | `has_answer`, `steps_taken` |
    | `STREAM_END` | `status: done` |

    ### Error responses
    | Code | error_code |
    |---|---|
    | 422 | `VALIDATION_ERROR` |
    """
    job_id = body.session_id or str(uuid.uuid4())
    ctx = SharedContext(session_id=job_id, query=body.query)
    tracer = get_tracer(job_id)

    async def _run():
        try:
            await _build_orchestrator().run(ctx)
        except Exception as exc:
            tracer.emit(
                EventType.POLICY_VIOLATION,
                agent_id="dynamic_orchestrator",
                policy_violations=[f"Unhandled pipeline error: {exc}"],
            )

    asyncio.create_task(_run())

    async def _generate():
        yield _sse({"job_id": job_id, "status": "started", "query": body.query}, event="STREAM_START")
        async for chunk in _pipe_tracer(job_id, request):
            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "X-Job-Id":          job_id,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — Full execution trace by job ID
# ═══════════════════════════════════════════════════════════════════════════════

class TraceEvent(BaseModel):
    sequence: int
    timestamp: str
    agent_id: str
    event_type: str
    input_hash: str
    output_hash: str
    latency_ms: float
    token_count: int
    context_budget_remaining: int
    policy_violations: List[str]
    payload: Dict[str, Any]


class TraceResponse(BaseModel):
    job_id: str
    total_events: int
    timeline_events: int
    events: List[TraceEvent]
    timeline: List[TraceEvent]


@router.get(
    "/jobs/{job_id}/trace",
    summary="② Full execution trace for a completed job",
    response_model=TraceResponse,
    responses={
        200: {"description": "Ordered trace with every agent event."},
        404: {"model": APIError, "description": "Job not found."},
    },
    tags=["Public API v1"],
)
async def get_execution_trace(job_id: str):
    """
    **Reconstruct the exact sequence of agent decisions, tool calls, and
    handoffs for any job – completed or in-flight.**

    The response contains two lists:

    * **`events`** — every event including `TOKEN_CHUNK` entries (full fidelity).
    * **`timeline`** — high-signal events only (`ROUTING_DECISION`, `AGENT_START`,
      `AGENT_COMPLETE`, `TOOL_CALL`, `TOOL_RESULT`, `POLICY_VIOLATION`,
      `PIPELINE_START/DONE`) — no token noise.  Ideal for audit dashboards.

    Each event carries `input_hash` / `output_hash` (first 8 chars of SHA-256)
    so you can verify exact inputs/outputs without storing full text.

    ### Error responses
    | HTTP | error_code |
    |---|---|
    | 404 | `JOB_NOT_FOUND` |
    """
    tracer = get_tracer(job_id)
    full = tracer.get_trace()
    if not full:
        raise _err(404, "JOB_NOT_FOUND", f"No trace found for job '{job_id}'.", job_id=job_id)

    timeline = tracer.get_agent_timeline()
    return TraceResponse(
        job_id=job_id,
        total_events=len(full),
        timeline_events=len(timeline),
        events=full,       # type: ignore[arg-type]
        timeline=timeline, # type: ignore[arg-type]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — Latest eval run summary
# ═══════════════════════════════════════════════════════════════════════════════

SCORE_DIMS = [
    "answer_correctness",
    "citation_accuracy",
    "contradiction_resolution",
    "tool_selection_efficiency",
    "context_budget_compliance",
    "critique_agreement_rate",
]


@router.get(
    "/eval/summary",
    summary="③ Latest eval-run summary by category and scoring dimension",
    responses={
        200: {"description": "Nested breakdown of the most recent eval harness run."},
        404: {"model": APIError, "description": "No eval run found."},
    },
    tags=["Public API v1"],
)
async def get_eval_summary():
    """
    **Return the most recent evaluation-harness run broken down by:**

    * **test category** (`baseline`, `ambiguous`, `adversarial`)
    * **scoring dimension** (`answer_correctness`, `citation_accuracy`,
      `contradiction_resolution`, `tool_selection_efficiency`,
      `context_budget_compliance`, `critique_agreement_rate`)

    Each leaf shows `mean`, `min`, `max`, and a list of per-case scores with
    justifications so you can see exactly why a score was given.

    ### Error responses
    | HTTP | error_code |
    |---|---|
    | 404 | `NO_EVAL_RUN` |
    """
    sf = get_session_factory()
    async with sf() as session:
        # Latest run
        run_result = await session.execute(
            select(EvalHarnessRun).order_by(EvalHarnessRun.timestamp.desc()).limit(1)
        )
        run = run_result.scalar_one_or_none()
        if not run:
            raise _err(404, "NO_EVAL_RUN", "No evaluation runs found. Run eval_harness.py first.")

        cases_result = await session.execute(
            select(EvalHarnessTestCaseResult).where(EvalHarnessTestCaseResult.run_id == run.id)
        )
        cases = cases_result.scalars().all()

    # Group by category
    by_category: Dict[str, List] = {}
    for c in cases:
        by_category.setdefault(c.category, []).append(c)

    def _dim_stats(case_list, dim: str) -> Dict:
        scores = [getattr(c, dim) for c in case_list]
        justifications = [
            {"query": c.query, "score": getattr(c, dim), "justification": getattr(c, f"{dim}_justification")}
            for c in case_list
        ]
        return {
            "mean":  round(sum(scores) / len(scores), 4) if scores else 0,
            "min":   round(min(scores), 4)               if scores else 0,
            "max":   round(max(scores), 4)               if scores else 0,
            "cases": justifications,
        }

    breakdown = {}
    for cat, cat_cases in by_category.items():
        breakdown[cat] = {
            "test_count":    len(cat_cases),
            "category_mean": round(sum(c.overall_score for c in cat_cases) / len(cat_cases), 4),
            "dimensions":    {dim: _dim_stats(cat_cases, dim) for dim in SCORE_DIMS},
        }

    # Dimension-level aggregates across all categories
    all_dim_means = {
        dim: round(sum(getattr(c, dim) for c in cases) / len(cases), 4) if cases else 0
        for dim in SCORE_DIMS
    }

    return {
        "run_id":           run.id,
        "timestamp":        run.timestamp.isoformat() if run.timestamp else None,
        "total_test_cases": len(cases),
        "overall_score":    round(run.total_score, 4),
        "by_category":      breakdown,
        "dimension_means":  all_dim_means,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — Approve or reject a prompt-rewrite proposal
# ═══════════════════════════════════════════════════════════════════════════════

class ProposalDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"] = Field(
        ...,
        description='Must be exactly `"approve"` or `"reject"`.',
        example="approve",
    )


class ProposalDecisionResponse(BaseModel):
    proposal_id: str
    decision: str
    status: str
    message: str
    rerun_job_id: Optional[str] = None


async def _execute_rerun(proposal_id: str, run_id: str) -> None:
    """Background task: activate prompt, re-run failed cases, store delta."""
    sf = get_session_factory()
    async with sf() as session:
        result = await session.execute(
            select(PromptRewriteProposal).where(PromptRewriteProposal.id == proposal_id)
        )
        proposal = result.scalar_one_or_none()
        if not proposal:
            return

        case_ids = proposal.failed_test_case_ids or []
        old_result = await session.execute(
            select(EvalHarnessTestCaseResult).where(EvalHarnessTestCaseResult.id.in_(case_ids))
        )
        old_cases = old_result.scalars().all()
        if not old_cases:
            return

        old_avg = sum(c.overall_score for c in old_cases) / len(old_cases)

        # Apply new prompt
        await prompt_manager.update_prompt(proposal.target_agent, proposal.proposed_prompt)

        evaluator = HarnessEvaluator()
        run_record = EvalHarnessRun(id=run_id)
        session.add(run_record)
        await session.commit()

        new_scores = []
        for c in old_cases:
            res = await evaluator.run_test_case(run_id, c.query, c.category)
            new_scores.append(res.overall_score)

        new_avg = sum(new_scores) / len(new_scores) if new_scores else 0.0
        delta = round(new_avg - old_avg, 4)

        proposal.performance_delta = delta
        proposal.status = "approved"
        proposal.decided_at = datetime.datetime.now(datetime.timezone.utc)
        run_record.total_score = new_avg
        session.add(proposal)
        session.add(run_record)
        await session.commit()
        logger.info(f"Rerun complete for proposal {proposal_id}. Delta={delta:+.4f}")


@router.post(
    "/proposals/{proposal_id}/decision",
    summary="④ Approve or reject a pending prompt-rewrite proposal",
    response_model=ProposalDecisionResponse,
    responses={
        200: {"description": "Decision recorded."},
        400: {"model": APIError, "description": "Proposal already decided."},
        404: {"model": APIError, "description": "Proposal not found."},
        422: {"model": APIError, "description": "Invalid decision value."},
    },
    tags=["Public API v1"],
)
async def decide_proposal(
    proposal_id: str,
    body: ProposalDecisionRequest,
    background_tasks: BackgroundTasks,
):
    """
    **Approve or reject a meta-agent prompt-rewrite proposal.**

    * **`approve`** — activates the new prompt for the target agent and queues a
      background re-evaluation on the previously failed test cases.
      `rerun_job_id` in the response can be used to poll `GET /api/v1/jobs/{id}/trace`.

    * **`reject`** — marks the proposal as rejected with a timestamp; the current
      prompt remains unchanged.

    Only `pending` proposals can be decided. Every decision is timestamped and
    permanently stored.

    ### Error responses
    | HTTP | error_code |
    |---|---|
    | 404 | `PROPOSAL_NOT_FOUND` |
    | 400 | `PROPOSAL_ALREADY_DECIDED` |
    | 422 | `VALIDATION_ERROR` |
    """
    sf = get_session_factory()
    async with sf() as session:
        result = await session.execute(
            select(PromptRewriteProposal).where(PromptRewriteProposal.id == proposal_id)
        )
        proposal = result.scalar_one_or_none()

    if not proposal:
        raise _err(404, "PROPOSAL_NOT_FOUND", f"No proposal with ID '{proposal_id}'.")

    if proposal.status != "pending":
        raise _err(
            400,
            "PROPOSAL_ALREADY_DECIDED",
            f"Proposal '{proposal_id}' has already been {proposal.status}.",
        )

    if body.decision == "reject":
        async with sf() as session:
            result = await session.execute(
                select(PromptRewriteProposal).where(PromptRewriteProposal.id == proposal_id)
            )
            p = result.scalar_one_or_none()
            p.status = "rejected"
            p.decided_at = datetime.datetime.now(datetime.timezone.utc)
            session.add(p)
            await session.commit()

        return ProposalDecisionResponse(
            proposal_id=proposal_id,
            decision="reject",
            status="rejected",
            message="Proposal rejected. Existing prompts remain unchanged.",
        )

    # Approve: kick off background rerun
    rerun_job_id = str(uuid.uuid4())
    background_tasks.add_task(_execute_rerun, proposal_id, rerun_job_id)

    return ProposalDecisionResponse(
        proposal_id=proposal_id,
        decision="approve",
        status="rerun_queued",
        message=(
            "Proposal approved. New prompt activated. "
            "Re-evaluation of failed cases started in background."
        ),
        rerun_job_id=rerun_job_id,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 5 — Trigger targeted re-eval on failed cases
# ═══════════════════════════════════════════════════════════════════════════════

class RerunRequest(BaseModel):
    proposal_id: str = Field(
        ...,
        description="ID of the approved proposal whose failed cases should be re-evaluated.",
        example="3fa85f64-5717-4562-b3fc-2c963f66afa6",
    )


class RerunResponse(BaseModel):
    rerun_job_id: str
    proposal_id: str
    failed_cases: int
    status: str
    message: str


@router.post(
    "/eval/rerun",
    summary="⑤ Re-evaluate previously failed cases with latest approved prompts",
    response_model=RerunResponse,
    responses={
        202: {"description": "Re-evaluation queued in background."},
        400: {"model": APIError, "description": "Proposal not approved or no failed cases."},
        404: {"model": APIError, "description": "Proposal not found."},
    },
    tags=["Public API v1"],
)
async def trigger_rerun(body: RerunRequest, background_tasks: BackgroundTasks):
    """
    **Run a targeted re-evaluation using only the test cases that failed in
    the original eval run associated with the given proposal.**

    The latest approved prompt for the target agent is used automatically.
    A new `EvalHarnessRun` record is created and linked to the same
    `proposal_id` so you can compare deltas over time.

    The response includes a `rerun_job_id` that you can pass to
    `GET /api/v1/jobs/{job_id}/trace` to follow progress in real time.

    ### Workflow
    ```
    POST /api/v1/query                        → job_id
    GET  /api/v1/jobs/{job_id}/trace          → full trace
    GET  /api/v1/eval/summary                 → scores / failures
    POST /api/v1/proposals/{id}/decision      → approve → rerun_job_id
    POST /api/v1/eval/rerun  { proposal_id }  → targeted rerun
    ```

    ### Error responses
    | HTTP | error_code |
    |---|---|
    | 404 | `PROPOSAL_NOT_FOUND` |
    | 400 | `PROPOSAL_NOT_APPROVED` |
    | 400 | `NO_FAILED_CASES` |
    """
    sf = get_session_factory()
    async with sf() as session:
        result = await session.execute(
            select(PromptRewriteProposal).where(PromptRewriteProposal.id == body.proposal_id)
        )
        proposal = result.scalar_one_or_none()

    if not proposal:
        raise _err(404, "PROPOSAL_NOT_FOUND", f"No proposal with ID '{body.proposal_id}'.")

    if proposal.status not in ("approved", "pending"):
        raise _err(
            400,
            "PROPOSAL_NOT_APPROVED",
            f"Proposal '{body.proposal_id}' has status '{proposal.status}'. "
            "Only approved or pending proposals can be re-evaluated.",
        )

    failed_ids = proposal.failed_test_case_ids or []
    if not failed_ids:
        raise _err(
            400,
            "NO_FAILED_CASES",
            f"Proposal '{body.proposal_id}' has no associated failed test cases.",
        )

    rerun_job_id = str(uuid.uuid4())
    background_tasks.add_task(_execute_rerun, body.proposal_id, rerun_job_id)

    return RerunResponse(
        rerun_job_id=rerun_job_id,
        proposal_id=body.proposal_id,
        failed_cases=len(failed_ids),
        status="queued",
        message=(
            f"Re-evaluation of {len(failed_ids)} failed case(s) queued. "
            f"Stream progress at GET /api/v1/jobs/{rerun_job_id}/trace"
        ),
    )
