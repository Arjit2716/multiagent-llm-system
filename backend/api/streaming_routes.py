"""
Streaming & Observability API
==============================
Endpoints:
  GET  /api/v1/stream/{job_id}          – SSE: real-time token-by-token stream
  GET  /api/v1/trace/{job_id}           – Full execution trace (all events ordered)
  GET  /api/v1/trace/{job_id}/timeline  – Agent-level timeline only (no TOKEN_CHUNK noise)
  GET  /api/v1/trace/{job_id}/budget    – Context budget status per agent
  GET  /api/v1/jobs                     – List all tracked job IDs
  POST /api/v1/orchestration/run-stream – Run pipeline and stream output via SSE
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agents.shared_context import SharedContext
from backend.api.orchestration_routes import _build_orchestrator
from backend.core.execution_trace import EventType, ExecutionTracer, get_tracer, list_jobs
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["streaming", "observability"])


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_event(data: dict, event: str = "message") -> str:
    """Format a dict as an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_tracer(tracer: ExecutionTracer, request: Request) -> AsyncGenerator[str, None]:
    """Pull events from the tracer's SSE queue and yield them as SSE lines."""
    q = tracer.subscribe()
    try:
        # Replay already-stored events first (catch-up for late subscribers)
        for event in tracer.get_trace():
            if await request.is_disconnected():
                return
            yield _sse_event(event, event=event["event_type"])

        # Then stream new events live
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                event_type = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
                yield _sse_event(event.to_dict(), event=event_type)
                # Stop streaming after pipeline is done
                if event.event_type == EventType.PIPELINE_DONE:
                    yield _sse_event({"job_id": tracer.job_id, "status": "done"}, event="STREAM_END")
                    break
            except asyncio.TimeoutError:
                # Heartbeat to keep connection alive
                yield ": heartbeat\n\n"
    finally:
        tracer.unsubscribe(q)


# ── SSE Stream endpoint ───────────────────────────────────────────────────────

@router.get("/stream/{job_id}", summary="Real-time SSE stream for a pipeline job")
async def stream_job(job_id: str, request: Request):
    """
    Server-Sent Events stream for a running pipeline.

    Events emitted:
      PIPELINE_START    – job accepted
      ROUTING_DECISION  – which agent was selected + why + budget remaining
      AGENT_START       – agent began executing
      TOOL_CALL / TOOL_RESULT – tool invocation in flight / result
      TOKEN_CHUNK       – raw token from the active LLM call (text: chunk)
      AGENT_COMPLETE    – agent done (latency_ms, token_count, output_hash)
      POLICY_VIOLATION  – budget overflow / safety failure
      PIPELINE_DONE     – pipeline finished
    """
    tracer = get_tracer(job_id)
    return StreamingResponse(
        _stream_tracer(tracer, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


# ── Run + Stream ──────────────────────────────────────────────────────────────

class StreamRunRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=5000)
    session_id: Optional[str] = None


@router.post(
    "/orchestration/run-stream",
    summary="Start pipeline and stream output via SSE",
)
async def run_and_stream(request_body: StreamRunRequest, request: Request):
    """
    Submit a query, start the pipeline in a background task, and immediately
    begin streaming events via SSE. The client receives token-by-token output
    plus full observability events (routing decisions, budget, tool calls).
    """
    job_id = request_body.session_id or str(uuid.uuid4())
    ctx = SharedContext(session_id=job_id, query=request_body.query)
    tracer = get_tracer(job_id)

    logger.info("stream_run_started", job_id=job_id, query=request_body.query[:80])

    async def _run_pipeline():
        orchestrator = _build_orchestrator()
        try:
            await orchestrator.run(ctx)
        except Exception as e:
            tracer.emit(
                EventType.POLICY_VIOLATION,
                agent_id="dynamic_orchestrator",
                policy_violations=[f"Top-level pipeline error: {e}"],
            )

    # Start pipeline in background; stream events immediately
    task = asyncio.create_task(_run_pipeline())

    async def _generate():
        q = tracer.subscribe()
        try:
            yield _sse_event({"job_id": job_id, "status": "started", "query": request_body.query}, event="STREAM_START")
            while True:
                if await request.is_disconnected():
                    task.cancel()
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    event_type = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
                    yield _sse_event(event.to_dict(), event=event_type)
                    if event.event_type == EventType.PIPELINE_DONE:
                        yield _sse_event({"job_id": job_id, "status": "done"}, event="STREAM_END")
                        break
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            tracer.unsubscribe(q)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
            "X-Job-Id":         job_id,
        },
    )


# ── Trace query endpoints ─────────────────────────────────────────────────────

@router.get("/trace/{job_id}", summary="Full execution trace for a job")
async def get_full_trace(job_id: str):
    """
    Returns the complete ordered sequence of all events for a job:
    pipeline start → routing decisions → agent starts → tool calls →
    token chunks → agent completions → pipeline done.

    Each event includes:
      timestamp, agent_id, event_type, input_hash, output_hash,
      latency_ms, token_count, context_budget_remaining, policy_violations
    """
    tracer = get_tracer(job_id)
    trace = tracer.get_trace()
    if not trace:
        raise HTTPException(status_code=404, detail=f"No trace found for job '{job_id}'")
    return {
        "job_id": job_id,
        "total_events": len(trace),
        "events": trace,
    }


@router.get("/trace/{job_id}/timeline", summary="Agent-level timeline (no token noise)")
async def get_timeline(job_id: str):
    """
    Returns only high-signal events: PIPELINE_START/DONE, ROUTING_DECISION,
    AGENT_START/COMPLETE, TOOL_CALL/RESULT, POLICY_VIOLATION.
    TOKEN_CHUNK events are excluded. Ideal for audit / replay views.
    """
    tracer = get_tracer(job_id)
    timeline = tracer.get_agent_timeline()
    if not timeline:
        raise HTTPException(status_code=404, detail=f"No timeline for job '{job_id}'")
    return {
        "job_id": job_id,
        "timeline_events": len(timeline),
        "timeline": timeline,
    }


@router.get("/trace/{job_id}/budget", summary="Context budget status per agent")
async def get_budget_status(job_id: str):
    """
    Reconstructs the per-agent context budget consumption from the trace,
    highlighting any POLICY_VIOLATION events caused by budget overflow.
    """
    tracer = get_tracer(job_id)
    trace = tracer.get_trace()
    if not trace:
        raise HTTPException(status_code=404, detail=f"No trace for job '{job_id}'")

    agent_budgets = {}
    violations = []

    for evt in trace:
        a = evt.get("agent_id", "")
        if evt["event_type"] == "AGENT_START":
            agent_budgets.setdefault(a, {"allocated": 0, "used": 0})
            agent_budgets[a]["allocated"] = evt.get("payload", {}).get("allocated_budget", 0)
            agent_budgets[a]["budget_at_start"] = evt.get("context_budget_remaining", -1)
        elif evt["event_type"] == "AGENT_COMPLETE":
            agent_budgets.setdefault(a, {"allocated": 0, "used": 0})
            agent_budgets[a]["used"] = evt.get("token_count", 0)
            agent_budgets[a]["latency_ms"] = evt.get("latency_ms", 0)
            agent_budgets[a]["budget_at_end"] = evt.get("context_budget_remaining", -1)
        elif evt["event_type"] == "POLICY_VIOLATION":
            violations.append({
                "agent_id": a,
                "timestamp": evt["timestamp"],
                "violations": evt.get("policy_violations", []),
            })

    return {
        "job_id": job_id,
        "agent_budget_summary": agent_budgets,
        "policy_violations": violations,
        "total_violations": len(violations),
    }


@router.get("/jobs", summary="List all tracked job IDs")
async def list_tracked_jobs():
    """List all job IDs for which a trace exists in memory."""
    jobs = list_jobs()
    return {"total": len(jobs), "job_ids": jobs}
