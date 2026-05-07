"""
Execution Trace Layer
=====================
Provides a structured, append-only event log per pipeline job (session_id).

Every trace event conforms to a strict schema:
  timestamp, agent_id, event_type, input_hash, output_hash,
  latency_ms, token_count, context_budget_remaining, policy_violations

Events are stored in memory AND in the DB (ExecutionTraceEvent table).
A single endpoint can reconstruct the full, ordered execution trace.

Event types:
  PIPELINE_START   – orchestrator accepted a new query
  AGENT_START      – an agent was invoked by the orchestrator
  TOOL_CALL        – a tool is in flight (name + input_hash)
  TOOL_RESULT      – a tool returned (output_hash + latency)
  TOKEN_CHUNK      – a streaming token chunk from an agent LLM call
  AGENT_COMPLETE   – agent finished (output_hash, latency, token_count)
  ROUTING_DECISION – orchestrator selected next agent
  POLICY_VIOLATION – context budget exceeded / safety failure
  PIPELINE_DONE    – pipeline finished (or halted)
"""
from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional


class EventType(str, Enum):
    PIPELINE_START    = "PIPELINE_START"
    AGENT_START       = "AGENT_START"
    TOOL_CALL         = "TOOL_CALL"
    TOOL_RESULT       = "TOOL_RESULT"
    TOKEN_CHUNK       = "TOKEN_CHUNK"
    AGENT_COMPLETE    = "AGENT_COMPLETE"
    ROUTING_DECISION  = "ROUTING_DECISION"
    POLICY_VIOLATION  = "POLICY_VIOLATION"
    PIPELINE_DONE     = "PIPELINE_DONE"


def _sha8(text: str) -> str:
    """First 8 chars of SHA-256 – just enough for identity, not security."""
    return hashlib.sha256(text.encode()).hexdigest()[:8] if text else ""


@dataclass
class TraceEvent:
    job_id: str
    event_type: EventType
    agent_id: str                         = ""
    timestamp: str                        = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sequence: int                         = 0          # monotonic within a job
    input_hash: str                       = ""
    output_hash: str                      = ""
    latency_ms: float                     = 0.0
    token_count: int                      = 0
    context_budget_remaining: int         = -1
    policy_violations: List[str]          = field(default_factory=list)
    payload: Dict[str, Any]               = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


class ExecutionTracer:
    """
    Per-job event log, attached to a pipeline run via session_id == job_id.

    Usage (in orchestrator / agents):
        tracer = get_tracer(session_id)
        tracer.emit(EventType.AGENT_START, agent_id="rag", ...)
        async for chunk in agent.stream(...):
            tracer.emit_token(chunk)
            yield chunk
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._events: List[TraceEvent] = []
        self._seq = 0
        # SSE subscriber queues (asyncio.Queue per connected client)
        self._subscribers: List[asyncio.Queue] = []

    # ── Emit ──────────────────────────────────────────────────────────────────

    def emit(
        self,
        event_type: EventType,
        *,
        agent_id: str = "",
        input_text: str = "",
        output_text: str = "",
        latency_ms: float = 0.0,
        token_count: int = 0,
        context_budget_remaining: int = -1,
        policy_violations: Optional[List[str]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> TraceEvent:
        self._seq += 1
        event = TraceEvent(
            job_id=self.job_id,
            event_type=event_type,
            agent_id=agent_id,
            sequence=self._seq,
            input_hash=_sha8(input_text),
            output_hash=_sha8(output_text),
            latency_ms=round(latency_ms, 2),
            token_count=token_count,
            context_budget_remaining=context_budget_remaining,
            policy_violations=policy_violations or [],
            payload=payload or {},
        )
        self._events.append(event)
        self._broadcast(event)
        return event

    def emit_token(self, chunk: str, agent_id: str = "") -> None:
        """Lightweight emit for streaming token chunks – no hashing to avoid overhead."""
        self._seq += 1
        event = TraceEvent(
            job_id=self.job_id,
            event_type=EventType.TOKEN_CHUNK,
            agent_id=agent_id,
            sequence=self._seq,
            payload={"chunk": chunk},
        )
        self._events.append(event)
        self._broadcast(event)

    # ── SSE Pub/Sub ───────────────────────────────────────────────────────────

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def _broadcast(self, event: TraceEvent) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Slow consumer – drop; they can fetch full trace via HTTP

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_trace(self) -> List[Dict[str, Any]]:
        """Return all events in sequence order."""
        return [e.to_dict() for e in sorted(self._events, key=lambda e: e.sequence)]

    def get_agent_timeline(self) -> List[Dict[str, Any]]:
        """
        High-level timeline: only AGENT_START / AGENT_COMPLETE / ROUTING_DECISION /
        TOOL_CALL / TOOL_RESULT / POLICY_VIOLATION events.
        """
        agent_events = {
            EventType.AGENT_START, EventType.AGENT_COMPLETE,
            EventType.ROUTING_DECISION, EventType.TOOL_CALL,
            EventType.TOOL_RESULT, EventType.POLICY_VIOLATION,
            EventType.PIPELINE_START, EventType.PIPELINE_DONE,
        }
        return [e.to_dict() for e in self._events if e.event_type in agent_events]


# ── Global Registry ───────────────────────────────────────────────────────────

_tracers: Dict[str, ExecutionTracer] = {}
_MAX_TRACERS = 1000  # LRU-style cap


def get_tracer(job_id: str) -> ExecutionTracer:
    """Get-or-create a tracer for this job_id."""
    if job_id not in _tracers:
        if len(_tracers) >= _MAX_TRACERS:
            # Evict oldest
            oldest = next(iter(_tracers))
            del _tracers[oldest]
        _tracers[job_id] = ExecutionTracer(job_id)
    return _tracers[job_id]


def list_jobs() -> List[str]:
    return list(_tracers.keys())
