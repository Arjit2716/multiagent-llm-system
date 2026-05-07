"""
FastAPI routes for the dynamic multi-agent orchestration pipeline.
Exposes: run, status, session inspection, document ingestion.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from backend.agents.decomposition import DecompositionAgent
from backend.agents.dynamic_orchestrator import DynamicOrchestrator
from backend.agents.rag_agent import RAGAgent
from backend.agents.critique_agent import CritiqueAgent
from backend.agents.synthesis_agent import SynthesisAgent
from backend.agents.doc_store import get_doc_store, Document
from backend.agents.shared_context import AgentRole, SharedContext
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/orchestration", tags=["orchestration"])

# In-process session store (Redis in production)
_sessions: Dict[str, SharedContext] = {}


def _build_orchestrator() -> DynamicOrchestrator:
    """Construct and wire a fresh DynamicOrchestrator for each session."""
    orch = DynamicOrchestrator(max_steps=8)

    decomp   = DecompositionAgent()
    rag      = RAGAgent(doc_store=get_doc_store())
    critique = CritiqueAgent()
    synth    = SynthesisAgent()

    orch.register(AgentRole.DECOMPOSITION, decomp)
    orch.register(AgentRole.RAG,           rag)
    orch.register(AgentRole.CRITIQUE,      critique)
    orch.register(AgentRole.SYNTHESIS,     synth)

    return orch


# ── Request / Response Schemas ────────────────────────────────────────────────

class OrchestrationRequest(BaseModel):
    query:      str = Field(..., min_length=5, max_length=5000)
    session_id: Optional[str] = None


class OrchestrationResponse(BaseModel):
    session_id:        str
    status:            str
    final_answer:      Optional[str]
    routing_decisions: List[Dict]
    provenance_map:    Optional[Dict]
    critique_flags:    List[Dict]
    retrieved_chunks:  List[Dict]
    hop_trace:         List[Dict]
    error_log:         List[Dict]
    snapshot:          Dict


class DocIngestRequest(BaseModel):
    doc_id: str
    title:  str
    text:   str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/run", response_model=OrchestrationResponse)
async def run_orchestration(request: OrchestrationRequest):
    """
    Run the full dynamic multi-agent pipeline on a query.

    Pipeline (decided at runtime by the orchestrator):
    Decomposition → RAG (multi-hop) → Critique → Synthesis
    """
    session_id = request.session_id or str(uuid.uuid4())
    ctx = SharedContext(session_id=session_id, query=request.query)

    logger.info("orchestration_request", session_id=session_id, query=request.query[:80])

    orchestrator = _build_orchestrator()
    try:
        await orchestrator.run(ctx)
    except Exception as e:
        logger.error("orchestration_top_level_error", error=str(e))
        ctx.log_error("top_level", str(e))

    _sessions[session_id] = ctx

    return _context_to_response(ctx)


@router.get("/session/{session_id}", response_model=OrchestrationResponse)
async def get_session(session_id: str):
    """Retrieve a completed or in-progress orchestration session."""
    ctx = _sessions.get(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _context_to_response(ctx)


@router.get("/session/{session_id}/routing-log")
async def get_routing_log(session_id: str):
    """Return the full routing decision log for a session (audit trail)."""
    ctx = _sessions.get(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "decisions": [
            {
                "step":                  i + 1,
                "agent":                 d.selected_agent.value,
                "context_budget":        d.context_budget,
                "priority_score":        d.priority_score,
                "reasoning":             d.reasoning,
                "alternatives":          d.alternatives_considered,
                "preconditions":         d.preconditions,
                "timestamp":             d.timestamp.isoformat(),
            }
            for i, d in enumerate(ctx.routing_decisions)
        ],
    }


@router.get("/session/{session_id}/provenance")
async def get_provenance(session_id: str):
    """Return the sentence-level provenance map."""
    ctx = _sessions.get(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")
    if not ctx.provenance_map:
        raise HTTPException(status_code=404, detail="No provenance map — synthesis not complete")
    pm = ctx.provenance_map
    return {
        "session_id": session_id,
        "readable": pm.to_readable(),
        "sentences": [s.model_dump() for s in pm.sentences],
        "contradiction_resolutions": pm.contradiction_resolutions,
    }


@router.get("/session/{session_id}/critique")
async def get_critique(session_id: str):
    """Return all span-level critique flags for a session."""
    ctx = _sessions.get(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "flags": [
            {
                "flag_id":        f.flag_id,
                "target_agent":   f.target_agent.value,
                "issue_type":     f.issue_type,
                "flagged_span":   f.flagged_span.model_dump(),
                "critique_text":  f.critique_text,
                "confidence":     f.confidence_score,
                "suggested_fix":  f.suggested_fix,
            }
            for f in ctx.critique_flags
        ],
        "per_agent_confidence": ctx.per_agent_confidence,
    }


@router.get("/session/{session_id}/dependency-graph")
async def get_dependency_graph(session_id: str):
    """Return the decomposition agent's dependency graph."""
    ctx = _sessions.get(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")
    if not ctx.dependency_graph:
        raise HTTPException(status_code=404, detail="Decomposition not yet complete")
    graph = ctx.dependency_graph
    return {
        "session_id": session_id,
        "tasks": {
            tid: {
                "task_id":     t.task_id,
                "task_type":   t.task_type.value,
                "description": t.description,
                "depends_on":  t.depends_on,
                "status":      t.status.value,
                "assigned_to": t.assigned_to.value if t.assigned_to else None,
                "output":      t.output,
            }
            for tid, t in graph.tasks.items()
        },
        "topological_order": graph.topological_order(),
        "has_cycle": graph.has_cycle(),
    }


@router.post("/documents/ingest")
async def ingest_document(request: DocIngestRequest):
    """Ingest a document into the RAG agent's document store."""
    store = get_doc_store()
    doc = Document(doc_id=request.doc_id, title=request.title, text=request.text)
    n_chunks = store.ingest(doc)
    return {"doc_id": request.doc_id, "chunks_created": n_chunks, "total_chunks": len(store)}


@router.get("/documents/stats")
async def doc_stats():
    store = get_doc_store()
    return {"total_chunks": len(store)}


@router.get("/sessions")
async def list_sessions():
    return {
        "sessions": [
            {
                "session_id": sid,
                "query":      ctx.query[:100],
                "completed":  ctx.completed_agents,
                "has_answer": ctx.final_answer is not None,
                "flags":      len(ctx.critique_flags),
            }
            for sid, ctx in list(_sessions.items())[-20:]
        ]
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _context_to_response(ctx: SharedContext) -> OrchestrationResponse:
    return OrchestrationResponse(
        session_id=ctx.session_id,
        status="complete" if ctx.final_answer else "partial",
        final_answer=ctx.final_answer,
        routing_decisions=ctx.get_routing_summary(),
        provenance_map=(
            {
                "sentences": [s.model_dump() for s in ctx.provenance_map.sentences],
                "contradiction_resolutions": ctx.provenance_map.contradiction_resolutions,
                "readable": ctx.provenance_map.to_readable(),
            }
            if ctx.provenance_map else None
        ),
        critique_flags=[
            {
                "flag_id":       f.flag_id,
                "target_agent":  f.target_agent.value,
                "issue_type":    f.issue_type,
                "span":          f.flagged_span.text_snippet,
                "critique":      f.critique_text,
                "confidence":    f.confidence_score,
                "fix":           f.suggested_fix,
            }
            for f in ctx.critique_flags
        ],
        retrieved_chunks=[
            {
                "chunk_id":  c.chunk_id,
                "source":    c.source,
                "hop":       c.hop_index,
                "score":     c.relevance_score,
                "snippet":   c.text[:150],
            }
            for c in ctx.retrieved_chunks.values()
        ],
        hop_trace=ctx.rag_hop_trace,
        error_log=ctx.error_log,
        snapshot=ctx.snapshot(),
    )
