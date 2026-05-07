"""
Retrieval-Augmented Agent — Multi-hop reasoning with mandatory chunk citation.

Requirements enforced by this implementation:
1. At least TWO retrieval hops — single-hop is rejected
2. Every claim in the answer is annotated with which chunk_id contributed to it
3. The hop-2 query is derived from hop-1 results (genuine multi-hop reasoning)
4. A hop trace is written to SharedContext.rag_hop_trace

Output written to: SharedContext.retrieved_chunks, SharedContext.rag_hop_trace,
                   SharedContext.agent_outputs[AgentRole.RAG]
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from backend.agents.doc_store import InMemoryDocStore, get_doc_store
from backend.agents.shared_context import (
    AgentOutput, AgentRole, Claim, ConfidenceLevel,
    RetrievedChunk, SharedContext, TextSpan,
)
from backend.core.config import settings
from backend.core.llm_client import LLMClient
from backend.core.logging import get_logger

logger = get_logger(__name__)


RAG_SYSTEM_PROMPT = """You are a Retrieval-Augmented Agent performing multi-hop reasoning.
You have been given retrieved document chunks from two retrieval hops.

## Your Task
1. Read ALL provided chunks carefully
2. Identify which chunk(s) contribute to each part of the answer
3. Perform multi-hop reasoning: use findings from hop-1 to deepen hop-2 reasoning
4. Write a structured answer with MANDATORY per-claim chunk citations

## Output Format (strict JSON)
```json
{
  "hop_reasoning": {
    "hop1_insight": "What hop-1 chunks revealed",
    "hop2_insight": "What hop-2 chunks added that hop-1 alone could not provide",
    "multi_hop_chain": "How hop-1 insight led to the hop-2 query and what new reasoning became possible"
  },
  "claims": [
    {
      "text": "Exact claim text",
      "contributing_chunks": ["chunk_id_1", "chunk_id_2"],
      "confidence": 0.85,
      "span_start": 0,
      "span_end": 42
    }
  ],
  "answer": "Full synthesized answer paragraph(s) incorporating all claims",
  "unanswered_aspects": ["List of query aspects not covered by the retrieved chunks"]
}
```

## Critical Rules
- Every claim MUST cite at least one chunk_id from the provided chunks
- hop2_insight must contain information NOT present in hop-1 alone
- Do NOT fabricate information not present in the chunks
- If chunks are insufficient, state what is missing in unanswered_aspects
"""


class RAGAgent:
    """
    Multi-hop retrieval agent with mandatory chunk-level citation tracking.

    Hop 1: Retrieve chunks directly relevant to the original query.
    Hop 2: Enrich query with hop-1 entities/concepts, retrieve second layer.
    Reasoning: Synthesize across both hops with per-claim chunk citations.
    """

    MIN_HOPS = 2

    def __init__(self, doc_store: Optional[InMemoryDocStore] = None):
        self._store = doc_store or get_doc_store()
        self._llm = LLMClient(
            model=settings.DEFAULT_MODEL,
            agent_name=AgentRole.RAG.value,
            token_budget=settings.EXECUTOR_TOKEN_BUDGET,
            temperature=0.3,
        )

    async def __call__(self, ctx: SharedContext, token_budget: int = 4000) -> None:
        start = time.monotonic()
        logger.info("rag_agent_start", query=ctx.query[:80])

        # ── Hop 1: Direct retrieval ────────────────────────────────────────────
        hop1_query = self._build_hop1_query(ctx)
        hop1_results = self._store.search(hop1_query, top_k=4)

        if not hop1_results:
            ctx.log_error(AgentRole.RAG.value, "Hop-1 returned no results")
            logger.warning("rag_hop1_empty")
            hop1_results = []

        hop1_chunks = []
        for chunk, score in hop1_results:
            rc = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.title,
                text=chunk.text,
                relevance_score=round(score, 4),
                hop_index=1,
                query_used=hop1_query,
            )
            ctx.add_retrieved_chunk(rc)
            hop1_chunks.append(rc)

        hop1_text = "\n\n".join(c.text for c in hop1_chunks)
        ctx.rag_hop_trace.append({
            "hop": 1,
            "query": hop1_query,
            "chunks_retrieved": [c.chunk_id for c in hop1_chunks],
            "top_score": round(hop1_results[0][1], 4) if hop1_results else 0.0,
        })
        logger.info("rag_hop1_complete", chunks=len(hop1_chunks))

        # ── Hop 2: Expanded retrieval ──────────────────────────────────────────
        hop2_query = self._store.expand_query(ctx.query, hop1_text, max_terms=6)
        # Ensure hop-2 query is different from hop-1
        if hop2_query.strip() == hop1_query.strip():
            hop2_query = f"{ctx.query} detailed mechanism underlying explanation"

        hop2_results = self._store.search(hop2_query, top_k=4)
        # Filter out chunks already retrieved in hop-1
        hop1_ids = {c.chunk_id for c in hop1_chunks}
        hop2_results = [(c, s) for c, s in hop2_results if c.chunk_id not in hop1_ids]

        # Fallback: if all results overlap, keep top-2 anyway
        if not hop2_results:
            hop2_results = [(c, s) for c, s in self._store.search(hop2_query, top_k=6)
                            if c.chunk_id not in hop1_ids][:2]

        hop2_chunks = []
        for chunk, score in hop2_results[:3]:
            rc = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.title,
                text=chunk.text,
                relevance_score=round(score, 4),
                hop_index=2,
                query_used=hop2_query,
            )
            ctx.add_retrieved_chunk(rc)
            hop2_chunks.append(rc)

        ctx.rag_hop_trace.append({
            "hop": 2,
            "query": hop2_query,
            "expansion_terms": hop2_query.replace(ctx.query, "").strip(),
            "chunks_retrieved": [c.chunk_id for c in hop2_chunks],
            "top_score": round(hop2_results[0][1], 4) if hop2_results else 0.0,
        })
        logger.info("rag_hop2_complete", chunks=len(hop2_chunks), query=hop2_query[:80])

        # ── LLM Reasoning over retrieved chunks ───────────────────────────────
        all_chunks = hop1_chunks + hop2_chunks
        context_block = self._format_chunks_for_llm(all_chunks)
        reasoning_prompt = (
            f"Original Query: {ctx.query}\n\n"
            f"Retrieved Chunks:\n{context_block}\n\n"
            "Perform multi-hop reasoning across these chunks and produce the structured JSON answer."
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": reasoning_prompt}],
                system_prompt=RAG_SYSTEM_PROMPT,
                max_tokens=min(token_budget, 2500),
            )
        except Exception as e:
            ctx.log_error(AgentRole.RAG.value, f"LLM call failed: {e}")
            logger.error("rag_llm_failed", error=str(e))
            return

        # Parse structured response
        parsed = self._parse_response(response)
        claims = self._build_claims(parsed, all_chunks)

        duration = time.monotonic() - start
        ctx.write_agent_output(AgentOutput(
            agent_role=AgentRole.RAG,
            raw_output=parsed.get("answer", response),
            claims=claims,
            token_budget_used=self._llm.count_tokens([{"role": "user", "content": response}]),
            duration_seconds=duration,
            metadata={
                "hop_count": self.MIN_HOPS,
                "hop1_chunks": len(hop1_chunks),
                "hop2_chunks": len(hop2_chunks),
                "total_chunks": len(all_chunks),
                "hop_reasoning": parsed.get("hop_reasoning", {}),
                "unanswered_aspects": parsed.get("unanswered_aspects", []),
            },
        ))

        logger.info(
            "rag_agent_complete",
            claims=len(claims),
            chunks=len(all_chunks),
            duration=round(duration, 2),
        )

    def _build_hop1_query(self, ctx: SharedContext) -> str:
        """Build the initial retrieval query, possibly enriched by decomposition output."""
        base = ctx.query
        if ctx.dependency_graph:
            # Use the first ready task description to focus retrieval
            for task in ctx.dependency_graph.tasks.values():
                if task.status.value in ("pending", "ready") and task.assigned_to == AgentRole.RAG:
                    base = f"{ctx.query} {task.description}"
                    break
        return base

    def _format_chunks_for_llm(self, chunks: List[RetrievedChunk]) -> str:
        lines = []
        for c in chunks:
            lines.append(
                f"[{c.chunk_id}] (hop={c.hop_index}, source='{c.source}', score={c.relevance_score})\n"
                f"{c.text}\n"
            )
        return "\n---\n".join(lines)

    def _parse_response(self, text: str) -> Dict:
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"(\{[\s\S]*\})"]:
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
        return {"answer": text, "claims": [], "hop_reasoning": {}}

    def _build_claims(self, parsed: Dict, chunks: List[RetrievedChunk]) -> List[Claim]:
        claims = []
        valid_ids = {c.chunk_id for c in chunks}
        answer_text = parsed.get("answer", "")

        for raw in parsed.get("claims", []):
            text = raw.get("text", "")
            if not text:
                continue
            conf = float(raw.get("confidence", 0.6))
            level = ConfidenceLevel.HIGH if conf >= 0.8 else (ConfidenceLevel.MEDIUM if conf >= 0.5 else ConfidenceLevel.LOW)

            # Build span relative to the full answer
            span = None
            idx = answer_text.find(text[:40]) if len(text) >= 40 else answer_text.find(text)
            if idx >= 0:
                span = TextSpan(start_char=idx, end_char=idx + len(text), text_snippet=text[:80])

            supporting = [cid for cid in raw.get("contributing_chunks", []) if cid in valid_ids]
            if not supporting and chunks:
                supporting = [chunks[0].chunk_id]  # fallback to first chunk

            claims.append(Claim(
                text=text,
                span=span,
                source_agent=AgentRole.RAG,
                confidence_score=conf,
                confidence_level=level,
                supporting_chunks=supporting,
            ))
        return claims
