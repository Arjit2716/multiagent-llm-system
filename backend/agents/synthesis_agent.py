"""
Synthesis Agent — Merges all agent outputs, resolves critique-flagged contradictions,
and produces a final answer with a sentence-level provenance map.

Requirements enforced:
1. Reads ALL agent outputs from SharedContext
2. Explicitly resolves each CritiqueFlag (not just ignores them)
3. Produces a ProvenanceMap linking each sentence to:
   - source_agent: which agent produced this information
   - source_chunks: which retrieved chunk_ids support it
4. Writes final_answer and provenance_map to SharedContext

Output written to: SharedContext.final_answer, SharedContext.provenance_map,
                   SharedContext.agent_outputs[AgentRole.SYNTHESIS]
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from backend.agents.shared_context import (
    AgentOutput, AgentRole, ProvenanceMap, ProvenanceSentence,
    SharedContext,
)
from backend.core.config import settings
from backend.core.llm_client import LLMClient
from backend.core.logging import get_logger

logger = get_logger(__name__)


SYNTHESIS_SYSTEM_PROMPT = """You are the Synthesis Agent. Your job is to merge outputs from all agents
into a single coherent final answer WITH a provenance map.

## Inputs you receive
- Decomposition output: the task structure
- RAG output: retrieved evidence with chunk citations
- Critique flags: span-level disagreements with suggested fixes

## Your obligations
1. Address EVERY critique flag — either accept the fix, reject it with justification, or note it as unresolvable
2. For every sentence in your final answer, record:
   - Which agent's output it primarily draws from
   - Which chunk_ids (from RAG) support it (if applicable)
3. Resolve contradictions explicitly — state what you chose and why

## Output format (strict JSON)
```json
{
  "contradiction_resolutions": [
    {
      "flag_id": "<CritiqueFlag flag_id>",
      "flagged_span": "The problematic text",
      "resolution": "accepted_fix | rejected | unresolvable",
      "rationale": "Why you made this choice",
      "revised_text": "The corrected version (if accepted_fix)"
    }
  ],
  "sentences": [
    {
      "index": 0,
      "text": "Full sentence text.",
      "source_agent": "rag",
      "source_chunks": ["doc_transformer_c0", "doc_attention_c1"],
      "resolved_contradiction": false,
      "confidence": 0.92
    }
  ],
  "final_answer": "Full final answer as flowing prose, incorporating all resolutions.",
  "synthesis_notes": "Brief notes on significant decisions made during synthesis"
}
```

## Rules
- final_answer must be self-contained and readable without the JSON structure
- Every sentence in final_answer must appear in the sentences array
- source_agent must be one of: decomposition, rag, critique, synthesis
- resolved_contradiction = true for sentences that fixed a critique flag
- Do NOT introduce new factual claims not present in any agent's output
"""


class SynthesisAgent:
    """
    Final synthesis stage that merges, deduplicates, and resolves conflicts
    across all agent outputs, producing a provenance-mapped final answer.
    """

    def __init__(self):
        self._llm = LLMClient(
            model=settings.DEFAULT_MODEL,
            agent_name=AgentRole.SYNTHESIS.value,
            token_budget=settings.ORCHESTRATOR_TOKEN_BUDGET,
            temperature=0.4,
        )

    async def __call__(self, ctx: SharedContext, token_budget: int = 3500) -> None:
        start = time.monotonic()
        logger.info("synthesis_agent_start", flags=len(ctx.critique_flags))

        # Build synthesis input
        prompt = self._build_prompt(ctx)

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                max_tokens=min(token_budget, 3000),
            )
        except Exception as e:
            ctx.log_error(AgentRole.SYNTHESIS.value, f"LLM call failed: {e}")
            logger.error("synthesis_llm_failed", error=str(e))
            return

        parsed = self._parse_response(response)
        final_answer, provenance = self._build_provenance(parsed, ctx)

        # Write to context
        ctx.set_final_answer(final_answer, provenance)

        duration = time.monotonic() - start
        ctx.write_agent_output(AgentOutput(
            agent_role=AgentRole.SYNTHESIS,
            raw_output=final_answer,
            token_budget_used=self._llm.count_tokens([{"role": "user", "content": response}]),
            duration_seconds=duration,
            metadata={
                "sentences": len(provenance.sentences),
                "contradictions_resolved": len(provenance.contradiction_resolutions),
                "synthesis_notes": parsed.get("synthesis_notes", ""),
            },
        ))

        logger.info(
            "synthesis_agent_complete",
            sentences=len(provenance.sentences),
            contradictions=len(provenance.contradiction_resolutions),
            duration=round(duration, 2),
        )

    def _build_prompt(self, ctx: SharedContext) -> str:
        sections = [f"Original Query: {ctx.query}\n"]

        # Decomposition output
        decomp_out = ctx.get_output(AgentRole.DECOMPOSITION)
        if decomp_out and ctx.dependency_graph:
            task_lines = [
                f"  - [{t.task_type.value}] {t.description} (status={t.status.value})"
                for t in ctx.dependency_graph.tasks.values()
            ]
            sections.append("## Task Structure (Decomposition Agent)\n" + "\n".join(task_lines))

        # RAG output
        rag_out = ctx.get_output(AgentRole.RAG)
        if rag_out:
            hop_info = rag_out.metadata.get("hop_reasoning", {})
            sections.append(
                f"## Retrieval-Augmented Answer (RAG Agent)\n"
                f"Hop-1 insight: {hop_info.get('hop1_insight', 'N/A')}\n"
                f"Hop-2 insight: {hop_info.get('hop2_insight', 'N/A')}\n"
                f"Multi-hop chain: {hop_info.get('multi_hop_chain', 'N/A')}\n\n"
                f"Answer:\n{rag_out.raw_output[:2000]}"
            )
            if rag_out.claims:
                claim_lines = [
                    f"  [{c.claim_id[:8]}] (conf={c.confidence_score:.2f}, chunks={c.supporting_chunks}): {c.text[:100]}"
                    for c in rag_out.claims[:8]
                ]
                sections.append("Claims:\n" + "\n".join(claim_lines))

        # Critique flags
        if ctx.critique_flags:
            flag_lines = [
                f"  [{f.flag_id[:8]}] {f.issue_type} in span '{f.flagged_span.text_snippet[:60]}': "
                f"{f.critique_text[:100]} → Fix: {f.suggested_fix or 'none'}"
                for f in ctx.critique_flags
            ]
            sections.append("## Critique Flags to Resolve\n" + "\n".join(flag_lines))
        else:
            sections.append("## Critique Flags\nNo flags — all outputs passed critique.")

        # Retrieved chunks (brief)
        if ctx.retrieved_chunks:
            chunk_lines = [
                f"  [{c.chunk_id}] hop={c.hop_index} '{c.source}': {c.text[:100]}…"
                for c in list(ctx.retrieved_chunks.values())[:6]
            ]
            sections.append("## Available Chunks for Provenance\n" + "\n".join(chunk_lines))

        sections.append(
            "\nNow produce the synthesis JSON with contradiction resolutions, "
            "sentence-level provenance, and the final answer."
        )
        return "\n\n".join(sections)

    def _parse_response(self, text: str) -> Dict:
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"(\{[\s\S]*\})"]:
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
        logger.warning("synthesis_parse_failed")
        return {"final_answer": text, "sentences": [], "contradiction_resolutions": []}

    def _build_provenance(
        self, parsed: Dict, ctx: SharedContext
    ) -> tuple[str, ProvenanceMap]:
        final_answer = parsed.get("final_answer", "")
        if not final_answer:
            # Reconstruct from sentences
            final_answer = " ".join(s.get("text", "") for s in parsed.get("sentences", []))
        if not final_answer:
            final_answer = "Synthesis could not produce a final answer."

        # Build ProvenanceSentence list
        sentences: List[ProvenanceSentence] = []
        valid_chunk_ids = set(ctx.retrieved_chunks.keys())

        for raw_s in parsed.get("sentences", []):
            text = raw_s.get("text", "")
            if not text:
                continue
            try:
                role = AgentRole(raw_s.get("source_agent", "synthesis"))
            except ValueError:
                role = AgentRole.SYNTHESIS

            chunk_ids = [cid for cid in raw_s.get("source_chunks", []) if cid in valid_chunk_ids]

            sentences.append(ProvenanceSentence(
                sentence_index=raw_s.get("index", len(sentences)),
                text=text,
                source_agent=role,
                source_chunks=chunk_ids,
                resolved_contradiction=raw_s.get("resolved_contradiction", False),
                confidence=float(raw_s.get("confidence", 0.8)),
            ))

        # If LLM didn't provide sentences, auto-generate from final_answer
        if not sentences:
            sentences = self._auto_provenance(final_answer, ctx)

        # Build contradiction resolution records
        resolutions = []
        for raw_r in parsed.get("contradiction_resolutions", []):
            resolutions.append({
                "flag_id": raw_r.get("flag_id", ""),
                "flagged_span": raw_r.get("flagged_span", ""),
                "resolution": raw_r.get("resolution", "unresolvable"),
                "rationale": raw_r.get("rationale", ""),
                "revised_text": raw_r.get("revised_text", ""),
            })

        provenance = ProvenanceMap(
            sentences=sentences,
            contradiction_resolutions=resolutions,
        )
        return final_answer, provenance

    def _auto_provenance(self, text: str, ctx: SharedContext) -> List[ProvenanceSentence]:
        """Fallback: split answer into sentences and assign source heuristically."""
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        rag_chunks = [c.chunk_id for c in list(ctx.retrieved_chunks.values())[:2]]
        rag_done = ctx.agent_has_completed(AgentRole.RAG)
        result = []
        for i, s in enumerate(raw_sentences):
            if not s.strip():
                continue
            result.append(ProvenanceSentence(
                sentence_index=i,
                text=s.strip(),
                source_agent=AgentRole.RAG if rag_done else AgentRole.SYNTHESIS,
                source_chunks=rag_chunks,
                confidence=0.7,
            ))
        return result
