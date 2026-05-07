"""
Critique Agent — Span-level review of every other agent's output.

Requirements enforced:
1. Reviews EACH agent's output individually
2. Assigns a per-CLAIM confidence score (not one score per agent)
3. Flags SPECIFIC TEXT SPANS it disagrees with, not the output as a whole
4. Flags carry issue_type, critique_text, and suggested_fix

Output written to: SharedContext.critique_flags, SharedContext.per_agent_confidence,
                   SharedContext.agent_outputs[AgentRole.CRITIQUE]
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from backend.agents.shared_context import (
    AgentOutput, AgentRole, Claim, ConfidenceLevel,
    CritiqueFlag, SharedContext, TextSpan,
)
from backend.core.config import settings
from backend.core.llm_client import LLMClient
from backend.core.logging import get_logger

logger = get_logger(__name__)


CRITIQUE_SYSTEM_PROMPT = """You are the Critique Agent. Your role is rigorous, span-level review.

## What you must do
1. Read the agent's output carefully
2. Identify specific SPANS of text (not the whole output) that have issues
3. For each span, assign a confidence score to the CLAIM made in that span
4. Categorize the issue type precisely

## Issue types
- factual_error      — the claim is demonstrably incorrect
- unsupported_claim  — claim made without evidence from retrieved chunks
- contradiction      — contradicts another part of the output or another agent's output
- vague              — too imprecise to be verifiable or useful
- hallucination      — confident statement about something not in the source chunks
- overclaiming       — exaggerates what the evidence actually supports

## Output format (strict JSON)
```json
{
  "agent_reviewed": "<agent role>",
  "overall_assessment": "Brief summary of quality",
  "flags": [
    {
      "flagged_text": "Exact substring from the output",
      "start_char": 42,
      "end_char": 98,
      "issue_type": "unsupported_claim",
      "confidence_score": 0.35,
      "critique_text": "Why this specific span is problematic",
      "suggested_fix": "How this could be corrected"
    }
  ],
  "per_claim_scores": [
    {
      "claim_snippet": "First 60 chars of the claim",
      "score": 0.85,
      "rationale": "Why this score"
    }
  ],
  "average_confidence": 0.72
}
```

## Rules
- Flag SPANS, not the overall output. Empty flags list means no issues found.
- Confidence scores: 0.0 = definitely wrong, 1.0 = highly confident and correct
- Be specific: reference the exact text you disagree with
- If the output is correct and well-supported, return an empty flags list
- Do NOT flag stylistic issues — only factual, logical, or evidential problems
"""


class CritiqueAgent:
    """
    Span-level critique agent that reviews every other agent's output.

    For each completed agent (decomposition, rag, etc.):
    - Extracts per-claim confidence scores
    - Identifies problematic text spans with specific issue types
    - Writes CritiqueFlags to SharedContext

    The Synthesis agent uses these flags to resolve contradictions.
    """

    def __init__(self):
        self._llm = LLMClient(
            model=settings.DEFAULT_MODEL,
            agent_name=AgentRole.CRITIQUE.value,
            token_budget=settings.CRITIC_TOKEN_BUDGET,
            temperature=0.15,   # Very low — consistency matters for critique
        )

    async def __call__(self, ctx: SharedContext, token_budget: int = 3000) -> None:
        start = time.monotonic()
        logger.info("critique_agent_start", agents_to_review=ctx.completed_agents)

        agents_to_review = [
            role for role in [AgentRole.DECOMPOSITION, AgentRole.RAG]
            if ctx.agent_has_completed(role)
        ]

        all_flags: List[CritiqueFlag] = []
        all_reviews: List[Dict] = []
        avg_confidences: Dict[str, float] = {}

        budget_per_agent = max(800, token_budget // max(len(agents_to_review), 1))

        for role in agents_to_review:
            output = ctx.get_output(role)
            if not output:
                continue

            review = await self._review_agent(ctx, role, output, budget_per_agent)
            if not review:
                continue

            all_reviews.append(review)
            agent_flags = self._build_flags(review, role, output.raw_output)
            for flag in agent_flags:
                ctx.add_critique_flag(flag)
            all_flags.extend(agent_flags)

            avg_conf = float(review.get("average_confidence", 0.7))
            ctx.per_agent_confidence[role.value] = avg_conf
            avg_confidences[role.value] = avg_conf

            logger.info(
                "critique_reviewed",
                agent=role.value,
                flags=len(agent_flags),
                avg_confidence=round(avg_conf, 3),
            )

        duration = time.monotonic() - start
        summary_text = self._build_summary(all_reviews, all_flags, avg_confidences)

        ctx.write_agent_output(AgentOutput(
            agent_role=AgentRole.CRITIQUE,
            raw_output=summary_text,
            token_budget_used=self._llm.count_tokens([{"role": "user", "content": summary_text}]),
            duration_seconds=duration,
            metadata={
                "agents_reviewed": [r.value for r in agents_to_review],
                "total_flags": len(all_flags),
                "per_agent_confidence": avg_confidences,
                "reviews": all_reviews,
            },
        ))

        logger.info(
            "critique_agent_complete",
            agents_reviewed=len(agents_to_review),
            total_flags=len(all_flags),
            duration=round(duration, 2),
        )

    async def _review_agent(
        self, ctx: SharedContext, role: AgentRole,
        output: AgentOutput, max_tokens: int
    ) -> Optional[Dict]:
        """Run LLM critique on a single agent's output."""
        # Build context: include retrieved chunks if reviewing RAG output
        chunk_context = ""
        if role == AgentRole.RAG and ctx.retrieved_chunks:
            chunks_sample = list(ctx.retrieved_chunks.values())[:4]
            chunk_context = "\n\nSource chunks (for fact-checking):\n" + "\n---\n".join(
                f"[{c.chunk_id}] {c.text[:300]}" for c in chunks_sample
            )

        prompt = (
            f"Review the following output from the '{role.value}' agent.\n\n"
            f"Original query: {ctx.query}\n\n"
            f"Agent output:\n{output.raw_output[:2500]}"
            f"{chunk_context}"
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=CRITIQUE_SYSTEM_PROMPT,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("critique_llm_failed", agent=role.value, error=str(e))
            ctx.log_error(AgentRole.CRITIQUE.value, f"Failed to review {role.value}: {e}")
            return None

        return self._parse_review(response)

    def _parse_review(self, text: str) -> Optional[Dict]:
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"(\{[\s\S]*\})"]:
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
        logger.warning("critique_parse_failed")
        return None

    def _build_flags(
        self, review: Dict, target_role: AgentRole, raw_output: str
    ) -> List[CritiqueFlag]:
        flags = []
        for raw_flag in review.get("flags", []):
            flagged_text = raw_flag.get("flagged_text", "")
            if not flagged_text:
                continue

            # Locate span in the raw output
            idx = raw_output.find(flagged_text[:60])
            if idx < 0:
                # Fuzzy: try first 30 chars
                idx = raw_output.find(flagged_text[:30])

            if idx >= 0:
                end = idx + len(flagged_text)
                span = TextSpan(
                    start_char=idx,
                    end_char=end,
                    text_snippet=flagged_text[:100],
                )
            else:
                # Use LLM-provided offsets as fallback
                span = TextSpan(
                    start_char=raw_flag.get("start_char", 0),
                    end_char=raw_flag.get("end_char", len(flagged_text)),
                    text_snippet=flagged_text[:100],
                )

            flags.append(CritiqueFlag(
                target_agent=target_role,
                flagged_span=span,
                issue_type=raw_flag.get("issue_type", "vague"),
                critique_text=raw_flag.get("critique_text", ""),
                confidence_score=float(raw_flag.get("confidence_score", 0.5)),
                suggested_fix=raw_flag.get("suggested_fix"),
            ))
        return flags

    def _build_summary(
        self,
        reviews: List[Dict],
        flags: List[CritiqueFlag],
        confidences: Dict[str, float],
    ) -> str:
        lines = ["=== Critique Summary ===\n"]
        for r in reviews:
            agent = r.get("agent_reviewed", "unknown")
            conf = confidences.get(agent, 0.0)
            assessment = r.get("overall_assessment", "")
            lines.append(f"Agent: {agent} | Avg Confidence: {conf:.2f}")
            lines.append(f"Assessment: {assessment}")
        lines.append(f"\nTotal Flags: {len(flags)}")
        for f in flags:
            lines.append(
                f"  [{f.issue_type}] {f.flagged_span.text_snippet[:60]} "
                f"(conf={f.confidence_score:.2f}) → {f.critique_text[:80]}"
            )
        return "\n".join(lines)
