"""
Dynamic Orchestrator — Runtime routing engine with structured reasoning.

The orchestrator does NOT follow a hardcoded chain. Instead, it:
1. Inspects the current SharedContext state
2. Reasons via LLM about which agent to invoke next and why
3. Allocates a context-window budget for that invocation
4. Logs the routing decision with full justification
5. Mediates the handoff — agents never call each other directly

Routing is driven by a structured reasoning prompt that considers:
- Which agents have already run
- What output exists in context
- What is still missing or needs critique
- Token budget remaining in the session
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from backend.agents.shared_context import (
    AgentRole, AgentOutput, RoutingDecision, SharedContext,
)
from backend.core.config import settings
from backend.core.llm_client import LLMClient
from backend.core.logging import get_logger
from backend.core import metrics

logger = get_logger(__name__)


# ── Routing Prompt ────────────────────────────────────────────────────────────

ORCHESTRATOR_ROUTING_PROMPT = """You are the Dynamic Orchestrator of a multi-agent reasoning pipeline.
Your job is to decide at runtime which agent to invoke next, based on the current pipeline state.

## Available Agents
| Agent | Role | When to invoke |
|---|---|---|
| decomposition | Breaks query into typed sub-tasks with a dependency DAG | Always first, unless already done |
| rag | Multi-hop retrieval + citation across document chunks | After decomposition; for factual/reasoning subtasks |
| critique | Per-claim confidence scoring + span-level disagreement flagging | After rag (and optionally after synthesis for re-review) |
| synthesis | Merges outputs, resolves contradictions, builds provenance map | After critique, as the final step |

## Current Pipeline State
{state_json}

## Remaining Token Budget
{budget_remaining} tokens

## Instructions
Analyze the current state and decide:
1. Which agent to invoke next (must be justified)
2. What token budget to allocate to it (must not exceed remaining budget)
3. What preconditions triggered this decision

Return ONLY valid JSON:
```json
{{
  "selected_agent": "<decomposition|rag|critique|synthesis>",
  "context_budget": <integer tokens to allocate>,
  "priority_score": <0.0-1.0 urgency>,
  "reasoning": "<detailed justification — be specific about what state triggered this>",
  "alternatives_considered": ["<agent>: <why not chosen>"],
  "preconditions": ["<state condition that led to this decision>"],
  "stop_pipeline": false
}}
```

If the pipeline is complete (synthesis done), set `stop_pipeline: true`.
"""


# ── Per-Agent Budget Defaults ─────────────────────────────────────────────────

DEFAULT_BUDGETS: Dict[AgentRole, int] = {
    AgentRole.DECOMPOSITION: 2000,
    AgentRole.RAG:           4000,
    AgentRole.CRITIQUE:      3000,
    AgentRole.SYNTHESIS:     3500,
}

# Maximum total tokens the orchestrator will spend in one pipeline run
SESSION_TOKEN_CAP = 20_000


class DynamicOrchestrator:
    """
    Runtime routing engine. No hardcoded sequence — every decision is
    made via LLM reasoning over the current SharedContext state.

    Key guarantees:
    - All routing decisions logged in SharedContext.routing_decisions
    - Agents are invoked via registered callables, never directly
    - Token budget is tracked and enforced across the session
    - Routing loop has a safety max-step guard
    """

    def __init__(
        self,
        max_steps: int = 10,
        session_token_cap: int = SESSION_TOKEN_CAP,
    ):
        self.max_steps = max_steps
        self.session_token_cap = session_token_cap
        self._tokens_used = 0

        # Agent registry: AgentRole → async callable(SharedContext) → None
        self._agent_registry: Dict[AgentRole, Callable] = {}

        self._llm = LLMClient(
            model=settings.DEFAULT_MODEL,
            agent_name="dynamic_orchestrator",
            token_budget=2000,
            temperature=0.2,
        )

    def register(self, role: AgentRole, fn: Callable) -> None:
        """Register an agent callable. The callable receives SharedContext and modifies it."""
        self._agent_registry[role] = fn
        logger.info("agent_registered_in_orchestrator", role=role.value)

    def _build_state_json(self, ctx: SharedContext) -> str:
        """Summarise current SharedContext state for the routing LLM."""
        state = {
            "completed_agents": ctx.completed_agents,
            "pending_agents": [r.value for r in AgentRole if r.value not in ctx.completed_agents and r != AgentRole.ORCHESTRATOR],
            "retrieved_chunks_count": len(ctx.retrieved_chunks),
            "critique_flags_count": len(ctx.critique_flags),
            "has_dependency_graph": ctx.dependency_graph is not None,
            "dependency_tasks": (
                {
                    tid: {
                        "type": t.task_type.value,
                        "status": t.status.value,
                        "depends_on": t.depends_on,
                    }
                    for tid, t in ctx.dependency_graph.tasks.items()
                }
                if ctx.dependency_graph else {}
            ),
            "has_final_answer": ctx.final_answer is not None,
            "error_log": ctx.error_log[-3:] if ctx.error_log else [],
            "per_agent_confidence": ctx.per_agent_confidence,
            "tokens_used_so_far": self._tokens_used,
            "tokens_remaining": self.session_token_cap - self._tokens_used,
        }
        return json.dumps(state, indent=2)

    async def _decide_next_agent(
        self, ctx: SharedContext
    ) -> Optional[RoutingDecision]:
        """
        Ask the LLM to decide which agent to invoke next.
        Returns a RoutingDecision or None if pipeline should stop.
        """
        budget_remaining = self.session_token_cap - self._tokens_used
        if budget_remaining <= 500:
            logger.warning("orchestrator_budget_exhausted", remaining=budget_remaining)
            return None

        state_json = self._build_state_json(ctx)
        prompt = ORCHESTRATOR_ROUTING_PROMPT.format(
            state_json=state_json,
            budget_remaining=budget_remaining,
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a meticulous multi-agent pipeline orchestrator. Your routing decisions must be grounded in the concrete pipeline state provided.",
                max_tokens=600,
            )
            self._tokens_used += 600  # conservative estimate
        except Exception as e:
            logger.error("orchestrator_routing_failed", error=str(e))
            ctx.log_error("orchestrator", f"Routing LLM call failed: {e}")
            return None

        # Parse routing decision
        decision_data = self._extract_json(response)
        if not decision_data or decision_data.get("stop_pipeline"):
            logger.info("orchestrator_stop_signal")
            return None

        try:
            role = AgentRole(decision_data["selected_agent"])
        except (KeyError, ValueError) as e:
            logger.error("orchestrator_invalid_agent", raw=decision_data, error=str(e))
            return None

        budget = min(
            int(decision_data.get("context_budget", DEFAULT_BUDGETS[role])),
            budget_remaining,
        )

        decision = RoutingDecision(
            selected_agent=role,
            context_budget=budget,
            reasoning=decision_data.get("reasoning", ""),
            alternatives_considered=decision_data.get("alternatives_considered", []),
            priority_score=float(decision_data.get("priority_score", 0.5)),
            preconditions=decision_data.get("preconditions", []),
        )
        ctx.append_routing_decision(decision)

        logger.info(
            "routing_decision_made",
            agent=role.value,
            budget=budget,
            priority=decision.priority_score,
            reasoning_excerpt=decision.reasoning[:120],
        )
        return decision

    async def run(self, ctx: SharedContext) -> SharedContext:
        """
        Main orchestration loop.

        1. Decide next agent via LLM reasoning
        2. Invoke the registered callable for that agent
        3. Track tokens and log decision
        4. Repeat until pipeline is complete or budget exhausted
        """
        logger.info("orchestration_started", session_id=ctx.session_id, query=ctx.query[:80])
        metrics.agent_tasks_total.labels(agent_name="dynamic_orchestrator", status="started").inc()

        step = 0
        while step < self.max_steps:
            step += 1

            # Safety: check if synthesis is done
            if ctx.agent_has_completed(AgentRole.SYNTHESIS) and ctx.final_answer:
                logger.info("orchestration_complete", steps=step, session=ctx.session_id)
                break

            decision = await self._decide_next_agent(ctx)
            if decision is None:
                logger.info("orchestration_halted", reason="stop_signal_or_budget", steps=step)
                break

            agent_fn = self._agent_registry.get(decision.selected_agent)
            if not agent_fn:
                ctx.log_error(
                    "orchestrator",
                    f"No callable registered for agent: {decision.selected_agent.value}",
                )
                logger.error("no_callable_for_agent", agent=decision.selected_agent.value)
                break

            logger.info(
                "invoking_agent",
                step=step,
                agent=decision.selected_agent.value,
                budget=decision.context_budget,
            )

            t0 = time.monotonic()
            try:
                await agent_fn(ctx, token_budget=decision.context_budget)
                duration = time.monotonic() - t0
                self._tokens_used += decision.context_budget
                logger.info(
                    "agent_invocation_complete",
                    agent=decision.selected_agent.value,
                    duration=round(duration, 2),
                    tokens_used=self._tokens_used,
                )
            except Exception as e:
                duration = time.monotonic() - t0
                ctx.log_error(decision.selected_agent.value, str(e))
                logger.error(
                    "agent_invocation_failed",
                    agent=decision.selected_agent.value,
                    error=str(e),
                )
                # Don't break — orchestrator may recover by routing to a different agent

        metrics.agent_tasks_total.labels(agent_name="dynamic_orchestrator", status="done").inc()
        logger.info("orchestration_finished", snapshot=ctx.snapshot())
        return ctx

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        import re
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"(\{[\s\S]*\})"]:
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
        return None
