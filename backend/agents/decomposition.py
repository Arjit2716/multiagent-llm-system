"""
Decomposition Agent — Breaks ambiguous queries into typed sub-tasks with
an explicit dependency graph (DAG). Dependent sub-tasks do not execute
until all their dependencies have resolved.

Output written to: SharedContext.dependency_graph
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from backend.agents.shared_context import (
    AgentOutput, AgentRole, DependencyGraph,
    SharedContext, SubTask, SubTaskStatus, SubTaskType,
)
from backend.core.config import settings
from backend.core.llm_client import LLMClient
from backend.core.logging import get_logger

logger = get_logger(__name__)


DECOMPOSITION_SYSTEM_PROMPT = """You are the Decomposition Agent.

Your job: analyse the input query and break it into typed sub-tasks with an explicit dependency graph.

## Sub-task types
- factual       — a fact that can be looked up
- reasoning     — multi-step inference, requires prior facts
- summarization — condensing retrieved information
- comparison    — comparing two or more things
- causal        — explaining cause-effect relationships
- procedural    — step-by-step how-to
- ambiguous     — needs clarification before answering

## Output schema (strict JSON, no prose)
```json
{
  "query_analysis": "Brief analysis of the query's ambiguity and structure",
  "subtasks": [
    {
      "task_id": "t1",
      "task_type": "factual",
      "description": "What exactly needs to be done",
      "depends_on": [],
      "assigned_to": "rag"
    },
    {
      "task_id": "t2",
      "task_type": "reasoning",
      "description": "Reasoning that requires t1 to be done first",
      "depends_on": ["t1"],
      "assigned_to": "rag"
    }
  ],
  "dependency_rationale": "Explain why the dependency relationships exist"
}
```

## Rules
1. task_ids must be unique strings (t1, t2, t3 …)
2. depends_on must reference previously defined task_ids only (no forward references)
3. The graph must be a DAG (no cycles)
4. assigned_to must be one of: decomposition, rag, critique, synthesis
5. At least 2 tasks required; at least 1 dependency edge required
6. For ambiguous queries, add a clarification sub-task first
"""


class DecompositionAgent:
    """
    Breaks the input query into a typed dependency graph of sub-tasks.

    The graph is written once to SharedContext.dependency_graph.
    The orchestrator then executes tasks in topological order,
    only unblocking a task when all its dependencies are DONE.
    """

    def __init__(self):
        self._llm = LLMClient(
            model=settings.DEFAULT_MODEL,
            agent_name=AgentRole.DECOMPOSITION.value,
            token_budget=settings.PLANNER_TOKEN_BUDGET,
            temperature=0.3,
        )

    async def __call__(self, ctx: SharedContext, token_budget: int = 2000) -> None:
        """Entry point called by the orchestrator."""
        start = time.monotonic()
        logger.info("decomposition_agent_start", query=ctx.query[:80])

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": f"Query to decompose:\n\n{ctx.query}"}],
                system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
                max_tokens=token_budget,
            )
        except Exception as e:
            ctx.log_error(AgentRole.DECOMPOSITION.value, f"LLM call failed: {e}")
            logger.error("decomposition_llm_failed", error=str(e))
            return

        # Parse the structured plan
        plan = self._parse_plan(response)
        if not plan:
            ctx.log_error(AgentRole.DECOMPOSITION.value, "Could not parse decomposition plan")
            # Fallback: single task, no dependencies
            plan = self._fallback_plan(ctx.query)

        # Build dependency graph
        graph = DependencyGraph()
        task_id_map: Dict[str, str] = {}  # local id (t1) → real uuid

        for raw_task in plan.get("subtasks", []):
            local_id = raw_task.get("task_id", str(uuid.uuid4()))
            try:
                task_type = SubTaskType(raw_task.get("task_type", "factual"))
            except ValueError:
                task_type = SubTaskType.FACTUAL

            try:
                assigned = AgentRole(raw_task.get("assigned_to", "rag"))
            except ValueError:
                assigned = AgentRole.RAG

            task = SubTask(
                task_id=str(uuid.uuid4()),
                task_type=task_type,
                description=raw_task.get("description", ""),
                depends_on=[],   # resolved after all tasks are created
                status=SubTaskStatus.PENDING,
                assigned_to=assigned,
            )
            task_id_map[local_id] = task.task_id
            graph.tasks[task.task_id] = task

        # Resolve depends_on to real UUIDs
        for raw_task in plan.get("subtasks", []):
            local_id = raw_task.get("task_id", "")
            real_id  = task_id_map.get(local_id)
            if not real_id:
                continue
            resolved_deps = []
            for dep_local in raw_task.get("depends_on", []):
                dep_real = task_id_map.get(dep_local)
                if dep_real:
                    resolved_deps.append(dep_real)
            graph.tasks[real_id].depends_on = resolved_deps

        # Validate: no cycles
        if graph.has_cycle():
            logger.warning("decomposition_cycle_detected", removing_all_deps=True)
            for t in graph.tasks.values():
                t.depends_on = []

        # Update status of immediately ready tasks
        completed: set = set()
        for task in graph.tasks.values():
            if task.is_ready(completed):
                task.status = SubTaskStatus.READY

        # Write to context
        ctx.set_dependency_graph(graph)

        duration = time.monotonic() - start
        ctx.write_agent_output(AgentOutput(
            agent_role=AgentRole.DECOMPOSITION,
            raw_output=response,
            token_budget_used=self._llm.count_tokens([{"role": "user", "content": response}]),
            duration_seconds=duration,
            metadata={
                "task_count": len(graph.tasks),
                "query_analysis": plan.get("query_analysis", ""),
                "dependency_rationale": plan.get("dependency_rationale", ""),
                "topological_order": graph.topological_order(),
            },
        ))

        logger.info(
            "decomposition_complete",
            tasks=len(graph.tasks),
            duration=round(duration, 2),
            order=graph.topological_order(),
        )

    def _parse_plan(self, text: str) -> Optional[Dict]:
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"(\{[\s\S]*\})"]:
            m = re.search(pattern, text)
            if m:
                try:
                    data = json.loads(m.group(1))
                    if "subtasks" in data and isinstance(data["subtasks"], list):
                        return data
                except json.JSONDecodeError:
                    continue
        return None

    def _fallback_plan(self, query: str) -> Dict:
        logger.warning("decomposition_using_fallback")
        return {
            "query_analysis": "Fallback: could not parse structured plan",
            "subtasks": [
                {"task_id": "t1", "task_type": "factual",   "description": f"Retrieve information about: {query[:200]}", "depends_on": [], "assigned_to": "rag"},
                {"task_id": "t2", "task_type": "reasoning",  "description": "Synthesize retrieved information",          "depends_on": ["t1"], "assigned_to": "rag"},
            ],
            "dependency_rationale": "Retrieval must precede synthesis",
        }
