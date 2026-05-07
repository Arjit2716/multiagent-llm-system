"""
Shared Context Object — the single source of truth for all inter-agent communication.

ALL agents read from and write to this object exclusively.
Agents MUST NOT call each other directly; only the orchestrator mediates handoffs.

Schema design principles:
- Append-only writes per agent (no agent overwrites another agent's output)
- Immutable routing log (decisions are appended, never modified)
- Full provenance tracking from query → chunk → agent → sentence
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    ORCHESTRATOR   = "orchestrator"
    DECOMPOSITION  = "decomposition"
    RAG            = "rag"
    CRITIQUE       = "critique"
    SYNTHESIS      = "synthesis"


class SubTaskType(str, Enum):
    FACTUAL       = "factual"       # Verifiable fact retrieval
    REASONING     = "reasoning"     # Multi-step inference
    SUMMARIZATION = "summarization" # Condensing information
    COMPARISON    = "comparison"    # Contrasting two or more entities
    CAUSAL        = "causal"        # Why/because reasoning
    PROCEDURAL    = "procedural"    # How-to / step-by-step
    AMBIGUOUS     = "ambiguous"     # Needs clarification


class SubTaskStatus(str, Enum):
    PENDING   = "pending"    # Waiting for dependencies
    READY     = "ready"      # Dependencies resolved, can execute
    RUNNING   = "running"    # Currently being processed
    DONE      = "done"       # Completed successfully
    FAILED    = "failed"     # Execution failed


class ConfidenceLevel(str, Enum):
    HIGH   = "high"    # ≥ 0.8
    MEDIUM = "medium"  # 0.5–0.79
    LOW    = "low"     # < 0.5


# ── Atomic Data Models ────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single document chunk retrieved by the RAG agent."""
    chunk_id:      str = Field(default_factory=lambda: str(uuid.uuid4()))
    source:        str                            # Document title / URL / identifier
    text:          str                            # Raw chunk text
    relevance_score: float = 0.0                 # Similarity score
    hop_index:     int = 0                        # Which retrieval hop produced this
    query_used:    str = ""                       # The query that retrieved this chunk
    metadata:      Dict[str, Any] = Field(default_factory=dict)


class TextSpan(BaseModel):
    """Identifies a specific span of text within an agent's output."""
    start_char:   int
    end_char:     int
    text_snippet: str   # The actual substring (for human readability)


class Claim(BaseModel):
    """A single verifiable claim extracted from an agent output."""
    claim_id:         str = Field(default_factory=lambda: str(uuid.uuid4()))
    text:             str
    span:             Optional[TextSpan] = None
    source_agent:     AgentRole
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    supporting_chunks: List[str] = Field(default_factory=list)  # chunk_ids


class CritiqueFlag(BaseModel):
    """
    A specific disagreement raised by the Critique agent.
    Flags a TEXT SPAN, not the whole output — span-level granularity is required.
    """
    flag_id:          str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_agent:     AgentRole
    flagged_span:     TextSpan
    issue_type:       str   # "factual_error", "unsupported_claim", "contradiction", "vague"
    critique_text:    str   # Explanation of the problem
    confidence_score: float = Field(ge=0.0, le=1.0)
    suggested_fix:    Optional[str] = None


class SubTask(BaseModel):
    """
    A typed sub-task produced by the Decomposition agent.
    Dependencies form a DAG; a task is READY only when all its deps are DONE.
    """
    task_id:      str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type:    SubTaskType
    description:  str
    depends_on:   List[str] = Field(default_factory=list)  # task_ids this depends on
    status:       SubTaskStatus = SubTaskStatus.PENDING
    assigned_to:  Optional[AgentRole] = None
    output:       Optional[str] = None
    metadata:     Dict[str, Any] = Field(default_factory=dict)

    def is_ready(self, completed_ids: Set[str]) -> bool:
        """Returns True when all dependency tasks have completed."""
        return all(dep in completed_ids for dep in self.depends_on)


class RoutingDecision(BaseModel):
    """
    One routing decision made by the Dynamic Orchestrator.
    Records the reasoning behind the decision for auditability.
    """
    decision_id:      str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:        datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    selected_agent:   AgentRole
    context_budget:   int                       # Token budget allocated to this invocation
    reasoning:        str                       # LLM-generated justification
    alternatives_considered: List[str] = Field(default_factory=list)  # Other agents considered
    priority_score:   float = 0.0               # Orchestrator's internal priority for this agent
    preconditions:    List[str] = Field(default_factory=list)  # What state triggered this


class AgentOutput(BaseModel):
    """Structured output written by an agent into the shared context."""
    agent_role:   AgentRole
    produced_at:  datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_output:   str                           # The agent's full text response
    claims:       List[Claim] = Field(default_factory=list)
    token_budget_used: int = 0
    duration_seconds:  float = 0.0
    metadata:     Dict[str, Any] = Field(default_factory=dict)


class ProvenanceSentence(BaseModel):
    """Maps one sentence in the final answer to its source agent and chunks."""
    sentence_index: int
    text:           str
    source_agent:   AgentRole
    source_chunks:  List[str] = Field(default_factory=list)   # chunk_ids
    resolved_contradiction: bool = False                       # True if a flag was resolved here
    confidence:     float = Field(ge=0.0, le=1.0, default=0.8)


class ProvenanceMap(BaseModel):
    """Full provenance trace for the synthesis agent's final answer."""
    sentences:         List[ProvenanceSentence] = Field(default_factory=list)
    contradiction_resolutions: List[Dict[str, Any]] = Field(default_factory=list)

    def to_readable(self) -> str:
        lines = ["=== Provenance Map ==="]
        for s in self.sentences:
            chunks = ", ".join(s.source_chunks) if s.source_chunks else "none"
            lines.append(
                f"[{s.sentence_index}] Agent={s.source_agent.value} | "
                f"Chunks=[{chunks}] | Conf={s.confidence:.2f}"
                + (" [contradiction resolved]" if s.resolved_contradiction else "")
            )
        return "\n".join(lines)


# ── Dependency Graph ──────────────────────────────────────────────────────────

class DependencyGraph(BaseModel):
    """
    DAG of sub-tasks produced by the Decomposition agent.
    Provides topological ordering and readiness tracking.
    """
    tasks: Dict[str, SubTask] = Field(default_factory=dict)  # task_id → SubTask

    def add_task(self, task: SubTask) -> None:
        self.tasks[task.task_id] = task

    def get_ready_tasks(self) -> List[SubTask]:
        """Return tasks that are PENDING and have all dependencies DONE."""
        completed = {tid for tid, t in self.tasks.items() if t.status == SubTaskStatus.DONE}
        return [
            t for t in self.tasks.values()
            if t.status == SubTaskStatus.PENDING and t.is_ready(completed)
        ]

    def mark_done(self, task_id: str, output: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].status = SubTaskStatus.DONE
            self.tasks[task_id].output = output

    def mark_failed(self, task_id: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].status = SubTaskStatus.FAILED

    def all_done(self) -> bool:
        return all(t.status in (SubTaskStatus.DONE, SubTaskStatus.FAILED) for t in self.tasks.values())

    def topological_order(self) -> List[str]:
        """Kahn's algorithm for topological sort."""
        in_degree: Dict[str, int] = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.depends_on:
                if dep in in_degree:
                    in_degree[task.task_id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for tid, task in self.tasks.items():
                if node in task.depends_on:
                    in_degree[tid] -= 1
                    if in_degree[tid] == 0:
                        queue.append(tid)
        return order

    def has_cycle(self) -> bool:
        return len(self.topological_order()) != len(self.tasks)


# ── Master Shared Context ─────────────────────────────────────────────────────

class SharedContext(BaseModel):
    """
    THE shared communication object for the entire multi-agent pipeline.

    RULES enforced by schema:
    1. Agents write to their designated output slot only.
    2. Routing decisions are orchestrator-only, append-only.
    3. Retrieved chunks are written by the RAG agent and readable by all.
    4. Critique flags are written by the Critique agent only.
    5. The provenance map is written by the Synthesis agent only.
    """
    # Identity
    session_id:   str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at:   datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query:        str

    # Per-agent outputs (append-only by respective agent)
    agent_outputs: Dict[str, AgentOutput] = Field(default_factory=dict)  # AgentRole.value → output

    # Orchestrator routing log (append-only)
    routing_decisions: List[RoutingDecision] = Field(default_factory=list)

    # Decomposition agent output
    dependency_graph: Optional[DependencyGraph] = None

    # RAG agent output — chunks indexed by chunk_id
    retrieved_chunks: Dict[str, RetrievedChunk] = Field(default_factory=dict)
    rag_hop_trace:    List[Dict[str, Any]] = Field(default_factory=list)  # reasoning trace per hop

    # Critique agent output
    critique_flags: List[CritiqueFlag] = Field(default_factory=list)
    per_agent_confidence: Dict[str, float] = Field(default_factory=dict)  # agent → avg confidence

    # Synthesis agent output
    final_answer:   Optional[str] = None
    provenance_map: Optional[ProvenanceMap] = None

    # State flags
    completed_agents: List[str] = Field(default_factory=list)  # AgentRole.value list
    error_log:        List[Dict[str, Any]] = Field(default_factory=list)

    # ── Write Methods (agents call these, not direct attribute mutation) ──────

    def write_agent_output(self, output: AgentOutput) -> None:
        """Agent writes its output. Idempotent — overwrites on retry."""
        self.agent_outputs[output.agent_role.value] = output
        if output.agent_role.value not in self.completed_agents:
            self.completed_agents.append(output.agent_role.value)

    def append_routing_decision(self, decision: RoutingDecision) -> None:
        """Orchestrator appends a routing decision."""
        self.routing_decisions.append(decision)

    def add_retrieved_chunk(self, chunk: RetrievedChunk) -> None:
        """RAG agent adds a retrieved chunk."""
        self.retrieved_chunks[chunk.chunk_id] = chunk

    def add_critique_flag(self, flag: CritiqueFlag) -> None:
        """Critique agent adds a span-level flag."""
        self.critique_flags.append(flag)

    def set_dependency_graph(self, graph: DependencyGraph) -> None:
        """Decomposition agent writes the dependency graph (once)."""
        if self.dependency_graph is not None:
            raise ValueError("Dependency graph already set — cannot overwrite")
        self.dependency_graph = graph

    def set_final_answer(self, answer: str, provenance: ProvenanceMap) -> None:
        """Synthesis agent writes the final answer with provenance."""
        self.final_answer = answer
        self.provenance_map = provenance

    def log_error(self, agent: str, error: str) -> None:
        self.error_log.append({"agent": agent, "error": error, "at": datetime.now(timezone.utc).isoformat()})

    # ── Read Helpers ──────────────────────────────────────────────────────────

    def get_output(self, role: AgentRole) -> Optional[AgentOutput]:
        return self.agent_outputs.get(role.value)

    def get_flags_for_agent(self, role: AgentRole) -> List[CritiqueFlag]:
        return [f for f in self.critique_flags if f.target_agent == role]

    def get_chunks_by_hop(self, hop: int) -> List[RetrievedChunk]:
        return [c for c in self.retrieved_chunks.values() if c.hop_index == hop]

    def agent_has_completed(self, role: AgentRole) -> bool:
        return role.value in self.completed_agents

    def get_routing_summary(self) -> List[Dict]:
        return [
            {
                "step": i + 1,
                "agent": d.selected_agent.value,
                "reasoning": d.reasoning[:200],
                "budget": d.context_budget,
                "priority": d.priority_score,
            }
            for i, d in enumerate(self.routing_decisions)
        ]

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot for logging/debugging."""
        return {
            "session_id": self.session_id,
            "query": self.query[:200],
            "completed_agents": self.completed_agents,
            "routing_steps": len(self.routing_decisions),
            "retrieved_chunks": len(self.retrieved_chunks),
            "critique_flags": len(self.critique_flags),
            "has_final_answer": self.final_answer is not None,
            "errors": len(self.error_log),
        }
