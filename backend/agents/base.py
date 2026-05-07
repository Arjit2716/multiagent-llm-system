"""
Base Agent class defining the interface and shared behavior for all agents.
Implements ReAct (Reason + Act) loop, memory management, and telemetry.
"""
import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncGenerator

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.llm_client import LLMClient
from backend.core import metrics

logger = get_logger(__name__)


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentMessage:
    """Represents a message in the agent's conversation history."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    token_count: Optional[int] = None

    def to_dict(self) -> Dict:
        d = {"role": self.role, "content": self.content}
        if self.tool_name:
            d["name"] = self.tool_name
        return d


@dataclass
class AgentResult:
    """Structured result returned by every agent."""
    agent_name: str
    task_id: str
    status: str  # "success", "error", "timeout"
    output: Any
    reasoning: Optional[str] = None
    tool_calls: List[Dict] = field(default_factory=list)
    tokens_used: int = 0
    duration_seconds: float = 0.0
    eval_score: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "status": self.status,
            "output": self.output,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "duration_seconds": round(self.duration_seconds, 3),
            "eval_score": self.eval_score,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base for all agents in the system.
    
    Each agent has:
    - A dedicated LLM client with its own token budget
    - A conversation history (working memory)
    - Status tracking and metrics instrumentation
    - A ReAct loop implementation
    """

    def __init__(
        self,
        name: str,
        model: Optional[str] = None,
        token_budget: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        self.name = name
        self.agent_id = str(uuid.uuid4())
        self.status = AgentStatus.IDLE
        self.system_prompt = system_prompt or self.default_system_prompt()
        self.history: List[AgentMessage] = []
        self.token_budget = token_budget or settings.MAX_TOKENS
        self.llm = LLMClient(
            model=model or settings.DEFAULT_MODEL,
            agent_name=name,
            token_budget=self.token_budget,
            temperature=temperature,
        )
        from backend.core.context_manager import context_manager
        context_manager.declare_budget(self.agent_id, self.token_budget)
        
        self._start_time: Optional[float] = None
        metrics.active_agents.labels(agent_name=self.name).inc()
        logger.info("agent_initialized", name=self.name, agent_id=self.agent_id)

    def default_system_prompt(self) -> str:
        """Subclasses override this to provide role-specific system prompts."""
        return f"You are {self.name}, a helpful AI assistant."

    @abstractmethod
    async def execute(self, task: str, context: Optional[Dict] = None) -> AgentResult:
        """Execute a task and return structured results."""
        ...

    def add_message(self, role: str, content: str, tool_name: Optional[str] = None) -> None:
        """Add a message to working memory."""
        msg = AgentMessage(role=role, content=content, tool_name=tool_name)
        self.history.append(msg)
        # Agents should check remaining budget before adding, but if they ignore it and overflow, 
        # ContextBudgetManager will catch and log it during assemble_context.

    async def get_messages(self) -> List[Dict]:
        """Return history as list of dicts for LLM API."""
        from backend.core.context_manager import context_manager
        raw_msgs = [m.to_dict() for m in self.history]
        compressed_msgs = await context_manager.assemble_context(self.agent_id, raw_msgs)
        return compressed_msgs

    def clear_history(self) -> None:
        """Reset working memory."""
        self.history.clear()

    def set_status(self, status: AgentStatus) -> None:
        self.status = status
        logger.debug("agent_status_changed", name=self.name, status=status.value)

    async def _run_with_metrics(self, task: str, context: Optional[Dict] = None) -> AgentResult:
        """Wrapper that records timing and status metrics."""
        self.set_status(AgentStatus.RUNNING)
        self._start_time = time.monotonic()
        task_id = str(uuid.uuid4())

        try:
            result = await self.execute(task, context)
            result.task_id = task_id
            result.duration_seconds = time.monotonic() - self._start_time
            metrics.agent_tasks_total.labels(
                agent_name=self.name, status="success"
            ).inc()
            metrics.agent_task_duration.labels(agent_name=self.name).observe(
                result.duration_seconds
            )
            self.set_status(AgentStatus.COMPLETED)
            logger.info(
                "agent_task_complete",
                name=self.name,
                task_id=task_id,
                duration=round(result.duration_seconds, 2),
                tokens=result.tokens_used,
            )
            return result
        except Exception as e:
            duration = time.monotonic() - self._start_time
            metrics.agent_tasks_total.labels(
                agent_name=self.name, status="error"
            ).inc()
            self.set_status(AgentStatus.ERROR)
            logger.error("agent_task_failed", name=self.name, error=str(e), task_id=task_id)
            return AgentResult(
                agent_name=self.name,
                task_id=task_id,
                status="error",
                output=None,
                metadata={"error": str(e)},
                duration_seconds=duration,
            )
        finally:
            metrics.active_agents.labels(agent_name=self.name).dec()
            metrics.active_agents.labels(agent_name=self.name).inc()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} status={self.status.value}>"
