"""
SQLAlchemy async database models and session management.
"""
import uuid
from datetime import datetime
from typing import Optional, AsyncGenerator

from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, Integer, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class TaskRecord(Base):
    """Persists task execution records."""
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    task: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    eval_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tool_calls: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class EvaluationRecord(Base):
    """Stores evaluation scores for analysis."""
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    overall_score: Mapped[float] = mapped_column(Float)
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    completeness: Mapped[float] = mapped_column(Float, default=0.0)
    coherence: Mapped[float] = mapped_column(Float, default=0.0)
    safety: Mapped[float] = mapped_column(Float, default=1.0)
    efficiency: Mapped[float] = mapped_column(Float, default=0.0)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    issues: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    suggestions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    agent_name: Mapped[str] = mapped_column(String(50), default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AdversarialTestRecord(Base):
    """Stores adversarial test results."""
    __tablename__ = "adversarial_tests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    test_id: Mapped[str] = mapped_column(String(20))
    attack_type: Mapped[str] = mapped_column(String(50))
    payload: Mapped[str] = mapped_column(Text)
    model_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    detected: Mapped[bool] = mapped_column(Boolean, default=False)
    bypassed: Mapped[bool] = mapped_column(Boolean, default=False)
    severity: Mapped[str] = mapped_column(String(20), default="medium")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvalHarnessRun(Base):
    """Stores a full execution of the evaluation harness."""
    __tablename__ = "eval_harness_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    total_score: Mapped[float] = mapped_column(Float, default=0.0)


class EvalHarnessTestCaseResult(Base):
    """Stores the result of a single test case within a harness run."""
    __tablename__ = "eval_harness_test_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    query: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50))
    exact_prompts: Mapped[dict] = mapped_column(JSON)
    exact_tool_calls: Mapped[dict] = mapped_column(JSON)
    exact_outputs: Mapped[dict] = mapped_column(JSON)
    
    answer_correctness: Mapped[float] = mapped_column(Float, default=0.0)
    answer_correctness_justification: Mapped[str] = mapped_column(Text, default="")
    
    citation_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    citation_accuracy_justification: Mapped[str] = mapped_column(Text, default="")
    
    contradiction_resolution: Mapped[float] = mapped_column(Float, default=0.0)
    contradiction_resolution_justification: Mapped[str] = mapped_column(Text, default="")
    
    tool_selection_efficiency: Mapped[float] = mapped_column(Float, default=0.0)
    tool_selection_efficiency_justification: Mapped[str] = mapped_column(Text, default="")
    
    context_budget_compliance: Mapped[float] = mapped_column(Float, default=0.0)
    context_budget_compliance_justification: Mapped[str] = mapped_column(Text, default="")
    
    critique_agreement_rate: Mapped[float] = mapped_column(Float, default=0.0)
    critique_agreement_rate_justification: Mapped[str] = mapped_column(Text, default="")
    
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ActivePrompt(Base):
    """Stores the currently active system prompt for each agent."""
    __tablename__ = "active_prompts"
    
    agent_name: Mapped[str] = mapped_column(String(50), primary_key=True)
    prompt_text: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PromptRewriteProposal(Base):
    """Stores meta-agent proposals for prompt improvement."""
    __tablename__ = "prompt_rewrite_proposals"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    eval_run_id: Mapped[str] = mapped_column(String(36), index=True)
    target_agent: Mapped[str] = mapped_column(String(50))
    failed_test_case_ids: Mapped[dict] = mapped_column(JSON) # List of test case IDs that triggered this
    
    original_prompt: Mapped[str] = mapped_column(Text)
    proposed_prompt: Mapped[str] = mapped_column(Text)
    diff_text: Mapped[str] = mapped_column(Text)
    justification: Mapped[str] = mapped_column(Text)
    
    status: Mapped[str] = mapped_column(String(20), default="pending") # pending, approved, rejected
    performance_delta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


# ── Database Engine Setup ─────────────────────────────────────────────────────

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        url = settings.DATABASE_URL
        is_sqlite = "sqlite" in url
        kwargs = {"echo": settings.DEBUG}
        if not is_sqlite:
            kwargs["pool_size"] = settings.DATABASE_POOL_SIZE
            kwargs["max_overflow"] = settings.DATABASE_MAX_OVERFLOW
        _engine = create_async_engine(url, **kwargs)
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized")


async def close_db() -> None:
    """Close database connections."""
    engine = get_engine()
    await engine.dispose()
    logger.info("database_closed")
