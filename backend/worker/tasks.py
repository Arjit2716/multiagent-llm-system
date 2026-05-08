"""
Celery tasks – background agent job processing.

All tasks are designed to be idempotent and safe to retry.
Results are stored in Redis with a 1-hour TTL.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional

from backend.worker.celery_app import app
from backend.core.logging import get_logger

logger = get_logger(__name__)


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@app.task(
    bind=True,
    name="backend.worker.tasks.run_agent_job",
    queue="agent_jobs",
    max_retries=2,
    default_retry_delay=5,
    acks_late=True,
)
def run_agent_job(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a multi-agent pipeline job asynchronously.

    Args:
        query:      The user query to run through the pipeline.
        session_id: Optional job ID; a UUID is minted if not supplied.

    Returns:
        Dict with job_id, status, final_answer, error_log.
    """
    job_id = session_id or str(uuid.uuid4())
    logger.info(f"[worker] Starting agent job {job_id}: {query[:80]}")

    async def _run():
        from backend.agents.shared_context import SharedContext
        from backend.api.orchestration_routes import _build_orchestrator
        from backend.core.execution_trace import get_tracer, EventType

        ctx = SharedContext(session_id=job_id, query=query)
        orchestrator = _build_orchestrator()
        try:
            await orchestrator.run(ctx)
            return {
                "job_id":       job_id,
                "status":       "complete",
                "final_answer": ctx.final_answer,
                "error_log":    ctx.error_log,
            }
        except Exception as exc:
            logger.error(f"[worker] Job {job_id} failed: {exc}")
            return {
                "job_id":  job_id,
                "status":  "failed",
                "error":   str(exc),
                "error_log": ctx.error_log,
            }

    try:
        result = _run_async(_run())
        logger.info(f"[worker] Job {job_id} finished with status={result['status']}")
        return result
    except Exception as exc:
        logger.error(f"[worker] Job {job_id} raised unhandled exception: {exc}")
        raise self.retry(exc=exc)


@app.task(
    bind=True,
    name="backend.worker.tasks.run_eval_suite",
    queue="eval_jobs",
    max_retries=1,
    default_retry_delay=10,
    acks_late=True,
    time_limit=600,      # 10-minute hard limit for full 15-case suite
    soft_time_limit=540,
)
def run_eval_suite(self) -> Dict[str, Any]:
    """
    Run the full 15-case evaluation harness as a background job.

    Triggered automatically after each pipeline run or on demand via
    POST /api/v1/eval/rerun.

    Returns:
        Dict with run_id, total_score, and per-case breakdown path.
    """
    logger.info("[worker] Starting eval harness suite")

    async def _run():
        from backend.db.models import init_db
        from backend.evaluation.eval_harness import HarnessEvaluator
        await init_db()
        evaluator = HarnessEvaluator()
        result = await evaluator.run_full_suite()
        return result

    try:
        result = _run_async(_run())
        logger.info(f"[worker] Eval suite done. Score={result.get('total_score', 'N/A')}")
        return result
    except Exception as exc:
        logger.error(f"[worker] Eval suite failed: {exc}")
        raise self.retry(exc=exc)
