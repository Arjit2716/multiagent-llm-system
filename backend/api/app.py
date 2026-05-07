"""
FastAPI application - main entry point.
REST API + WebSocket + SSE for real-time multi-agent orchestration.
"""
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.logging import configure_logging, get_logger, set_correlation_id
from backend.core.metrics import get_metrics_output, CONTENT_TYPE_LATEST
from backend.core.circuit_breaker import get_all_circuit_breaker_metrics
from backend.agents.orchestrator import OrchestratorAgent
from backend.agents.planner import PlannerAgent
from backend.agents.executor import ExecutorAgent
from backend.agents.critic import CriticAgent
from backend.tools.registry import get_global_registry, register_default_tools
from backend.evaluation.eval_loop import SelfImprovingEvalLoop
from backend.evaluation.adversarial import AdversarialTester, AttackType
from backend.db.models import init_db, close_db, get_db, TaskRecord, EvaluationRecord, AdversarialTestRecord

configure_logging(log_level=settings.LOG_LEVEL, json_output=not settings.DEBUG)
logger = get_logger(__name__)

# ── Global system instances ───────────────────────────────────────────────────

orchestrator: Optional[OrchestratorAgent] = None
eval_loop: Optional[SelfImprovingEvalLoop] = None
adversarial_tester: Optional[AdversarialTester] = None
connected_websockets: List[WebSocket] = []


async def broadcast_event(event: Dict) -> None:
    """Broadcast an event to all connected WebSocket clients."""
    if not connected_websockets:
        return
    message = json.dumps({"timestamp": datetime.utcnow().isoformat(), **event})
    dead = []
    for ws in connected_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_websockets.remove(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup → yield → shutdown."""
    global orchestrator, eval_loop, adversarial_tester

    logger.info("system_starting", version=settings.APP_VERSION)

    # Initialize database
    try:
        await init_db()
    except Exception as e:
        logger.warning("db_init_failed", error=str(e))

    # Set up tool registry
    registry = get_global_registry()
    register_default_tools(registry)
    logger.info("tools_registered", count=len(registry))

    # Initialize agents
    planner = PlannerAgent()
    executor = ExecutorAgent(tool_registry=registry)
    critic = CriticAgent()

    orchestrator = OrchestratorAgent()
    orchestrator.register_agent("planner", planner)
    orchestrator.register_agent("executor", executor)
    orchestrator.register_agent("critic", critic)

    # Initialize evaluation loop
    eval_loop = SelfImprovingEvalLoop()

    # Initialize adversarial tester
    async def _agent_callable(prompt: str) -> str:
        result = await orchestrator._run_with_metrics(prompt)
        return str(result.output or "")

    adversarial_tester = AdversarialTester(agent_callable=_agent_callable)

    logger.info("system_ready")
    yield

    # Shutdown
    logger.info("system_shutting_down")
    try:
        await close_db()
    except Exception:
        pass


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade multi-agent LLM orchestration system with self-improving evaluation",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=10000, description="Task description")
    session_id: Optional[str] = Field(default=None)
    context: Optional[Dict] = Field(default=None)
    stream: bool = Field(default=False, description="Stream response via SSE")
    enable_eval: bool = Field(default=True, description="Run critic evaluation")


class TaskResponse(BaseModel):
    task_id: str
    status: str
    output: Optional[str] = None
    reasoning: Optional[str] = None
    tokens_used: int = 0
    duration_seconds: float = 0.0
    eval_score: Optional[float] = None
    tool_calls: List[Dict] = []
    eval_result: Optional[Dict] = None


class AdversarialTestRequest(BaseModel):
    attack_types: Optional[List[str]] = None
    severity_filter: Optional[str] = None
    run_async: bool = False


# ── Health & Observability ────────────────────────────────────────────────────

@app.get("/health", tags=["observability"])
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "agents": orchestrator.get_status_report() if orchestrator else None,
    }


@app.get("/metrics", tags=["observability"])
async def prometheus_metrics():
    """Expose Prometheus metrics."""
    return StreamingResponse(
        iter([get_metrics_output()]),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/api/v1/circuit-breakers", tags=["observability"])
async def circuit_breaker_status():
    return {"circuit_breakers": get_all_circuit_breaker_metrics()}


@app.get("/api/v1/tools", tags=["tools"])
async def list_tools():
    registry = get_global_registry()
    return {"tools": registry.list_tools(), "total": len(registry)}


# ── Core Agent API ────────────────────────────────────────────────────────────

@app.post("/api/v1/tasks", response_model=TaskResponse, tags=["agents"])
async def run_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a task to the multi-agent orchestrator.
    Returns structured results with optional evaluation scores.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    cid = set_correlation_id()
    session_id = request.session_id or str(uuid.uuid4())

    logger.info("task_received", task=request.task[:100], session_id=session_id, cid=cid)

    # Broadcast task start
    await broadcast_event({
        "type": "task_started",
        "task": request.task[:100],
        "session_id": session_id,
    })

    # Execute task
    result = await orchestrator._run_with_metrics(request.task, request.context)

    # Run evaluation if requested
    eval_result_dict = None
    if request.enable_eval and result.status == "success":
        critic = orchestrator._agent_registry.get("critic")
        if critic:
            eval_result = await critic.evaluate_output(request.task, result.output)
            eval_result_dict = eval_result.to_dict()
            result.eval_score = eval_result.overall_score

            # Record in eval loop
            if eval_loop:
                eval_loop.record_evaluation(
                    task=request.task,
                    output=str(result.output or ""),
                    score=eval_result.overall_score,
                    passed=eval_result.passed,
                    issues=eval_result.issues_found,
                    suggestions=eval_result.improvement_suggestions,
                    agent_name="orchestrator",
                )

    # Persist to database
    try:
        record = TaskRecord(
            session_id=session_id,
            task=request.task,
            status=result.status,
            output=str(result.output or ""),
            reasoning=result.reasoning,
            tokens_used=result.tokens_used,
            duration_seconds=result.duration_seconds,
            eval_score=result.eval_score,
            tool_calls=result.tool_calls,
            metadata_=result.metadata,
            completed_at=datetime.utcnow(),
        )
        db.add(record)
    except Exception as e:
        logger.warning("db_persist_failed", error=str(e))

    # Broadcast completion
    await broadcast_event({
        "type": "task_completed",
        "task_id": result.task_id,
        "status": result.status,
        "eval_score": result.eval_score,
        "tokens_used": result.tokens_used,
    })

    return TaskResponse(
        task_id=result.task_id,
        status=result.status,
        output=str(result.output) if result.output else None,
        reasoning=result.reasoning,
        tokens_used=result.tokens_used,
        duration_seconds=result.duration_seconds,
        eval_score=result.eval_score,
        tool_calls=result.tool_calls,
        eval_result=eval_result_dict,
    )


@app.post("/api/v1/tasks/stream", tags=["agents"])
async def stream_task(request: TaskRequest):
    """
    Stream a task response using Server-Sent Events (SSE).
    Tokens are emitted as they're generated.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'start', 'task': request.task[:100]})}\n\n"
            
            # Use planner first
            planner = orchestrator._agent_registry.get("planner")
            if planner:
                plan_result = await planner._run_with_metrics(request.task, request.context)
                yield f"data: {json.dumps({'type': 'plan', 'data': plan_result.output})}\n\n"

            # Stream from executor
            executor = orchestrator._agent_registry.get("executor")
            if executor:
                messages = [{"role": "user", "content": request.task}]
                async for chunk in executor.llm.stream(messages, system_prompt=executor.system_prompt):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Evaluation API ────────────────────────────────────────────────────────────

@app.get("/api/v1/eval/metrics", tags=["evaluation"])
async def get_eval_metrics():
    """Get current evaluation metrics and self-improvement status."""
    if not eval_loop:
        raise HTTPException(status_code=503, detail="Eval loop not initialized")
    return eval_loop.get_metrics_report()


@app.post("/api/v1/eval/improve", tags=["evaluation"])
async def trigger_improvement_cycle():
    """Manually trigger the self-improvement cycle."""
    if not eval_loop:
        raise HTTPException(status_code=503, detail="Eval loop not initialized")
    status = await eval_loop.run_improvement_cycle()
    return status


# ── Adversarial Testing API ───────────────────────────────────────────────────

@app.post("/api/v1/adversarial/run", tags=["adversarial"])
async def run_adversarial_tests(request: AdversarialTestRequest, background_tasks: BackgroundTasks):
    """
    Run adversarial robustness tests against the agent system.
    Can be run asynchronously for full suite.
    """
    if not adversarial_tester:
        raise HTTPException(status_code=503, detail="Adversarial tester not initialized")

    attack_type_filter = None
    if request.attack_types:
        try:
            attack_type_filter = [AttackType(t) for t in request.attack_types]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid attack type: {e}")

    if request.run_async:
        background_tasks.add_task(
            adversarial_tester.run_all_tests,
            attack_types=attack_type_filter,
            severity_filter=request.severity_filter,
        )
        return {"status": "started", "message": "Adversarial tests running in background"}

    report = await adversarial_tester.run_all_tests(
        attack_types=attack_type_filter,
        severity_filter=request.severity_filter,
    )
    return report


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming to the dashboard.
    Broadcasts: task_started, task_completed, eval_scores, agent_status.
    """
    await websocket.accept()
    connected_websockets.append(websocket)
    logger.info("websocket_connected", total=len(connected_websockets))

    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Connected to Multi-Agent System",
            "agents": orchestrator.get_status_report() if orchestrator else {},
            "eval_metrics": eval_loop.get_metrics_report() if eval_loop else {},
        }))

        # Keep connection alive and handle client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(data)
                
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif msg.get("type") == "get_status":
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "agents": orchestrator.get_status_report() if orchestrator else {},
                        "eval_metrics": eval_loop.get_metrics_report() if eval_loop else {},
                        "circuit_breakers": get_all_circuit_breaker_metrics(),
                    }))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("websocket_error", error=str(e))
    finally:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)
        logger.info("websocket_disconnected", total=len(connected_websockets))
