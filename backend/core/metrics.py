"""
Prometheus metrics registry for the multi-agent system.
Tracks LLM calls, agent performance, tool usage, and eval scores.
"""
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

# Create a custom registry to avoid conflicts in tests
REGISTRY = CollectorRegistry(auto_describe=True)

# ── LLM Call Metrics ──────────────────────────────────────────────────────────
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["model", "agent", "status"],
    registry=REGISTRY,
)

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM API response latency in seconds",
    ["model", "agent"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=REGISTRY,
)

llm_tokens_used = Counter(
    "llm_tokens_used_total",
    "Total tokens consumed",
    ["model", "agent", "token_type"],  # token_type: prompt/completion
    registry=REGISTRY,
)

llm_cost_dollars = Counter(
    "llm_cost_dollars_total",
    "Estimated LLM cost in USD",
    ["model", "agent"],
    registry=REGISTRY,
)

# ── Agent Metrics ─────────────────────────────────────────────────────────────
agent_tasks_total = Counter(
    "agent_tasks_total",
    "Total tasks handled per agent",
    ["agent_name", "status"],
    registry=REGISTRY,
)

agent_task_duration = Histogram(
    "agent_task_duration_seconds",
    "Time taken per agent task",
    ["agent_name"],
    buckets=[0.5, 1.0, 5.0, 15.0, 60.0, 180.0],
    registry=REGISTRY,
)

active_agents = Gauge(
    "active_agents",
    "Number of currently active agents",
    ["agent_name"],
    registry=REGISTRY,
)

# ── Tool Metrics ──────────────────────────────────────────────────────────────
tool_calls_total = Counter(
    "tool_calls_total",
    "Total tool invocations",
    ["tool_name", "status"],
    registry=REGISTRY,
)

tool_execution_duration = Histogram(
    "tool_execution_duration_seconds",
    "Tool execution time",
    ["tool_name"],
    buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 30.0],
    registry=REGISTRY,
)

# ── Evaluation Metrics ────────────────────────────────────────────────────────
eval_score = Histogram(
    "eval_score",
    "Evaluation scores from the critic agent",
    ["metric"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY,
)

eval_iterations_total = Counter(
    "eval_iterations_total",
    "Total evaluation iterations (self-improvement loop)",
    ["outcome"],  # outcome: passed, failed, improved
    registry=REGISTRY,
)

adversarial_attacks_total = Counter(
    "adversarial_attacks_total",
    "Total adversarial attack tests run",
    ["attack_type", "outcome"],  # outcome: detected, bypassed
    registry=REGISTRY,
)

# ── System Metrics ────────────────────────────────────────────────────────────
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["name"],
    registry=REGISTRY,
)

active_sessions = Gauge(
    "active_sessions_total",
    "Number of active user sessions",
    registry=REGISTRY,
)

memory_cache_hits = Counter(
    "memory_cache_hits_total",
    "Redis cache hit/miss counts",
    ["cache_type", "result"],  # result: hit, miss
    registry=REGISTRY,
)


def get_metrics_output() -> bytes:
    """Return Prometheus text format metrics."""
    return generate_latest(REGISTRY)
