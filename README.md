# 🧠 Real-Time Multi-Agent LLM Orchestration & Evaluation System

> **Production-grade multi-agent system** with self-improving evaluation loop, dynamic tool orchestration, and adversarial robustness testing.

[![Tests](https://github.com/Arjit2716/multiagent-llm-system/actions/workflows/test.yml/badge.svg)](https://github.com/Arjit2716/multiagent-llm-system/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](./docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│          API Gateway (FastAPI + WebSocket)        │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────▼───────────┐
        │   Orchestrator Agent  │  ← Task decomposition, routing
        └──┬────────┬───────┬──┘
           │        │       │
      ┌────▼──┐ ┌───▼───┐ ┌▼──────┐
      │Planner│ │Execut.│ │Critic │  ← Specialized agents
      └───────┘ └───┬───┘ └───────┘
                    │
          ┌─────────▼──────────┐
          │  Dynamic Tool Reg. │  ← Web search, code exec, calc, memory
          └────────────────────┘
                    │
          ┌─────────▼──────────┐
          │  Self-Improving    │  ← Eval loop, few-shot injection
          │  Evaluation Loop   │
          └────────────────────┘
                    │
          ┌─────────▼──────────┐
          │  Adversarial Tests │  ← Prompt injection, jailbreak, hallucination
          └────────────────────┘
```

## 🔑 Key Features

| Feature | Description |
|---|---|
| **Multi-Agent Orchestration** | Orchestrator → Planner → Executor → Critic pipeline |
| **ReAct Loop** | Reason + Act pattern in Planner agent |
| **Dynamic Tool Registry** | Hot-pluggable tools: web search, code exec, calculator, memory |
| **Self-Improving Eval Loop** | Sliding window quality monitoring → few-shot injection on degradation |
| **Adversarial Testing** | 15+ attack patterns: prompt injection, jailbreak, hallucination probes |
| **Circuit Breaker** | Exponential backoff + circuit breaker for LLM API failures |
| **Token Budget Enforcement** | Per-agent token limits with overflow protection |
| **LLM Fallback Routing** | Primary → fallback model (e.g. GPT-4o → Llama 3) |
| **Real-time Dashboard** | React + WebSocket + SSE streaming frontend |
| **Full Observability** | Prometheus metrics + Grafana dashboards |
| **Sandboxed Code Execution** | AST-level security analysis before subprocess execution |

## 🚀 Quick Start

### 1. Clone & Configure
```bash
git clone https://github.com/Arjit2716/multiagent-llm-system.git
cd multiagent-llm-system
cp .env.example .env
# Edit .env and add your API key(s)
```

### 2. Run with Docker Compose
```bash
docker-compose up -d
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API Docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin/admin) |

### 3. Local Dev (no Docker)
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn backend.api.app:app --reload --port 8000

# Frontend
cd frontend
npm install && npm run dev
```

## 📡 API Reference

### Submit a Task
```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Explain transformer attention and calculate parameters in a 7B model", "enable_eval": true}'
```

### Stream a Task (SSE)
```bash
curl -N http://localhost:8000/api/v1/tasks/stream \
  -H "Content-Type: application/json" \
  -d '{"task": "Search for quantum computing news"}'
```

### Run Adversarial Tests
```bash
curl -X POST http://localhost:8000/api/v1/adversarial/run \
  -H "Content-Type: application/json" \
  -d '{"attack_types": ["prompt_injection", "jailbreak"]}'
```

### Trigger Self-Improvement Cycle
```bash
curl -X POST http://localhost:8000/api/v1/eval/improve
```

## 🧪 Running Tests
```bash
pytest tests/ -v --cov=backend
```

## 🔒 Security Design
- **Sandboxed code execution**: AST-level import blocking, subprocess isolation
- **Circuit breaker**: Prevents cascade failures on LLM API outages
- **Adversarial detection**: 15+ attack patterns tested automatically
- **Token budget enforcement**: Prevents runaway token consumption
- **Input validation**: Pydantic schemas on all endpoints

## 📊 Observability
- **Prometheus metrics**: LLM latency, token usage, eval scores, circuit breaker state
- **Structured logging**: JSON logs with correlation IDs across all agents
- **WebSocket events**: Real-time task and agent status to dashboard

## 🏗️ Production Constraints Addressed
| Constraint | Solution |
|---|---|
| LLM API failures | Circuit breaker + exponential backoff + fallback model |
| Context window limits | Per-agent token budgets + history trimming |
| Cost management | Token counting + cost estimation per model |
| Hallucinations | Critic agent with LLM-as-Judge scoring |
| Prompt injection | Adversarial test suite + detection patterns |
| Latency | Streaming SSE + async throughout |
| Scalability | Celery task queue + Redis + PostgreSQL |

## License
MIT
