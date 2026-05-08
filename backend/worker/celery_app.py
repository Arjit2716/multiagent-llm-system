"""
Celery worker entry point.

Discovers all tasks under backend.worker.tasks and starts consuming
from the Redis queue. One worker process handles all agent jobs.

Start with:
  celery -A backend.worker.celery_app worker --loglevel=info --concurrency=4
"""
import os
from celery import Celery

# Config is driven entirely by environment variables – no hardcoding.
REDIS_URL        = os.environ.get("REDIS_URL",         "redis://redis:6379/0")
BROKER_URL       = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/1")
RESULT_BACKEND   = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/2")

app = Celery(
    "multiagent_worker",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["backend.worker.tasks"],
)

app.conf.update(
    task_serializer          = "json",
    result_serializer        = "json",
    accept_content           = ["json"],
    timezone                 = "UTC",
    enable_utc               = True,
    task_track_started       = True,
    task_acks_late           = True,           # re-queue on worker crash
    worker_prefetch_multiplier = 1,            # fair dispatch
    result_expires           = 3600,           # 1-hour result TTL
    task_routes              = {
        "backend.worker.tasks.run_agent_job": {"queue": "agent_jobs"},
        "backend.worker.tasks.run_eval_suite": {"queue": "eval_jobs"},
    },
    task_default_queue       = "agent_jobs",
)

if __name__ == "__main__":
    app.start()
