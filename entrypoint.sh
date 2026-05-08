#!/bin/sh
# entrypoint.sh — dispatches to API server or Celery worker based on SERVICE_ROLE env var.
# No credentials are accepted here; all config comes from the container environment.
set -e

: "${SERVICE_ROLE:=api}"

echo "[entrypoint] SERVICE_ROLE=${SERVICE_ROLE}"

case "$SERVICE_ROLE" in
  api)
    echo "[entrypoint] Starting API server..."
    exec uvicorn backend.api.app:app \
      --host 0.0.0.0 \
      --port 8000 \
      --workers "${UVICORN_WORKERS:-2}" \
      --log-level "${LOG_LEVEL:-info}"
    ;;
  worker)
    echo "[entrypoint] Starting Celery worker..."
    exec celery -A backend.worker.celery_app worker \
      --loglevel="${LOG_LEVEL:-info}" \
      --concurrency="${CELERY_CONCURRENCY:-4}" \
      --queues="agent_jobs,eval_jobs" \
      --hostname="worker@%h"
    ;;
  *)
    echo "[entrypoint] ERROR: Unknown SERVICE_ROLE '${SERVICE_ROLE}'. Use 'api' or 'worker'."
    exit 1
    ;;
esac
