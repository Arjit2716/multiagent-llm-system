"""
Memory tools: In-memory key-value store backed by Redis for agent memory.
Allows agents to persist and retrieve information across steps.
"""
import json
import time
from typing import Any, Optional

from backend.tools.registry import BaseTool, ToolParameter, ToolResult
from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core import metrics

logger = get_logger(__name__)

# In-process fallback when Redis is unavailable
_local_memory: dict = {}


async def _get_redis():
    """Get Redis client, return None if unavailable."""
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await client.ping()
        return client
    except Exception:
        return None


class MemoryStoreTool(BaseTool):
    """
    Store information in agent memory (Redis-backed with local fallback).
    Memories expire after TTL seconds.
    """

    name = "memory_store"
    description = "Store a key-value pair in agent memory for later retrieval. Memories persist across agent steps."
    category = "memory"
    parameters = [
        ToolParameter("key", "str", "Memory key to store under", required=True),
        ToolParameter("value", "str", "Value to store", required=True),
        ToolParameter("ttl", "int", "Time-to-live in seconds (default: 3600)", required=False, default=3600),
        ToolParameter("namespace", "str", "Namespace for memory isolation", required=False, default="default"),
    ]

    async def run(self, key: str, value: str, ttl: int = 3600, namespace: str = "default") -> ToolResult:
        full_key = f"memory:{namespace}:{key}"
        payload = json.dumps({"value": value, "stored_at": time.time()})

        redis = await _get_redis()
        if redis:
            try:
                await redis.setex(full_key, ttl, payload)
                metrics.memory_cache_hits.labels(cache_type="redis", result="write").inc()
                await redis.aclose()
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output={"key": full_key, "stored": True, "backend": "redis"},
                )
            except Exception as e:
                logger.warning("redis_store_failed", error=str(e))
        
        # Local fallback
        _local_memory[full_key] = {"payload": payload, "expires_at": time.time() + ttl}
        metrics.memory_cache_hits.labels(cache_type="local", result="write").inc()
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"key": full_key, "stored": True, "backend": "local"},
        )


class MemoryRetrieveTool(BaseTool):
    """
    Retrieve information from agent memory by key.
    """

    name = "memory_retrieve"
    description = "Retrieve a previously stored value from agent memory by key."
    category = "memory"
    parameters = [
        ToolParameter("key", "str", "Memory key to retrieve", required=True),
        ToolParameter("namespace", "str", "Namespace for memory isolation", required=False, default="default"),
    ]

    async def run(self, key: str, namespace: str = "default") -> ToolResult:
        full_key = f"memory:{namespace}:{key}"

        redis = await _get_redis()
        if redis:
            try:
                raw = await redis.get(full_key)
                await redis.aclose()
                if raw:
                    data = json.loads(raw)
                    metrics.memory_cache_hits.labels(cache_type="redis", result="hit").inc()
                    return ToolResult(
                        tool_name=self.name,
                        success=True,
                        output={"key": full_key, "value": data["value"], "found": True},
                    )
                else:
                    metrics.memory_cache_hits.labels(cache_type="redis", result="miss").inc()
            except Exception as e:
                logger.warning("redis_retrieve_failed", error=str(e))

        # Local fallback
        entry = _local_memory.get(full_key)
        if entry and entry["expires_at"] > time.time():
            data = json.loads(entry["payload"])
            metrics.memory_cache_hits.labels(cache_type="local", result="hit").inc()
            return ToolResult(
                tool_name=self.name,
                success=True,
                output={"key": full_key, "value": data["value"], "found": True},
            )

        metrics.memory_cache_hits.labels(cache_type="local", result="miss").inc()
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"key": full_key, "value": None, "found": False},
            metadata={"message": f"No memory found for key: {full_key}"},
        )
