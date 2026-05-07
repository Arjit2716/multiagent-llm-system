"""
Central configuration management using pydantic-settings.
All values can be overridden via environment variables.
"""
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Application ---
    APP_NAME: str = "MultiAgent LLM System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # --- API Keys ---
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)
    GROQ_API_KEY: Optional[str] = Field(default=None)
    GEMINI_API_KEY: Optional[str] = Field(default=None)

    # --- LLM Configuration ---
    DEFAULT_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str = "groq/llama3-8b-8192"
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.7
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    CIRCUIT_BREAKER_THRESHOLD: int = 5   # failures before tripping
    CIRCUIT_BREAKER_TIMEOUT: int = 60     # seconds before reset

    # --- Token Budget ---
    ORCHESTRATOR_TOKEN_BUDGET: int = 8000
    PLANNER_TOKEN_BUDGET: int = 4000
    EXECUTOR_TOKEN_BUDGET: int = 6000
    CRITIC_TOKEN_BUDGET: int = 3000

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://multiagent:multiagent@postgres:5432/multiagent_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # --- Redis ---
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_TTL: int = 3600  # seconds

    # --- Celery ---
    CELERY_BROKER_URL: str = "redis://redis:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/2"

    # --- Evaluation ---
    EVAL_THRESHOLD_SCORE: float = 0.7
    EVAL_MAX_ITERATIONS: int = 3
    ENABLE_ADVERSARIAL_TESTING: bool = True
    ADVERSARIAL_INJECTION_RATE: float = 0.1  # 10% of requests

    # --- Tool Execution ---
    TOOL_EXECUTION_TIMEOUT: int = 30  # seconds
    CODE_EXECUTION_TIMEOUT: int = 10  # seconds
    MAX_TOOL_RETRIES: int = 2

    # --- Monitoring ---
    PROMETHEUS_PORT: int = 8001
    ENABLE_TRACING: bool = True
    JAEGER_HOST: str = "jaeger"
    JAEGER_PORT: int = 6831

    # --- Security ---
    SECRET_KEY: str = "change-me-in-production-use-strong-secret"
    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # --- Rate Limiting ---
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
