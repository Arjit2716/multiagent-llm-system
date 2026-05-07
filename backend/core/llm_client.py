"""
LLM Client wrapper using LiteLLM for provider-agnostic completions.
Handles token counting, cost estimation, streaming, and fallback routing.
"""
import time
from typing import AsyncGenerator, Dict, List, Optional, Any
import litellm
from litellm import acompletion, token_counter

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.circuit_breaker import retry_with_backoff, get_circuit_breaker
from backend.core import metrics

logger = get_logger(__name__)

# Cost per 1K tokens (USD) - kept minimal, real values loaded at runtime
MODEL_COSTS = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "groq/llama3-8b-8192": {"input": 0.00005, "output": 0.00008},
    "groq/llama3-70b-8192": {"input": 0.00059, "output": 0.00079},
}

# Configure LiteLLM
litellm.set_verbose = settings.DEBUG
litellm.drop_params = True  # Drop unsupported params per model


class TokenBudgetExceeded(Exception):
    """Raised when a request would exceed the configured token budget."""
    pass


class LLMMessage:
    def __init__(self, role: str, content: str, name: Optional[str] = None):
        self.role = role
        self.content = content
        self.name = name

    def to_dict(self) -> Dict:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


class LLMClient:
    """
    Provider-agnostic LLM client with:
    - Automatic fallback routing
    - Token budget enforcement
    - Cost tracking
    - Streaming support
    - Circuit breaker integration
    """

    def __init__(
        self,
        model: str = None,
        agent_name: str = "unknown",
        token_budget: int = None,
        temperature: float = None,
    ):
        self.model = model or settings.DEFAULT_MODEL
        self.fallback_model = settings.FALLBACK_MODEL
        self.agent_name = agent_name
        self.token_budget = token_budget or settings.MAX_TOKENS
        self.temperature = temperature or settings.TEMPERATURE
        self._circuit_breaker = get_circuit_breaker(f"llm_{self.model}")

    def count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens for a message list using litellm's token counter."""
        try:
            return token_counter(model=self.model, messages=messages)
        except Exception:
            # Fallback: rough estimate
            total_chars = sum(len(m.get("content", "")) for m in messages)
            return total_chars // 4

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        costs = MODEL_COSTS.get(model, {"input": 0.001, "output": 0.002})
        return (prompt_tokens / 1000 * costs["input"]) + (completion_tokens / 1000 * costs["output"])

    def _record_metrics(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        status: str,
    ) -> None:
        metrics.llm_requests_total.labels(
            model=model, agent=self.agent_name, status=status
        ).inc()
        metrics.llm_latency_seconds.labels(
            model=model, agent=self.agent_name
        ).observe(latency)
        metrics.llm_tokens_used.labels(
            model=model, agent=self.agent_name, token_type="prompt"
        ).inc(prompt_tokens)
        metrics.llm_tokens_used.labels(
            model=model, agent=self.agent_name, token_type="completion"
        ).inc(completion_tokens)
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
        metrics.llm_cost_dollars.labels(
            model=model, agent=self.agent_name
        ).inc(cost)

    async def complete(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        use_fallback: bool = True,
    ) -> str:
        """
        Make a completion request with automatic fallback.
        Enforces token budget and records all metrics.
        """
        from backend.core.prompt_manager import prompt_manager
        
        full_messages = []
        if system_prompt:
            active_system_prompt = await prompt_manager.get_prompt(self.agent_name, system_prompt)
            full_messages.append({"role": "system", "content": active_system_prompt})
        full_messages.extend(messages)

        # Token budget check
        prompt_tokens = self.count_tokens(full_messages)
        max_out = max_tokens or min(self.token_budget - prompt_tokens, settings.MAX_TOKENS)
        if max_out <= 0:
            raise TokenBudgetExceeded(
                f"Prompt uses {prompt_tokens} tokens, exceeding budget of {self.token_budget}"
            )

        kwargs = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": max_out,
            "temperature": self.temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        async def _call(**kw):
            return await acompletion(**kw)

        start = time.monotonic()
        try:
            response = await retry_with_backoff(
                _call,
                circuit_breaker=self._circuit_breaker,
                **kwargs,
            )
            latency = time.monotonic() - start
            usage = response.usage
            self._record_metrics(
                self.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                latency,
                "success",
            )
            content = response.choices[0].message.content
            logger.debug(
                "llm_complete",
                model=self.model,
                agent=self.agent_name,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency=round(latency, 3),
            )
            return content
        except Exception as primary_error:
            latency = time.monotonic() - start
            self._record_metrics(self.model, prompt_tokens, 0, latency, "error")
            logger.warning(
                "llm_primary_failed",
                model=self.model,
                error=str(primary_error),
                using_fallback=use_fallback,
            )
            if not use_fallback or self.model == self.fallback_model:
                raise

            # Try fallback model
            kwargs["model"] = self.fallback_model
            fallback_cb = get_circuit_breaker(f"llm_{self.fallback_model}")
            start = time.monotonic()
            try:
                response = await retry_with_backoff(
                    _call, circuit_breaker=fallback_cb, **kwargs
                )
                latency = time.monotonic() - start
                usage = response.usage
                self._record_metrics(
                    self.fallback_model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    latency,
                    "fallback",
                )
                return response.choices[0].message.content
            except Exception as fallback_error:
                self._record_metrics(
                    self.fallback_model, prompt_tokens, 0, time.monotonic() - start, "error"
                )
                raise RuntimeError(
                    f"Both primary ({primary_error}) and fallback ({fallback_error}) failed"
                )

    async def stream(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream completions token by token, emitting trace events per chunk."""
        from backend.core.prompt_manager import prompt_manager
        from backend.core.execution_trace import get_tracer, EventType

        full_messages = []
        if system_prompt:
            active_system_prompt = await prompt_manager.get_prompt(self.agent_name, system_prompt)
            full_messages.append({"role": "system", "content": active_system_prompt})
        full_messages.extend(messages)

        prompt_tokens = self.count_tokens(full_messages)
        max_out = max_tokens or min(self.token_budget - prompt_tokens, settings.MAX_TOKENS)
        
        # Determine job_id from context (passed as hint in messages if available)
        job_id = next(
            (m.get("job_id", "") for m in messages if isinstance(m, dict) and "job_id" in m),
            ""
        )

        start = time.monotonic()
        try:
            response = await acompletion(
                model=self.model,
                messages=full_messages,
                max_tokens=max_out,
                temperature=self.temperature,
                stream=True,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
                    if job_id:
                        get_tracer(job_id).emit_token(delta.content, agent_id=self.agent_name)
        except Exception as e:
            logger.error("llm_stream_error", error=str(e))
            raise
        finally:
            latency = time.monotonic() - start
            metrics.llm_latency_seconds.labels(
                model=self.model, agent=self.agent_name
            ).observe(latency)
