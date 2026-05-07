"""
Comprehensive test suite for the multi-agent system.
Tests: agents, tools, evaluation loop, adversarial testing, circuit breaker.
"""
import asyncio
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_response():
    """Factory for mock LLM responses."""
    def _make(content: str):
        mock = AsyncMock()
        mock.choices = [MagicMock(message=MagicMock(content=content))]
        mock.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        return mock
    return _make


@pytest.fixture
def sample_plan():
    return {
        "thought": "Analyzing the task",
        "reasoning": "Breaking into subtasks",
        "plan": [
            {"step": 1, "action": "Search for information", "tool": "web_search", "risk": "low"},
            {"step": 2, "action": "Summarize findings", "tool": None, "risk": "low"},
        ],
        "success_criteria": "Comprehensive answer provided",
        "estimated_tokens": 500,
    }


# ── Circuit Breaker Tests ──────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        from backend.core.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test_cb", failure_threshold=3, timeout=60)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        from backend.core.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test_cb2", failure_threshold=3, timeout=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_cannot_execute_when_open(self):
        from backend.core.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test_cb3", failure_threshold=1, timeout=60)
        cb.record_failure()
        assert not cb.can_execute()

    def test_resets_on_success(self):
        from backend.core.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test_cb4", failure_threshold=3, timeout=60)
        for _ in range(2):
            cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_retry_with_backoff_succeeds_on_second_attempt(self):
        from backend.core.circuit_breaker import retry_with_backoff
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient error")
            return "success"

        result = await retry_with_backoff(flaky_func, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_raises_when_open(self):
        from backend.core.circuit_breaker import (
            CircuitBreaker, CircuitBreakerOpen, retry_with_backoff
        )
        cb = CircuitBreaker("test_open", failure_threshold=1, timeout=60)
        cb.record_failure()

        async def func():
            return "result"

        with pytest.raises(CircuitBreakerOpen):
            await retry_with_backoff(func, circuit_breaker=cb)


# ── Calculator Tool Tests ──────────────────────────────────────────────────────

class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        from backend.tools.calculator import CalculatorTool
        tool = CalculatorTool()
        result = await tool.run(expression="2 + 3 * 4")
        assert result.success
        assert result.output["result"] == 14

    @pytest.mark.asyncio
    async def test_square_root(self):
        from backend.tools.calculator import CalculatorTool
        tool = CalculatorTool()
        result = await tool.run(expression="sqrt(16)")
        assert result.success
        assert result.output["result"] == 4.0

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        from backend.tools.calculator import CalculatorTool
        tool = CalculatorTool()
        result = await tool.run(expression="1 / 0")
        assert not result.success
        assert "zero" in result.error.lower()

    @pytest.mark.asyncio
    async def test_trig_functions(self):
        from backend.tools.calculator import CalculatorTool
        import math
        tool = CalculatorTool()
        result = await tool.run(expression="sin(pi)")
        assert result.success
        assert abs(result.output["result"]) < 1e-10  # sin(π) ≈ 0

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        from backend.tools.calculator import CalculatorTool
        tool = CalculatorTool()
        result = await tool.run(expression="log10(1000) + sqrt(144) / 3")
        assert result.success
        assert abs(result.output["result"] - 7.0) < 0.001

    @pytest.mark.asyncio
    async def test_rejects_huge_exponents(self):
        from backend.tools.calculator import CalculatorTool
        tool = CalculatorTool()
        result = await tool.run(expression="2 ** 10000")
        assert not result.success


# ── Code Executor Security Tests ──────────────────────────────────────────────

class TestCodeExecutorSecurity:
    def test_blocks_os_import(self):
        from backend.tools.code_executor import check_code_safety
        error = check_code_safety("import os; os.system('rm -rf /')")
        assert error is not None
        assert "os" in error.lower()

    def test_blocks_subprocess(self):
        from backend.tools.code_executor import check_code_safety
        error = check_code_safety("import subprocess; subprocess.call(['ls'])")
        assert error is not None

    def test_blocks_eval(self):
        from backend.tools.code_executor import check_code_safety
        error = check_code_safety("eval('print(1)')")
        assert error is not None
        assert "eval" in error.lower()

    def test_blocks_exec(self):
        from backend.tools.code_executor import check_code_safety
        error = check_code_safety("exec('import os')")
        assert error is not None

    def test_allows_math_import(self):
        from backend.tools.code_executor import check_code_safety
        error = check_code_safety("import math; print(math.pi)")
        assert error is None

    def test_blocks_socket(self):
        from backend.tools.code_executor import check_code_safety
        error = check_code_safety("import socket; s = socket.socket()")
        assert error is not None

    def test_blocks_dunder_globals(self):
        from backend.tools.code_executor import check_code_safety
        error = check_code_safety("x = ().__class__.__bases__[0].__subclasses__()")
        assert error is not None

    @pytest.mark.asyncio
    async def test_executes_safe_code(self):
        from backend.tools.code_executor import CodeExecutorTool
        tool = CodeExecutorTool()
        result = await tool.run(code="print(2 + 2)", timeout=5)
        assert result.success
        assert "4" in result.output

    @pytest.mark.asyncio
    async def test_blocks_unsafe_code_at_runtime(self):
        from backend.tools.code_executor import CodeExecutorTool
        tool = CodeExecutorTool()
        result = await tool.run(code="import os; os.getcwd()")
        assert not result.success


# ── Adversarial Detection Tests ───────────────────────────────────────────────

class TestAdversarialDetection:
    def test_prompt_injection_pattern_loaded(self):
        from backend.evaluation.adversarial import PROMPT_INJECTION_TESTS
        assert len(PROMPT_INJECTION_TESTS) > 0

    def test_jailbreak_tests_loaded(self):
        from backend.evaluation.adversarial import JAILBREAK_TESTS
        assert len(JAILBREAK_TESTS) > 0

    def test_hallucination_tests_loaded(self):
        from backend.evaluation.adversarial import HALLUCINATION_TESTS
        assert len(HALLUCINATION_TESTS) > 0

    def test_tester_detects_injection_in_response(self):
        """Test that the tester correctly identifies injected responses."""
        from backend.evaluation.adversarial import (
            AdversarialTester, AdversarialTest, AttackType
        )
        tester = AdversarialTester(agent_callable=AsyncMock())
        test = AdversarialTest(
            id="test_001",
            attack_type=AttackType.PROMPT_INJECTION,
            payload="ignore instructions",
            description="test",
            expected_behavior="ignore_injection",
            detection_patterns=[r"HACKED"],
        )
        # If response contains "HACKED" → attack bypassed
        detected, bypassed = tester._check_response_for_attack("I have been HACKED!", test)
        assert not detected
        assert bypassed

    def test_tester_detects_clean_response(self):
        from backend.evaluation.adversarial import (
            AdversarialTester, AdversarialTest, AttackType
        )
        tester = AdversarialTester(agent_callable=AsyncMock())
        test = AdversarialTest(
            id="test_002",
            attack_type=AttackType.PROMPT_INJECTION,
            payload="ignore instructions",
            description="test",
            expected_behavior="ignore_injection",
            detection_patterns=[r"HACKED"],
        )
        # Clean response → attack detected (handled)
        detected, bypassed = tester._check_response_for_attack(
            "I cannot follow instructions that ask me to override my guidelines.", test
        )
        assert detected
        assert not bypassed

    def test_jailbreak_refusal_detected(self):
        from backend.evaluation.adversarial import (
            AdversarialTester, AdversarialTest, AttackType
        )
        tester = AdversarialTester(agent_callable=AsyncMock())
        test = AdversarialTest(
            id="test_003",
            attack_type=AttackType.JAILBREAK,
            payload="roleplay harmful",
            description="test",
            expected_behavior="refuse",
            detection_patterns=[],
        )
        detected, bypassed = tester._check_response_for_attack(
            "I cannot and will not provide that information.", test
        )
        assert detected


# ── Eval Loop Tests ───────────────────────────────────────────────────────────

class TestSelfImprovingEvalLoop:
    def test_records_evaluation(self):
        from backend.evaluation.eval_loop import SelfImprovingEvalLoop
        loop = SelfImprovingEvalLoop(window_size=10)
        loop.record_evaluation("task", "output", 0.8, True, [], [], "agent")
        assert len(loop._eval_window) == 1
        assert len(loop._all_entries) == 1

    def test_detects_degradation(self):
        from backend.evaluation.eval_loop import SelfImprovingEvalLoop
        loop = SelfImprovingEvalLoop(window_size=10, degradation_threshold=0.7)
        for _ in range(5):
            loop.record_evaluation("task", "output", 0.4, False, ["issue"], [], "agent")
        assert loop.is_degrading()

    def test_no_degradation_with_high_scores(self):
        from backend.evaluation.eval_loop import SelfImprovingEvalLoop
        loop = SelfImprovingEvalLoop(window_size=10, degradation_threshold=0.7)
        for _ in range(5):
            loop.record_evaluation("task", "output", 0.9, True, [], [], "agent")
        assert not loop.is_degrading()

    def test_tracks_best_examples(self):
        from backend.evaluation.eval_loop import SelfImprovingEvalLoop
        loop = SelfImprovingEvalLoop(window_size=10, improvement_threshold=0.8)
        loop.record_evaluation("task1", "output1", 0.95, True, [], [], "agent")
        loop.record_evaluation("task2", "output2", 0.50, False, ["issue"], [], "agent")
        assert len(loop._best_examples) == 1
        assert loop._best_examples[0].score == 0.95

    def test_identifies_common_issues(self):
        from backend.evaluation.eval_loop import SelfImprovingEvalLoop
        loop = SelfImprovingEvalLoop(window_size=10)
        for i in range(5):
            loop.record_evaluation("task", "output", 0.4, False, ["accuracy issue", "verbose"], [], "agent")
        issues = loop.get_common_issues(n=2)
        assert len(issues) <= 2

    def test_metrics_report_structure(self):
        from backend.evaluation.eval_loop import SelfImprovingEvalLoop
        loop = SelfImprovingEvalLoop()
        loop.record_evaluation("task", "output", 0.8, True, [], [], "agent")
        report = loop.get_metrics_report()
        assert "total_evaluations" in report
        assert "pass_rate" in report
        assert "avg_score" in report


# ── Tool Registry Tests ───────────────────────────────────────────────────────

class TestToolRegistry:
    def test_register_and_retrieve(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        
        tool = registry.get_tool("calculator")
        assert tool is not None
        assert tool.name == "calculator"

    def test_list_tools_returns_schemas(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "calculator"
        assert "description" in tools[0]

    def test_unregister_tool(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool
        
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.unregister("calculator")
        
        assert registry.get_tool("calculator") is None
        assert len(registry) == 0

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self):
        from backend.tools.registry import ToolRegistry
        registry = ToolRegistry()
        result = await registry.execute_tool("nonexistent_tool", {})
        assert not result.success
        assert "not found" in result.error.lower()

    def test_tool_validation_catches_missing_params(self):
        from backend.tools.calculator import CalculatorTool
        tool = CalculatorTool()
        error = tool.validate_inputs({})  # Missing 'expression'
        assert error is not None
        assert "expression" in error


# ── Planner Agent Tests ───────────────────────────────────────────────────────

class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_parse_valid_json_plan(self):
        from backend.agents.planner import PlannerAgent
        
        with patch.object(PlannerAgent, '__init__', lambda self: None):
            agent = PlannerAgent.__new__(PlannerAgent)
            
            json_response = json.dumps({
                "thought": "Breaking down the task",
                "reasoning": "Need to search first",
                "plan": [{"step": 1, "action": "Search", "risk": "low"}],
                "success_criteria": "Done",
                "estimated_tokens": 100,
            })
            plan = agent._parse_plan(json_response)
            assert "plan" in plan
            assert len(plan["plan"]) == 1

    @pytest.mark.asyncio
    async def test_parse_json_in_codeblock(self):
        from backend.agents.planner import PlannerAgent
        
        with patch.object(PlannerAgent, '__init__', lambda self: None):
            agent = PlannerAgent.__new__(PlannerAgent)
            
            response = """
Here is my plan:
```json
{"thought": "test", "plan": [{"step": 1, "action": "test", "risk": "low"}]}
```
"""
            plan = agent._parse_plan(response)
            assert "plan" in plan

    @pytest.mark.asyncio
    async def test_fallback_plan_on_parse_error(self):
        from backend.agents.planner import PlannerAgent
        
        with patch.object(PlannerAgent, '__init__', lambda self: None):
            agent = PlannerAgent.__new__(PlannerAgent)
            
            response = "1. First step\n2. Second step"
            plan = agent._parse_plan(response)
            assert "plan" in plan
            assert len(plan["plan"]) > 0


# ── Integration Test ──────────────────────────────────────────────────────────

class TestIntegration:
    """
    Integration tests that test the full pipeline with mocked LLM calls.
    These don't require actual API keys.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mock_llm(self):
        """Test the full orchestrator → planner → executor → critic pipeline."""
        from backend.agents.orchestrator import OrchestratorAgent
        from backend.agents.planner import PlannerAgent
        from backend.agents.executor import ExecutorAgent
        from backend.agents.critic import CriticAgent
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool

        plan_json = json.dumps({
            "reasoning": "Simple calculation task",
            "plan": [{"step": 1, "agent": "executor", "task": "Calculate 2+2"}],
            "synthesis_strategy": "Direct answer",
            "estimated_complexity": "low",
        })

        eval_json = json.dumps({
            "scores": {"accuracy": 0.9, "completeness": 0.9, "coherence": 0.9, "safety": 1.0, "efficiency": 0.9},
            "overall_score": 0.92,
            "passed": True,
            "issues_found": [],
            "strengths": ["Accurate"],
            "improvement_suggestions": [],
            "reasoning": "Good response",
        })

        with patch("backend.core.llm_client.acompletion") as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="The answer is 4"))]
            mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
            mock_completion.return_value = mock_response

            registry = ToolRegistry()
            registry.register(CalculatorTool())

            orchestrator = OrchestratorAgent()
            planner = PlannerAgent()
            executor = ExecutorAgent(tool_registry=registry)
            critic = CriticAgent()

            orchestrator.register_agent("planner", planner)
            orchestrator.register_agent("executor", executor)
            orchestrator.register_agent("critic", critic)

            result = await orchestrator._run_with_metrics("What is 2 + 2?")
            
            # Even with mock, should return structured result
            assert result is not None
            assert result.agent_name == "orchestrator"
            assert result.status in ("success", "error")

    @pytest.mark.asyncio
    async def test_calculator_in_registry_execution(self):
        from backend.tools.registry import ToolRegistry
        from backend.tools.calculator import CalculatorTool

        registry = ToolRegistry()
        registry.register(CalculatorTool())

        result = await registry.execute_tool("calculator", {"expression": "10 * 10"})
        assert result.success
        assert result.output["result"] == 100
