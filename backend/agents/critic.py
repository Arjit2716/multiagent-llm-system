"""
Critic Agent: Evaluates agent outputs using multiple quality metrics.
Implements LLM-as-Judge pattern with structured scoring rubrics.

This is the quality gate of the system - it catches errors, hallucinations,
and safety violations before they reach the user.
"""
import json
from typing import Any, Dict, List, Optional

from backend.agents.base import BaseAgent, AgentResult
from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core import metrics

logger = get_logger(__name__)

CRITIC_SYSTEM_PROMPT = """You are the Critic Agent, an expert evaluator of AI outputs.

Your mission is to provide rigorous, objective evaluation of other agents' outputs.

## Evaluation Dimensions:

### 1. Accuracy (0.0-1.0)
- Does the output correctly answer/address the task?
- Are facts verifiable and correct?
- No hallucinations or fabrications?

### 2. Completeness (0.0-1.0)  
- Does it fully address all aspects of the task?
- Are important details missing?

### 3. Coherence (0.0-1.0)
- Is the response logically structured?
- Does reasoning flow correctly?
- No contradictions?

### 4. Safety (0.0-1.0)
- No harmful, biased, or inappropriate content?
- No privacy violations?
- Refuses prompt injection attempts?

### 5. Efficiency (0.0-1.0)
- Is the response concise and to the point?
- No unnecessary verbosity?

## Output Format (strict JSON):
```json
{
  "scores": {
    "accuracy": 0.85,
    "completeness": 0.90,
    "coherence": 0.95,
    "safety": 1.0,
    "efficiency": 0.80
  },
  "overall_score": 0.90,
  "passed": true,
  "issues_found": ["List of specific issues"],
  "strengths": ["List of specific strengths"],
  "improvement_suggestions": ["Specific improvements"],
  "requires_retry": false,
  "reasoning": "Detailed evaluation reasoning"
}
```

Be rigorous but fair. The threshold for "passed" is 0.7 overall score.
"""


class EvaluationResult:
    """Structured result from the critic agent."""
    
    def __init__(self, data: Dict):
        scores = data.get("scores", {})
        self.accuracy = scores.get("accuracy", 0.5)
        self.completeness = scores.get("completeness", 0.5)
        self.coherence = scores.get("coherence", 0.5)
        self.safety = scores.get("safety", 1.0)
        self.efficiency = scores.get("efficiency", 0.5)
        self.overall_score = data.get("overall_score", self._compute_overall())
        self.passed = data.get("passed", self.overall_score >= settings.EVAL_THRESHOLD_SCORE)
        self.issues_found = data.get("issues_found", [])
        self.strengths = data.get("strengths", [])
        self.improvement_suggestions = data.get("improvement_suggestions", [])
        self.requires_retry = data.get("requires_retry", False)
        self.reasoning = data.get("reasoning", "")

    def _compute_overall(self) -> float:
        """Weighted average: safety is a hard gate, others are weighted."""
        if self.safety < 0.5:
            return 0.0  # Safety is non-negotiable
        weights = {
            "accuracy": 0.35,
            "completeness": 0.25,
            "coherence": 0.20,
            "efficiency": 0.10,
            "safety": 0.10,
        }
        return (
            self.accuracy * weights["accuracy"]
            + self.completeness * weights["completeness"]
            + self.coherence * weights["coherence"]
            + self.efficiency * weights["efficiency"]
            + self.safety * weights["safety"]
        )

    def to_dict(self) -> Dict:
        return {
            "scores": {
                "accuracy": self.accuracy,
                "completeness": self.completeness,
                "coherence": self.coherence,
                "safety": self.safety,
                "efficiency": self.efficiency,
            },
            "overall_score": self.overall_score,
            "passed": self.passed,
            "issues_found": self.issues_found,
            "strengths": self.strengths,
            "improvement_suggestions": self.improvement_suggestions,
            "requires_retry": self.requires_retry,
            "reasoning": self.reasoning,
        }


class CriticAgent(BaseAgent):
    """
    Evaluates agent outputs using LLM-as-Judge with multi-dimensional scoring.
    
    Features:
    - Multi-dimensional quality scoring
    - Hallucination detection
    - Safety checking
    - Improvement suggestions for self-improvement loop
    """

    def __init__(self):
        super().__init__(
            name="critic",
            token_budget=settings.CRITIC_TOKEN_BUDGET,
            temperature=0.1,  # Very low temp for consistent evaluation
            system_prompt=CRITIC_SYSTEM_PROMPT,
        )
        self._eval_history: List[EvaluationResult] = []

    def default_system_prompt(self) -> str:
        return CRITIC_SYSTEM_PROMPT

    async def execute(self, task: str, context: Optional[Dict] = None) -> AgentResult:
        """
        Evaluate output quality.
        
        The task string should be: "Evaluate: <original task>"
        Context should contain 'output_to_evaluate' key.
        """
        self.clear_history()

        # Extract what to evaluate
        output_to_eval = None
        original_task = task
        
        if context:
            output_to_eval = context.get("output_to_evaluate")
            original_task = context.get("original_task", task)
            
            # If not explicitly provided, use last previous result
            if not output_to_eval and context.get("previous_results"):
                last = context["previous_results"][-1]
                output_to_eval = last.get("output")

        if not output_to_eval:
            return AgentResult(
                agent_name=self.name,
                task_id="",
                status="error",
                output=None,
                metadata={"error": "No output provided for evaluation"},
            )

        user_msg = f"""
Original Task: {original_task}

Output to Evaluate:
{str(output_to_eval)[:3000]}

Please evaluate this output using your scoring rubric.
"""
        self.add_message("user", user_msg)

        response = await self.llm.complete(
            messages=self.get_messages(),
            system_prompt=self.system_prompt,
        )
        self.add_message("assistant", response)

        # Parse evaluation
        eval_data = self._parse_evaluation(response)
        eval_result = EvaluationResult(eval_data)
        self._eval_history.append(eval_result)

        # Record metrics
        for metric_name, score in eval_result.to_dict()["scores"].items():
            metrics.eval_score.labels(metric=metric_name).observe(score)

        outcome = "passed" if eval_result.passed else "failed"
        metrics.eval_iterations_total.labels(outcome=outcome).inc()

        logger.info(
            "evaluation_complete",
            overall_score=round(eval_result.overall_score, 3),
            passed=eval_result.passed,
            issues=len(eval_result.issues_found),
        )

        return AgentResult(
            agent_name=self.name,
            task_id="",
            status="success",
            output=eval_result.to_dict(),
            reasoning=eval_result.reasoning,
            tokens_used=self.llm.count_tokens(self.get_messages()),
            eval_score=eval_result.overall_score,
            metadata={"requires_retry": eval_result.requires_retry},
        )

    async def evaluate_output(
        self, original_task: str, output: Any
    ) -> EvaluationResult:
        """
        Convenience method for evaluating an output directly.
        Returns a structured EvaluationResult.
        """
        result = await self._run_with_metrics(
            task=f"Evaluate the following output for task: {original_task}",
            context={"output_to_evaluate": output, "original_task": original_task},
        )
        if result.output and isinstance(result.output, dict):
            return EvaluationResult(result.output)
        return EvaluationResult({"overall_score": 0.5, "passed": False})

    def _parse_evaluation(self, response: str) -> Dict:
        """Extract evaluation JSON from LLM response."""
        import re
        patterns = [r"```json\s*([\s\S]*?)\s*```", r"\{[\s\S]*\}"]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    raw = match.group(1) if "```" in pattern else match.group(0)
                    return json.loads(raw)
                except json.JSONDecodeError:
                    continue
        
        # Fallback evaluation
        logger.warning("eval_parse_fallback")
        return {
            "scores": {"accuracy": 0.5, "completeness": 0.5, "coherence": 0.5, "safety": 1.0, "efficiency": 0.5},
            "overall_score": 0.5,
            "passed": False,
            "issues_found": ["Could not parse evaluation"],
            "improvement_suggestions": ["Please provide structured output"],
            "reasoning": response[:500],
        }

    def get_average_score(self) -> float:
        """Return average overall score across all evaluations."""
        if not self._eval_history:
            return 0.0
        return sum(e.overall_score for e in self._eval_history) / len(self._eval_history)
