"""
Planner Agent: Creates structured, step-by-step execution plans.
Implements the ReAct (Reason + Act) loop for systematic problem solving.
"""
import json
from typing import Any, Dict, List, Optional

from backend.agents.base import BaseAgent, AgentResult
from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the Planner Agent. Your expertise is in:
1. Breaking down complex problems into clear, actionable steps
2. Identifying dependencies between tasks
3. Estimating effort and risk for each step
4. Creating contingency plans for likely failures

## ReAct Loop:
For each task, follow this structure:
- **Thought**: Analyze what needs to be done
- **Reasoning**: Explain your approach
- **Plan**: Numbered list of concrete steps
- **Dependencies**: What each step depends on
- **Risks**: Potential failure points and mitigations

## Output Format (strict JSON):
```json
{
  "thought": "Your analysis of the problem",
  "reasoning": "Why you chose this approach",
  "plan": [
    {
      "step": 1,
      "action": "Specific action to take",
      "tool": "tool_name_if_needed",
      "expected_output": "What this step should produce",
      "depends_on": [],
      "risk": "low|medium|high",
      "fallback": "What to do if this fails"
    }
  ],
  "success_criteria": "How we know the task is complete",
  "estimated_tokens": 1000
}
```

Always reason through the problem before proposing a plan.
"""


class PlannerAgent(BaseAgent):
    """
    Creates structured execution plans using ReAct reasoning.
    
    The Planner is the strategic brain - it thinks before acting,
    identifies risks, and provides fallback strategies.
    """

    def __init__(self):
        super().__init__(
            name="planner",
            token_budget=settings.PLANNER_TOKEN_BUDGET,
            temperature=0.4,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

    def default_system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT

    async def execute(self, task: str, context: Optional[Dict] = None) -> AgentResult:
        """Generate a structured execution plan for the given task."""
        self.clear_history()

        context_info = ""
        if context and context.get("previous_results"):
            prev = context["previous_results"]
            context_info = f"\n\nPrevious step results:\n{json.dumps(prev, indent=2)}"

        user_msg = f"""Task: {task}{context_info}

Please analyze this task and create a detailed execution plan following your instructions."""

        self.add_message("user", user_msg)

        response = await self.llm.complete(
            messages=self.get_messages(),
            system_prompt=self.system_prompt,
        )
        self.add_message("assistant", response)

        # Parse the plan
        plan = self._parse_plan(response)
        token_count = self.llm.count_tokens(self.get_messages())

        return AgentResult(
            agent_name=self.name,
            task_id="",
            status="success",
            output=plan,
            reasoning=plan.get("thought", "") + " " + plan.get("reasoning", ""),
            tokens_used=token_count,
            metadata={"raw_response": response},
        )

    def _parse_plan(self, response: str) -> Dict:
        """Extract and validate the JSON plan from LLM response."""
        import re

        # Try JSON extraction
        patterns = [r"```json\s*([\s\S]*?)\s*```", r"\{[\s\S]*\}"]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    raw = match.group(1) if "```" in pattern else match.group(0)
                    plan = json.loads(raw)
                    # Validate required fields
                    if "plan" in plan and isinstance(plan["plan"], list):
                        return plan
                except json.JSONDecodeError:
                    continue

        # Fallback: extract steps from text
        logger.warning("plan_parse_fallback", reason="Could not parse JSON plan")
        steps = []
        lines = response.split("\n")
        for i, line in enumerate(lines):
            if line.strip() and (line.strip()[0].isdigit() or line.startswith("-")):
                steps.append({
                    "step": i + 1,
                    "action": line.strip().lstrip("0123456789.-) "),
                    "risk": "medium",
                })

        return {
            "thought": "Plan extracted from text",
            "reasoning": "JSON parsing failed, extracted steps from text",
            "plan": steps or [{"step": 1, "action": response, "risk": "medium"}],
            "success_criteria": "Task completed",
            "estimated_tokens": 1000,
        }
