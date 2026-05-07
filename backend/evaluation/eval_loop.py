"""
Self-Improving Evaluation Loop.
Monitors output quality and feeds improvement signals back into the system.

Architecture:
- Collects evaluation scores from the Critic agent
- Detects quality degradation via sliding window
- Generates improvement suggestions via LLM
- Injects few-shot examples from successful runs into system prompts
"""
import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.llm_client import LLMClient
from backend.core import metrics

logger = get_logger(__name__)


@dataclass
class EvalEntry:
    """Single evaluation record."""
    timestamp: float
    task: str
    output: str
    score: float
    passed: bool
    issues: List[str]
    suggestions: List[str]
    agent_name: str


@dataclass
class ImprovementSignal:
    """Signal generated when quality drops below threshold."""
    trigger_score: float
    window_avg_score: float
    common_issues: List[str]
    few_shot_examples: List[Dict]
    system_prompt_patches: Dict[str, str]  # agent_name -> patch


class SelfImprovingEvalLoop:
    """
    Continuous self-improvement loop that:
    1. Tracks evaluation scores over time
    2. Detects quality degradation
    3. Generates improvement signals
    4. Injects few-shot examples into agent system prompts
    5. Tracks improvement over iterations
    
    This implements a simplified RL feedback loop without requiring
    full model fine-tuning.
    """

    def __init__(
        self,
        window_size: int = 20,
        degradation_threshold: float = 0.65,
        improvement_threshold: float = 0.80,
    ):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.improvement_threshold = improvement_threshold
        
        self._eval_window: Deque[EvalEntry] = deque(maxlen=window_size)
        self._all_entries: List[EvalEntry] = []
        self._improvement_signals: List[ImprovementSignal] = []
        self._best_examples: List[EvalEntry] = []  # High-scoring examples for few-shot
        self._iteration_count: int = 0
        
        # LLM for generating improvement suggestions
        self._llm = LLMClient(
            model=settings.DEFAULT_MODEL,
            agent_name="eval_loop",
            token_budget=3000,
            temperature=0.3,
        )

    def record_evaluation(
        self,
        task: str,
        output: str,
        score: float,
        passed: bool,
        issues: List[str],
        suggestions: List[str],
        agent_name: str = "unknown",
    ) -> None:
        """Record an evaluation result and check for improvement triggers."""
        entry = EvalEntry(
            timestamp=time.time(),
            task=task[:500],
            output=output[:1000],
            score=score,
            passed=passed,
            issues=issues,
            suggestions=suggestions,
            agent_name=agent_name,
        )
        self._eval_window.append(entry)
        self._all_entries.append(entry)
        self._iteration_count += 1

        # Track best examples for few-shot learning
        if score >= self.improvement_threshold:
            self._best_examples.append(entry)
            # Keep only top 10
            self._best_examples.sort(key=lambda e: e.score, reverse=True)
            self._best_examples = self._best_examples[:10]

        logger.debug(
            "eval_recorded",
            score=round(score, 3),
            passed=passed,
            window_size=len(self._eval_window),
        )

    def get_window_average(self) -> Optional[float]:
        """Compute average score in the current window."""
        if not self._eval_window:
            return None
        return sum(e.score for e in self._eval_window) / len(self._eval_window)

    def is_degrading(self) -> bool:
        """Check if quality is degrading below threshold."""
        avg = self.get_window_average()
        return avg is not None and avg < self.degradation_threshold

    def get_common_issues(self, n: int = 5) -> List[str]:
        """Identify most frequently occurring issues in recent evaluations."""
        issue_counts: Dict[str, int] = {}
        for entry in self._eval_window:
            for issue in entry.issues:
                # Normalize issue text
                normalized = issue.strip().lower()[:100]
                issue_counts[normalized] = issue_counts.get(normalized, 0) + 1
        
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, _ in sorted_issues[:n]]

    def get_few_shot_examples(self, n: int = 3) -> List[Dict]:
        """Get the best examples for few-shot injection."""
        examples = []
        for entry in self._best_examples[:n]:
            examples.append({
                "task": entry.task,
                "output": entry.output[:500],
                "score": entry.score,
                "agent": entry.agent_name,
            })
        return examples

    async def generate_improvement_signal(self) -> Optional[ImprovementSignal]:
        """
        Generate an improvement signal using LLM analysis.
        Called when degradation is detected.
        """
        if not self.is_degrading():
            return None

        common_issues = self.get_common_issues()
        few_shots = self.get_few_shot_examples()
        window_avg = self.get_window_average()

        analysis_prompt = f"""
Analyze these recurring quality issues in a multi-agent LLM system:

Window average score: {window_avg:.3f} (threshold: {self.degradation_threshold})
Degradation threshold: {self.degradation_threshold}

Most common issues:
{json.dumps(common_issues, indent=2)}

Recent failed examples:
{json.dumps([{"task": e.task, "issues": e.issues} for e in list(self._eval_window)[-5:] if not e.passed], indent=2)}

Best performing examples:
{json.dumps([{"task": e.task, "score": e.score} for e in self._best_examples[:3]], indent=2)}

Generate specific improvements as JSON:
{{
  "root_cause": "What's causing the quality drop",
  "system_prompt_patches": {{
    "executor": "Additional instruction to add to executor",
    "planner": "Additional instruction to add to planner"
  }},
  "training_examples": [
    {{"input": "example input", "ideal_output": "example output", "reason": "why this is good"}}
  ],
  "immediate_actions": ["list of immediate fixes"]
}}
"""
        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": analysis_prompt}],
                system_prompt="You are an expert at diagnosing and improving LLM system quality.",
            )
            
            import re
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                analysis = json.loads(match.group(0))
            else:
                analysis = {"root_cause": "Parse error", "system_prompt_patches": {}, "immediate_actions": []}

            signal = ImprovementSignal(
                trigger_score=window_avg,
                window_avg_score=window_avg,
                common_issues=common_issues,
                few_shot_examples=few_shots,
                system_prompt_patches=analysis.get("system_prompt_patches", {}),
            )
            self._improvement_signals.append(signal)
            
            metrics.eval_iterations_total.labels(outcome="improved").inc()
            logger.info(
                "improvement_signal_generated",
                root_cause=analysis.get("root_cause", ""),
                patches=list(analysis.get("system_prompt_patches", {}).keys()),
            )
            return signal
        except Exception as e:
            logger.error("improvement_signal_failed", error=str(e))
            return None

    async def run_improvement_cycle(self) -> Dict:
        """
        Full improvement cycle:
        1. Check if degrading
        2. Generate signal if needed
        3. Return patches for agents to apply
        """
        window_avg = self.get_window_average()
        
        status = {
            "iteration": self._iteration_count,
            "window_size": len(self._eval_window),
            "window_avg_score": round(window_avg, 3) if window_avg else None,
            "is_degrading": self.is_degrading(),
            "best_examples_count": len(self._best_examples),
            "improvement_signals_generated": len(self._improvement_signals),
        }

        if self.is_degrading() and len(self._eval_window) >= 5:
            logger.warning(
                "quality_degradation_detected",
                window_avg=round(window_avg, 3),
                threshold=self.degradation_threshold,
            )
            signal = await self.generate_improvement_signal()
            if signal:
                status["improvement_signal"] = {
                    "patches": signal.system_prompt_patches,
                    "few_shot_count": len(signal.few_shot_examples),
                    "common_issues": signal.common_issues,
                }

        return status

    def get_metrics_report(self) -> Dict:
        """Return comprehensive metrics for the dashboard."""
        if not self._all_entries:
            return {"total_evaluations": 0}

        scores = [e.score for e in self._all_entries]
        passed = [e for e in self._all_entries if e.passed]
        
        return {
            "total_evaluations": len(self._all_entries),
            "pass_rate": len(passed) / len(self._all_entries),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "recent_window_avg": self.get_window_average(),
            "is_degrading": self.is_degrading(),
            "best_examples_count": len(self._best_examples),
            "improvement_cycles": len(self._improvement_signals),
            "score_trend": [round(e.score, 3) for e in list(self._eval_window)[-10:]],
        }
