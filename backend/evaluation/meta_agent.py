import json
import difflib
from typing import List, Dict, Any, Tuple
from sqlalchemy import select

from backend.core.llm_client import LLMClient
from backend.core.config import settings
from backend.core.logging import get_logger
from backend.db.models import get_session_factory, EvalHarnessTestCaseResult, PromptRewriteProposal
from backend.core.prompt_manager import prompt_manager

logger = get_logger(__name__)

class MetaAgent:
    """
    Self-Improving Prompt Loop Meta-Agent.
    Reads an evaluation run, finds the worst-performing agent/prompt, and proposes a structured rewrite.
    """
    
    def __init__(self):
        self.llm = LLMClient(
            model=settings.DEFAULT_MODEL,
            agent_name="meta_agent",
            token_budget=10000,
            temperature=0.3
        )
        self.session_factory = get_session_factory()
        
        # Mapping evaluation dimensions to responsible agents
        self.dimension_agent_map = {
            "answer_correctness": "synthesis",
            "citation_accuracy": "rag",
            "contradiction_resolution": "synthesis",
            "tool_selection_efficiency": "dynamic_orchestrator",
            "critique_agreement_rate": "synthesis",
            "context_budget_compliance": "dynamic_orchestrator"
        }

    async def analyze_eval_run(self, run_id: str) -> None:
        """Analyze a run and propose rewrites for the worst performing prompts."""
        logger.info(f"Meta-Agent analyzing run {run_id}")
        
        async with self.session_factory() as session:
            result = await session.execute(
                select(EvalHarnessTestCaseResult).where(EvalHarnessTestCaseResult.run_id == run_id)
            )
            cases = result.scalars().all()
            
        if not cases:
            logger.warning(f"No test cases found for run {run_id}")
            return
            
        # 1. Aggregate scores by dimension
        dim_scores = {
            "answer_correctness": [],
            "citation_accuracy": [],
            "contradiction_resolution": [],
            "tool_selection_efficiency": [],
            "critique_agreement_rate": [],
            "context_budget_compliance": []
        }
        
        failures_by_dim = {dim: [] for dim in dim_scores.keys()}
        
        for c in cases:
            dim_scores["answer_correctness"].append(c.answer_correctness)
            dim_scores["citation_accuracy"].append(c.citation_accuracy)
            dim_scores["contradiction_resolution"].append(c.contradiction_resolution)
            dim_scores["tool_selection_efficiency"].append(c.tool_selection_efficiency)
            dim_scores["critique_agreement_rate"].append(c.critique_agreement_rate)
            dim_scores["context_budget_compliance"].append(c.context_budget_compliance)
            
            if c.answer_correctness < 1.0: failures_by_dim["answer_correctness"].append(c)
            if c.citation_accuracy < 1.0: failures_by_dim["citation_accuracy"].append(c)
            if c.contradiction_resolution < 1.0: failures_by_dim["contradiction_resolution"].append(c)
            if c.tool_selection_efficiency < 1.0: failures_by_dim["tool_selection_efficiency"].append(c)
            if c.critique_agreement_rate < 1.0: failures_by_dim["critique_agreement_rate"].append(c)
            if c.context_budget_compliance < 1.0: failures_by_dim["context_budget_compliance"].append(c)

        # Find dimension with worst average score
        avg_scores = {dim: sum(scores)/len(scores) if scores else 1.0 for dim, scores in dim_scores.items()}
        worst_dim = min(avg_scores, key=avg_scores.get)
        
        if avg_scores[worst_dim] >= 0.95:
            logger.info("No significant failures detected. Meta-Agent skipping rewrite.")
            return
            
        target_agent = self.dimension_agent_map.get(worst_dim, "dynamic_orchestrator")
        logger.info(f"Worst dimension is {worst_dim} ({avg_scores[worst_dim]:.2f}). Target agent: {target_agent}")
        
        # Extract failure context
        failed_cases = failures_by_dim[worst_dim]
        failed_case_ids = [c.id for c in failed_cases]
        
        failure_context = []
        for c in failed_cases[:3]: # take top 3 failures to avoid token overflow
            justification = getattr(c, f"{worst_dim}_justification", "")
            failure_context.append({
                "query": c.query,
                "score": getattr(c, worst_dim),
                "justification": justification,
                "exact_outputs": c.exact_outputs
            })
            
        # Get current prompt
        # Fallbacks (ideally imported from their respective modules, but for safety we define default placeholders here if not fetched)
        # To fetch exactly, we'd need to import them. We'll simulate fetching current active:
        from backend.agents.dynamic_orchestrator import ORCHESTRATOR_ROUTING_PROMPT
        from backend.agents.synthesis_agent import SYNTHESIS_SYSTEM_PROMPT
        from backend.agents.rag_agent import RAG_SYSTEM_PROMPT
        from backend.agents.critique_agent import CRITIQUE_SYSTEM_PROMPT
        from backend.agents.decomposition import DECOMPOSITION_SYSTEM_PROMPT
        
        fallback_prompts = {
            "dynamic_orchestrator": ORCHESTRATOR_ROUTING_PROMPT,
            "synthesis": SYNTHESIS_SYSTEM_PROMPT,
            "rag": RAG_SYSTEM_PROMPT,
            "critique": CRITIQUE_SYSTEM_PROMPT,
            "decomposition": DECOMPOSITION_SYSTEM_PROMPT
        }
        
        original_prompt = await prompt_manager.get_prompt(target_agent, fallback_prompts.get(target_agent, "You are a helpful AI assistant."))
        
        # Propose rewrite
        prompt = f"""
You are the Meta-Agent responsible for self-improving the system.
The agent "{target_agent}" is severely underperforming on the dimension "{worst_dim}".
Average score: {avg_scores[worst_dim]:.2f}/1.0

### FAILURE CASES CONTEXT ###
{json.dumps(failure_context, indent=2)}

### CURRENT SYSTEM PROMPT ###
{original_prompt}

### INSTRUCTIONS ###
Analyze the failure justifications and rewrite the CURRENT SYSTEM PROMPT to explicitly prevent these failures.
Make the new prompt more robust, adding explicit safety, formatting, or reasoning rules. Do not change the JSON schema if the agent outputs JSON.

Output ONLY a JSON object with:
"proposed_prompt": "The complete rewritten prompt string",
"justification": "Why this change will fix the failures"

```json
"""
        
        try:
            res = await self.llm.complete([{"role": "user", "content": prompt}], max_tokens=1500)
            import re
            m = re.search(r"\{[\s\S]*\}", res)
            if not m:
                raise ValueError("Could not parse JSON from Meta-Agent")
                
            data = json.loads(m.group(0))
            proposed_prompt = data["proposed_prompt"]
            justification = data["justification"]
            
            # Create Diff
            d = difflib.unified_diff(
                original_prompt.splitlines(keepends=True),
                proposed_prompt.splitlines(keepends=True),
                fromfile='original',
                tofile='proposed'
            )
            diff_text = "".join(d)
            
            # Save to DB
            proposal = PromptRewriteProposal(
                eval_run_id=run_id,
                target_agent=target_agent,
                failed_test_case_ids=failed_case_ids,
                original_prompt=original_prompt,
                proposed_prompt=proposed_prompt,
                diff_text=diff_text,
                justification=justification
            )
            async with self.session_factory() as session:
                session.add(proposal)
                await session.commit()
                
            logger.info(f"Saved rewrite proposal for {target_agent}")
            
        except Exception as e:
            logger.error(f"Meta-Agent failed to propose rewrite: {e}")
