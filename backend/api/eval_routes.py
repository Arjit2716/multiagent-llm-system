import uuid
import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from sqlalchemy import select

from backend.db.models import get_session_factory, PromptRewriteProposal, EvalHarnessTestCaseResult, EvalHarnessRun
from backend.core.prompt_manager import prompt_manager
from backend.evaluation.eval_harness import HarnessEvaluator
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/eval", tags=["eval"])

@router.get("/proposals")
async def list_proposals():
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(PromptRewriteProposal).order_by(PromptRewriteProposal.created_at.desc())
        )
        proposals = result.scalars().all()
        return [
            {
                "id": p.id,
                "eval_run_id": p.eval_run_id,
                "target_agent": p.target_agent,
                "failed_cases_count": len(p.failed_test_case_ids),
                "original_prompt": p.original_prompt,
                "proposed_prompt": p.proposed_prompt,
                "diff_text": p.diff_text,
                "justification": p.justification,
                "status": p.status,
                "performance_delta": p.performance_delta,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "decided_at": p.decided_at.isoformat() if p.decided_at else None
            }
            for p in proposals
        ]

async def _rerun_failed_cases(proposal_id: str):
    logger.info(f"Re-running failed cases for proposal {proposal_id}")
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        result = await session.execute(
            select(PromptRewriteProposal).where(PromptRewriteProposal.id == proposal_id)
        )
        proposal = result.scalar_one_or_none()
        if not proposal:
            return
            
        case_ids = proposal.failed_test_case_ids
        
        # Get old scores
        old_results = await session.execute(
            select(EvalHarnessTestCaseResult).where(EvalHarnessTestCaseResult.id.in_(case_ids))
        )
        old_cases = old_results.scalars().all()
        if not old_cases:
            return
            
        old_avg = sum(c.overall_score for c in old_cases) / len(old_cases)
        
        # Activate new prompt
        await prompt_manager.update_prompt(proposal.target_agent, proposal.proposed_prompt)
        
        # Run tests again
        evaluator = HarnessEvaluator()
        new_run_id = str(uuid.uuid4())
        
        run_record = EvalHarnessRun(id=new_run_id)
        session.add(run_record)
        await session.commit()
        
        new_scores = []
        for c in old_cases:
            res = await evaluator.run_test_case(new_run_id, c.query, c.category)
            new_scores.append(res.overall_score)
            
        new_avg = sum(new_scores) / len(new_scores)
        delta = new_avg - old_avg
        
        # Update proposal
        proposal.performance_delta = delta
        proposal.status = "approved"
        proposal.decided_at = datetime.datetime.now(datetime.timezone.utc)
        
        session.add(proposal)
        
        # Update run record total score
        run_record.total_score = new_avg
        session.add(run_record)
        
        await session.commit()
        logger.info(f"Proposal {proposal_id} evaluated. Delta: {delta:.3f}")

@router.post("/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: str, background_tasks: BackgroundTasks):
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(PromptRewriteProposal).where(PromptRewriteProposal.id == proposal_id)
        )
        proposal = result.scalar_one_or_none()
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found")
            
        if proposal.status != "pending":
            raise HTTPException(status_code=400, detail=f"Proposal already {proposal.status}")
            
        # Queue background task to run eval and update delta
        background_tasks.add_task(_rerun_failed_cases, proposal_id)
        
        return {"status": "approval processing started", "proposal_id": proposal_id}

@router.post("/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: str):
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(PromptRewriteProposal).where(PromptRewriteProposal.id == proposal_id)
        )
        proposal = result.scalar_one_or_none()
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found")
            
        if proposal.status != "pending":
            raise HTTPException(status_code=400, detail=f"Proposal already {proposal.status}")
            
        proposal.status = "rejected"
        proposal.decided_at = datetime.datetime.now(datetime.timezone.utc)
        session.add(proposal)
        await session.commit()
        
        return {"status": "rejected", "proposal_id": proposal_id}
