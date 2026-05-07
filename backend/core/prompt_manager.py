import asyncio
from typing import Dict, Optional

from sqlalchemy import select
from backend.db.models import get_session_factory, ActivePrompt
from backend.core.logging import get_logger

logger = get_logger(__name__)

class PromptManager:
    """Manages active system prompts, pulling overrides from the DB if they exist."""
    
    def __init__(self):
        self._session_factory = get_session_factory()
        self._cache: Dict[str, str] = {}
        
    async def get_prompt(self, agent_name: str, fallback_prompt: str) -> str:
        """Get the active prompt for an agent, or use the fallback if none is set in DB."""
        if agent_name in self._cache:
            return self._cache[agent_name]
            
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(ActivePrompt).where(ActivePrompt.agent_name == agent_name)
                )
                active = result.scalar_one_or_none()
                if active:
                    self._cache[agent_name] = active.prompt_text
                    return active.prompt_text
        except Exception as e:
            logger.error(f"Error fetching prompt for {agent_name}: {e}")
            
        self._cache[agent_name] = fallback_prompt
        return fallback_prompt

    async def update_prompt(self, agent_name: str, new_prompt: str) -> None:
        """Set a new active prompt for an agent."""
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(ActivePrompt).where(ActivePrompt.agent_name == agent_name)
                )
                active = result.scalar_one_or_none()
                if active:
                    active.prompt_text = new_prompt
                else:
                    active = ActivePrompt(agent_name=agent_name, prompt_text=new_prompt)
                    session.add(active)
                await session.commit()
                self._cache[agent_name] = new_prompt
                logger.info(f"Updated active prompt for {agent_name}")
        except Exception as e:
            logger.error(f"Error updating prompt for {agent_name}: {e}")
            raise

prompt_manager = PromptManager()
