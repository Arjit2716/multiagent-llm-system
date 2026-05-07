import json
import logging
import tiktoken
from typing import List, Dict, Any

from backend.core.logging import get_logger
from backend.core.llm_client import LLMClient
from backend.core.config import settings

logger = get_logger(__name__)

class CompressionAgent:
    """Agent responsible for summarizing older context to fit within the budget."""
    def __init__(self):
        self.llm = LLMClient(
            model=settings.DEFAULT_MODEL,
            agent_name="compression_agent",
            token_budget=4000,
            temperature=0.1
        )
    
    async def compress(self, messages: List[Dict]) -> List[Dict]:
        """
        Compresses a list of messages.
        Lossless for structured data (tool outputs, scores, citations).
        Lossy for conversational filler.
        """
        structured = []
        conversational = []
        
        for msg in messages:
            if msg.get("role") == "tool" or self._is_structured(msg.get("content", "")):
                structured.append(msg)
            else:
                conversational.append(msg)
                
        if not conversational:
            return messages
            
        text_to_compress = json.dumps(conversational)
        prompt = (
            "Summarize the following conversational history concisely. "
            "Keep the essential meaning but remove filler words and redundant information.\n\n"
            f"{text_to_compress}"
        )
        
        try:
            summary = await self.llm.complete([{"role": "user", "content": prompt}], max_tokens=500)
            compressed_msg = {"role": "system", "content": f"Compressed older conversation: {summary}"}
            
            # Reconstruct list: put compressed system msg first, then structured data
            new_messages = [compressed_msg] + structured
            return new_messages
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return messages

    def _is_structured(self, content: str) -> bool:
        if not content:
            return False
        content_stripped = content.strip()
        if (content_stripped.startswith("{") and content_stripped.endswith("}")) or \
           (content_stripped.startswith("[") and content_stripped.endswith("]")):
            try:
                json.loads(content_stripped)
                return True
            except:
                pass
        if "confidence_score" in content or "chunk_id" in content or "claim_id" in content:
            return True
        return False

class ContextBudgetManager:
    """Tracks token consumption per agent per turn and manages compression."""
    def __init__(self):
        self.budgets: Dict[str, int] = {}
        self.usage: Dict[str, int] = {}
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback if tiktoken fails
            self.encoder = None
        self.compression_agent = CompressionAgent()
        
    def declare_budget(self, agent_id: str, budget: int) -> None:
        """Each agent must declare its max context budget before execution."""
        self.budgets[agent_id] = budget
        self.usage[agent_id] = 0
        
    def check_remaining_budget(self, agent_id: str) -> int:
        """Expose a method that any agent can call to check remaining budget before adding to its context."""
        if agent_id not in self.budgets:
            return 0
        return max(0, self.budgets[agent_id] - self.usage.get(agent_id, 0))
        
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text.split())  # simple fallback
        
    def record_usage(self, agent_id: str, tokens: int) -> None:
        if agent_id in self.usage:
            self.usage[agent_id] += tokens

    async def assemble_context(self, agent_id: str, messages: List[Dict]) -> List[Dict]:
        """
        If assembled context exceeds budget, automatically summarize using a compression agent.
        Agents that ignore budget constraints and overflow are caught and logged as policy violations, not silently truncated.
        """
        if agent_id not in self.budgets:
            return messages
            
        budget = self.budgets[agent_id]
        total_tokens = sum(self.count_tokens(json.dumps(m)) for m in messages)
        
        if total_tokens > budget:
            logger.error("POLICY_VIOLATION_CONTEXT_OVERFLOW", 
                         agent_id=agent_id, 
                         tokens=total_tokens, 
                         budget=budget, 
                         msg="Agent ignored budget constraints and overflowed. Triggering compression.")
            
            # Automatically summarize older context using compression agent
            compressed_messages = await self.compression_agent.compress(messages)
            new_tokens = sum(self.count_tokens(json.dumps(m)) for m in compressed_messages)
            
            self.usage[agent_id] = new_tokens
            return compressed_messages
            
        self.usage[agent_id] = total_tokens
        return messages

# Global singleton
context_manager = ContextBudgetManager()
