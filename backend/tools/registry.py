"""
Dynamic Tool Registry - Plugin architecture for hot-pluggable tools.
Tools are registered with capability descriptions for LLM-based selection.
"""
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core import metrics

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """Standardized result from any tool execution."""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ToolParameter:
    """Describes a single tool parameter for LLM selection."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class BaseTool(ABC):
    """
    Abstract base for all tools in the system.
    
    Tools are sandboxed with timeouts and input validation.
    Each tool provides a capability description for LLM-based selection.
    """

    name: str = "base_tool"
    description: str = "Base tool"
    parameters: List[ToolParameter] = []
    category: str = "general"  # search, code, data, communication, etc.

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """Execute the tool. Must be overridden by subclasses."""
        ...

    def validate_inputs(self, kwargs: Dict) -> Optional[str]:
        """Validate inputs against parameter schema. Returns error message or None."""
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return f"Missing required parameter: '{param.name}'"
            if param.name in kwargs:
                val = kwargs[param.name]
                if param.type == "int" and not isinstance(val, int):
                    try:
                        kwargs[param.name] = int(val)
                    except (TypeError, ValueError):
                        return f"Parameter '{param.name}' must be an integer"
                elif param.type == "float" and not isinstance(val, (int, float)):
                    try:
                        kwargs[param.name] = float(val)
                    except (TypeError, ValueError):
                        return f"Parameter '{param.name}' must be a number"
        return None

    async def execute_with_timeout(self, **kwargs) -> ToolResult:
        """Execute tool with timeout and metrics tracking."""
        validation_error = self.validate_inputs(kwargs)
        if validation_error:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Validation error: {validation_error}",
            )

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.run(**kwargs),
                timeout=settings.TOOL_EXECUTION_TIMEOUT,
            )
            result.execution_time = time.monotonic() - start
            metrics.tool_calls_total.labels(
                tool_name=self.name, status="success"
            ).inc()
            metrics.tool_execution_duration.labels(tool_name=self.name).observe(
                result.execution_time
            )
            return result
        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            metrics.tool_calls_total.labels(tool_name=self.name, status="timeout").inc()
            logger.warning("tool_timeout", tool=self.name, timeout=settings.TOOL_EXECUTION_TIMEOUT)
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Tool '{self.name}' timed out after {settings.TOOL_EXECUTION_TIMEOUT}s",
                execution_time=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            metrics.tool_calls_total.labels(tool_name=self.name, status="error").inc()
            logger.error("tool_execution_error", tool=self.name, error=str(e))
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=str(e),
                execution_time=duration,
            )

    def get_schema(self) -> Dict:
        """Return tool schema for LLM tool selection."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": {
                p.name: {
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in self.parameters
            },
        }


class ToolRegistry:
    """
    Hot-pluggable tool registry with dynamic discovery.
    
    Tools can be registered at startup or dynamically at runtime.
    The registry provides capability-based tool selection descriptions.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        logger.info("tool_registry_initialized")

    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        logger.info("tool_registered", name=tool.name, category=tool.category)

    def register_class(self, tool_class: Type[BaseTool]) -> None:
        """Instantiate and register a tool from its class."""
        self.register(tool_class())

    def unregister(self, tool_name: str) -> bool:
        """Remove a tool from the registry."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info("tool_unregistered", name=tool_name)
            return True
        return False

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict]:
        """List all registered tools with their schemas."""
        return [tool.get_schema() for tool in self._tools.values()]

    def list_by_category(self, category: str) -> List[BaseTool]:
        """Get tools filtered by category."""
        return [t for t in self._tools.values() if t.category == category]

    async def execute_tool(self, tool_name: str, params: Dict) -> ToolResult:
        """Execute a tool by name with given parameters."""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found in registry. Available: {list(self._tools.keys())}",
            )
        return await tool.execute_with_timeout(**params)

    def get_capabilities_description(self) -> str:
        """
        Generate a human-readable description of all tool capabilities.
        Used by agents to select appropriate tools.
        """
        if not self._tools:
            return "No tools available."
        
        categories: Dict[str, List] = {}
        for tool in self._tools.values():
            categories.setdefault(tool.category, []).append(tool)
        
        lines = ["## Available Tools by Category:\n"]
        for category, tools in categories.items():
            lines.append(f"### {category.title()}")
            for tool in tools:
                lines.append(f"- **{tool.name}**: {tool.description}")
            lines.append("")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={list(self._tools.keys())}>"


# Global singleton registry
_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_default_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools to the registry."""
    from backend.tools.search import WebSearchTool, WikipediaTool
    from backend.tools.code_executor import CodeExecutorTool
    from backend.tools.calculator import CalculatorTool
    from backend.tools.memory import MemoryStoreTool, MemoryRetrieveTool

    for tool_class in [
        WebSearchTool,
        WikipediaTool,
        CodeExecutorTool,
        CalculatorTool,
        MemoryStoreTool,
        MemoryRetrieveTool,
    ]:
        try:
            registry.register_class(tool_class)
        except Exception as e:
            logger.warning("tool_registration_failed", tool=tool_class.__name__, error=str(e))
