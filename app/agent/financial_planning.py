from typing import Dict, Any, Optional, List, TypedDict
import logging
import asyncio
from pydantic import BaseModel, validator, BaseSettings
import aiohttp
from collections import deque
import contextlib
import structlog
from asyncio import Queue
from dataclasses import dataclass
from typing import Set

from app.agent.base import BaseAgent
from app.tool import (
    MarketDataTool,
    PropertyAnalysisTool,
    SuperannuationAnalysisTool,
    EstateAnalysisTool,
    InsuranceAnalysisTool,
    GoogleSearch
)

logger = logging.getLogger(__name__)

class ToolCall(BaseModel):
    tool_name: str
    params: Dict[str, Any]
    retry_attempts: int = 1
    retry_delay: float = 1.0

    @validator('retry_attempts')
    def validate_attempts(cls, v):
        if v < 1:
            raise ValueError("retry_attempts must be >= 1")
        return v

class ToolResult(BaseModel):
    tool_name: str
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    success: bool

    @validator('result', 'error')
    def validate_result_error(cls, v, values):
        if values.get('success') and not v:
            raise ValueError("Success requires result")
        if not values.get('success') and not v:
            raise ValueError("Failure requires error message")
        return v

class AgentConfig(BaseSettings):
    concurrency_limit: int = 3
    default_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    context_history_size: int = 100
    
    class Config:
        env_prefix = 'FINANCIAL_AGENT_'

@dataclass
class ToolDependency:
    required_tools: Set[str]
    optional_tools: Set[str] = frozenset()

class FinancialPlanningAgent(BaseAgent):
    """Enhanced Financial Planning Agent with improved error handling and tool management."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        super().__init__(groq_api_key=None)
        self.available_tools = [
            MarketDataTool(),
            PropertyAnalysisTool(),
            SuperannuationAnalysisTool(),
            EstateAnalysisTool(),
            InsuranceAnalysisTool(),
            GoogleSearch()
        ]
        # Validate tools after initialization
        if not self.validate_tools():
            raise ValueError("Tool validation failed for FinancialPlanningAgent")
        self._context_history = deque(maxlen=self.config.context_history_size)  # Limit memory usage
        self._tool_results_cache = {}  # Cache recent tool results
        self._progress_queue: Queue = Queue()

    async def process_query(self, query: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        try:
            async with asyncio.timeout(timeout or self.config.default_timeout) as timeout_context:
                return await self._process_query_internal(query)
        except asyncio.TimeoutError:
            logger.error("Query processing timed out")
            return {"error": "Operation timed out", "success": False}

    async def shutdown(self):
        """Graceful shutdown of agent resources."""
        tasks = []
        for tool in self.available_tools:
            if hasattr(tool, 'shutdown'):
                tasks.append(tool.shutdown())
        
        if tasks:
            await asyncio.gather(*tasks)
        
        # Clear caches and context
        self._tool_results_cache.clear()
        self._context_history.clear()

    async def _execute_parallel_tools(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        # Now with validated input 

    async def __aenter__(self):
        """Setup async resources."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async resources."""
        if hasattr(self, '_session'):
            await self._session.close() 

    async def _run_single_tool_call(self, call: ToolCall, call_idx: int) -> Dict[str, Any]:
        await self._progress_queue.put({
            "step": call_idx,
            "tool": call.tool_name,
            "status": "started"
        })
        log = logger.bind(
            tool_name=call.tool_name,
            call_idx=call_idx,
            params=call.params
        )
        try:
            result = # ... existing code ...
            await self._progress_queue.put({
                "step": call_idx,
                "tool": call.tool_name,
                "status": "completed"
            })
            return result
        except Exception as e:
            await self._progress_queue.put({
                "step": call_idx,
                "tool": call.tool_name,
                "status": "failed",
                "error": str(e)
            })
            raise 

    def validate_tools(self) -> bool:
        """
        Implement your tool validation logic here.
        This could involve checking that all required tools are present and properly configured.
        """
        required_tools = {'market_data', 'property_analysis'}
        available_tools = {tool.name for tool in self.available_tools}
        return required_tools.issubset(available_tools)

    async def think(self, query: str) -> Dict[str, Any]:
        """
        Implement your thinking/reasoning logic here.
        This could involve calling an LLM or other decision-making process.
        """
        # Example implementation
        return {
            "next_actions": [
                {
                    "tool_calls": [
                        {
                            "tool_name": "market_data",
                            "params": {"symbol": "AAPL"}
                        }
                    ]
                }
            ]
        }

    def update_context(self, context: Dict[str, Any]) -> None:
        """
        Implement your context update logic here.
        This could involve storing state in a database or memory.
        """
        if not hasattr(self, '_context'):
            self._context = []
        self._context.append(context) 