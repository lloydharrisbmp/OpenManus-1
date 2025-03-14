from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Memory, Message, ROLE_TYPE
from app.reasoning.groq_client import GroqReasoner
from app.config.tool_config import ToolConfig

logger = logging.getLogger(__name__)

class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    duplicate_threshold: int = 2

    groq_api_key: Optional[str] = Field(None, description="Groq API key")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses

    def __init__(self, **data):
        super().__init__(**data)
        self.config = ToolConfig()
        self.reasoner = GroqReasoner(api_key=self.groq_api_key or self.config.groq_api_key)
        self.available_tools = []
        self.context = {
            "history": [],
            "state": {},
            "start_time": datetime.utcnow().isoformat()
        }
        if not self.validate_tools():
            raise ValueError("Tool validation failed during initialization")

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE, # type: ignore
        content: str,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        msg_factory = message_map[role]
        msg = msg_factory(content, **kwargs) if role == "tool" else msg_factory(content)
        self.memory.add_message(msg)

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                # Check for stuck state
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")

        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value

    def validate_tools(self) -> bool:
        """Validate all registered tools have required attributes and methods."""
        try:
            for tool in self.available_tools:
                if not hasattr(tool, "execute"):
                    logger.error(f"Tool {tool.name} missing execute method")
                    return False
                if not hasattr(tool, "parameters"):
                    logger.error(f"Tool {tool.name} missing parameters definition")
                    return False
                if not hasattr(tool, "name") or not tool.name:
                    logger.error("Tool missing name attribute")
                    return False
            return True
        except Exception as e:
            logger.error(f"Tool validation error: {e}", exc_info=True)
            return False

    async def safe_execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute a tool with error handling and logging."""
        start_time = datetime.utcnow()
        tool_name = tool_call.get("tool")
        
        try:
            tool = next((t for t in self.available_tools if t.name == tool_name), None)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")

            # Get tool-specific configuration
            tool_config = self.config.get_tool_config(tool_name)
            
            # Merge tool_call parameters with configuration
            parameters = {**tool_config, **tool_call.get("parameters", {})}
            
            # Execute the tool
            result = await tool.execute(**parameters)
            
            # Update context with execution result
            self.update_context({
                "tool": tool_name,
                "parameters": parameters,
                "result": result,
                "execution_time": (datetime.utcnow() - start_time).total_seconds()
            })
            
            return result
            
        except Exception as e:
            error_detail = {
                "tool": tool_name,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": (datetime.utcnow() - start_time).total_seconds()
            }
            logger.error(f"Tool execution failed: {error_detail}", exc_info=True)
            self.update_context({"error": error_detail})
            return {"error": str(e), "success": False}

    async def execute_parallel_tools(self, task: str) -> Dict[str, Any]:
        """Execute multiple tools in parallel with proper error handling."""
        try:
            tool_descriptions = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
                for tool in self.available_tools
            ]
            
            execution_plan = await self.reasoner.parallel_tool_planning(
                tools=tool_descriptions,
                task=task
            )
            
            # Execute tools in parallel with concurrency limit
            semaphore = asyncio.Semaphore(self.config.max_concurrent_tools)
            async with asyncio.TaskGroup() as tg:
                tasks = []
                for tool_call in execution_plan.get("tool_calls", []):
                    async def _execute_with_semaphore(tool_call):
                        async with semaphore:
                            return await self.safe_execute_tool(tool_call)
                    
                    task = tg.create_task(_execute_with_semaphore(tool_call))
                    tasks.append(task)
            
            results = [t.result() for t in tasks]
            
            return {
                "plan": execution_plan,
                "results": results,
                "success": all(not r.get("error") for r in results)
            }
            
        except Exception as e:
            logger.error(f"Parallel tool execution failed: {e}", exc_info=True)
            return {"error": str(e), "success": False}

    def update_context(self, result: Dict[str, Any]) -> None:
        """Update agent context with execution results."""
        self.context["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            **result
        })
        if "state_updates" in result:
            self.context["state"].update(result["state_updates"])

    async def think(self, query: str) -> Dict[str, Any]:
        """Enhanced thinking process with error handling."""
        try:
            tool_descriptions = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
                for tool in self.available_tools
            ]
            
            reasoning_result = await self.reasoner.reason(
                context=self.context,
                available_tools=tool_descriptions,
                query=query
            )
            
            # Update context with reasoning result
            self.update_context({
                "query": query,
                "reasoning": reasoning_result
            })
            
            return reasoning_result
            
        except Exception as e:
            error_detail = {
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__
            }
            logger.error(f"Reasoning failed: {error_detail}", exc_info=True)
            self.update_context({"error": error_detail})
            return {"error": str(e), "success": False}
