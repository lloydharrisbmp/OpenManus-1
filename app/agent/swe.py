from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json
from pathlib import Path

from pydantic import Field, BaseModel
import asyncio

from app.agent.toolcall import ToolCallAgent
from app.prompt.financial_planner import NEXT_STEP_TEMPLATE, SYSTEM_PROMPT
from app.schema import Message
from app.tool import Bash, StrReplaceEditor, Terminate, ToolCollection
from app.tool.financial_tools import (
    AustralianMarketAnalysisTool,
    MarketAnalysisTool,
    PortfolioOptimizationTool,
    ReportGeneratorTool,
    TaxOptimizationTool,
)
from app.tool.document_analyzer import DocumentAnalyzerTool
from app.tool.website_generator import WebsiteGeneratorTool
from app.tool.tool_creator import ToolCreatorTool
from app.logger import logger

class ToolExecutionMetrics(BaseModel):
    """Track metrics for tool execution within the agent."""
    tool_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    response_length: int = 0
    has_visualization: bool = False

class FinancialPlanningAgent(ToolCallAgent):
    """An advanced AI agent specializing in Australian financial planning and investment advice for high net worth clients."""

    name: str = "aus_financial_planner"
    description: str = """A sophisticated financial planning AI that specializes in providing comprehensive advice for high net worth Australian clients, with expertise in:
    - Complex entity and tax structure optimization (trusts, companies, SMSFs)
    - ASX and international investment research and analysis
    - Portfolio optimization with tax-aware strategies
    - Australian regulatory compliance (ASIC, ATO requirements)
    - Customized financial reporting and strategy development
    - SMSF strategy and compliance
    - Estate planning and succession
    - Website and document generation"""

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_TEMPLATE
    last_observation: Optional[str] = None
    thinking_steps: List[str] = Field(default_factory=list)
    current_section: Optional[str] = None
    completed_tasks: List[str] = Field(default_factory=list)

    max_steps: int = 5  # Reduced from 30 to 5 for testing purposes

    bash: Bash = Field(default_factory=Bash)
    working_dir: str = "."

    current_metrics: Optional[ToolExecutionMetrics] = None
    execution_history: List[ToolExecutionMetrics] = Field(default_factory=list)
    visualization_paths: List[str] = Field(default_factory=list)

    def __init__(self, **kwargs):
        # Initialize tools first
        tools = [
            AustralianMarketAnalysisTool(),
            MarketAnalysisTool(),  # Keep for international markets
            PortfolioOptimizationTool(),
            TaxOptimizationTool(),
            ReportGeneratorTool(),
            DocumentAnalyzerTool(),
            WebsiteGeneratorTool(),
            ToolCreatorTool(),
            Bash(),
            StrReplaceEditor(),
            Terminate()
        ]
        kwargs['available_tools'] = ToolCollection(*tools)
        super().__init__(**kwargs)
        self.tool_creator = next(
            (tool for tool in tools if isinstance(tool, ToolCreatorTool)),
            None
        )

    async def think(self) -> bool:
        """Process current state and decide next action"""
        # Update working directory
        self.working_dir = await self.bash.execute("pwd")
        
        # Create a modified template with single braces
        modified_template = NEXT_STEP_TEMPLATE.replace("{{", "{").replace("}}", "}")
        
        # Now format with single braces
        self.next_step_prompt = modified_template.format(
            observation=getattr(self, 'last_observation', '') or '',
            open_file='',
            working_dir=self.working_dir
        )
        
        # Store thinking step before executing it
        thinking_step = f"Step {len(self.thinking_steps) + 1}: {self.next_step_prompt}"
        self.thinking_steps.append(thinking_step)
        
        # For Gemini models, use a simpler approach without forcing tool calls
        if self.llm.api_type == "gemini":
            if self.next_step_prompt:
                user_msg = Message.user_message(self.next_step_prompt)
                self.messages += [user_msg]
                
            # Get a direct response from Gemini
            content = await self.llm.ask(
                messages=self.messages,
                system_msgs=[Message.system_message(self.system_prompt)]
                if self.system_prompt
                else None,
            )
            
            # Log response info
            logger.info(f"✨ {self.name}'s thoughts: {content}")
            
            # Create and add assistant message
            assistant_msg = Message.assistant_message(content)
            self.memory.add_message(assistant_msg)
            
            # No tool calls for Gemini mode, but return True to continue the conversation
            return True
        else:
            # For other models, use the standard tool call approach
            return await super().think()
        
    async def observe(self, observation: str) -> None:
        """Store the observation for future reference"""
        await super().observe(observation)
        self.last_observation = observation

    async def process_message(self, message: str) -> str:
        """Process a message with enhanced tracking and visualization support."""
        try:
            # Reset tracking for new message
            self.current_section = "Understanding Request"
            self.completed_tasks = []
            self.visualization_paths = []
            
            # Analyze if we need to create or improve tools
            if self.tool_creator and "create tool" in message.lower():
                result = await self.tool_creator.execute(
                    request=message,
                    visualization_required="visualize" in message.lower() or "plot" in message.lower(),
                    analytics_required="analyze" in message.lower() or "report" in message.lower()
                )
                if result["status"] == "success":
                    self.completed_tasks.append(f"Created new tool: {result['tool_info']['name']}")
            
            # Run the agent with the message
            response = await self.run(message)
            
            # Format response with progress and visualization information
            formatted_response = self._format_response_with_progress(response)
            
            # Add visualization paths if any were generated
            if self.visualization_paths:
                viz_section = "\n\n## Generated Visualizations:\n"
                for path in self.visualization_paths:
                    viz_section += f"- [View Visualization]({path})\n"
                formatted_response += viz_section
            
            return formatted_response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _format_response_with_progress(self, response: str) -> str:
        """Format response with enhanced progress tracking and metrics."""
        formatted_response = []
        
        # Add current section if available
        if self.current_section:
            formatted_response.append(f"## Working on: {self.current_section}")
        
        # Add main response
        formatted_response.append(response)
        
        # Add completed tasks
        if self.completed_tasks:
            formatted_response.append("\n### Completed Tasks:")
            for task in self.completed_tasks:
                formatted_response.append(f"✓ {task}")
        
        # Add execution metrics if available
        if self.current_metrics and self.current_metrics.end_time:
            execution_time = (self.current_metrics.end_time - self.current_metrics.start_time).total_seconds()
            formatted_response.append(f"\n### Execution Metrics:")
            formatted_response.append(f"- Tool: {self.current_metrics.tool_name}")
            formatted_response.append(f"- Execution Time: {execution_time:.2f} seconds")
            formatted_response.append(f"- Status: {'✓ Success' if self.current_metrics.success else '✗ Failed'}")
            
            if self.current_metrics.error_message:
                formatted_response.append(f"- Error: {self.current_metrics.error_message}")
        
        return "\n\n".join(formatted_response)

    async def execute_tool(self, tool_call) -> str:
        """Execute tool with enhanced metrics tracking and visualization handling."""
        try:
            # Initialize metrics
            self.current_metrics = ToolExecutionMetrics(
                tool_name=tool_call.function.name,
                start_time=datetime.now()
            )
            
            result = await super().execute_tool(tool_call)
            
            # Try to parse result as JSON
            try:
                result_data = json.loads(result)
                
                # Track visualizations
                if isinstance(result_data, dict):
                    if "visualization_path" in result_data:
                        self.visualization_paths.append(result_data["visualization_path"])
                    elif "file_url" in result_data:
                        self.visualization_paths.append(result_data["file_url"])
                
                # Update metrics
                self.current_metrics.success = result_data.get("status") == "success"
                self.current_metrics.error_message = result_data.get("message") if not self.current_metrics.success else None
                
            except json.JSONDecodeError:
                # Result is not JSON, just track basic metrics
                self.current_metrics.success = "error" not in result.lower()
                
            self.current_metrics.end_time = datetime.now()
            self.current_metrics.response_length = len(result)
            self.current_metrics.has_visualization = bool(self.visualization_paths)
            
            # Store metrics
            self.execution_history.append(self.current_metrics)
            
            return result
            
        except Exception as e:
            if self.current_metrics:
                self.current_metrics.success = False
                self.current_metrics.error_message = str(e)
                self.current_metrics.end_time = datetime.now()
                self.execution_history.append(self.current_metrics)
            
            logger.error(f"Error in execute_tool: {str(e)}")
            return f"Error executing tool: {str(e)}"
