from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json
from pathlib import Path

from pydantic import Field, BaseModel
import asyncio

from app.agent.toolcall import ToolCallAgent
from app.prompt.financial_planner import NEXT_STEP_TEMPLATE, SYSTEM_PROMPT
from app.schema import Message
from app.tool import BashTool, StrReplaceEditor, Terminate, ToolCollection
from app.tool.financial_tools import (
    AustralianMarketAnalysisTool,
    MarketAnalysisTool,
    PortfolioOptimizationTool,
    ReportGeneratorTool,
    TaxOptimizationTool,
)
from app.tool.document_analyzer import DocumentAnalyzerTool
from app.tool.tool_creator import ToolCreatorTool
from app.tool.property_analyzer import PropertyAnalyzerTool
from app.tool.financial_integrations import FinancialIntegrationsTool
from app.tool.superannuation_analyzer import SuperannuationAnalyzerTool
from app.tool.web_search import WebSearchTool
from app.tool.website_generator import WebsiteGeneratorTool
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

    bash: BashTool = Field(default_factory=BashTool)
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
            BashTool(),
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
        """Process an observation and update agent state."""
        self.last_observation = observation
        if hasattr(self, "update_memory") and callable(self.update_memory):
            self.update_memory(role="system", content=f"Observation: {observation}")

    async def process_message(self, message: str) -> str:
        """Process a message with enhanced tracking and visualization support."""
        try:
            # Reset tracking for new message
            self.current_section = "Understanding Request"
            self.completed_tasks = []
            self.visualization_paths = []
            
            # Update the agent's memory with the user message
            self.update_memory(role="user", content=message)
            
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
            response = await self.run()  # No need to pass message again as it's already in memory
            
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
        """Execute a tool call and track metrics."""
        if isinstance(tool_call, dict) and 'name' in tool_call and 'arguments' in tool_call:
            tool_name = tool_call['name']
            try:
                args = json.loads(tool_call['arguments']) if tool_call['arguments'] else {}
            except json.JSONDecodeError:
                args = {}

            # Record start of execution
            self.current_metrics = ToolExecutionMetrics(
                tool_name=tool_name,
                start_time=datetime.now()
            )

            try:
                # Find the tool in available_tools
                tool = None
                if hasattr(self, 'available_tools') and isinstance(self.available_tools, ToolCollection):
                    tool = self.available_tools.get_tool(tool_name)
                
                if tool:
                    result = await tool.execute(**args)
                    
                    # Update metrics
                    if self.current_metrics:
                        self.current_metrics.end_time = datetime.now()
                        self.current_metrics.success = True
                        # Check if result is a string or has a specific attribute
                        if hasattr(result, 'output'):
                            response = str(result.output)
                            self.current_metrics.response_length = len(response)
                        elif isinstance(result, str):
                            response = result
                            self.current_metrics.response_length = len(response)
                        elif isinstance(result, dict):
                            response = json.dumps(result)
                            self.current_metrics.response_length = len(response)
                            # Check for visualization paths in result
                            if 'visualization_path' in result or 'chart_path' in result or 'report_path' in result:
                                path = result.get('visualization_path') or result.get('chart_path') or result.get('report_path')
                                if path and path not in self.visualization_paths:
                                    self.visualization_paths.append(path)
                                    self.current_metrics.has_visualization = True
                        else:
                            response = str(result)
                            self.current_metrics.response_length = len(response)
                        
                        self.execution_history.append(self.current_metrics)
                        self.current_metrics = None
                    
                    return f"Tool {tool_name} executed successfully with result: {result}"
                else:
                    if self.current_metrics:
                        self.current_metrics.end_time = datetime.now()
                        self.current_metrics.success = False
                        self.current_metrics.error_message = f"Tool {tool_name} not found"
                        self.execution_history.append(self.current_metrics)
                        self.current_metrics = None
                    
                    return f"Error: Tool {tool_name} not found"
            except Exception as e:
                if self.current_metrics:
                    self.current_metrics.end_time = datetime.now()
                    self.current_metrics.success = False
                    self.current_metrics.error_message = str(e)
                    self.execution_history.append(self.current_metrics)
                    self.current_metrics = None
                
                return f"Error executing tool {tool_name}: {str(e)}"
        else:
            return "Invalid tool call format"
