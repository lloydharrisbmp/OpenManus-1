from typing import List, Optional
import json

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.financial_planner import NEXT_STEP_TEMPLATE, SYSTEM_PROMPT
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

    available_tools: ToolCollection = ToolCollection(
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
    )
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    max_steps: int = 30  # Increased to handle complex financial analysis tasks

    bash: Bash = Field(default_factory=Bash)
    working_dir: str = "."

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
            
        return await super().think()
        
    async def observe(self, observation: str) -> None:
        """Store the observation for future reference"""
        await super().observe(observation)
        self.last_observation = observation

    async def process_message(self, message: str) -> str:
        """Process a message and return a response."""
        try:
            # Reset tracking for a new message
            self.current_section = "Understanding Request"
            self.completed_tasks = []
            
            # Determine initial section based on the message
            if "tax" in message.lower() or "structure" in message.lower():
                self.current_section = "Tax Analysis"
            elif "market" in message.lower() or "investment" in message.lower():
                self.current_section = "Market Research"
            elif "estate" in message.lower() or "succession" in message.lower():
                self.current_section = "Estate Planning"
            elif "smsf" in message.lower() or "super" in message.lower():
                self.current_section = "SMSF Strategy"
            elif "report" in message.lower() or "document" in message.lower():
                self.current_section = "Document Creation"
            
            # Run the agent with the message
            response = await self.run(message)
            
            # Format response with progress information
            formatted_response = self._format_response_with_progress(response)
            return formatted_response
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _format_response_with_progress(self, response: str) -> str:
        """Format the response with progress tracking information."""
        # Only add section and task information if not already included
        if "## Working on:" not in response and self.current_section:
            response = f"## Working on: {self.current_section}\n\n{response}"
        
        # Add completed tasks if any
        if self.completed_tasks:
            task_info = "\n\n"
            for task in self.completed_tasks:
                task_info += f"✓ Task completed: {task}\n"
            
            # Insert task info before the last paragraph if possible
            parts = response.rsplit("\n\n", 1)
            if len(parts) > 1:
                response = parts[0] + task_info + "\n\n" + parts[1]
            else:
                response += task_info
        
        return response
    
    async def execute_tool(self, tool_call) -> str:
        """Override to track task completion and improve website URL display."""
        try:
            result = await super().execute_tool(tool_call)
            
            # If this was a website generation, format the output nicely
            if tool_call.function.name == "website_generator":
                try:
                    data = json.loads(result)
                    if data.get("file_url"):
                        self.completed_tasks.append("Generated website")
                        return f"""✨ Website generated successfully!

📂 Access your website:
1. Direct link: {data['file_url']}
2. File path: {data['index_file']}
3. Command: open {data['index_file']}

The website is also available in the 'Generated Documents' sidebar."""
                except:
                    pass
            
            return result
        except Exception as e:
            logger.error(f"Error in execute_tool: {str(e)}")
            return f"Error executing tool: {str(e)}"
