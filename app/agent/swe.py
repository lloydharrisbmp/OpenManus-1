from typing import List, Optional

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

    available_tools: ToolCollection = ToolCollection(
        AustralianMarketAnalysisTool(),
        MarketAnalysisTool(),  # Keep for international markets
        PortfolioOptimizationTool(),
        TaxOptimizationTool(),
        ReportGeneratorTool(),
        DocumentAnalyzerTool(),
        WebsiteGeneratorTool(),
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
