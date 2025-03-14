from typing import List

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
    - Estate planning and succession"""

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_TEMPLATE

    available_tools: ToolCollection = ToolCollection(
        AustralianMarketAnalysisTool(),
        MarketAnalysisTool(),  # Keep for international markets
        PortfolioOptimizationTool(),
        TaxOptimizationTool(),
        ReportGeneratorTool(),
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
        self.next_step_prompt = self.next_step_prompt.format(
            current_dir=self.working_dir
        )

        return await super().think()
