from typing import Dict, Any, Optional
import logging

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

class FinancialPlanningAgent(BaseAgent):
    """Enhanced Financial Planning Agent with improved error handling and tool management."""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        super().__init__(groq_api_key=groq_api_key)
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

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a financial planning query with enhanced error handling."""
        try:
            # Get reasoning about the query
            reasoning = await self.think(query)
            if "error" in reasoning:
                return reasoning

            results = []
            for action in reasoning.get("next_actions", []):
                if "tool_calls" in action:
                    # Execute tools in parallel when possible
                    tool_results = await self.execute_parallel_tools(action["tool_calls"])
                    results.append(tool_results)

            response = {
                "reasoning": reasoning,
                "results": results,
                "success": True
            }

            # Update context with final response
            self.update_context({
                "query": query,
                "response": response
            })

            return response

        except Exception as e:
            error_detail = {
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__
            }
            logger.error(f"Query processing failed: {error_detail}", exc_info=True)
            self.update_context({"error": error_detail})
            return {"error": str(e), "success": False} 