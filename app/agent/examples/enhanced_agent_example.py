"""
Enhanced Agent Example

This example demonstrates how to create a custom agent using the supercharged BaseAgent class
with its enhanced concurrency and error handling capabilities.
"""

import asyncio
from typing import Dict, Any, List

from app.agent.base import BaseAgent
from app.logger import logger

class FinancialAdvisorAgent(BaseAgent):
    """
    Example financial advisor agent that showcases the enhanced BaseAgent capabilities.
    
    This agent takes advantage of:
    - Concurrency for parallel tool execution
    - Enhanced error handling
    - Reasoning transparency with thinking steps
    - Better context management
    """
    
    def __init__(self, **kwargs):
        # Set default name and system prompt if not provided
        kwargs["name"] = kwargs.get("name", "Financial Advisor")
        kwargs["system_prompt"] = kwargs.get("system_prompt", 
            "You are an Australian financial planning assistant. Provide accurate advice "
            "based on Australian tax laws, superannuation regulations, and financial best practices."
        )
        
        # Initialize with a higher concurrency limit for parallel operations
        kwargs["concurrency_limit"] = kwargs.get("concurrency_limit", 5)
        
        super().__init__(**kwargs)
        
        # Add any domain-specific configuration here
        self.financial_data_sources = {
            "asx": "Australian Stock Exchange",
            "rba": "Reserve Bank of Australia",
            "ato": "Australian Taxation Office"
        }
        
    async def step(self) -> str:
        """
        A single step in the agent's workflow. This implementation demonstrates how
        to use enhanced concurrency, error handling, and the LLM for complex tasks.
        """
        # If this is the first step, plan the approach
        if self.current_step == 1:
            return await self._plan_approach()
            
        # If this is the second step, gather information (using concurrency)
        elif self.current_step == 2:
            return await self._gather_information()
            
        # For subsequent steps, analyze and provide advice
        else:
            return await self._analyze_and_advise()
    
    async def _plan_approach(self) -> str:
        """Plan the approach to answering the user's financial question."""
        # Use the reasoner to think about the approach
        planning_result = await self.think(
            "What financial information do I need to answer the user's question effectively?"
        )
        
        # Build a custom prompt for better planning
        planning_prompt = (
            "Based on the user's question, identify the key financial topics involved, "
            "relevant Australian regulations, and outline your plan to address their needs. "
            "Be specific about what information you need to gather."
        )
        
        # Get LLM's response for the plan
        full_prompt = f"{planning_prompt}\n\n{self._build_conversation_prompt()}"
        plan = await self.llm.generate(full_prompt)
        
        # Update memory with the plan
        self.update_memory("assistant", plan)
        
        return f"Planned approach: {plan}"
    
    async def _gather_information(self) -> str:
        """
        Gather information from multiple sources in parallel.
        This demonstrates the enhanced concurrency capabilities.
        """
        try:
            # Extract topics from the conversation for focused research
            topics_result = await self.think("What specific financial topics should I research?")
            if "topics" not in topics_result or not topics_result.get("success", False):
                # Fallback if thinking doesn't produce useful results
                topics = ["retirement", "superannuation", "taxation"]
                logger.warning(f"Using default topics: {topics}")
            else:
                topics = topics_result.get("topics", [])
            
            # Simulate parallel tools execution for gathering info from different sources
            # In a real implementation, these would be actual tool calls to different APIs
            info_gathering_tasks = [
                self._get_info_from_source("asx", topics),
                self._get_info_from_source("rba", topics),
                self._get_info_from_source("ato", topics)
            ]
            
            # Execute all information gathering tasks in parallel with proper concurrency control
            results = await asyncio.gather(*info_gathering_tasks)
            
            # Combine and process results
            combined_info = self._process_gathered_info(results)
            
            # Update memory with gathered information
            self.update_memory(
                "system", 
                f"Gathered information: {combined_info}",
                source="research"
            )
            
            return f"Gathered information from {len(results)} sources"
            
        except Exception as e:
            logger.error(f"Error gathering information: {str(e)}", exc_info=True)
            self.update_context({
                "error": str(e),
                "step": "information_gathering"
            })
            return f"Error gathering information: {str(e)}"
    
    async def _analyze_and_advise(self) -> str:
        """Analyze gathered information and provide financial advice."""
        # Construct a prompt that uses all gathered context
        analysis_prompt = (
            "Based on all the information we've gathered and the user's specific question, "
            "provide comprehensive financial advice tailored to their situation. "
            "Include specific references to Australian tax laws and regulations where appropriate. "
            "Structure your response with clear recommendations and next steps."
        )
        
        # Get LLM's advice
        full_prompt = f"{analysis_prompt}\n\n{self._build_conversation_prompt()}"
        advice = await self.llm.generate(full_prompt)
        
        # Update memory with the advice
        self.update_memory("assistant", advice)
        
        return advice
    
    async def _get_info_from_source(self, source: str, topics: List[str]) -> Dict[str, Any]:
        """
        Simulated function to get information from a specific source.
        In a real implementation, this would call actual APIs or tools.
        """
        # Simulate API delay for realism
        await asyncio.sleep(0.5)
        
        # Simulate success/occasional failure
        if source == "asx" and "investment" in topics:
            source_info = {
                "source": source,
                "data": {
                    "market_update": "ASX 200 is currently trading at 7,500 points",
                    "volatility": "Market volatility has been decreasing",
                    "sectors": ["Technology", "Finance", "Healthcare"]
                },
                "success": True
            }
        elif source == "rba":
            source_info = {
                "source": source,
                "data": {
                    "cash_rate": "The current RBA cash rate is 4.25%",
                    "inflation": "Inflation rate is currently 3.5%",
                    "outlook": "Economic growth is projected at 2.8% for the next year"
                },
                "success": True
            }
        elif source == "ato" and any(t in topics for t in ["tax", "taxation", "superannuation"]):
            source_info = {
                "source": source,
                "data": {
                    "tax_rates": "Current tax rates range from 19% to 45%",
                    "super_contribution_cap": "The concessional contribution cap is $27,500",
                    "recent_changes": "Changes to the taxation of super for balances above $3M"
                },
                "success": True
            }
        else:
            source_info = {
                "source": source,
                "data": {},
                "error": f"No relevant information found for topics {topics}",
                "success": False
            }
        
        # Add to agent context
        self.update_context({
            "information_source": source,
            "information_result": source_info
        })
        
        return source_info
    
    def _process_gathered_info(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and combine information from multiple sources."""
        combined_data = {}
        success_count = 0
        
        for result in results:
            if result.get("success", False):
                source = result.get("source", "unknown")
                combined_data[source] = result.get("data", {})
                success_count += 1
        
        processed_info = {
            "total_sources": len(results),
            "successful_sources": success_count,
            "data": combined_data
        }
        
        # Add to thinking steps for transparency
        self.thinking_steps.append(f"Processed information from {success_count}/{len(results)} sources")
        
        return processed_info


async def main():
    """Example usage of the FinancialAdvisorAgent."""
    # Create the agent
    agent = FinancialAdvisorAgent(
        description="Australian Financial Planning Assistant",
        max_steps=5  # Limit steps for demonstration
    )
    
    # Run the agent with a user query
    query = "I'm 45 years old with $200,000 in my super. What strategies should I consider to maximize my retirement savings in the next 10 years?"
    
    print(f"User Query: {query}")
    print("\nRunning financial advisor agent...\n")
    
    result = await agent.run(query)
    
    print("\n=== Agent Execution Result ===")
    print(result)
    
    print("\n=== Agent Thinking Steps ===")
    for i, step in enumerate(agent.thinking_steps, 1):
        print(f"{i}. {step}")


if __name__ == "__main__":
    asyncio.run(main()) 