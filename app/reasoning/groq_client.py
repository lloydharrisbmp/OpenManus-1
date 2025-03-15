import os
from typing import Dict, Any, List, Optional
import logging
from groq import Groq

logger = logging.getLogger(__name__)

class GroqReasoner:
    """
    A reasoning layer using Groq's qwen-qwq-32b model for enhanced decision making
    and tool orchestration.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq client with API key.
        
        Args:
            api_key (Optional[str]): Groq API key. If None, tries to get from environment.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        self.model = "qwen-qwq-32b"
        
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

    async def reason(self, 
                    context: Dict[str, Any],
                    available_tools: List[Dict[str, Any]],
                    query: str) -> Dict[str, Any]:
        """
        Use the Groq model to reason about the next actions to take.
        Falls back to a simple response if Groq is not available.
        """
        if not self.client:
            # Fallback response when Groq is not available
            return {
                "analysis": "Groq reasoning is not available. Proceeding with basic execution.",
                "next_actions": ["Execute tools sequentially"],
                "tool_calls": [],
                "explanations": ["No Groq API key provided. Using fallback logic."]
            }
            
        try:
            # Construct the prompt
            tools_desc = "\n".join([
                f"Tool: {tool['name']}\nDescription: {tool['description']}\n"
                for tool in available_tools
            ])
            
            prompt = f"""Given the following context and available tools, reason about the best course of action:

Context:
{context}

Available Tools:
{tools_desc}

Query:
{query}

Provide your reasoning and recommended actions in JSON format including:
1. Analysis of the situation
2. Recommended next actions
3. Specific tool calls to make (if any)
4. Explanations for each decision
"""

            # Get completion from Groq
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            # Parse and validate response
            response = completion.choices[0].message.content
            logger.info(f"Groq reasoning response: {response}")
            
            return response

        except Exception as e:
            logger.error(f"Error in Groq reasoning: {e}", exc_info=True)
            return {
                "error": str(e),
                "success": False
            }

    async def parallel_tool_planning(self,
                                   tools: List[Dict[str, Any]],
                                   task: str) -> Dict[str, Any]:
        """
        Plan parallel tool execution for complex tasks.
        Falls back to sequential execution if Groq is not available.
        """
        if not self.client:
            # Fallback to sequential execution plan
            return {
                "tool_calls": [{"tool": tool["name"], "parameters": {}} for tool in tools],
                "dependencies": {},
                "parallel_groups": [[tool["name"]] for tool in tools],
                "error_handling": {"strategy": "continue-on-error"}
            }
            
        try:
            prompt = f"""Given these tools and this task, create a parallel execution plan:

Tools:
{tools}

Task:
{task}

Create a JSON execution plan that includes:
1. Tools that can be run in parallel
2. Dependencies between tool calls
3. Expected inputs and outputs
4. Error handling considerations
"""

            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            return completion.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in parallel tool planning: {e}", exc_info=True)
            return {
                "error": str(e),
                "success": False
            } 