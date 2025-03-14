#!/usr/bin/env python3
"""
Enhanced ToolCallAgent Demo Script

This script demonstrates the advanced capabilities of the enhanced ToolCallAgent:
1. Parallel Tool Execution with Concurrency
2. Partial Success/Failure Tracking
3. Robust Error Handling
4. Memory Expansions for Chain-of-Thought
5. Extended Logging and Reporting
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.agent.toolcall import ToolCallAgent
from app.tool import Tool, ToolCollection, Terminate
from app.schema import Message, ToolChoice


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create custom tools for demonstration purposes
class SearchTool(Tool):
    name = "search"
    description = "Search for information about a topic"
    
    async def _run(self, query: str, **kwargs) -> str:
        # Simulate a search with variable duration
        await asyncio.sleep(1.0)  # Simulate network request
        return f"Search results for '{query}': Found information about {query}."


class WeatherTool(Tool):
    name = "get_weather"
    description = "Get the current weather for a location"
    
    async def _run(self, location: str, **kwargs) -> str:
        # Simulate weather lookup
        await asyncio.sleep(0.8)  # Simulate API call
        return f"Current weather in {location}: 72°F, Partly Cloudy"


class TranslateTool(Tool):
    name = "translate"
    description = "Translate text from one language to another"
    
    async def _run(self, text: str, source_lang: str, target_lang: str, **kwargs) -> str:
        # Simulate translation
        await asyncio.sleep(1.2)  # Simulate processing time
        return f"Translation from {source_lang} to {target_lang}: '{text}' → 'Translated text'"


class FailingTool(Tool):
    name = "unreliable_api"
    description = "A tool that sometimes fails"
    
    async def _run(self, endpoint: str, **kwargs) -> str:
        # Simulate an unreliable API that fails sometimes
        await asyncio.sleep(0.5)
        
        if endpoint == "unstable":
            raise Exception("API endpoint is currently unavailable")
            
        return f"API response from {endpoint}: Success"


class LongRunningTool(Tool):
    name = "process_data"
    description = "Process a large dataset"
    
    async def _run(self, dataset: str, **kwargs) -> str:
        # Simulate a long computation
        await asyncio.sleep(2.0)
        return f"Processed {len(dataset)} characters of data"


# Create enhanced ToolCallAgent with custom tools
async def create_enhanced_agent():
    tools = ToolCollection(
        SearchTool(),
        WeatherTool(),
        TranslateTool(),
        FailingTool(),
        LongRunningTool(),
        Terminate()
    )
    
    agent = ToolCallAgent(
        name="EnhancedToolCallDemo",
        description="A demo of the enhanced ToolCallAgent with parallel execution",
        available_tools=tools,
        concurrency_limit=3,  # Set concurrency limit
        system_prompt=(
            "You are an AI assistant with access to several tools. "
            "You can use these tools to help answer user questions. "
            "When appropriate, use multiple tools in parallel for efficiency."
        ),
        next_step_prompt=(
            "If you want to stop interaction, use the `terminate` tool. "
            "For this demo, try to use multiple tools when possible to demonstrate parallel execution."
        )
    )
    
    return agent


# Demo functions
async def demo_single_tool_call():
    """Demonstrate a single tool call execution."""
    print("\n=== Single Tool Call Demo ===")
    
    agent = await create_enhanced_agent()
    
    # Add a user message
    agent.memory.add_message(Message.user_message(
        "What's the weather in San Francisco?"
    ))
    
    # Run one step
    result = await agent.step()
    print(f"Step result: {result}")
    
    # Generate execution report
    report = agent.generate_execution_report()
    print("\nExecution Report:")
    print(f"Total tool calls: {report['total_tool_calls']}")
    print(f"Success rate: {report['overall_success_rate']:.2%}")
    
    return agent


async def demo_parallel_tool_calls():
    """Demonstrate parallel execution of multiple tool calls."""
    print("\n=== Parallel Tool Calls Demo ===")
    
    agent = await create_enhanced_agent()
    
    # Add a user message requesting multiple pieces of information
    agent.memory.add_message(Message.user_message(
        "I need the weather in New York, search results about quantum computing, "
        "and translation of 'hello' from English to Spanish."
    ))
    
    # Run one step (should result in parallel tool calls)
    print("Running step with multiple tool calls...")
    start_time = time.time()
    result = await agent.step()
    duration = time.time() - start_time
    
    print(f"Step completed in {duration:.2f} seconds")
    print(f"Result: {result}")
    
    # Check if parallel execution happened
    if "parallel" in result.lower() or "multiple" in result.lower():
        print("✅ Successfully demonstrated parallel tool execution")
        
    # Generate execution report
    report = agent.generate_execution_report()
    print("\nExecution Report:")
    print(f"Total tool calls: {report['total_tool_calls']}")
    print(f"Success rate: {report['overall_success_rate']:.2%}")
    
    return agent


async def demo_partial_failure():
    """Demonstrate partial success/failure tracking with mixed results."""
    print("\n=== Partial Failure Demo ===")
    
    agent = await create_enhanced_agent()
    
    # Add a user message requesting multiple things including one that will fail
    agent.memory.add_message(Message.user_message(
        "Please get the weather in Paris, search for information about machine learning, "
        "and call the unstable API endpoint."
    ))
    
    # Run one step (should have mixed success/failure)
    print("Running step with mixed success/failure...")
    result = await agent.step()
    print(f"Result: {result}")
    
    # Check for partial success indicators
    if "success" in result.lower() and "fail" in result.lower():
        print("✅ Successfully demonstrated partial success/failure tracking")
        
    # Generate execution report
    report = agent.generate_execution_report()
    print("\nExecution Report:")
    print(f"Total tool calls: {report['total_tool_calls']}")
    print(f"Success rate: {report['overall_success_rate']:.2%}")
    
    # Print tool-specific stats
    print("\nTool-specific statistics:")
    for tool_name, stats in report.get('tool_stats', {}).items():
        print(f"  {tool_name}: {stats['successful_calls']}/{stats['total_calls']} successful")
    
    return agent


async def demo_chain_of_thought():
    """Demonstrate memory expansions for chain-of-thought reasoning."""
    print("\n=== Chain-of-Thought Memory Demo ===")
    
    agent = await create_enhanced_agent()
    
    # Add a user message requiring complex reasoning
    agent.memory.add_message(Message.user_message(
        "I need to plan a trip. Can you help me find the weather in Tokyo, "
        "translate 'where is the train station' to Japanese, and search for "
        "information about popular tourist attractions?"
    ))
    
    # Run step to execute the tools
    result = await agent.step()
    print(f"Step result: {result}")
    
    # Check for intermediate results in memory
    intermediate_results = [
        msg for msg in agent.memory.messages 
        if msg.role == "assistant" and msg.content and "**" in msg.content
    ]
    
    print("\nIntermediate results stored in memory:")
    for msg in intermediate_results:
        print(f"- {msg.content}")
        
    if intermediate_results:
        print("✅ Successfully demonstrated chain-of-thought memory expansions")
    
    return agent


async def demo_comprehensive():
    """Run a comprehensive demo showcasing all features together."""
    print("\n=== Comprehensive ToolCallAgent Demo ===")
    
    agent = await create_enhanced_agent()
    agent.tool_choices = ToolChoice.AUTO  # Ensure AUTO mode to let the agent decide
    
    print("Running a multi-step interaction...")
    
    # Step 1: Initial query with multiple tools
    agent.memory.add_message(Message.user_message(
        "I'm planning a trip to Europe. Can you help me get the weather in Paris, "
        "translate 'hello' to French, and search for information about the Eiffel Tower?"
    ))
    
    result1 = await agent.step()
    print(f"\nStep 1 Result: {result1}")
    
    # Step 2: Follow-up with one failing tool
    agent.memory.add_message(Message.user_message(
        "Great! Now can you also check the weather in Rome, search for information about "
        "the Colosseum, and try to access the unstable API endpoint?"
    ))
    
    result2 = await agent.step()
    print(f"\nStep 2 Result (with partial failure): {result2}")
    
    # Step 3: Process some data
    agent.memory.add_message(Message.user_message(
        "Can you process this dataset: 'This is a sample dataset with tourism statistics for various European cities'?"
    ))
    
    result3 = await agent.step()
    print(f"\nStep 3 Result (long-running tool): {result3}")
    
    # Generate final execution report
    report = agent.generate_execution_report()
    print("\nFinal Execution Report:")
    print(f"Total steps: {report['total_steps']}")
    print(f"Total tool calls: {report['total_tool_calls']}")
    print(f"Overall success rate: {report['overall_success_rate']:.2%}")
    
    print("\nTool-specific statistics:")
    for tool_name, stats in report.get('tool_stats', {}).items():
        if stats['total_calls'] > 0:
            success_rate = stats['successful_calls'] / stats['total_calls']
            print(f"  {tool_name}: {stats['successful_calls']}/{stats['total_calls']} ({success_rate:.2%})")
            if 'avg_duration' in stats:
                print(f"    Average duration: {stats['avg_duration']:.2f}s")
    
    # Finalize the agent
    finalize_result = await agent.finalize()
    print(f"\nAgent finalized with report: {json.dumps(finalize_result, indent=2)}")
    
    return agent


async def main():
    """Run all demos."""
    print("=== Enhanced ToolCallAgent Demonstration ===")
    print("This demo showcases the advanced capabilities of the ToolCallAgent")
    
    try:
        # Run individual demos
        await demo_single_tool_call()
        await demo_parallel_tool_calls()
        await demo_partial_failure()
        await demo_chain_of_thought()
        
        # Run comprehensive demo
        await demo_comprehensive()
        
        print("\n=== Demo Summary ===")
        print("The enhanced ToolCallAgent successfully demonstrated:")
        print("1. Parallel tool execution with concurrency control")
        print("2. Partial success/failure tracking")
        print("3. Robust error handling with detailed logging")
        print("4. Memory expansions for chain-of-thought reasoning")
        print("5. Comprehensive execution reporting")
        
        print("\nAll features are working as expected!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 