"""
Enhanced Manus Agent Demo

This script demonstrates the enhanced Manus agent's capabilities for:
1. Parallel tool execution
2. Error handling and partial success tracking 
3. Execution telemetry
4. Unified tool cleanup

Run this script to see the enhanced Manus in action.
"""

import asyncio
import json
from datetime import datetime
import sys
import os

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.agent.manus import Manus
from app.logger import logger


async def demo_parallel_execution():
    """
    Demonstrate the parallel execution capabilities of the enhanced Manus agent.
    """
    print("\n=== DEMONSTRATING PARALLEL TOOL EXECUTION ===\n")
    
    # Create a Manus agent with a higher concurrency limit for demonstration
    manus = Manus(concurrency_limit=5)
    
    # Create a list of simulated tool calls that would work with real tools
    tool_calls = [
        {
            "tool_name": "GoogleSearch",
            "args": {"query": "Australian retirement planning strategies"}
        },
        {
            "tool_name": "PythonExecute",
            "args": {"code": "import time\ntime.sleep(1)\nprint('Hello from Python!')\n2 + 2"}
        },
        {
            "tool_name": "FileSaver",
            "args": {"content": "This is a test file", "filename": "test_output.txt"}
        }
    ]
    
    print(f"Running {len(tool_calls)} tools in parallel...")
    start_time = datetime.now()
    
    # Execute the tools in parallel
    results = await manus.run_parallel_tools(tool_calls)
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    print(f"\nExecution completed in {execution_time:.2f} seconds")
    print(f"Success: {results['success_count']}, Failures: {results['failure_count']}")
    
    # Print each result
    for i, result_data in enumerate(results["results"]):
        tool = result_data.get("tool", "Unknown")
        success = result_data.get("success", False)
        status = "SUCCESS" if success else "FAILED"
        print(f"\nTool {i+1}: {tool} - {status}")
        
        if success:
            # Truncate long results for readability
            result_str = result_data.get("result", "")
            if len(result_str) > 100:
                result_str = result_str[:100] + "... [truncated]"
            print(f"Result: {result_str}")
        else:
            print(f"Error: {result_data.get('error', 'Unknown error')}")
    
    # Demonstrate accessing execution history
    print("\n=== TOOL EXECUTION HISTORY ===\n")
    for tool_name, history in manus.tool_execution_history.items():
        success_count = sum(1 for exec_data in history if exec_data["result"].get("success", False))
        print(f"{tool_name}: {success_count}/{len(history)} successful executions")
    
    # Clean up
    await manus.finalize()
    return manus


async def demo_error_handling():
    """
    Demonstrate the error handling capabilities of the enhanced Manus agent.
    """
    print("\n=== DEMONSTRATING ERROR HANDLING ===\n")
    
    # Create a Manus agent
    manus = Manus()
    
    # Create a list of tool calls with some that will succeed and some that will fail
    tool_calls = [
        {
            "tool_name": "PythonExecute",
            "args": {"code": "print('This should succeed')"}
        },
        {
            "tool_name": "PythonExecute",
            "args": {"code": "this_will_cause_an_error()"}  # This will fail
        },
        {
            "tool_name": "FileSaver",
            "args": {"content": "This should succeed", "filename": "success.txt"}
        },
        {
            "tool_name": "NonExistentTool",  # This tool doesn't exist
            "args": {"param": "value"}
        }
    ]
    
    print(f"Running {len(tool_calls)} tools with some errors...")
    
    # Execute the tools in parallel
    results = await manus.run_parallel_tools(tool_calls)
    
    print(f"\nPartial execution results:")
    print(f"Success: {results['success_count']}, Failures: {results['failure_count']}")
    
    # Print each result
    for i, result_data in enumerate(results["results"]):
        tool = result_data.get("tool", "Unknown")
        success = result_data.get("success", False)
        status = "SUCCESS" if success else "FAILED"
        print(f"\nTool {i+1}: {tool} - {status}")
        
        if success:
            print(f"Result: {result_data.get('result', '')[:50]}...")
        else:
            print(f"Error: {result_data.get('error', 'Unknown error')}")
    
    # Show step execution stats
    print("\n=== STEP EXECUTION STATISTICS ===\n")
    for i, stats in enumerate(manus.step_execution_stats):
        print(f"Step {i+1}:")
        print(f"  Total tools: {stats['total_tools']}")
        print(f"  Success: {stats['success_count']}")
        print(f"  Failures: {stats['failure_count']}")
    
    # Clean up
    await manus.finalize()
    return manus


async def demo_tool_integration():
    """
    Demonstrate using the enhanced Manus agent with the run() method
    for a more realistic usage scenario.
    """
    print("\n=== DEMONSTRATING MANUS AGENT INTEGRATION ===\n")
    
    # Create a Manus agent
    manus = Manus(
        concurrency_limit=3,
        max_steps=5  # Limit for demonstration
    )
    
    print("Running Manus agent on a task that requires multiple tools...")
    
    # In a real scenario, the agent would determine which tools to use based on the request
    request = ("Find information about Australian superannuation, "
               "save it to a file, and create a Python script to analyze the data.")
    
    try:
        print(f"User request: {request}")
        print("\nProcessing request...\n")
        
        # This would normally trigger the agent's think-act cycle
        # For demo purposes, we'll simulate a few steps
        
        print("This would trigger the agent to:")
        print("1. Use GoogleSearch to find information about Australian superannuation")
        print("2. Use FileSaver to save the information to a file")
        print("3. Use PythonExecute to create and run an analysis script")
        print("4. Execute all of these tools efficiently in parallel when possible")
        
        # In a real scenario, you would use:
        # result = await manus.run(request)
        # print(result)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Clean up
    cleanup_results = await manus.finalize()
    print("\nCleanup results:")
    print(json.dumps(cleanup_results, indent=2))
    
    return manus


async def generate_execution_report(manus):
    """
    Generate a comprehensive execution report from Manus agent stats.
    """
    # Check if there's any execution data
    if not manus.tool_execution_history:
        return {"error": "No execution data available"}
    
    total_tools = sum(len(history) for history in manus.tool_execution_history.values())
    if total_tools == 0:
        return {"error": "No tool executions recorded"}
    
    successful_executions = sum(
        sum(1 for exec_data in history if exec_data["result"].get("success", False))
        for history in manus.tool_execution_history.values()
    )
    
    report = {
        "total_steps": len(manus.step_execution_stats),
        "total_tool_calls": total_tools,
        "success_rate": successful_executions / total_tools if total_tools > 0 else 0,
        "tools_breakdown": {}
    }
    
    # Add tool-specific stats
    for tool_name, history in manus.tool_execution_history.items():
        if not history:
            continue
            
        success_count = sum(1 for exec_data in history if exec_data["result"].get("success", False))
        report["tools_breakdown"][tool_name] = {
            "calls": len(history),
            "success_rate": success_count / len(history) if history else 0,
            "success_count": success_count,
            "failure_count": len(history) - success_count
        }
    
    return report


async def main():
    """Run the enhanced Manus agent demo."""
    print("\n=== ENHANCED MANUS AGENT DEMONSTRATION ===\n")
    print("This demo showcases the enhanced capabilities of the Manus agent.")
    
    # Demo 1: Parallel Execution
    manus1 = await demo_parallel_execution()
    
    # Demo 2: Error Handling
    manus2 = await demo_error_handling()
    
    # Demo 3: Tool Integration
    manus3 = await demo_tool_integration()
    
    # Generate report for one of the demos
    print("\n=== EXECUTION REPORT ===\n")
    report = await generate_execution_report(manus2)
    print(json.dumps(report, indent=2))
    
    print("\n=== DEMO COMPLETE ===\n")
    print("The enhanced Manus agent offers:")
    print("1. Parallel tool execution with concurrency control")
    print("2. Robust error handling with partial success tracking")
    print("3. Detailed execution telemetry")
    print("4. Unified tool cleanup")


if __name__ == "__main__":
    asyncio.run(main()) 