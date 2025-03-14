# Enhanced ToolCallAgent

## Overview

The enhanced ToolCallAgent is a powerful, concurrent, and error-resilient implementation that extends the base ReActAgent with advanced capabilities for parallel tool execution, detailed execution tracking, and sophisticated error handling.

This agent specializes in executing LLM-decided tool (function) calls, with the ability to run multiple tool calls concurrently for improved efficiency while maintaining robust error handling and execution statistics.

## Key Features

### 1. Parallel Tool Execution

The enhanced ToolCallAgent can execute multiple tool calls in parallel, dramatically improving performance:

- **Concurrency Control**: Uses `asyncio.Semaphore` to limit concurrent tasks to `concurrency_limit` (default: 3)
- **Smart Path Selection**: Automatically chooses between parallel and sequential execution based on the number of tools
- **Exception Safety**: Maintains robust execution even when individual tool calls fail

```python
# Example: Multiple tool calls executed in parallel
async def act(self) -> str:
    if len(self.tool_calls) > 1:
        # Use parallel execution for multiple tool calls
        results_map = await self._execute_parallel_tool_calls(self.tool_calls)
        return self._summarize_parallel_outcomes(results_map)
    else:
        # Use sequential execution for a single tool call
        return await self._execute_single_tool_call(self.tool_calls[0])
```

### 2. Partial Success/Failure Tracking

Robust handling of mixed success/failure scenarios when executing multiple tools:

- **Per-Tool Success Tracking**: Records success or failure for each individual tool call
- **Partial Results Utilization**: Continues with partial successes even if some tools fail
- **Comprehensive Summary**: Generates a clear summary of which tools succeeded and failed

```python
# Example of partial success summary
"Executed 3 tool calls with 2 successes and 1 failures:
✅ Tool Call fc_1: Succeeded
❌ Tool Call fc_2: Error: API endpoint is currently unavailable
✅ Tool Call fc_3: Succeeded"
```

### 3. Execution Telemetry

The agent maintains detailed execution histories for comprehensive tracking:

- **Tool Execution History**: Records success/failure rates per tool
- **Performance Metrics**: Tracks execution duration for performance analysis
- **Generated Reports**: Creates detailed execution summaries with success rates and timing information

```python
# Example of generating an execution report
report = agent.generate_execution_report()
print(f"Success rate: {report['overall_success_rate']:.2%}")
print(f"Total tool calls: {report['total_tool_calls']}")
```

### 4. Memory-Enhanced Chain-of-Thought

Stores intermediate results and partial reasoning steps for improved reasoning:

- **Intermediate Results Storage**: Stores partial results in memory for chain-of-thought reasoning
- **Structured Memory**: Maintains a dictionary of key-value results for easy access
- **Step-by-Step Tracking**: Records the progression of reasoning throughout execution

```python
# Example of storing intermediate results
self.store_intermediate_result("parallel_tool_results", {
    "total_count": len(calls),
    "success_count": 2,
    "failure_count": 1,
    "total_duration": 1.5
})
```

### 5. Enhanced Error Handling

Sophisticated error handling prevents tool failures from crashing the entire agent:

- **Detailed Error Logging**: Records errors with full stack traces and contexts
- **Graceful Degradation**: Continues execution with partial results when possible
- **Error Classification**: Distinguishes between different types of failures for appropriate handling

## Implementation Details

### Core Components

1. **Parallel Execution Engine**:
   - `_execute_parallel_tool_calls()`: Executes multiple tool calls in parallel with concurrency control
   - `_execute_single_tool_call()`: Optimized path for single tool call execution
   - `_summarize_parallel_outcomes()`: Generates a human-readable summary of execution results

2. **Tool Execution Tracking**:
   - `_track_tool_execution()`: Records detailed execution statistics for each tool
   - `generate_execution_report()`: Creates comprehensive execution reports with success rates and performance metrics

3. **Enhanced Memory**:
   - `store_intermediate_result()`: Stores chain-of-thought or intermediate data for improved reasoning
   - `partial_results_memory`: Dictionary storage for structured access to intermediate results

4. **Resource Management**:
   - `finalize()`: Unified cleanup of resources and report generation
   - `_handle_special_tool()`: Enhanced handling of special tools like termination

### Example Usage

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection, SearchTool, WeatherTool, Terminate

# Create a tool collection
tools = ToolCollection(
    SearchTool(),
    WeatherTool(),
    Terminate()
)

# Create the enhanced agent
agent = ToolCallAgent(
    name="TravelAssistant",
    description="An agent that helps with travel planning",
    available_tools=tools,
    concurrency_limit=3,  # Set parallel execution limit
    system_prompt="You are a helpful travel assistant."
)

# Run the agent
user_query = "What's the weather in London and can you search for tourist attractions there?"
agent.memory.add_message(Message.user_message(user_query))
result = await agent.step()

# Generate execution report
report = agent.generate_execution_report()
print(f"Tool calls: {report['total_tool_calls']}, Success rate: {report['overall_success_rate']:.2%}")

# Finalize and cleanup
await agent.finalize()
```

## Advanced Features

### Tool Execution Reports

Generate comprehensive reports about tool execution:

```python
# Generate detailed execution report
report = agent.generate_execution_report()

# Print tool-specific statistics
for tool_name, stats in report['tool_stats'].items():
    print(f"{tool_name}: {stats['successful_calls']}/{stats['total_calls']} successful")
    if 'avg_duration' in stats:
        print(f"  Average duration: {stats['avg_duration']:.2f}s")
```

### Chain-of-Thought Integration

Store intermediate results to enhance reasoning capabilities:

```python
# Store an intermediate result
agent.store_intermediate_result("search_results", {
    "query": "Paris tourist attractions",
    "result_count": 5,
    "top_result": "Eiffel Tower"
})

# Access stored results
if "search_results" in agent.partial_results_memory:
    search_data = agent.partial_results_memory["search_results"]
    print(f"Found {search_data['result_count']} results for {search_data['query']}")
```

## Performance Considerations

- The default concurrency limit is set to 3, which balances performance with resource usage
- For I/O-bound tools (like API calls), increasing the concurrency limit may improve performance
- For CPU-bound tools, keeping a lower concurrency limit is recommended
- The parallel execution path is only used when multiple tool calls are present; single tool calls use an optimized direct path

## Future Enhancements

Potential future improvements include:

1. **Dynamic Concurrency**: Automatically adjust concurrency limits based on system load
2. **Tool-Specific Retry Logic**: Implement automatic retries for transient failures with tool-specific policies
3. **Tool Priority Scheduling**: Execute high-priority tools before lower-priority ones when resources are limited
4. **Advanced Visualization**: Create visual representations of tool execution patterns and performance metrics
5. **Predictive Tool Selection**: Use execution history to predict which tools are likely to succeed or fail

## Demo Script

A comprehensive demo script is available at `app/examples/enhanced_toolcall_demo.py`, which showcases:

1. Single tool call execution
2. Parallel tool execution with concurrency control
3. Partial success/failure tracking with mixed results
4. Memory-enhanced chain-of-thought reasoning
5. Comprehensive execution reporting

Run the demo with:

```bash
python app/examples/enhanced_toolcall_demo.py
``` 