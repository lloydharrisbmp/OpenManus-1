# Enhanced ReActAgent

## Overview

The enhanced ReActAgent is a powerful, concurrent, and error-resilient implementation that extends the base agent framework with advanced capabilities for parallel task execution, detailed execution histories, and sophisticated error handling.

This agent follows the ReAct (Reasoning + Acting) pattern, where it alternates between reasoning about a problem and taking actions to solve it. The enhanced version adds robust concurrency support, partial success/failure tracking, sophisticated stuck-state detection, automatic retry logic, dynamic concurrency adjustment, tool priority scheduling, and visualization tools.

## Key Features

### 1. Parallel Task Execution

The enhanced ReActAgent can execute multiple subtasks in parallel, dramatically improving performance:

- **Concurrency Control**: Uses `asyncio.Semaphore` to limit concurrent tasks to `concurrency_limit` (default: 3)
- **Task-Based Parallelism**: Simplifies running multiple independent actions concurrently
- **Exception Safety**: Maintains robust execution even when individual tasks fail

```python
# Example: executing multiple tasks in parallel
async def act(self) -> str:
    tasks = [
        lambda: self._call_tool("search", {"query": "retirement planning"}),
        lambda: self._call_tool("data_fetch", {"source": "financial_data"}),
        lambda: self._call_tool("analyze", {"data_type": "portfolio"})
    ]
    results = await self._execute_concurrent_tasks(tasks)
    summary = self.handle_partial_failures(results)
    return f"Completed with {summary['success_count']} successes, {summary['failure_count']} failures"
```

### 2. Comprehensive Error Handling

Robust error handling prevents a single task failure from breaking the entire workflow:

- **Detailed Error Logging**: Records errors with full stack traces and contexts
- **Partial Success Tracking**: Continues execution even if some tasks fail
- **Meaningful Error Reports**: Provides clear information about what went wrong

### 3. Execution Telemetry

The agent maintains detailed execution histories for comprehensive tracking:

- **Step Execution History**: Tracks each step's success or failure
- **Tool Execution History**: Records success/failure rates per tool
- **Generated Reports**: Creates detailed execution summaries for analysis

```python
# Example of generating an execution report
report = await agent.generate_execution_report()
print(f"Success rate: {report['success_steps']/report['total_steps']}")
```

### 4. Stuck-State Detection

Intelligent detection of repeated outputs or recurring failures:

- **Content Repetition Detection**: Identifies when the agent repeats the same outputs
- **Failure Pattern Recognition**: Detects when the agent is stuck in an error loop
- **Automatic Recovery**: Takes corrective action when stuck states are detected

### 5. Automatic Retry Logic

Smart retry capability for transient failures:

- **Exponential Backoff**: Implements exponential backoff to reduce load during failures
- **Configurable Retries**: Allows setting different retry policies per tool or task
- **Jitter Strategy**: Adds randomized jitter to prevent thundering herd problems

```python
# Example: Configuring retry settings for a set of tasks
retry_config = {
    "max_retries": 5,
    "base_delay": 1.0,
    "backoff_factor": 2.0,
    "jitter": 0.1
}
results = await agent._execute_concurrent_tasks(tasks, retry_config=retry_config)
```

### 6. Dynamic Concurrency Adjustment

Self-tuning concurrency based on system load:

- **CPU-Aware Scaling**: Reduces concurrency when CPU usage is high
- **Memory-Aware Scaling**: Scales back when memory pressure increases
- **Performance Tracking**: Records system metrics to optimize future runs

```python
# Example: Setting dynamic concurrency parameters
agent.dynamic_concurrency = True
agent.min_concurrency = 1
agent.max_concurrency = 8
agent.cpu_threshold_high = 75.0  # Reduce when CPU > 75%
agent.cpu_threshold_low = 30.0   # Increase when CPU < 30%
```

### 7. Tool Priority Scheduling

Intelligent scheduling of tasks based on priority:

- **Priority Assignment**: Assigns priority levels to different tools or task types
- **Critical-First Execution**: Executes high-priority tasks before lower-priority ones
- **Resource Allocation**: Ensures important tasks get resources even under load

```python
# Example: Setting tool priorities
agent.set_tool_priority("critical_data_fetch", 10)  # Highest priority
agent.set_tool_priority("background_analysis", 1)   # Lower priority
```

### 8. Visualization Tools

Built-in visualization capabilities for monitoring and analysis:

- **Execution Timeline**: Generates visual representations of step execution
- **Performance Insights**: Creates charts showing tool performance metrics
- **Resource Usage**: Tracks and visualizes system resource utilization

```python
# Example: Generating visualizations
visualization_files = await agent.generate_visualizations()
print(f"Generated visualization files: {visualization_files}")
```

## Implementation Details

### Core Components

1. **Parallel Execution Engine**:
   - `_execute_concurrent_tasks()`: Executes multiple tasks in parallel with concurrency control, retries, and priorities
   - `handle_partial_failures()`: Processes results of concurrent execution

2. **Enhanced Step Execution**:
   - Overridden `step()`: Adds error handling, stuck-state detection, and execution tracking
   - `_record_step_execution()`: Records detailed step execution stats

3. **Stuck-State Detection**:
   - `_detect_stuck_state()`: Analyzes execution history to identify loops or repetition
   - `_track_assistant_output()`: Tracks outputs to detect repetition

4. **Memory Enhancements**:
   - `store_intermediate_result()`: Stores chain-of-thought or intermediate data
   - Advanced tracking of execution history and tool performance

5. **Retry Logic**:
   - Exponential backoff with jitter for transient failures
   - Per-task configurable retry policies

6. **Dynamic Concurrency**:
   - `_update_system_metrics()`: Tracks CPU and memory usage
   - `_get_effective_concurrency_limit()`: Adjusts concurrency based on system load

7. **Priority Scheduling**:
   - `set_tool_priority()`: Sets priority levels for tools
   - Priority-based task ordering in `_execute_concurrent_tasks()`

8. **Visualization Generation**:
   - `generate_visualizations()`: Creates JSON files for visualization
   - `visualize_execution()`: Returns structured visualization data

### Example Usage

```python
from app.agent.react import ReActAgent

class MyCustomAgent(ReActAgent):
    async def think(self) -> bool:
        # Your thinking logic here
        return True  # Return True to proceed to act()
        
    async def act(self) -> str:
        # Create tasks with different priorities
        tasks = [lambda: self._some_async_task(x) for x in data_points]
        priorities = [10, 5, 3]  # Higher numbers = higher priority
        
        # Execute with retries and priority scheduling
        results = await self._execute_concurrent_tasks(
            tasks, 
            priorities=priorities,
            retry_config={"max_retries": 3}
        )
        summary = self.handle_partial_failures(results)
        
        # Store intermediate results for chain-of-thought
        self.store_intermediate_result("analysis_summary", summary)
        
        return "Completed analysis with mixed results"

# Using the agent
agent = MyCustomAgent()
agent.set_tool_priority("critical_tool", 10)  # Set priority for a specific tool
agent.dynamic_concurrency = True  # Enable dynamic concurrency adjustment
result = await agent.run("Analyze these financial data points")
report = await agent.generate_execution_report()
visualizations = await agent.generate_visualizations()
```

## Advanced Features

### Automatic Retry with Exponential Backoff

Configure sophisticated retry logic for transient failures:

```python
# Custom retry configuration
retry_config = {
    "max_retries": 5,             # Maximum number of retry attempts
    "base_delay": 1.0,            # Initial delay in seconds
    "backoff_factor": 2.0,        # Multiply delay by this factor each retry
    "jitter": 0.1                 # Random jitter factor to add/subtract
}

# Use in task execution
results = await agent._execute_concurrent_tasks(tasks, retry_config=retry_config)
```

### Dynamic Concurrency Adjustment

The agent automatically tunes concurrency based on system load:

```python
# Enable and configure dynamic concurrency
agent.dynamic_concurrency = True
agent.min_concurrency = 1         # Minimum concurrency limit
agent.max_concurrency = 10        # Maximum concurrency limit
agent.cpu_threshold_high = 80.0   # Reduce concurrency when CPU exceeds this percentage
agent.cpu_threshold_low = 30.0    # Increase concurrency when CPU is below this percentage
agent.memory_threshold = 85.0     # Reduce concurrency when memory exceeds this percentage
```

### Priority-Based Task Scheduling

Ensure critical tasks run first when resources are limited:

```python
# Set tool priorities (higher numbers = higher priority)
agent.set_tool_priority("data_fetch", 10)
agent.set_tool_priority("analysis", 5)
agent.set_tool_priority("background_task", 1)

# Execute tasks with specific priorities
priorities = [10, 5, 1]  # Priority for each task
results = await agent._execute_concurrent_tasks(tasks, priorities=priorities)
```

### Visualization and Monitoring

Generate visualizations for execution analysis:

```python
# Configure visualization settings
agent.visualization_enabled = True
agent.visualization_output_dir = "path/to/visualizations"

# Generate visualizations
files = await agent.generate_visualizations()

# Get visualization data programmatically
viz_data = await agent.visualize_execution(output_format="json")
```

## Performance Considerations

- The default concurrency limit is set to 3, which balances performance with resource usage
- Dynamic concurrency automatically adjusts based on system load
- For I/O-bound tasks (like web searches), you might increase max_concurrency
- For CPU-bound tasks, keep max_concurrency lower
- Priority scheduling ensures critical tasks complete even under load

## Future Enhancements

Potential future improvements include:

1. **Machine Learning for Concurrency**: Using ML to predict optimal concurrency settings
2. **Advanced Visualization Dashboard**: Interactive dashboard for real-time monitoring
3. **Distributed Task Execution**: Extending to multi-node execution for larger workloads
4. **Predictive Failure Analysis**: Using history to predict and prevent failures
5. **Custom Visualization Formats**: Supporting additional output formats like HTML and SVG 