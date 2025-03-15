# Supercharged BaseAgent

## Overview

The BaseAgent has been enhanced with several powerful capabilities to improve robustness, concurrency support, error handling, and transparency. This supercharged version maintains backward compatibility while allowing for more advanced agent implementations.

## Key Enhancements

1. **Concurrency Support**
   - New concurrency framework for parallel execution of tasks within a step
   - Configurable concurrency limit via `concurrency_limit` parameter
   - Support for parallel tool execution with proper error handling

2. **Partial Success/Failure Tracking**
   - Detailed logging of each step's result, capturing any errors
   - Graceful handling of failures while maintaining partial results
   - Step-level error context capture for debugging

3. **Improved Logging & Error Handling**
   - Clear logging of transitions between steps
   - Detailed error logging with stack traces
   - Agent-specific logging with name prefixes

4. **Stuck State Detection**
   - Enhanced logic for detecting repeated messages
   - Automatic prompt adjustments to get unstuck
   - Configurable duplicate threshold

5. **Better Memory & Prompt Construction**
   - New `_build_conversation_prompt()` method for standardized prompt generation
   - Improved memory integration with LLM prompts
   - Support for system prompts and next-step instructions

6. **Context Maintenance**
   - Enhanced context tracking for debugging and auditing
   - Structured storage of execution history and state
   - Timestamp tracking for performance analysis

7. **Transparent Reasoning**
   - New `thinking_steps` attribute that tracks reasoning process
   - Visibility into agent's thought process for debugging and explanation
   - Transparent logging of reasoning failures

## Usage

### Basic Implementation

```python
from app.agent.base import BaseAgent

class MyAgent(BaseAgent):
    async def step(self) -> str:
        # Implement your agent's step logic here
        response = await self._basic_llm_step()
        return f"Step completed with response: {response}"
```

### With Enhanced Concurrency

```python
import asyncio
from app.agent.base import BaseAgent

class ParallelTaskAgent(BaseAgent):
    async def step(self) -> str:
        # Example of a step that runs multiple operations in parallel
        tasks = [
            self._task1(),
            self._task2(),
            self._task3()
        ]
        
        # Execute all tasks with concurrency control
        results = await asyncio.gather(*tasks)
        
        # Process and return results
        return f"Completed {len(results)} parallel tasks"
    
    async def _task1(self):
        # Your task implementation
        pass
```

### Using the Basic LLM Step Helper

```python
from app.agent.base import BaseAgent

class SimpleAgent(BaseAgent):
    async def step(self) -> str:
        # Use the helper method for simple LLM-based steps
        response = await self._basic_llm_step()
        return response
```

### With Thinking Process Visibility

```python
from app.agent.base import BaseAgent

class TransparentAgent(BaseAgent):
    async def step(self) -> str:
        # Add to thinking steps for transparency
        self.thinking_steps.append("Considering the next action...")
        
        # Use the reasoning module
        reasoning = await self.think("What should I do next?")
        
        # Decision based on reasoning
        if reasoning.get("success", False):
            self.thinking_steps.append(f"Decided to: {reasoning.get('decision')}")
            # Take action based on reasoning
            return await self._execute_action(reasoning)
        else:
            self.thinking_steps.append("Reasoning failed, taking default action")
            return await self._basic_llm_step()
```

## Complete Example

See `app/agent/examples/enhanced_agent_example.py` for a comprehensive example that demonstrates multiple enhanced features working together.

## Migration Guide

If you have existing agents based on the previous BaseAgent:

1. They will continue to work with the enhanced BaseAgent
2. To take advantage of new features:
   - Add concurrency control via the `concurrency_limit` parameter
   - Use `self._basic_llm_step()` for simple LLM interactions
   - Add to `self.thinking_steps` for transparency
   - Use `self.update_context()` for better debugging

## Future Development

The supercharged BaseAgent is designed for extensibility. Future enhancements may include:

- Dynamic concurrency adjustment based on system load
- Automatic retry strategies for failed steps
- Enhanced memory management for long-running agents
- Integration with monitoring and observability systems
- Automatic tool selection based on task requirements
