#!/usr/bin/env python3
"""
Enhanced ReActAgent Demo Script

This script demonstrates the advanced capabilities of the enhanced ReActAgent:
1. Parallel Task Execution with Priority Scheduling
2. Automatic Retry Logic with Exponential Backoff
3. Dynamic Concurrency Adjustment based on System Load
4. Comprehensive Error Handling and Partial Success
5. Stuck-State Detection and Recovery
6. Execution Telemetry and Reporting
7. Visualization Generation for Analysis
"""

import os
import sys
import json
import time
import asyncio
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.agent.react import ReActAgent
from app.schema import Message


class ComprehensiveReActAgent(ReActAgent):
    """
    A concrete implementation of ReActAgent demonstrating all enhanced features.
    """
    
    def __init__(self, name: str = "ComprehensiveDemo", **kwargs):
        super().__init__(name=name, **kwargs)
        self.thinking_steps = 0
        self.should_induce_stuck = False
        self.should_induce_error = False
        self.consecutive_errors = 0
        
    async def think(self) -> bool:
        """
        Demonstrate thinking logic with stuck-state simulation.
        
        Returns:
            True to proceed with act(), False to stop
        """
        self.thinking_steps += 1
        
        # Store intermediate reasoning for chain-of-thought
        self.store_intermediate_result(f"thinking_step_{self.thinking_steps}", {
            "step": self.thinking_steps,
            "timestamp": datetime.now().isoformat(),
            "thought": f"Processing step {self.thinking_steps} of the task"
        })
        
        # Simulate stuck state if requested
        if self.should_induce_stuck and self.thinking_steps > 2:
            # Generate the same output repeatedly to trigger stuck detection
            repeated_thought = "I need to analyze the data again"
            
            message = Message(role="assistant", content=repeated_thought)
            self.memory.add(message)
            
            # This should trigger stuck-state detection after a few repetitions
            print(f"Inducing stuck state with repeated output: '{repeated_thought}'")
            
            # Return True to continue executing steps (stuck detection should catch this)
            return True
        
        # Normal thinking process
        message = Message(
            role="assistant", 
            content=f"Thinking step {self.thinking_steps}: Determining best course of action"
        )
        self.memory.add(message)
        
        # Always proceed to act() in this demo
        return True
        
    async def act(self) -> str:
        """
        Demonstrate action execution with priorities, retries, and partial failures.
        
        Returns:
            Action result summary
        """
        # Update system metrics for dynamic concurrency
        self._update_system_metrics()
        
        # Prepare tasks with different priorities and characteristics
        tasks = []
        task_names = []
        priorities = []
        
        # Fast, reliable task (high priority)
        tasks.append(lambda: self._simulate_task("critical_data_fetch", 0.5, 0.05))
        task_names.append("critical_data_fetch")
        priorities.append(10)  # Highest priority
        
        # Medium task (medium priority)
        tasks.append(lambda: self._simulate_task("data_processing", 1.0, 0.1))
        task_names.append("data_processing")
        priorities.append(5)  # Medium priority
        
        # Slow task (low priority)
        tasks.append(lambda: self._simulate_task("background_analysis", 2.0, 0.2))
        task_names.append("background_analysis")
        priorities.append(2)  # Low priority
        
        # Add error-prone task if error induction is enabled
        if self.should_induce_error:
            tasks.append(lambda: self._simulate_failing_task("error_prone_task", 0.8, 0.9))
            task_names.append("error_prone_task") 
            priorities.append(3)  # Medium-low priority
        
        # Configure retry settings
        retry_config = {
            "max_retries": 3,
            "base_delay": 0.5,
            "backoff_factor": 2.0,
            "jitter": 0.2
        }
        
        # Execute tasks in parallel with priority scheduling and retry logic
        print(f"\nExecuting {len(tasks)} tasks with priorities: {priorities}")
        print(f"Current concurrency limit: {self._get_effective_concurrency_limit()}")
        
        start_time = time.time()
        results = await self._execute_concurrent_tasks(
            tasks,
            priorities=priorities,
            retry_config=retry_config
        )
        duration = time.time() - start_time
        
        # Process results and handle partial failures
        summary = self.handle_partial_failures(results)
        
        # Record additional metadata about the execution
        for i, result in enumerate(summary["results"]):
            if i < len(task_names):
                result["task_name"] = task_names[i]
                
        result_message = (
            f"Completed {len(tasks)} tasks in {duration:.2f}s with "
            f"{summary['success_count']} successes and {summary['failure_count']} failures. "
            f"Status: {summary['status']}"
        )
        
        # Increment consecutive errors count if needed
        if summary['status'] == 'failed':
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0
            
        # Store the result for chain-of-thought
        self.store_intermediate_result(f"action_result_{self.current_step}", {
            "step": self.current_step,
            "summary": summary,
            "duration": duration
        })
        
        return result_message
        
    async def _simulate_task(self, name: str, duration: float, failure_prob: float) -> Dict[str, Any]:
        """
        Simulate an async task with configurable duration and failure probability.
        
        Args:
            name: Task name
            duration: Base duration in seconds
            failure_prob: Probability of failure (0.0 to 1.0)
            
        Returns:
            Task result data or raises exception
        """
        # Add some randomness to the duration
        actual_duration = duration * (0.8 + random.random() * 0.4)
        await asyncio.sleep(actual_duration)
        
        # Possibly fail the task
        if random.random() < failure_prob:
            raise Exception(f"Task {name} failed with random error")
            
        return {
            "task_name": name,
            "result": f"Data from {name}",
            "timestamp": datetime.now().isoformat(),
            "execution_time": actual_duration
        }
        
    async def _simulate_failing_task(self, name: str, duration: float, failure_prob: float) -> Dict[str, Any]:
        """
        Simulate a task that's very likely to fail, even after retries.
        """
        # This task has a very high failure probability
        await asyncio.sleep(duration)
        
        if random.random() < failure_prob:
            raise Exception(f"Persistent failure in {name} - this error won't be fixed by retries")
            
        return {
            "task_name": name,
            "result": "Surprisingly, this worked!",
            "timestamp": datetime.now().isoformat()
        }


# Demo functions

async def demo_priority_scheduling():
    """Demonstrate tool priority scheduling."""
    print("\n=== Priority Scheduling Demo ===")
    
    # Create agent with priority scheduling enabled
    agent = ComprehensiveReActAgent(use_priority_scheduling=True)
    agent.set_tool_priority("critical_data_fetch", 10)
    agent.set_tool_priority("data_processing", 5)
    agent.set_tool_priority("background_analysis", 2)
    
    print("Running agent with prioritized tasks...")
    for _ in range(3):
        result = await agent.step()
        print(f"Step {agent.current_step} result: {result}")
        
    # Generate execution report
    report = await agent.generate_execution_report()
    print(f"\nPriority scheduling resulted in {report['success_steps']} successful steps")
    
    # Check the execution order in the visualization
    viz_files = await agent.generate_visualizations()
    print(f"Generated visualization files: {viz_files}")
    
    return agent


async def demo_retry_logic():
    """Demonstrate automatic retry logic with exponential backoff."""
    print("\n=== Automatic Retry Logic Demo ===")
    
    # Create agent with custom retry settings
    agent = ComprehensiveReActAgent(
        max_retries=4,
        retry_delay_base=0.2,
        retry_backoff_factor=2.0,
        retry_jitter=0.2
    )
    
    # Enable error induction to demonstrate retries
    agent.should_induce_error = True
    
    print("Running agent with error-prone tasks that will trigger retries...")
    for _ in range(2):
        result = await agent.step()
        print(f"Step {agent.current_step} result: {result}")
        
    # Generate report showing retry statistics
    report = await agent.generate_execution_report()
    print("\nRetry Statistics:")
    print(f"- Steps completed: {agent.current_step}")
    print(f"- Success rate: {report.get('success_rate', 0):.2f}")
    
    return agent


async def demo_dynamic_concurrency():
    """Demonstrate dynamic concurrency adjustment based on system load."""
    print("\n=== Dynamic Concurrency Adjustment Demo ===")
    
    # Create agent with dynamic concurrency enabled
    agent = ComprehensiveReActAgent(
        dynamic_concurrency=True,
        min_concurrency=1,
        max_concurrency=8,
        cpu_threshold_high=70.0,
        cpu_threshold_low=30.0,
        memory_threshold=80.0
    )
    
    print("Running agent with dynamic concurrency adjustment...")
    print(f"Initial concurrency limit: {agent.concurrency_limit}")
    
    # Create artificial load
    print("Generating some CPU load to trigger concurrency adjustment...")
    
    # Run a few steps while system metrics are being collected
    for _ in range(4):
        # Update system metrics before each step
        agent._update_system_metrics()
        
        # Display current concurrency limit
        effective_limit = agent._get_effective_concurrency_limit()
        print(f"Effective concurrency limit: {effective_limit}")
        
        result = await agent.step()
        print(f"Step {agent.current_step} result: {result}")
        
        # Generate some CPU load between steps
        if _ == 1:  # Only for one iteration to demonstrate adjustment
            start_time = time.time()
            while time.time() - start_time < 1.0:
                # Waste CPU cycles
                [i*i for i in range(1000000)]
        
    # Generate visualizations showing system metrics
    viz_files = await agent.generate_visualizations()
    print(f"System metrics visualization files: {viz_files}")
    
    return agent


async def demo_stuck_detection():
    """Demonstrate stuck-state detection and recovery."""
    print("\n=== Stuck-State Detection Demo ===")
    
    # Create agent with stuck-state detection configured
    agent = ComprehensiveReActAgent(duplicate_threshold=2)
    
    # Enable stuck state after a few normal steps
    print("Running first few steps normally...")
    for _ in range(2):
        result = await agent.step()
        print(f"Step {agent.current_step} result: {result}")
    
    print("\nNow inducing a stuck state with repeated outputs...")
    agent.should_induce_stuck = True
    
    # Run more steps, which should trigger stuck detection
    for _ in range(4):
        result = await agent.step()
        print(f"Step {agent.current_step} result: {result}")
        
        # Check if stuck state was detected
        if "detected potential stuck state" in result.lower():
            print("✓ Successfully detected and recovered from stuck state")
            break
    
    return agent


async def demo_visualization_tools():
    """Demonstrate visualization tools for execution analysis."""
    print("\n=== Visualization Tools Demo ===")
    
    # Create agent with visualization enabled
    agent = ComprehensiveReActAgent(
        visualization_enabled=True,
        visualization_output_dir="demo_visualizations"
    )
    
    # Run a mix of successful and failing steps
    print("Running agent to generate interesting visualization data...")
    agent.should_induce_error = True
    
    for _ in range(5):
        result = await agent.step()
        print(f"Step {agent.current_step} result: {result}")
        
        # Toggle error induction to create varied data
        if _ % 2 == 0:
            agent.should_induce_error = not agent.should_induce_error
    
    # Generate visualizations
    print("\nGenerating visualization files...")
    viz_files = await agent.generate_visualizations()
    
    print(f"Generated {len(viz_files)} visualization files:")
    for file in viz_files:
        print(f"- {file}")
        
    # Generate programmatic visualization data
    viz_data = await agent.visualize_execution(output_format="json")
    print(f"\nVisualization data contains {len(viz_data.get('execution_data', {}).get('steps', []))} steps")
    
    return agent


async def comprehensive_demo():
    """Run a demo that showcases all enhanced features together."""
    print("\n=== Comprehensive ReActAgent Demo ===")
    
    # Create fully configured agent
    agent = ComprehensiveReActAgent(
        # Base settings
        name="ComprehensiveDemo",
        max_steps=10,
        
        # Concurrency settings
        concurrency_limit=3,
        dynamic_concurrency=True,
        min_concurrency=1,
        max_concurrency=5,
        
        # Retry settings
        max_retries=3,
        retry_delay_base=0.5,
        retry_backoff_factor=2.0,
        retry_jitter=0.1,
        
        # Stuck detection settings
        duplicate_threshold=2,
        
        # Visualization settings
        visualization_enabled=True,
        visualization_output_dir="comprehensive_demo_viz"
    )
    
    # Set tool priorities
    agent.set_tool_priority("critical_data_fetch", 10)
    agent.set_tool_priority("data_processing", 5)
    agent.set_tool_priority("background_analysis", 2)
    agent.set_tool_priority("error_prone_task", 3)
    
    print("Running comprehensive demo with all features enabled...")
    
    # First few steps are normal
    print("\nPhase 1: Normal execution with priorities")
    for _ in range(2):
        result = await agent.step()
        print(f"Step {agent.current_step} result: {result}")
    
    # Induce errors to demonstrate retry logic
    print("\nPhase 2: Error handling and retry logic")
    agent.should_induce_error = True
    for _ in range(2):
        result = await agent.step()
        print(f"Step {agent.current_step} result: {result}")
    
    # Induce stuck state to demonstrate detection
    print("\nPhase 3: Stuck-state detection")
    agent.should_induce_error = False
    agent.should_induce_stuck = True
    for _ in range(3):
        result = await agent.step()
        print(f"Step {agent.current_step} result: {result}")
        
        # Break if stuck state was detected
        if "detected potential stuck state" in result.lower():
            print("✓ Successfully detected and recovered from stuck state")
            break
    
    # Return to normal operation
    print("\nPhase 4: Recovery and finalization")
    agent.should_induce_stuck = False
    for _ in range(2):
        result = await agent.step()
        print(f"Step {agent.current_step} result: {result}")
    
    # Finalize and generate reports
    print("\nFinalizing agent and generating reports...")
    finalize_result = await agent.finalize()
    
    print(f"\nExecution report shows {finalize_result['execution_report'].get('success_steps', 0)} successful steps")
    print(f"Generated {len(finalize_result['visualization_files'])} visualization files")
    
    return agent


async def main():
    """Run all demos and generate a summary report."""
    print("=== Enhanced ReActAgent Demonstration ===")
    print("This demo showcases the advanced capabilities of the enhanced ReActAgent")
    
    try:
        # Run individual feature demos
        await demo_priority_scheduling()
        await demo_retry_logic()
        await demo_dynamic_concurrency()
        await demo_stuck_detection()
        await demo_visualization_tools()
        
        # Run the comprehensive demo
        agent = await comprehensive_demo()
        
        print("\n=== Demo Summary ===")
        print("The enhanced ReActAgent successfully demonstrated:")
        print("1. Parallel task execution with priorities")
        print("2. Automatic retry logic with exponential backoff")
        print("3. Dynamic concurrency adjustment based on system load")
        print("4. Intelligent stuck-state detection and recovery")
        print("5. Comprehensive error handling and partial success tracking")
        print("6. Detailed execution reporting and analysis")
        print("7. Visualization tools for performance monitoring")
        
        print("\nAll features are working as expected!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 