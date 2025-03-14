from abc import ABC, abstractmethod
import asyncio
import logging
import time
import json
import os
import platform
import psutil
from typing import Any, Dict, List, Optional, Callable, Union, Awaitable, Tuple
from datetime import datetime

from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.schema import AgentState, Memory, Message

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent, ABC):
    """
    A supercharged ReAct agent that:
      - implements concurrency for multiple sub-tasks in act(),
      - partial success/failure tracking,
      - robust error handling and advanced logging,
      - memory expansions for chain-of-thought or partial tool results,
      - stuck-state detection for repeated outputs.
      - automatic retry logic for transient failures,
      - dynamic concurrency adjustment based on system load,
      - tool priority scheduling for critical tasks,
      - visualization tools for execution analysis.
    """

    name: str
    description: Optional[str] = None

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    llm: Optional[LLM] = Field(default_factory=LLM)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE

    max_steps: int = 10
    current_step: int = 0
    
    # Enhanced features
    concurrency_limit: int = 3
    duplicate_threshold: int = 2
    previous_outputs: List[str] = Field(default_factory=list)
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    tool_execution_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    potential_loop_counter: int = 0
    
    # Retry logic settings
    max_retries: int = 3
    retry_delay_base: float = 1.0  # Base delay in seconds
    retry_backoff_factor: float = 2.0  # Exponential backoff factor
    retry_jitter: float = 0.1  # Random jitter factor
    
    # Dynamic concurrency settings
    dynamic_concurrency: bool = True
    min_concurrency: int = 1
    max_concurrency: int = 10
    cpu_threshold_high: float = 80.0  # CPU usage % to reduce concurrency
    cpu_threshold_low: float = 30.0   # CPU usage % to increase concurrency
    memory_threshold: float = 85.0    # Memory usage % to reduce concurrency
    
    # Tool priority settings
    use_priority_scheduling: bool = True
    priority_levels: Dict[str, int] = Field(default_factory=dict)  # Tool name to priority (higher is more important)
    
    # Visualization settings
    visualization_enabled: bool = True
    visualization_output_dir: str = "visualizations"
    
    # Performance tracking
    performance_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    system_metrics_history: List[Dict[str, float]] = Field(default_factory=list)

    @abstractmethod
    async def think(self) -> bool:
        """
        Process the current state, memory, or conversation, and decide whether to proceed with an action.
        Return True if we should call `act()`, else False for no action.
        """

    @abstractmethod
    async def act(self) -> str:
        """
        Execute the decided actions.  
        If multiple sub-tasks need concurrency, we can do so here:
        
        Example:
            tasks = [self._run_subtask(t) for t in parallel_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        Return a summary of results or partial success/failure.
        """

    async def step(self) -> str:
        """
        Execute a single step: think + act.  
        Adds concurrency approach for multiple sub-tasks in `act()`.
        """
        if self.state == AgentState.ERROR:
            logger.warning(f"[{self.name}] Agent in error state, cannot proceed.")
            return "Agent in error state, cannot proceed."

        if self.current_step >= self.max_steps:
            self.state = AgentState.FINISHED
            logger.warning(f"[{self.name}] Reached max steps ({self.max_steps}).")
            return "Max steps reached, agent stopping."

        # Collect system metrics for dynamic concurrency
        if self.dynamic_concurrency:
            self._update_system_metrics()

        logger.info(f"[{self.name}] Step {self.current_step + 1}/{self.max_steps} started.")
        
        # Check stuck state before execution
        if self._detect_stuck_state():
            logger.warning(f"[{self.name}] Taking corrective action for stuck state")
            # Add a message to help break the loop
            stuck_message = (
                f"I notice we may be stuck in a loop. Let's try a different approach. "
                f"Previous steps have failed with similar patterns. "
                f"Let me reconsider the problem from a different angle."
            )
            self.memory.add_message(Message(role="user", content=stuck_message))
            # Reset the counter
            self.potential_loop_counter = 0
            
        start_time = time.time()
        
        try:
            should_act = await self.think()
            
            # Track the latest assistant output for stuck detection
            self._track_assistant_output()
            
            if not should_act:
                logger.debug(f"[{self.name}] think() returned False, no action needed.")
                self._record_step_execution({
                    "status": "think_only", 
                    "result": "No action taken",
                    "duration": time.time() - start_time
                })
                self.current_step += 1
                return "No action taken this step."
    
            # If we do have actions, run them
            result = await self.act()
            duration = time.time() - start_time
            
            self._record_step_execution({
                "status": "completed", 
                "result": result,
                "duration": duration
            })
            self.current_step += 1
    
            # Check if we've reached finishing conditions
            if self.current_step >= self.max_steps:
                self.state = AgentState.FINISHED
                logger.info(f"[{self.name}] Completed all steps or reached max step limit.")
                
            # Generate visualizations if enabled
            if self.visualization_enabled and self.current_step % 5 == 0:  # Every 5 steps
                await self.generate_visualizations()
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{self.name}] Error during step execution: {e}", exc_info=True)
            self._record_step_execution({
                "status": "error", 
                "error": str(e),
                "duration": duration
            })
            self.state = AgentState.ERROR
            return f"Error during step execution: {str(e)}"

    ############################################################################
    # UTILITY METHODS FOR SUBCLASSES
    ############################################################################

    def store_intermediate_result(self, key: str, value: Any) -> None:
        """
        Helper to store partial chain-of-thought or intermediate data in memory for ReAct synergy.
        """
        self.memory.add_message(Message(role="assistant", content=f"**{key}**: {value}"))
        logger.debug(f"[{self.name}] Storing intermediate result for {key}")

    def _track_assistant_output(self) -> None:
        """
        Track the last assistant message to detect repeated outputs.
        """
        assistant_messages = [m for m in self.memory.messages if m.role == "assistant"]
        if assistant_messages:
            last_content = assistant_messages[-1].content
            if len(self.previous_outputs) >= 10:  # Keep a reasonable history
                self.previous_outputs.pop(0)
            self.previous_outputs.append(last_content)

    def _detect_stuck_state(self) -> bool:
        """
        Check if the agent is stuck in a loop by detecting:
        1. Repeated identical assistant outputs
        2. Multiple failed tool executions
        3. Multiple steps with no progress
        
        Returns:
            True if a stuck state is detected, False otherwise
        """
        # Not enough history to detect a loop
        if len(self.previous_outputs) < 2:
            return False
            
        # Check for repeated identical content
        last_content = self.previous_outputs[-1]
        duplicate_count = 0
        
        for content in reversed(self.previous_outputs[:-1]):
            if content == last_content:
                duplicate_count += 1
                if duplicate_count >= self.duplicate_threshold:
                    self.potential_loop_counter += 1
                    logger.warning(f"[{self.name}] Detected potential stuck state: repeated identical content")
                    if self.potential_loop_counter >= 3:
                        logger.error(f"[{self.name}] Stuck state confirmed: Multiple repeated contents")
                        return True
                    break
        
        # Check for repeated failures in execution history
        if len(self.execution_history) >= 3:
            recent_executions = self.execution_history[-3:]
            if all(exec_data.get("status") == "error" for exec_data in recent_executions):
                self.potential_loop_counter += 1
                logger.warning(f"[{self.name}] Detected potential stuck state: consecutive execution failures")
                if self.potential_loop_counter >= 3:
                    logger.error(f"[{self.name}] Stuck state confirmed: Multiple consecutive failures")
                    return True
        else:
            # Reset counter if we don't see consistent patterns
            self.potential_loop_counter = 0
            
        return False

    def _record_step_execution(self, data: Dict[str, Any]) -> None:
        """
        Record execution data for the current step.
        """
        execution_data = {
            "step": self.current_step,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        self.execution_history.append(execution_data)

    def _track_tool_execution(self, tool_name: str, result: Dict[str, Any]) -> None:
        """
        Track tool execution results for analytics and debugging.
        """
        if tool_name not in self.tool_execution_history:
            self.tool_execution_history[tool_name] = []
            
        execution_data = {
            "step": self.current_step,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "status": "success" if result.get("success", False) else "failed"
        }
        
        # Track performance metrics for this tool
        if tool_name not in self.performance_metrics:
            self.performance_metrics[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "avg_duration": 0,
                "total_duration": 0
            }
            
        metrics = self.performance_metrics[tool_name]
        metrics["calls"] += 1
        
        if result.get("success", False):
            metrics["successes"] += 1
        else:
            metrics["failures"] += 1
            
        # Track duration if available
        if "duration" in result:
            duration = result["duration"]
            metrics["total_duration"] += duration
            metrics["avg_duration"] = metrics["total_duration"] / metrics["calls"]
            execution_data["duration"] = duration
        
        self.tool_execution_history[tool_name].append(execution_data)
    
    async def _execute_concurrent_tasks(
        self, 
        tasks: List[Callable[[], Awaitable[Any]]], 
        priorities: Optional[List[int]] = None,
        concurrency_limit: Optional[int] = None,
        retry_config: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Enhanced concurrent task execution with:
        - Dynamic concurrency adjustment based on system load
        - Priority scheduling for critical tasks
        - Automatic retry logic for transient failures
        
        Args:
            tasks: List of async callables to execute
            priorities: Optional list of priorities (higher = more important)
            concurrency_limit: Optional override for concurrency limit
            retry_config: Optional retry configuration override
            
        Returns:
            List of results, with exceptions preserved in place
        """
        # Determine effective concurrency limit
        effective_limit = self._get_effective_concurrency_limit(concurrency_limit)
        logger.info(f"[{self.name}] Running tasks with concurrency limit: {effective_limit}")
        
        sem = asyncio.Semaphore(effective_limit)
        results = [None] * len(tasks)  # Pre-allocate results list
        
        # Process priorities
        if priorities is None:
            # Default all tasks to priority 0
            effective_priorities = [0] * len(tasks)
        else:
            effective_priorities = priorities
            if len(effective_priorities) != len(tasks):
                raise ValueError("priorities list must be same length as tasks")
        
        # Create task entries with priorities
        task_entries = [
            {"index": i, "task": task, "priority": priority} 
            for i, (task, priority) in enumerate(zip(tasks, effective_priorities))
        ]
        
        # Sort by priority (higher first) if priority scheduling is enabled
        if self.use_priority_scheduling:
            task_entries.sort(key=lambda x: x["priority"], reverse=True)
            logger.debug(f"[{self.name}] Tasks sorted by priority: {[entry['priority'] for entry in task_entries]}")
        
        # Create and schedule all tasks
        async def _execute_with_retry_and_semaphore(entry):
            task_fn = entry["task"]
            index = entry["index"]
            priority = entry["priority"]
            
            async with sem:
                logger.debug(f"[{self.name}] Executing task {index} with priority {priority}")
                
                # Apply retry logic
                retry_settings = retry_config or {
                    "max_retries": self.max_retries,
                    "base_delay": self.retry_delay_base,
                    "backoff_factor": self.retry_backoff_factor,
                    "jitter": self.retry_jitter
                }
                
                for attempt in range(retry_settings["max_retries"] + 1):  # +1 for initial attempt
                    try:
                        start_time = time.time()
                        result = await task_fn()
                        duration = time.time() - start_time
                        
                        # Add duration to result if it's a dictionary
                        if isinstance(result, dict):
                            result["duration"] = duration
                        
                        logger.debug(f"[{self.name}] Task {index} completed successfully in {duration:.2f}s")
                        results[index] = result
                        return
                    except Exception as e:
                        if attempt < retry_settings["max_retries"]:
                            # Calculate delay with exponential backoff and jitter
                            delay = retry_settings["base_delay"] * (retry_settings["backoff_factor"] ** attempt)
                            jitter_amount = delay * retry_settings["jitter"] * (2 * asyncio.get_event_loop().time() - 1)
                            delay += jitter_amount
                            
                            logger.warning(
                                f"[{self.name}] Task {index} failed (attempt {attempt+1}/{retry_settings['max_retries']+1}), "
                                f"retrying in {delay:.2f}s: {e}"
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"[{self.name}] Task {index} failed after all retries: {e}", exc_info=True)
                            results[index] = e
        
        # Execute all tasks and wait for completion
        await asyncio.gather(*[_execute_with_retry_and_semaphore(entry) for entry in task_entries])
        
        return results

    def handle_partial_failures(self, results: List[Any]) -> Dict[str, Any]:
        """
        Summarize partial successes/failures from concurrent execution.
        
        Args:
            results: List of results from _execute_concurrent_tasks
            
        Returns:
            A dictionary with success_count, failure_count, and formatted results
        """
        summary = {
            "success_count": 0,
            "failure_count": 0,
            "partial_success": False,
            "results": []
        }
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                summary["failure_count"] += 1
                summary["results"].append({
                    "task_id": idx,
                    "success": False,
                    "error": str(result)
                })
            else:
                summary["success_count"] += 1
                summary["results"].append({
                    "task_id": idx,
                    "success": True,
                    "result": result
                })
        
        # Check if we had partial success
        if summary["success_count"] > 0 and summary["failure_count"] > 0:
            summary["partial_success"] = True
            summary["status"] = "partial_success"
        elif summary["success_count"] > 0:
            summary["status"] = "success"
        else:
            summary["status"] = "failed"
            
        return summary
    
    def set_tool_priority(self, tool_name: str, priority: int) -> None:
        """
        Set the execution priority for a specific tool.
        Higher priority (larger number) tools will be executed first.
        
        Args:
            tool_name: Name of the tool
            priority: Priority level (higher means more important)
        """
        self.priority_levels[tool_name] = priority
        logger.info(f"[{self.name}] Set priority for tool '{tool_name}' to {priority}")
    
    def get_tool_priority(self, tool_name: str) -> int:
        """
        Get the execution priority for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Priority level (higher means more important)
        """
        return self.priority_levels.get(tool_name, 0)  # Default priority 0
    
    def _update_system_metrics(self) -> None:
        """
        Update system metrics for dynamic concurrency adjustment.
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent
            }
            
            self.system_metrics_history.append(metrics)
            
            # Keep history from growing too large
            if len(self.system_metrics_history) > 100:
                self.system_metrics_history = self.system_metrics_history[-100:]
                
            logger.debug(f"[{self.name}] System metrics: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to update system metrics: {e}")
    
    def _get_effective_concurrency_limit(self, override_limit: Optional[int] = None) -> int:
        """
        Calculate the effective concurrency limit based on system load or override.
        
        Args:
            override_limit: Optional manual override for concurrency limit
            
        Returns:
            Effective concurrency limit to use
        """
        if override_limit is not None:
            return max(1, override_limit)  # Ensure at least 1
            
        if not self.dynamic_concurrency or not self.system_metrics_history:
            return self.concurrency_limit
            
        # Get latest metrics
        latest = self.system_metrics_history[-1]
        cpu_percent = latest["cpu_percent"]
        memory_percent = latest["memory_percent"]
        
        # Start with current limit
        effective_limit = self.concurrency_limit
        
        # Adjust based on CPU usage
        if cpu_percent > self.cpu_threshold_high:
            # Reduce concurrency due to high CPU
            effective_limit = max(self.min_concurrency, effective_limit - 1)
            logger.debug(f"[{self.name}] Reducing concurrency due to high CPU ({cpu_percent:.1f}%)")
        elif cpu_percent < self.cpu_threshold_low:
            # Increase concurrency due to low CPU
            effective_limit = min(self.max_concurrency, effective_limit + 1)
            logger.debug(f"[{self.name}] Increasing concurrency due to low CPU ({cpu_percent:.1f}%)")
            
        # Further adjust based on memory usage
        if memory_percent > self.memory_threshold:
            # Reduce concurrency due to high memory
            effective_limit = max(self.min_concurrency, effective_limit - 1)
            logger.debug(f"[{self.name}] Reducing concurrency due to high memory ({memory_percent:.1f}%)")
            
        return effective_limit
        
    async def generate_execution_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the agent's execution.
        
        Includes:
        - Overall statistics
        - Step-by-step summary
        - Tool usage breakdown
        - Identified issues
        - Performance metrics
        - System resource usage
        
        Returns:
            Dictionary with execution report data
        """
        # Skip if no execution history
        if not self.execution_history:
            return {"error": "No execution history available"}
            
        # Build basic report
        total_duration = sum(step.get("duration", 0) for step in self.execution_history)
        
        report = {
            "total_steps": self.current_step,
            "success_steps": sum(1 for step in self.execution_history if step.get("status") == "completed"),
            "error_steps": sum(1 for step in self.execution_history if step.get("status") == "error"),
            "think_only_steps": sum(1 for step in self.execution_history if step.get("status") == "think_only"),
            "total_duration": total_duration,
            "avg_step_duration": total_duration / len(self.execution_history) if self.execution_history else 0,
            "execution_timeline": self.execution_history,
        }
        
        # Add tool statistics and performance metrics
        if self.tool_execution_history:
            tool_stats = {}
            for tool_name, executions in self.tool_execution_history.items():
                successful = sum(1 for e in executions if e.get("status") == "success")
                tool_stats[tool_name] = {
                    "total_calls": len(executions),
                    "successful_calls": successful,
                    "failed_calls": len(executions) - successful,
                    "success_rate": successful / len(executions) if executions else 0,
                    "priority": self.get_tool_priority(tool_name),
                    "performance_metrics": self.performance_metrics.get(tool_name, {})
                }
            report["tool_statistics"] = tool_stats
        
        # Add system metrics summary if available
        if self.system_metrics_history:
            avg_cpu = sum(m["cpu_percent"] for m in self.system_metrics_history) / len(self.system_metrics_history)
            avg_memory = sum(m["memory_percent"] for m in self.system_metrics_history) / len(self.system_metrics_history)
            peak_cpu = max(m["cpu_percent"] for m in self.system_metrics_history)
            peak_memory = max(m["memory_percent"] for m in self.system_metrics_history)
            
            report["system_metrics"] = {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory,
                "peak_cpu_percent": peak_cpu,
                "peak_memory_percent": peak_memory,
                "concurrency_changes": self._count_concurrency_changes()
            }
        
        # Check for patterns of issues
        issue_patterns = []
        
        # Look for consecutive failures
        consecutive_failures = 0
        for step in self.execution_history:
            if step.get("status") == "error":
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                
            if consecutive_failures >= 3 and {"type": "consecutive_failures", "count": consecutive_failures} not in issue_patterns:
                issue_patterns.append({"type": "consecutive_failures", "count": consecutive_failures})
        
        # Check for repeat attempts with same tool
        if self.tool_execution_history:
            for tool_name, stats in report.get("tool_statistics", {}).items():
                if stats["total_calls"] >= 3 and stats["success_rate"] < 0.5:
                    issue_patterns.append({
                        "type": "high_failure_rate_tool",
                        "tool": tool_name,
                        "success_rate": stats["success_rate"]
                    })
        
        report["issue_patterns"] = issue_patterns
        
        # Add retry statistics if available
        retry_stats = self._calculate_retry_statistics()
        if retry_stats:
            report["retry_statistics"] = retry_stats
            
        return report
    
    def _count_concurrency_changes(self) -> int:
        """
        Count how many times the concurrency limit changed due to dynamic adjustment.
        """
        if len(self.execution_history) < 2:
            return 0
            
        # This is an estimation as we don't directly track concurrency changes
        count = 0
        for i in range(1, len(self.system_metrics_history)):
            prev = self.system_metrics_history[i-1]
            curr = self.system_metrics_history[i]
            
            # Detect significant changes in CPU or memory that would trigger a change
            cpu_trigger = (
                (prev["cpu_percent"] < self.cpu_threshold_high and curr["cpu_percent"] >= self.cpu_threshold_high) or
                (prev["cpu_percent"] >= self.cpu_threshold_low and curr["cpu_percent"] < self.cpu_threshold_low)
            )
            
            memory_trigger = (
                (prev["memory_percent"] < self.memory_threshold and curr["memory_percent"] >= self.memory_threshold) or
                (prev["memory_percent"] >= self.memory_threshold and curr["memory_percent"] < self.memory_threshold)
            )
            
            if cpu_trigger or memory_trigger:
                count += 1
                
        return count
        
    def _calculate_retry_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics about retries from the execution history.
        """
        # This requires additional tracking which we might not have yet
        # This is a placeholder for future implementation
        return {}
        
    async def generate_visualizations(self) -> List[str]:
        """
        Generate visualization files for agent execution analysis.
        
        Returns:
            List of generated file paths
        """
        if not self.visualization_enabled:
            return []
            
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.visualization_output_dir, exist_ok=True)
            
            # Base filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{self.name}_{timestamp}"
            
            generated_files = []
            
            # Generate execution timeline JSON file
            timeline_file = os.path.join(self.visualization_output_dir, f"{base_filename}_timeline.json")
            with open(timeline_file, 'w') as f:
                json.dump({
                    "agent": self.name,
                    "steps": self.current_step,
                    "history": self.execution_history
                }, f, indent=2)
            generated_files.append(timeline_file)
            
            # Generate tool statistics JSON file
            if self.tool_execution_history:
                tools_file = os.path.join(self.visualization_output_dir, f"{base_filename}_tools.json")
                
                # Prepare tool statistics
                tool_stats = {}
                for tool_name, executions in self.tool_execution_history.items():
                    successful = sum(1 for e in executions if e.get("status") == "success")
                    tool_stats[tool_name] = {
                        "total_calls": len(executions),
                        "successful_calls": successful,
                        "failed_calls": len(executions) - successful,
                        "success_rate": successful / len(executions) if executions else 0,
                        "priority": self.get_tool_priority(tool_name),
                        "executions": executions
                    }
                
                with open(tools_file, 'w') as f:
                    json.dump({
                        "agent": self.name,
                        "tool_statistics": tool_stats
                    }, f, indent=2)
                generated_files.append(tools_file)
            
            # Generate system metrics JSON file
            if self.system_metrics_history:
                metrics_file = os.path.join(self.visualization_output_dir, f"{base_filename}_system_metrics.json")
                with open(metrics_file, 'w') as f:
                    json.dump({
                        "agent": self.name,
                        "metrics": self.system_metrics_history
                    }, f, indent=2)
                generated_files.append(metrics_file)
                
            logger.info(f"[{self.name}] Generated {len(generated_files)} visualization files")
            return generated_files
            
        except Exception as e:
            logger.error(f"[{self.name}] Failed to generate visualizations: {e}", exc_info=True)
            return []
            
    async def visualize_execution(self, output_format: str = "json") -> Dict[str, Any]:
        """
        Generate a visualization of the agent's execution.
        
        Args:
            output_format: The output format ("json", "html", etc.)
            
        Returns:
            Visualization data in the specified format
        """
        # For now, we only support JSON output
        report = await self.generate_execution_report()
        
        visualization = {
            "agent_name": self.name,
            "execution_data": report,
            "format": output_format,
            "generated_at": datetime.now().isoformat()
        }
        
        return visualization
        
    async def finalize(self) -> Dict[str, Any]:
        """
        Perform final cleanup and generate execution report.
        Also generates visualizations for analysis.
        """
        logger.info(f"[{self.name}] Finalizing agent and generating report")
        
        # Generate execution report
        execution_report = await self.generate_execution_report()
        
        # Generate final visualizations
        visualization_files = []
        if self.visualization_enabled:
            visualization_files = await self.generate_visualizations()
        
        results = {
            "execution_report": execution_report,
            "visualization_files": visualization_files
        }
        
        return results
