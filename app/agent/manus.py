import asyncio
from typing import Any, Dict, List, Optional, Set
import json

from pydantic import Field
from loguru import logger

from app.agent.toolcall import ToolCallAgent
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_saver import FileSaver
from app.tool.google_search import GoogleSearch
from app.tool.python_execute import PythonExecute
from app.config import config
from app.schema import ToolCall


class Manus(ToolCallAgent):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends ToolCallAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    
    Enhanced with:
    - Concurrency for parallel tool execution
    - Partial success/failure tracking
    - Detailed logging and error handling
    - Unified tool cleanup
    - Retry logic for transient failures
    - Stuck-state detection
    - Enhanced status tracking
    - Comprehensive execution reporting
    """

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 2000
    max_steps: int = 20
    
    # New concurrency control
    concurrency_limit: int = 3
    
    # Retry configuration
    default_retry_attempts: int = 2
    default_retry_delay: float = 1.0

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), GoogleSearch(), BrowserUseTool(), FileSaver(), Terminate()
        )
    )
    
    # Track partial failures and execution stats
    step_execution_stats: List[Dict[str, Any]] = Field(default_factory=list)
    tool_execution_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    
    # Track previously seen content to detect loops
    previous_messages_hash: Set[str] = Field(default_factory=set)
    potential_loop_counter: int = Field(default=0)

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """
        Handle special tool calls with enhanced error handling and logging.
        """
        logger.info(f"[{self.name}] Handling special tool: {name}")
        
        # If the tool is the BrowserUseTool, do final cleanup
        if name == BrowserUseTool().name:
            try:
                await self.available_tools.get_tool(name).cleanup()
                logger.info(f"[{self.name}] Cleaned up browser tool successfully")
            except Exception as e:
                logger.error(f"[{self.name}] Error cleaning up browser tool: {e}", exc_info=True)
                
                # Track tool execution failure
                self._track_tool_execution(name, {"error": str(e), "success": False})
        
        # Call the parent implementation
        await super()._handle_special_tool(name, result, **kwargs)
    
    def _track_tool_execution(self, tool_name: str, result: Dict[str, Any]):
        """
        Track tool execution results for analytics and debugging.
        """
        if tool_name not in self.tool_execution_history:
            self.tool_execution_history[tool_name] = []
            
        execution_data = {
            "step": self.current_step,
            "result": result,
            "timestamp": config.get_current_time(),
            "status": "success" if result.get("success", False) else "failed"
        }
        
        self.tool_execution_history[tool_name].append(execution_data)
    
    async def execute_tool(
        self, 
        command: ToolCall,
        retry_attempts: Optional[int] = None,
        retry_delay: Optional[float] = None
    ) -> str:
        """
        Enhanced tool execution with retry logic, better error handling and tracking.
        Overrides the parent method to add detailed tracking and logging.
        
        Args:
            command: The tool call to execute
            retry_attempts: Number of retry attempts (defaults to self.default_retry_attempts)
            retry_delay: Delay between retries in seconds (defaults to self.default_retry_delay)
            
        Returns:
            The result of the tool execution
        """
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"
        
        # Use default retry settings if not specified
        retry_attempts = retry_attempts if retry_attempts is not None else self.default_retry_attempts
        retry_delay = retry_delay if retry_delay is not None else self.default_retry_delay
        
        # Parse arguments outside the retry loop to avoid repeated parsing failures
        try:
            args = json.loads(command.function.arguments or "{}")
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"[{self.name}] 📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            
            # Track JSON parsing failure
            self._track_tool_execution(name, {
                "success": False,
                "error": "JSONDecodeError",
                "arguments": command.function.arguments,
                "status": "failed"
            })
            
            return f"Error: {error_msg}"
        
        # Implement retry logic
        last_exception = None
        for attempt in range(retry_attempts):
            try:
                # Log the attempt if it's a retry
                if attempt > 0:
                    logger.info(f"[{self.name}] Retry attempt {attempt}/{retry_attempts-1} for tool '{name}'")
                
                # Execute the tool with additional logging
                logger.info(f"[{self.name}] Activating tool: '{name}' with arguments: {args}")
                start_time = config.get_current_time()
                result = await self.available_tools.execute(name=name, tool_input=args)
                end_time = config.get_current_time()
                execution_time = end_time - start_time

                # Track successful execution
                self._track_tool_execution(name, {
                    "success": True,
                    "execution_time": execution_time,
                    "args": args,
                    "attempt": attempt + 1,
                    "status": "success"
                })

                # Format result for display
                observation = (
                    f"Observed output of cmd `{name}` executed:\n{str(result)}"
                    if result
                    else f"Cmd `{name}` completed with no output"
                )

                # Handle special tools like `finish`
                await self._handle_special_tool(name=name, result=result)

                return observation
                
            except Exception as e:
                last_exception = e
                error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
                logger.warning(f"[{self.name}] {error_msg} (attempt {attempt+1}/{retry_attempts})")
                
                # If we have more attempts remaining, sleep before the next retry
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    # Log the final failure with full stack trace
                    logger.error(f"[{self.name}] All retry attempts failed for tool '{name}'", exc_info=True)
                    
                    # Track execution failure after all retries
                    self._track_tool_execution(name, {
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "attempt": attempt + 1,
                        "status": "failed_after_retries"
                    })
        
        # If we exit the loop, all retries failed
        return f"Error: {error_msg}"
    
    async def run_parallel_tools(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run multiple tool calls in parallel with concurrency control.
        
        Args:
            tool_calls: List of tool calls, each with 'tool_name' and 'args'
            
        Returns:
            Dictionary with results and success/failure stats
        """
        logger.info(f"[{self.name}] Running {len(tool_calls)} tools in parallel with concurrency limit {self.concurrency_limit}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        
        async def _execute_with_semaphore(tool_call):
            """Execute a single tool with semaphore control"""
            tool_name = tool_call.get("tool_name")
            args = tool_call.get("args", {})
            retry_attempts = tool_call.get("retry_attempts", self.default_retry_attempts)
            retry_delay = tool_call.get("retry_delay", self.default_retry_delay)
            
            async with semaphore:
                logger.info(f"[{self.name}] Executing tool '{tool_name}' in parallel task")
                try:
                    # Construct a ToolCall-like object for execute_tool
                    command = ToolCall(
                        id=f"parallel-{tool_name}-{config.get_current_time()}",
                        function=ToolCall.Function(
                            name=tool_name,
                            arguments=json.dumps(args)
                        )
                    )
                    
                    result = await self.execute_tool(
                        command, 
                        retry_attempts=retry_attempts,
                        retry_delay=retry_delay
                    )
                    return {"tool": tool_name, "result": result, "success": True, "status": "success"}
                except Exception as e:
                    logger.error(f"[{self.name}] Parallel execution of '{tool_name}' failed: {e}", exc_info=True)
                    return {"tool": tool_name, "error": str(e), "success": False, "status": "failed"}
        
        # Create and execute tasks for parallel execution
        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(_execute_with_semaphore(tool_call))
            tasks.append(task)
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process and summarize results
        final_report = {
            "success_count": 0,
            "failure_count": 0,
            "partial_success": False,
            "results": []
        }
        
        for result in results:
            if isinstance(result, Exception):
                final_report["failure_count"] += 1
                final_report["results"].append({
                    "error": str(result),
                    "success": False,
                    "status": "failed"
                })
            else:
                if result.get("success", False):
                    final_report["success_count"] += 1
                else:
                    final_report["failure_count"] += 1
                final_report["results"].append(result)
        
        # Determine if we had partial success (some tools succeeded, some failed)
        if final_report["success_count"] > 0 and final_report["failure_count"] > 0:
            final_report["partial_success"] = True
            final_report["status"] = "partial_success"
        elif final_report["success_count"] > 0:
            final_report["status"] = "success"
        else:
            final_report["status"] = "failed"
        
        # Record step execution stats
        self.step_execution_stats.append({
            "step": self.current_step,
            "total_tools": len(tool_calls),
            "success_count": final_report["success_count"],
            "failure_count": final_report["failure_count"],
            "partial_success": final_report["partial_success"],
            "status": final_report["status"],
            "timestamp": config.get_current_time()
        })
        
        return final_report
    
    def _check_for_stuck_state(self) -> bool:
        """
        Detect if the agent is stuck in a loop by checking:
        1. Repeated identical messages
        2. Multiple failed tool executions in a row
        3. Same tools being called repeatedly with no progress
        
        Returns:
            True if a stuck state is detected, False otherwise
        """
        # Not enough history to detect a loop
        if len(self.step_execution_stats) < 3:
            return False
            
        # Get the last few steps
        recent_steps = self.step_execution_stats[-3:]
        
        # Check for repeated failures
        if all(step["status"] == "failed" for step in recent_steps):
            self.potential_loop_counter += 1
            logger.warning(f"[{self.name}] Detected potential stuck state: {self.potential_loop_counter} consecutive steps with failures")
            if self.potential_loop_counter >= 3:
                logger.error(f"[{self.name}] Stuck state confirmed: Multiple consecutive failed steps")
                return True
        
        # Check for identical tool usage pattern (same tools being called repeatedly)
        # This requires comparing tool names, which would be in the execution history
        tool_patterns = []
        for step in recent_steps:
            step_tools = set()
            for tool_name, history in self.tool_execution_history.items():
                for entry in history:
                    if entry["step"] == step["step"]:
                        step_tools.add(tool_name)
            tool_patterns.append(frozenset(step_tools))
        
        # If all recent steps used exactly the same set of tools and all had some failures
        if len(set(tool_patterns)) == 1 and all(step["failure_count"] > 0 for step in recent_steps):
            self.potential_loop_counter += 1
            logger.warning(f"[{self.name}] Detected potential stuck state: {self.potential_loop_counter} consecutive steps with identical tool patterns")
            if self.potential_loop_counter >= 3:
                logger.error(f"[{self.name}] Stuck state confirmed: Identical failing tool patterns")
                return True
        else:
            # Reset counter if we don't see a consistent pattern
            self.potential_loop_counter = 0
        
        # Check if the most recent message content is repetitive
        if self.messages and len(self.messages) >= 2:
            # Create a hash of the most recent message content
            latest_msg_content = self.messages[-1].content or ""
            msg_hash = hash(latest_msg_content)
            
            # If we've seen this exact content before
            if msg_hash in self.previous_messages_hash:
                self.potential_loop_counter += 1
                logger.warning(f"[{self.name}] Detected potential stuck state: {self.potential_loop_counter} steps with repeated message content")
                if self.potential_loop_counter >= 3:
                    logger.error(f"[{self.name}] Stuck state confirmed: Repeated identical messages")
                    return True
            else:
                # Add this message hash to our set of seen messages
                self.previous_messages_hash.add(msg_hash)
                # Keep the set from growing too large
                if len(self.previous_messages_hash) > 10:
                    # Remove the oldest entry (approximately)
                    self.previous_messages_hash.pop()
        
        return False
    
    async def act(self) -> str:
        """
        Enhanced act method that supports parallel tool execution when appropriate.
        Also checks for stuck states and addresses them.
        
        This method overrides the parent method to add concurrency and
        better track partial successes and failures.
        """
        if not self.tool_calls:
            if self.tool_choices == "required":
                raise ValueError("Tool calls required but none provided")
                
            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"
        
        # Check for stuck state before execution
        if self._check_for_stuck_state():
            logger.warning(f"[{self.name}] Taking corrective action for stuck state")
            # Add a message to help break the loop
            stuck_message = (
                f"I notice we may be stuck in a loop. Let's try a different approach. "
                f"Previous steps have failed with similar patterns. "
                f"Let me reconsider the problem from a different angle."
            )
            self.memory.add_message(self.memory.create_user_message(stuck_message))
            # Reset the counter
            self.potential_loop_counter = 0
            # Return a message about the stuck state
            return f"Detected potential stuck state. Attempting to break the loop."
        
        # For a single tool call, use the standard approach
        if len(self.tool_calls) == 1:
            return await super().act()
        
        # For multiple tool calls, use parallel execution with concurrency
        logger.info(f"[{self.name}] Multiple tool calls detected ({len(self.tool_calls)}), using parallel execution")
        
        # Convert tool_calls to the format expected by run_parallel_tools
        parallel_tool_calls = []
        for command in self.tool_calls:
            try:
                args = json.loads(command.function.arguments or "{}")
                parallel_tool_calls.append({
                    "tool_name": command.function.name,
                    "args": args,
                    "retry_attempts": self.default_retry_attempts,
                    "retry_delay": self.default_retry_delay
                })
            except json.JSONDecodeError:
                logger.error(f"[{self.name}] Invalid JSON in arguments for {command.function.name}")
        
        # Execute tools in parallel
        execution_result = await self.run_parallel_tools(parallel_tool_calls)
        
        # Process results and add to memory
        results = []
        for idx, result_data in enumerate(execution_result["results"]):
            if idx < len(self.tool_calls):  # Safety check
                command = self.tool_calls[idx]
                result = result_data.get("result", f"Error: {result_data.get('error', 'Unknown error')}")
                
                if self.max_observe:
                    result = result[:self.max_observe]
                
                # Add tool response to memory
                tool_msg = self.memory.create_message(
                    "tool", 
                    content=result, 
                    tool_call_id=command.id,
                    tool_name=command.function.name
                )
                self.memory.add_message(tool_msg)
                results.append(result)
        
        # Summarize execution with status
        status_text = "success"
        if execution_result.get("partial_success", False):
            status_text = "partial success"
        elif execution_result["failure_count"] > 0 and execution_result["success_count"] == 0:
            status_text = "failure"
        
        summary = (f"Executed {len(parallel_tool_calls)} tools in parallel. "
                  f"Status: {status_text}, "
                  f"Success: {execution_result['success_count']}, "
                  f"Failures: {execution_result['failure_count']}")
        
        logger.info(f"[{self.name}] {summary}")
        
        return "\n\n".join([summary] + results)
    
    async def generate_execution_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the agent's execution history.
        
        This includes:
        - Overall statistics (success rates, tool usage, etc.)
        - Per-step execution summary
        - Per-tool performance analysis
        - Execution timeline
        - Identified issues or patterns
        
        Returns:
            A dictionary containing the execution report
        """
        # Skip if no execution history
        if not self.step_execution_stats:
            return {"error": "No execution history available"}
        
        # Calculate overall statistics
        total_tools_executed = sum(step["total_tools"] for step in self.step_execution_stats)
        total_success = sum(step["success_count"] for step in self.step_execution_stats)
        total_failures = sum(step["failure_count"] for step in self.step_execution_stats)
        
        # Success rate (avoid division by zero)
        overall_success_rate = (
            total_success / total_tools_executed if total_tools_executed > 0 else 0
        )
        
        # Per-tool statistics
        tool_stats = {}
        for tool_name, executions in self.tool_execution_history.items():
            successful_executions = sum(
                1 for exec_data in executions 
                if exec_data["result"].get("success", False)
            )
            
            total_executions = len(executions)
            avg_execution_time = 0
            if successful_executions > 0:
                # Calculate average execution time for successful executions
                execution_times = [
                    exec_data["result"].get("execution_time", 0)
                    for exec_data in executions
                    if exec_data["result"].get("success", False) 
                    and "execution_time" in exec_data["result"]
                ]
                if execution_times:
                    avg_execution_time = sum(execution_times) / len(execution_times)
            
            # Most common errors
            error_counts = {}
            for exec_data in executions:
                if not exec_data["result"].get("success", False):
                    error = exec_data["result"].get("error", "Unknown error")
                    error_type = exec_data["result"].get("error_type", "Unknown")
                    error_key = f"{error_type}: {error[:50]}"  # Truncate long errors
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            # Sort errors by frequency
            common_errors = sorted(
                [{"error": k, "count": v} for k, v in error_counts.items()],
                key=lambda x: x["count"],
                reverse=True
            )[:3]  # Top 3 most common errors
            
            tool_stats[tool_name] = {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": total_executions - successful_executions,
                "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
                "avg_execution_time": avg_execution_time,
                "common_errors": common_errors
            }
        
        # Step summary
        step_summary = []
        for step in self.step_execution_stats:
            step_summary.append({
                "step": step["step"],
                "total_tools": step["total_tools"],
                "success_count": step["success_count"],
                "failure_count": step["failure_count"],
                "status": step.get("status", "unknown"),
                "timestamp": step["timestamp"]
            })
        
        # Check for patterns of issues
        issue_patterns = []
        
        # Look for consecutive failures
        consecutive_failures = 0
        for step in self.step_execution_stats:
            if step["status"] == "failed":
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                
            if consecutive_failures >= 3 and {"type": "consecutive_failures", "count": consecutive_failures} not in issue_patterns:
                issue_patterns.append({"type": "consecutive_failures", "count": consecutive_failures})
        
        # Look for tools with high failure rates
        for tool_name, stats in tool_stats.items():
            if stats["total_executions"] >= 3 and stats["success_rate"] < 0.5:
                issue_patterns.append({
                    "type": "high_failure_rate_tool",
                    "tool": tool_name,
                    "success_rate": stats["success_rate"],
                    "total_executions": stats["total_executions"]
                })
        
        # Final report
        report = {
            "overall_stats": {
                "total_steps": len(self.step_execution_stats),
                "total_tool_executions": total_tools_executed,
                "successful_executions": total_success,
                "failed_executions": total_failures,
                "overall_success_rate": overall_success_rate,
                "tools_used": len(tool_stats)
            },
            "tool_stats": tool_stats,
            "step_summary": step_summary,
            "issue_patterns": issue_patterns,
            "execution_timeline": [
                {
                    "step": step["step"],
                    "timestamp": step["timestamp"],
                    "status": step.get("status", "unknown")
                }
                for step in self.step_execution_stats
            ]
        }
        
        return report
    
    async def finalize(self):
        """
        Perform final cleanup of all tools when the agent is done.
        Also generates a final execution report for analysis.
        """
        logger.info(f"[{self.name}] Finalizing agent and cleaning up all tools")
        
        cleanup_results = {}
        for tool_name, tool in self.available_tools.tool_map.items():
            if hasattr(tool, "cleanup") and callable(tool.cleanup):
                try:
                    logger.info(f"[{self.name}] Cleaning up tool: {tool_name}")
                    await tool.cleanup()
                    cleanup_results[tool_name] = {"success": True}
                except Exception as e:
                    logger.error(f"[{self.name}] Error cleaning up tool {tool_name}: {e}", exc_info=True)
                    cleanup_results[tool_name] = {"success": False, "error": str(e)}
        
        # Generate final execution report
        execution_report = await self.generate_execution_report()
        
        # Add final statistics
        total_tool_calls = sum(len(history) for history in self.tool_execution_history.values())
        successful_calls = sum(
            sum(1 for exec_data in history if exec_data["result"].get("success", False))
            for history in self.tool_execution_history.values()
        )
        
        logger.info(f"[{self.name}] Agent finalized. "
                   f"Total tool calls: {total_tool_calls}, "
                   f"Successful: {successful_calls}, "
                   f"Failed: {total_tool_calls - successful_calls}")
        
        # Include execution report in cleanup results
        cleanup_results["execution_report"] = execution_report
        
        return cleanup_results
