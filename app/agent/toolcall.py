import asyncio
import json
import psutil
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

from pydantic import Field

from app.agent.react import ReActAgent
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message, ToolCall, TOOL_CHOICE_TYPE, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class ToolMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    last_execution: Optional[datetime] = None
    error_count: int = 0
    consecutive_failures: int = 0


class ToolCallAgent(ReActAgent):
    """
    A supercharged agent class for handling tool/function calls with concurrency, partial success/failure,
    robust error handling, stuck-state detection, and advanced logging.
    """

    name: str = "toolcall"
    description: str = "An agent that can execute multiple tool calls in a single step."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None
    
    # Enhanced features
    concurrency_limit: int = 3
    tool_execution_history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    partial_results_memory: Dict[str, Any] = Field(default_factory=dict)

    # Add new configuration fields
    max_retries: int = Field(default=3, description="Maximum number of retry attempts for failed tools")
    retry_delay: float = Field(default=1.0, description="Base delay between retries (seconds)")
    min_concurrency: int = Field(default=1, description="Minimum concurrency limit")
    max_concurrency: int = Field(default=10, description="Maximum concurrency limit")
    cpu_threshold: float = Field(default=80.0, description="CPU usage threshold percentage")
    memory_threshold: float = Field(default=80.0, description="Memory usage threshold percentage")
    system_metrics_history: List[Dict[str, float]] = Field(default_factory=list)
    
    # Add error classification
    class ToolError(Exception):
        """Base class for tool execution errors"""
        pass
        
    class TransientError(ToolError):
        """Error that might resolve with retry"""
        pass
        
    class PermanentError(ToolError):
        """Error that won't resolve with retry"""
        pass

    # Add new fields
    tool_priorities: Dict[str, ToolPriority] = Field(
        default_factory=dict,
        description="Priority levels for different tools"
    )
    tool_metrics: Dict[str, ToolMetrics] = Field(
        default_factory=lambda: defaultdict(ToolMetrics)
    )
    performance_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Add configuration for monitoring
    metrics_alert_threshold: int = Field(default=5, description="Number of consecutive failures to trigger alert")
    slow_execution_threshold: float = Field(default=10.0, description="Threshold for slow execution alert (seconds)")
    
    # Add visualization configuration
    visualization_output_dir: str = Field(
        default="visualizations",
        description="Directory for storing visualization outputs"
    )
    
    async def think(self) -> bool:
        """
        Determine which tools to call or content to produce, then store them in `self.tool_calls`.
        ReAct-style logic: LLM picks a tool or returns content.
        """
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        # Use the LLM to decide which tools to call
        response = await self.llm.ask_tool(
            messages=self.messages,
            system_msgs=[Message.system_message(self.system_prompt)]
            if self.system_prompt
            else None,
            tools=self.available_tools.to_params(),
            tool_choice=self.tool_choices,
        )

        self.tool_calls = response.tool_calls
        logger.info(f"✨ {self.name} LLM reasoning: {response.content}")
        if self.tool_calls:
            logger.info(
                f"🧰 Tools decided: {[call.function.name for call in self.tool_calls]}"
            )

        # Create an assistant message with either content or the tool calls
        try:
            if self.tool_calls:
                assistant_msg = Message.from_tool_calls(
                    content=response.content, tool_calls=self.tool_calls
                )
            else:
                assistant_msg = Message.assistant_message(response.content)
            self.memory.add_message(assistant_msg)
            
            # Track the latest assistant output for stuck detection
            self._track_assistant_output()
            
        except Exception as e:
            logger.error(f"🚨 {self.name}'s thinking process hit a snag: {e}", exc_info=True)
            self.memory.add_message(
                Message.assistant_message(f"Error encountered: {str(e)}")
            )
            return False

        # Check tool_choices mode
        if self.tool_choices == ToolChoice.NONE:
            if self.tool_calls:
                logger.warning(
                    f"🤔 {self.name} tried to use tools when none is allowed!"
                )
            # Add the LLM's content if any
            return bool(response.content)

        if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
            # Must have tool calls; we might handle it in act()
            return True

        # In 'auto' mode
        if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
            return bool(response.content)

        return bool(self.tool_calls)

    async def act(self) -> str:
        """
        Execute the tool calls. If multiple calls are present, run them in parallel for speed.
        Return partial success/failure logs if any calls fail.
        """
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no calls
            return self.messages[-1].content or "No content or commands"

        if len(self.tool_calls) == 1:
            # Single call - use original execution path
            return await self._execute_single_tool_call(self.tool_calls[0])
        else:
            # Multiple calls => use concurrency
            logger.info(f"[{self.name}] Executing {len(self.tool_calls)} tool calls in parallel with concurrency limit {self.concurrency_limit}")
            results_map = await self._execute_parallel_tool_calls(self.tool_calls)
            
            # Summarize partial successes/failures
            summary_str = self._summarize_parallel_outcomes(results_map)
            return summary_str

    async def _execute_single_tool_call(self, call: ToolCall) -> str:
        """
        Execute a single tool call with robust error handling.
        
        Args:
            call: The tool call to execute
            
        Returns:
            The result of the tool execution
        """
        try:
            logger.info(f"[{self.name}] Executing single tool call: {call.function.name}")
            result = await self.execute_tool(call)
            
            if self.max_observe and isinstance(self.max_observe, int):
                result = result[: self.max_observe]

            logger.info(f"🎯 Tool '{call.function.name}' completed! Result: {result}")
            
            # Add the result as a tool message
            tool_msg = Message.tool_message(
                content=result, tool_call_id=call.id, name=call.function.name
            )
            self.memory.add_message(tool_msg)
            
            # Track tool execution for analytics
            self._track_tool_execution(call.function.name, {
                "success": True,
                "result": result,
                "duration": None  # We don't track duration for single calls yet
            })
            
            return result
        except Exception as e:
            error_msg = f"Error during tool execution: {str(e)}"
            logger.error(f"[{self.name}] Tool call {call.function.name} failed: {error_msg}", exc_info=True)
            
            # Add tool message with error
            tool_msg = Message.tool_message(
                content=error_msg, tool_call_id=call.id, name=call.function.name
            )
            self.memory.add_message(tool_msg)
            
            # Track failed execution
            self._track_tool_execution(call.function.name, {
                "success": False,
                "error": str(e),
                "duration": None
            })
            
            return error_msg

    async def _execute_parallel_tool_calls(self, calls: List[ToolCall]) -> Dict[str, str]:
        """Execute tool calls with priority scheduling."""
        # Update system metrics
        await self._update_system_metrics()
        effective_concurrency = self._get_effective_concurrency_limit()
        
        # Sort calls by priority
        prioritized_calls = sorted(
            calls,
            key=lambda x: self.get_tool_priority(x.function.name).value,
            reverse=True
        )
        
        sem = asyncio.Semaphore(effective_concurrency)
        start_time = asyncio.get_event_loop().time()
        
        logger.info(
            f"[{self.name}] Executing {len(calls)} tools with priority scheduling "
            f"(concurrency: {effective_concurrency})"
        )
        
        # Execute calls with priority handling
        tasks = [self._execute_prioritized_call(tc, sem) for tc in prioritized_calls]
        raw_results = await asyncio.gather(*tasks)
        total_duration = asyncio.get_event_loop().time() - start_time
        
        # Process results and update metrics
        results_map = {}
        for tc_id, tool_name, result, duration in raw_results:
            self._update_tool_metrics(tool_name, result, duration)
            results_map[tc_id] = result
            
        # Check for performance issues
        self._check_performance_alerts()
        
        return results_map
    
    async def _execute_prioritized_call(
        self, tc: ToolCall, sem: asyncio.Semaphore
    ) -> Tuple[str, str, str, float]:
        """Execute a single tool call with priority consideration."""
        priority = self.get_tool_priority(tc.function.name)
        tc_start_time = asyncio.get_event_loop().time()
        
        try:
            async with sem:
                logger.debug(
                    f"[{self.name}] Starting {priority.name} priority execution "
                    f"of tool: {tc.function.name}"
                )
                result = await self.execute_tool(tc)
                duration = asyncio.get_event_loop().time() - tc_start_time
                
                logger.debug(
                    f"[{self.name}] Completed {priority.name} priority tool "
                    f"{tc.function.name} in {duration:.2f}s"
                )
                return tc.id, tc.function.name, result, duration
                
        except Exception as e:
            duration = asyncio.get_event_loop().time() - tc_start_time
            logger.error(
                f"[{self.name}] {priority.name} priority tool {tc.function.name} "
                f"failed after {duration:.2f}s: {e}",
                exc_info=True
            )
            return tc.id, tc.function.name, f"Error: {str(e)}", duration
    
    def _update_tool_metrics(self, tool_name: str, result: str, duration: float) -> None:
        """Update metrics for a tool execution."""
        metrics = self.tool_metrics[tool_name]
        metrics.total_calls += 1
        metrics.total_duration += duration
        metrics.avg_duration = metrics.total_duration / metrics.total_calls
        metrics.last_execution = datetime.now()
        
        if result.startswith("Error:"):
            metrics.failed_calls += 1
            metrics.error_count += 1
            metrics.consecutive_failures += 1
        else:
            metrics.successful_calls += 1
            metrics.consecutive_failures = 0
    
    def _check_performance_alerts(self) -> None:
        """Check for performance issues and generate alerts."""
        current_time = datetime.now()
        
        for tool_name, metrics in self.tool_metrics.items():
            # Check for consecutive failures
            if metrics.consecutive_failures >= self.metrics_alert_threshold:
                self._add_performance_alert(
                    tool_name,
                    "consecutive_failures",
                    f"Tool has failed {metrics.consecutive_failures} times in a row"
                )
            
            # Check for slow execution
            if metrics.avg_duration > self.slow_execution_threshold:
                self._add_performance_alert(
                    tool_name,
                    "slow_execution",
                    f"Tool average execution time ({metrics.avg_duration:.2f}s) "
                    f"exceeds threshold ({self.slow_execution_threshold}s)"
                )
            
            # Check error rate
            if metrics.total_calls > 0:
                error_rate = metrics.error_count / metrics.total_calls
                if error_rate > 0.5:  # More than 50% errors
                    self._add_performance_alert(
                        tool_name,
                        "high_error_rate",
                        f"Tool has high error rate: {error_rate:.1%}"
                    )
    
    def _add_performance_alert(self, tool_name: str, alert_type: str, message: str) -> None:
        """Add a performance alert to the history."""
        alert = {
            "timestamp": self.get_current_timestamp(),
            "tool": tool_name,
            "type": alert_type,
            "message": message,
            "priority": self.get_tool_priority(tool_name).name
        }
        
        self.performance_alerts.append(alert)
        logger.warning(f"[{self.name}] Performance Alert: {message}")

    def _summarize_parallel_outcomes(self, results_map: Dict[str, str]) -> str:
        """
        Summarize partial success/failures from concurrency into a single output.
        
        Args:
            results_map: Map of tool_call_id to results
            
        Returns:
            A formatted summary string
        """
        success_count = sum(1 for content in results_map.values() if not content.startswith("Error:"))
        failure_count = len(results_map) - success_count
        
        summary = [f"Executed {len(results_map)} tool calls with {success_count} successes and {failure_count} failures:"]
        
        # Add details for each tool call
        for tc_id, content in results_map.items():
            if content.startswith("Error:"):
                summary.append(f"❌ Tool Call {tc_id}: {content}")
            else:
                summary.append(f"✅ Tool Call {tc_id}: Succeeded")
                
        # If we have too many results, truncate for readability
        if len(summary) > 10:
            summary = summary[:5] + ["..."] + summary[-5:]
            
        return "\n".join(summary)

    def _track_tool_execution(self, tool_name: str, result: Dict[str, Any]) -> None:
        """
        Track tool execution results for analytics and debugging.
        
        Args:
            tool_name: Name of the tool
            result: Execution result data
        """
        if tool_name not in self.tool_execution_history:
            self.tool_execution_history[tool_name] = []
            
        execution_data = {
            "step": self.current_step,
            "result": result,
            "timestamp": self.get_current_timestamp(),
            "status": "success" if result.get("success", False) else "failed"
        }
        
        # Add duration if available
        if "duration" in result and result["duration"] is not None:
            execution_data["duration"] = result["duration"]
        
        self.tool_execution_history[tool_name].append(execution_data)
        
    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()

    async def execute_tool(self, command: ToolCall) -> str:
        """
        Execute a single tool call with robust error handling and retry logic.
        
        Args:
            command: The tool call to execute
            
        Returns:
            The result of the tool execution, or an error message
        """
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            args = json.loads(command.function.arguments or "{}")
        except json.JSONDecodeError:
            return self._format_error(f"Error parsing arguments for {name}: Invalid JSON")

        logger.info(f"🔧 Activating tool '{name}' with retry logic...")
        
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            try:
                result = await self.available_tools.execute(name=name, tool_input=args)
                observation = (f"Observed output of `{name}`:\n{str(result)}" 
                             if result else f"Cmd `{name}` completed with no output")
                
                # Handle special tools like Terminate
                await self._handle_special_tool(name, result)
                
                return observation
                
            except self.TransientError as e:
                # Potentially recoverable error
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(f"[{self.name}] Transient error in tool {name} (attempt {attempt}): {e}")
                    await asyncio.sleep(delay)
                continue
                
            except self.PermanentError as e:
                # Non-recoverable error
                error_msg = f"⚠️ Tool '{name}' encountered a permanent error: {str(e)}"
                logger.error(error_msg)
                return self._format_error(error_msg)
                
            except Exception as e:
                # Unclassified error - treat as permanent
                error_msg = f"⚠️ Tool '{name}' encountered an error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return self._format_error(error_msg)
        
        # All retries exhausted
        if last_error:
            error_msg = f"⚠️ Tool '{name}' failed after {self.max_retries} attempts: {str(last_error)}"
            logger.error(error_msg)
            return self._format_error(error_msg)
            
        return self._format_error(f"⚠️ Tool '{name}' failed for unknown reasons")

    def _format_error(self, message: str) -> str:
        """Format error messages consistently"""
        return f"Error: {message}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """
        Handle special tools like Terminate.
        
        Args:
            name: Tool name
            result: Tool execution result
            **kwargs: Additional arguments
        """
        if name.lower() in [n.lower() for n in self.special_tool_names]:
            logger.info(f"🏁 Special tool '{name}' is finishing the agent run.")
            self.state = AgentState.FINISHED
            
            # Perform final cleanup if needed
            await self.finalize()

    def _is_special_tool(self, name: str) -> bool:
        """
        Check if the tool is in the special tools list.
        
        Args:
            name: Tool name
            
        Returns:
            True if the tool is special, False otherwise
        """
        return name.lower() in [n.lower() for n in self.special_tool_names]

    def _should_finish_execution(self, **kwargs) -> bool:
        """
        Determine if execution should finish.
        
        Returns:
            True if execution should finish, False otherwise
        """
        return True
        
    async def finalize(self) -> Dict[str, Any]:
        """
        Perform final cleanup and generate execution report.
        
        Returns:
            Execution report data
        """
        logger.info(f"[{self.name}] Finalizing agent and generating report")
        
        # Generate execution report
        execution_report = self.generate_execution_report()
        
        # Perform any additional cleanup
        # (Add any resource cleanup needed here)
        
        return execution_report
        
    def generate_execution_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the agent's execution.
        
        Returns:
            Dictionary with execution report data
        """
        # Calculate success/failure stats for tools
        tool_stats = {}
        total_calls = 0
        total_success = 0
        
        for tool_name, executions in self.tool_execution_history.items():
            successful = sum(1 for e in executions if e.get("status") == "success")
            failures = len(executions) - successful
            
            tool_stats[tool_name] = {
                "total_calls": len(executions),
                "successful_calls": successful,
                "failed_calls": failures,
                "success_rate": successful / len(executions) if executions else 0
            }
            
            # Calculate average duration if available
            durations = [e.get("duration") for e in executions if e.get("duration") is not None]
            if durations:
                tool_stats[tool_name]["avg_duration"] = sum(durations) / len(durations)
                
            total_calls += len(executions)
            total_success += successful
            
        # Build report
        report = {
            "agent_name": self.name,
            "total_steps": self.current_step,
            "tool_stats": tool_stats,
            "total_tool_calls": total_calls,
            "total_successful_calls": total_success,
            "overall_success_rate": total_success / total_calls if total_calls > 0 else 0,
            "execution_history": self.execution_history if hasattr(self, "execution_history") else [],
            "performance_alerts": self.performance_alerts
        }
        
        return report
        
    def store_intermediate_result(self, key: str, value: Any) -> None:
        """
        Store intermediate results for chain-of-thought or debugging.
        
        Args:
            key: Result key
            value: Result value
        """
        self.partial_results_memory[key] = value
        self.memory.add_message(Message(role="assistant", content=f"**{key}**: {str(value)}"))
        logger.debug(f"[{self.name}] Storing intermediate result for {key}")

    async def _update_system_metrics(self) -> None:
        """Update system metrics for dynamic concurrency adjustment."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metrics = {
                "timestamp": self.get_current_timestamp(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "current_concurrency": self.concurrency_limit
            }
            
            self.system_metrics_history.append(metrics)
            
            # Keep only recent history
            if len(self.system_metrics_history) > 100:
                self.system_metrics_history = self.system_metrics_history[-100:]
                
            logger.debug(f"[{self.name}] System metrics updated - CPU: {cpu_percent}%, Memory: {memory.percent}%")
            
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to update system metrics: {e}")

    def _get_effective_concurrency_limit(self) -> int:
        """
        Calculate effective concurrency limit based on system metrics.
        Uses a sliding scale between min and max concurrency based on resource usage.
        """
        try:
            if not self.system_metrics_history:
                return self.concurrency_limit
                
            # Get latest metrics
            latest = self.system_metrics_history[-1]
            cpu_percent = latest["cpu_percent"]
            memory_percent = latest["memory_percent"]
            
            # Calculate resource pressure (0-1 scale)
            cpu_pressure = max(0, min(1, cpu_percent / self.cpu_threshold))
            memory_pressure = max(0, min(1, memory_percent / self.memory_threshold))
            
            # Use the more constrained resource
            resource_pressure = max(cpu_pressure, memory_pressure)
            
            # Calculate new limit
            range_size = self.max_concurrency - self.min_concurrency
            effective_limit = self.max_concurrency - int(range_size * resource_pressure)
            
            # Ensure within bounds
            effective_limit = max(self.min_concurrency, min(effective_limit, self.max_concurrency))
            
            # Log if limit changed
            if effective_limit != self.concurrency_limit:
                logger.info(f"[{self.name}] Adjusting concurrency limit from {self.concurrency_limit} to {effective_limit} "
                          f"(CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)")
                
            return effective_limit
            
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating concurrency limit: {e}")
            return self.concurrency_limit

    def set_tool_priority(self, tool_name: str, priority: ToolPriority) -> None:
        """Set priority level for a tool."""
        self.tool_priorities[tool_name] = priority
        logger.info(f"[{self.name}] Set {tool_name} priority to {priority.name}")
    
    def get_tool_priority(self, tool_name: str) -> ToolPriority:
        """Get priority level for a tool."""
        return self.tool_priorities.get(tool_name, ToolPriority.MEDIUM)

    async def generate_visualizations(self) -> List[str]:
        """
        Generate comprehensive visualizations of agent execution.
        
        Returns:
            List of generated visualization file paths
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.visualization_output_dir, exist_ok=True)
            
            # Generate base filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{self.name}_{timestamp}"
            
            generated_files = []
            
            # 1. Tool Performance Timeline
            timeline_path = self._generate_tool_timeline(base_filename)
            if timeline_path:
                generated_files.append(timeline_path)
            
            # 2. System Resource Usage
            resources_path = self._generate_resource_usage(base_filename)
            if resources_path:
                generated_files.append(resources_path)
            
            # 3. Tool Success/Failure Analysis
            analysis_path = self._generate_tool_analysis(base_filename)
            if analysis_path:
                generated_files.append(analysis_path)
            
            # 4. Performance Alerts Timeline
            alerts_path = self._generate_alerts_timeline(base_filename)
            if alerts_path:
                generated_files.append(alerts_path)
            
            logger.info(f"[{self.name}] Generated {len(generated_files)} visualization files")
            return generated_files
            
        except Exception as e:
            logger.error(f"[{self.name}] Error generating visualizations: {e}", exc_info=True)
            return []
    
    def _generate_tool_timeline(self, base_filename: str) -> Optional[str]:
        """Generate timeline visualization of tool executions."""
        try:
            # Prepare data
            data = []
            for tool_name, metrics in self.tool_metrics.items():
                for execution in self.tool_execution_history.get(tool_name, []):
                    data.append({
                        "tool": tool_name,
                        "timestamp": execution.get("timestamp"),
                        "duration": execution.get("duration", 0),
                        "success": execution.get("status") == "success",
                        "priority": self.get_tool_priority(tool_name).name
                    })
            
            if not data:
                return None
                
            df = pd.DataFrame(data)
            
            # Create figure
            fig = go.Figure()
            
            # Add successful executions
            success_df = df[df["success"]]
            fig.add_trace(go.Scatter(
                x=success_df["timestamp"],
                y=success_df["duration"],
                mode="markers",
                name="Successful",
                marker=dict(
                    color="green",
                    size=10,
                    symbol="circle"
                ),
                text=success_df["tool"],
                hovertemplate="Tool: %{text}<br>Duration: %{y:.2f}s<br>Time: %{x}"
            ))
            
            # Add failed executions
            failed_df = df[~df["success"]]
            fig.add_trace(go.Scatter(
                x=failed_df["timestamp"],
                y=failed_df["duration"],
                mode="markers",
                name="Failed",
                marker=dict(
                    color="red",
                    size=10,
                    symbol="x"
                ),
                text=failed_df["tool"],
                hovertemplate="Tool: %{text}<br>Duration: %{y:.2f}s<br>Time: %{x}"
            ))
            
            # Update layout
            fig.update_layout(
                title="Tool Execution Timeline",
                xaxis_title="Time",
                yaxis_title="Duration (seconds)",
                showlegend=True
            )
            
            # Save figure
            output_path = os.path.join(self.visualization_output_dir, f"{base_filename}_timeline.html")
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"[{self.name}] Error generating timeline visualization: {e}", exc_info=True)
            return None
    
    def _generate_resource_usage(self, base_filename: str) -> Optional[str]:
        """Generate system resource usage visualization."""
        try:
            if not self.system_metrics_history:
                return None
                
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add CPU usage
            timestamps = [m["timestamp"] for m in self.system_metrics_history]
            cpu_values = [m["cpu_percent"] for m in self.system_metrics_history]
            memory_values = [m["memory_percent"] for m in self.system_metrics_history]
            concurrency = [m["current_concurrency"] for m in self.system_metrics_history]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cpu_values,
                    name="CPU Usage",
                    line=dict(color="blue")
                ),
                secondary_y=False
            )
            
            # Add memory usage
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=memory_values,
                    name="Memory Usage",
                    line=dict(color="red")
                ),
                secondary_y=False
            )
            
            # Add concurrency limit
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=concurrency,
                    name="Concurrency Limit",
                    line=dict(color="green", dash="dash")
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title="System Resource Usage",
                xaxis_title="Time",
                showlegend=True
            )
            
            fig.update_yaxes(title_text="Usage %", secondary_y=False)
            fig.update_yaxes(title_text="Concurrency Limit", secondary_y=True)
            
            # Save figure
            output_path = os.path.join(self.visualization_output_dir, f"{base_filename}_resources.html")
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"[{self.name}] Error generating resource visualization: {e}", exc_info=True)
            return None
    
    def _generate_tool_analysis(self, base_filename: str) -> Optional[str]:
        """Generate tool success/failure analysis visualization."""
        try:
            # Prepare data
            data = []
            for tool_name, metrics in self.tool_metrics.items():
                data.append({
                    "tool": tool_name,
                    "success_rate": metrics.successful_calls / metrics.total_calls if metrics.total_calls > 0 else 0,
                    "avg_duration": metrics.avg_duration,
                    "total_calls": metrics.total_calls,
                    "priority": self.get_tool_priority(tool_name).name
                })
            
            if not data:
                return None
                
            df = pd.DataFrame(data)
            
            # Create figure with multiple subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Tool Success Rates", "Average Execution Duration")
            )
            
            # Add success rate bars
            fig.add_trace(
                go.Bar(
                    x=df["tool"],
                    y=df["success_rate"],
                    name="Success Rate",
                    marker_color="green",
                    text=[f"{rate:.1%}" for rate in df["success_rate"]],
                    textposition="auto"
                ),
                row=1, col=1
            )
            
            # Add average duration bars
            fig.add_trace(
                go.Bar(
                    x=df["tool"],
                    y=df["avg_duration"],
                    name="Avg Duration",
                    marker_color="blue",
                    text=[f"{dur:.2f}s" for dur in df["avg_duration"]],
                    textposition="auto"
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=False,
                title_text="Tool Performance Analysis"
            )
            
            fig.update_yaxes(title_text="Success Rate", row=1, col=1)
            fig.update_yaxes(title_text="Average Duration (s)", row=2, col=1)
            
            # Save figure
            output_path = os.path.join(self.visualization_output_dir, f"{base_filename}_analysis.html")
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"[{self.name}] Error generating tool analysis: {e}", exc_info=True)
            return None
    
    def _generate_alerts_timeline(self, base_filename: str) -> Optional[str]:
        """Generate timeline visualization of performance alerts."""
        try:
            if not self.performance_alerts:
                return None
                
            # Create figure
            fig = go.Figure()
            
            # Group alerts by type
            alert_types = set(alert["type"] for alert in self.performance_alerts)
            
            for alert_type in alert_types:
                type_alerts = [a for a in self.performance_alerts if a["type"] == alert_type]
                
                fig.add_trace(go.Scatter(
                    x=[a["timestamp"] for a in type_alerts],
                    y=[1] * len(type_alerts),  # Use constant y for timeline
                    mode="markers",
                    name=alert_type,
                    text=[f"{a['tool']}: {a['message']}" for a in type_alerts],
                    hovertemplate="%{text}<br>Time: %{x}"
                ))
            
            # Update layout
            fig.update_layout(
                title="Performance Alerts Timeline",
                xaxis_title="Time",
                yaxis_visible=False,
                showlegend=True
            )
            
            # Save figure
            output_path = os.path.join(self.visualization_output_dir, f"{base_filename}_alerts.html")
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"[{self.name}] Error generating alerts visualization: {e}", exc_info=True)
            return None
