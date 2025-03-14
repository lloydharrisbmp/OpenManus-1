import asyncio
import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message, ToolCall, TOOL_CHOICE_TYPE, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


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
        """
        Run multiple calls concurrently with a concurrency limit.
        
        Args:
            calls: List of tool calls to execute in parallel
            
        Returns:
            A dict mapping tool_call_id -> result string (or error message)
        """
        sem = asyncio.Semaphore(self.concurrency_limit)
        start_time = asyncio.get_event_loop().time()

        async def run_call(tc: ToolCall) -> Tuple[str, str, str, float]:
            """
            Execute a tool call with the semaphore for concurrency control.
            
            Returns:
                A tuple (tc_id, tool_name, result_str, duration)
            """
            tc_start_time = asyncio.get_event_loop().time()
            try:
                async with sem:
                    logger.debug(f"[{self.name}] Starting parallel execution of tool: {tc.function.name}")
                    result = await self.execute_tool(tc)
                    duration = asyncio.get_event_loop().time() - tc_start_time
                    logger.debug(f"[{self.name}] Completed tool {tc.function.name} in {duration:.2f}s")
                    return tc.id, tc.function.name, result, duration
            except Exception as e:
                duration = asyncio.get_event_loop().time() - tc_start_time
                logger.error(f"[{self.name}] Tool {tc.function.name} failed after {duration:.2f}s: {e}", exc_info=True)
                return tc.id, tc.function.name, f"Error: {str(e)}", duration

        # Create tasks for parallel execution
        tasks = [run_call(tc) for tc in calls]
        raw_results = await asyncio.gather(*tasks)
        total_duration = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"[{self.name}] Completed all parallel tool calls in {total_duration:.2f}s")

        # Process results
        results_map = {}
        for tc_id, tool_name, result, duration in raw_results:
            # Apply max_observe limit if set
            if self.max_observe and isinstance(self.max_observe, int):
                result = result[:self.max_observe]
                
            # Determine success/failure
            is_success = not result.startswith("Error:")
            
            # Track execution statistics
            self._track_tool_execution(tool_name, {
                "success": is_success,
                "result": result if is_success else None,
                "error": result if not is_success else None,
                "duration": duration
            })
            
            # Add tool message to memory
            tool_msg = Message.tool_message(
                content=result, tool_call_id=tc_id, name=tool_name
            )
            self.memory.add_message(tool_msg)
            
            # Store in results map
            results_map[tc_id] = result
            
        # Store partial results in memory for chain-of-thought
        self.store_intermediate_result("parallel_tool_results", {
            "total_count": len(calls),
            "success_count": sum(1 for r in raw_results if not r[2].startswith("Error:")),
            "failure_count": sum(1 for r in raw_results if r[2].startswith("Error:")),
            "total_duration": total_duration
        })

        return results_map

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
        Execute a single tool call with robust error handling.
        If tool arguments are invalid JSON or other errors occur, capture them as partial failures.
        
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
            error_msg = f"Error parsing arguments for {name}: Invalid JSON"
            logger.error(f"[{self.name}] Invalid JSON arguments for '{name}': {command.function.arguments}")
            return error_msg

        logger.info(f"🔧 Activating tool '{name}' with concurrent approach...")

        try:
            result = await self.available_tools.execute(name=name, tool_input=args)
            observation = f"Observed output of `{name}`:\n{str(result)}" if result else f"Cmd `{name}` completed with no output"
            
            # Handle special tools like Terminate
            await self._handle_special_tool(name, result)
            
            return observation
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"

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
            "execution_history": self.execution_history if hasattr(self, "execution_history") else []
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
