import json
import time
from typing import Dict, List, Optional, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow, PlanStepStatus
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool


class PlanningFlow(BaseFlow):
    """A flow that manages planning and execution of tasks using agents."""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # Set executor keys before super().__init__
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # Set plan ID if provided
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # Initialize the planning tool if not provided
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # Call parent's init with the processed data
        super().__init__(agents, **data)

        # Set executor_keys to all agent keys if not specified
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """
        Get an appropriate executor agent for the current step.
        Can be extended to select agents based on step type/requirements.
        """
        # If step type is provided and matches an agent key, use that agent
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # Otherwise use the first available executor or fall back to primary agent
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # Fallback to primary agent
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """Execute the planning flow with agents."""
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # Create initial plan if input provided
            if input_text:
                await self._create_initial_plan(input_text)

                # Verify plan was created successfully
                if hasattr(self.planning_tool, 'plans') and isinstance(self.planning_tool.plans, dict):
                    if self.active_plan_id not in self.planning_tool.plans:
                        logger.error(
                            f"Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                        )
                        return f"Failed to create plan for: {input_text}"
                else:
                    logger.error("Planning tool has no plans attribute or it's not a dictionary")
                    return f"Failed to create plan for: {input_text}"

            result = ""
            while True:
                # Get current step to execute
                self.current_step_index, step_info = await self._get_current_step_info()

                # Exit if no more steps or plan completed
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # Execute current step with appropriate agent
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # Check if agent wants to terminate
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"

    async def _create_initial_plan(self, request: str) -> None:
        """Create an initial plan based on the request using the flow's LLM and PlanningTool."""
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        # Create a system message for plan creation
        system_message = Message.system_message(
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. "
            "Optimize for clarity and efficiency."
        )

        # Create a user message with the request
        user_message = Message.user_message(
            f"Create a reasonable plan with clear steps to accomplish the task: {request}"
        )

        # Call LLM with PlanningTool
        if hasattr(self.llm, 'ask_tool') and callable(self.llm.ask_tool):
            tool_param = None
            if hasattr(self.planning_tool, 'to_param') and callable(self.planning_tool.to_param):
                tool_param = self.planning_tool.to_param()
            else:
                logger.error("Planning tool doesn't have a valid to_param method")
                return
                
            if tool_param:
                try:
                    response = await self.llm.ask_tool(
                        messages=[user_message],
                        system_msgs=[system_message],
                        tools=[tool_param],
                        tool_choice=ToolChoice.AUTO,
                    )
                except Exception as e:
                    logger.error(f"Error calling LLM to create plan: {e}")
                    return
            else:
                logger.error("Failed to get tool parameters")
                return
        else:
            logger.error("LLM doesn't have a valid ask_tool method")
            return

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        Parse the current plan to identify the first non-completed step's index and info.
        Returns (None, None) if no active step is found.
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return None, None

        try:
            # Direct access to plan data from planning tool storage
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # Find first non-completed step
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # Extract step type/category if available
                    step_info = {"text": step}

                    # Try to extract step type from the text (e.g., [SEARCH] or [CODE])
                    import re

                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # Mark current step as in_progress
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"Error marking step as in_progress: {e}")
                        # Update step status directly if needed
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # No active step found

        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """Execute a single step using the appropriate agent."""
        if not executor or not step_info:
            return "No executor or step information available"

        # Set the prompt to include step details
        step_prompt = (
            f"Execute this specific step in the plan: {step_info.get('description', '')}\n"
            f"Remember to focus only on this step, not the entire plan. "
            f"Use the available tools to accomplish the task."
        )

        # Call the executor agent
        if hasattr(executor, 'execute') and callable(executor.execute):
            try:
                result = await executor.execute(step_prompt)
                return result
            except Exception as e:
                logger.error(f"Error executing step: {e}")
                return f"Error executing step: {str(e)}"
        else:
            return "Executor agent doesn't have a valid execute method"

    async def _mark_step_completed(self) -> None:
        """Mark the current step as completed in the plan."""
        if self.current_step_index is None or not self.active_plan_id:
            return

        if hasattr(self.planning_tool, 'execute') and callable(self.planning_tool.execute):
            try:
                await self.planning_tool.execute(
                    command="mark_step",
                    plan_id=self.active_plan_id,
                    step_index=self.current_step_index,
                    step_status="completed",
                )
                logger.info(
                    f"Marked step {self.current_step_index} in plan {self.active_plan_id} as completed"
                )
            except Exception as e:
                logger.error(f"Error marking step as completed: {e}")
        else:
            logger.error("Planning tool doesn't have a valid execute method")

    async def _get_plan_text(self) -> str:
        """Get the current plan as text."""
        if not self.active_plan_id:
            return "No active plan"

        # Try to get plan from planning tool
        if hasattr(self.planning_tool, 'execute') and callable(self.planning_tool.execute):
            try:
                result = await self.planning_tool.execute(
                    command="get",
                    plan_id=self.active_plan_id,
                )
                if isinstance(result, dict) and "plan" in result:
                    return result["plan"]
                elif hasattr(result, "output"):
                    return result.output
                else:
                    return str(result)
            except Exception as e:
                logger.error(f"Error getting plan from planning tool: {e}")

        # Fall back to generating plan text from stored plan
        return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """Generate a text representation of the plan from internal storage."""
        if not self.active_plan_id:
            return "No active plan"

        # Check if planning tool has plans attribute and it has our plan
        if not hasattr(self.planning_tool, 'plans') or not isinstance(self.planning_tool.plans, dict):
            return "Planning tool has no valid plans storage"

        plan_data = self.planning_tool.plans.get(self.active_plan_id)
        if not plan_data:
            return f"Plan with ID {self.active_plan_id} not found"

        title = plan_data.get("title", "Untitled Plan")
        steps = plan_data.get("steps", [])
        step_statuses = plan_data.get("step_statuses", [])
        step_notes = plan_data.get("step_notes", [])

        # Ensure step_statuses and step_notes match the number of steps
        while len(step_statuses) < len(steps):
            step_statuses.append(PlanStepStatus.NOT_STARTED.value)
        while len(step_notes) < len(steps):
            step_notes.append("")

        # Count steps by status
        status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

        for status in step_statuses:
            if status in status_counts:
                status_counts[status] += 1

        completed = status_counts[PlanStepStatus.COMPLETED.value]
        total = len(steps)
        progress = (completed / total) * 100 if total > 0 else 0

        plan_text = f"Plan: {title} (ID: {self.active_plan_id})\n"
        plan_text += "=" * len(plan_text) + "\n\n"

        plan_text += (
            f"Progress: {completed}/{total} steps completed ({progress:.1f}%)\n"
        )
        plan_text += f"Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, "
        plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n"
        plan_text += "Steps:\n"

        status_marks = PlanStepStatus.get_status_marks()

        for i, (step, status, notes) in enumerate(
            zip(steps, step_statuses, step_notes)
        ):
            # Use status marks to indicate step status
            status_mark = status_marks.get(
                status, status_marks[PlanStepStatus.NOT_STARTED.value]
            )

            plan_text += f"{i}. {status_mark} {step}\n"
            if notes:
                plan_text += f"   Notes: {notes}\n"

        return plan_text

    async def _finalize_plan(self) -> str:
        """Finalize the plan and provide a summary using the flow's LLM directly."""
        plan_text = await self._get_plan_text()

        # Create a summary using the flow's LLM directly
        try:
            system_message = Message.system_message(
                "You are a planning assistant. Your task is to summarize the completed plan."
            )

            user_message = Message.user_message(
                f"The plan has been completed. Here is the final plan status:\n\n{plan_text}\n\nPlease provide a summary of what was accomplished and any final thoughts."
            )

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"Plan completed:\n\n{response}"
        except Exception as e:
            logger.error(f"Error finalizing plan with LLM: {e}")

            # Fallback to using an agent for the summary
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                The plan has been completed. Here is the final plan status:

                {plan_text}

                Please provide a summary of what was accomplished and any final thoughts.
                """
                summary = await agent.run(summary_prompt)
                return f"Plan completed:\n\n{summary}"
            except Exception as e2:
                logger.error(f"Error finalizing plan with agent: {e2}")
                return "Plan completed. Error generating summary."
