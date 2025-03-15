from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json
from pathlib import Path
import os

from pydantic import Field, BaseModel
import asyncio
import shutil

from app.agent.toolcall import ToolCallAgent
from app.prompt.financial_planner import NEXT_STEP_TEMPLATE, SYSTEM_PROMPT
from app.schema import Message
from app.tool import BashTool, StrReplaceEditor, Terminate, ToolCollection
from app.tool.financial_tools import (
    AustralianMarketAnalysisTool,
    MarketAnalysisTool,
    PortfolioOptimizationTool,
    ReportGeneratorTool,
    TaxOptimizationTool,
)
from app.tool.document_analyzer import DocumentAnalyzerTool
from app.tool.tool_creator import ToolCreatorTool
from app.tool.property_analyzer import PropertyAnalyzerTool
from app.tool.financial_integrations import FinancialIntegrationsTool
from app.tool.superannuation_analyzer import SuperannuationAnalyzerTool
from app.tool.web_search import WebSearchTool
from app.tool.website_generator import WebsiteGeneratorTool
from app.logger import logger
from app.conversation_manager import ConversationManager

class ToolExecutionMetrics(BaseModel):
    """Track metrics for tool execution within the agent."""
    tool_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    response_length: int = 0
    has_visualization: bool = False

class FinancialPlanningAgent(ToolCallAgent):
    """An advanced AI agent specializing in Australian financial planning and investment advice for high net worth clients."""

    name: str = "aus_financial_planner"
    description: str = """A sophisticated financial planning AI that specializes in providing comprehensive advice for high net worth Australian clients, with expertise in:
    - Complex entity and tax structure optimization (trusts, companies, SMSFs)
    - ASX and international investment research and analysis
    - Portfolio optimization with tax-aware strategies
    - Australian regulatory compliance (ASIC, ATO requirements)
    - Customized financial reporting and strategy development
    - SMSF strategy and compliance
    - Estate planning and succession
    - Website and document generation"""

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_TEMPLATE
    last_observation: Optional[str] = None
    thinking_steps: List[str] = Field(default_factory=list)
    current_section: Optional[str] = None
    completed_tasks: List[str] = Field(default_factory=list)
    include_disclaimers: bool = True

    max_steps: int = 5  # Reduced from 30 to 5 for testing purposes

    bash: BashTool = Field(default_factory=BashTool)
    working_dir: str = "."

    current_metrics: Optional[ToolExecutionMetrics] = None
    execution_history: List[ToolExecutionMetrics] = Field(default_factory=list)
    visualization_paths: List[str] = Field(default_factory=list)
    
    # Add conversation manager
    conversation_manager: Optional[ConversationManager] = None
    current_conversation_id: Optional[str] = None

    def __init__(self, include_disclaimers: bool = True, conversation_title: Optional[str] = None, **kwargs):
        # Initialize tools first
        tools = [
            AustralianMarketAnalysisTool(),
            MarketAnalysisTool(),  # Keep for international markets
            PortfolioOptimizationTool(),
            TaxOptimizationTool(),
            ReportGeneratorTool(),
            DocumentAnalyzerTool(),
            WebsiteGeneratorTool(),
            ToolCreatorTool(),
            BashTool(),
            StrReplaceEditor(),
            Terminate()
        ]
        kwargs['available_tools'] = ToolCollection(*tools)
        
        # Set disclaimer flag before init
        object.__setattr__(self, 'include_disclaimers', include_disclaimers)
        
        # Initialize conversation manager
        object.__setattr__(self, 'conversation_manager', ConversationManager())
        
        # Start a new conversation
        cm = object.__getattribute__(self, 'conversation_manager')
        conversation_id = cm.start_new_conversation(title=conversation_title)
        object.__setattr__(self, 'current_conversation_id', conversation_id)
        
        # Modify system prompt to remove disclaimer requirements if needed
        if not include_disclaimers:
            modified_prompt = SYSTEM_PROMPT.replace("4. Include appropriate risk warnings and disclaimers\n", "")
            kwargs['system_prompt'] = modified_prompt
            
        super().__init__(**kwargs)
        self.tool_creator = next(
            (tool for tool in tools if isinstance(tool, ToolCreatorTool)),
            None
        )
        
        # Update tools to use conversation directory for outputs
        self._configure_tools_for_conversation()
    
    def _configure_tools_for_conversation(self):
        """Configure tools to use the conversation directory for outputs."""
        if not self.conversation_manager:
            return
            
        conversation_dir = self.conversation_manager.get_conversation_path()
        
        # Configure each tool that needs it
        for tool in self.available_tools.tools:
            # Check tool type and set output paths appropriately
            if hasattr(tool, 'output_dir'):
                tool.output_dir = conversation_dir
                
            # For specific tools with custom output directories
            if isinstance(tool, ReportGeneratorTool) and hasattr(tool, 'report_dir'):
                tool.report_dir = conversation_dir
                
            if isinstance(tool, WebsiteGeneratorTool) and hasattr(tool, 'output_dir'):
                tool.output_dir = conversation_dir
                
            # For dividend analyzer
            if hasattr(tool, 'parameters') and isinstance(tool.parameters, dict) and 'output_dir' in tool.parameters:
                tool.parameters['output_dir'] = str(conversation_dir)

    async def think(self) -> bool:
        """Process current state and decide next action"""
        # Update working directory
        self.working_dir = await self.bash.execute("pwd")
        
        # Create a modified template with single braces
        modified_template = NEXT_STEP_TEMPLATE.replace("{{", "{").replace("}}", "}")
        
        # Now format with single braces
        self.next_step_prompt = modified_template.format(
            observation=getattr(self, 'last_observation', '') or '',
            open_file='',
            working_dir=self.working_dir
        )
        
        # Get current model information
        model_info = self.llm.get_model_info() if hasattr(self.llm, 'get_model_info') else {
            'name': self.llm.api_type,
            'capabilities': ['text generation', 'code understanding', 'tool use'],
            'selection_reason': 'Default model for financial planning tasks'
        }
        
        # Store thinking step with enhanced information
        step_number = len(self.thinking_steps) + 1
        thinking_step = f"""Step {step_number}:
🧠 Current Task: {self.current_section or 'Processing request'}
🤖 Model: {model_info['name']}
📝 Capabilities: {', '.join(model_info['capabilities'])}
🎯 Selection Reason: {model_info['selection_reason']}
🔄 Action: {self.next_step_prompt}
"""
        self.thinking_steps.append(thinking_step)
        
        # For Gemini models, use a simpler approach without forcing tool calls
        if self.llm.api_type == "gemini":
            if self.next_step_prompt:
                user_msg = Message.user_message(self.next_step_prompt)
                self.messages += [user_msg]
                
            # Get a direct response from Gemini
            content = await self.llm.ask(
                messages=self.messages,
                system_msgs=[Message.system_message(self.system_prompt)]
                if self.system_prompt
                else None,
            )
            
            # Log response info with enhanced details
            logger.info(f"""✨ {self.name}'s thoughts:
Model: {model_info['name']}
Task: {self.current_section or 'Processing request'}
Response: {content}""")
            
            # Create and add assistant message
            assistant_msg = Message.assistant_message(content)
            self.memory.add_message(assistant_msg)
            
            # No tool calls for Gemini mode, but return True to continue the conversation
            return True
        else:
            # For other models, use the standard tool call approach
            return await super().think()
        
    async def observe(self, observation: str) -> None:
        """Process an observation and update agent state."""
        self.last_observation = observation
        if hasattr(self, "update_memory") and callable(self.update_memory):
            self.update_memory(role="system", content=f"Observation: {observation}")
            
            # Store observation in conversation history
            if self.conversation_manager and self.current_conversation_id:
                self.conversation_manager.add_message(
                    self.current_conversation_id,
                    "system", 
                    f"Observation: {observation}"
                )

    async def process_message(self, message: str) -> str:
        """Process a message with enhanced tracking, clarification, and visualization support."""
        try:
            # Reset tracking for new message
            self.current_section = "Understanding Request"
            self.completed_tasks = []
            self.visualization_paths = []
            
            # Update the agent's memory with the user message
            self.update_memory(role="user", content=message)
            
            # Store user message in conversation history
            if self.conversation_manager and self.current_conversation_id:
                self.conversation_manager.add_message(
                    self.current_conversation_id,
                    "user",
                    message
                )
            
            # Generate clarification questions first
            clarification_prompt = f"""Based on the user's request: "{message}"
            Generate 3-5 specific follow-up questions to ensure I fully understand the task and requirements.
            Focus on:
            1. Any ambiguous aspects that need clarification
            2. Specific preferences or constraints
            3. Expected outcomes or deliverables
            4. Any technical requirements or limitations
            
            Format the response as a clear numbered list of questions only."""
            
            clarification_response = await self.llm.ask(
                messages=[Message.user_message(clarification_prompt)],
                system_msgs=[Message.system_message(self.system_prompt)] if self.system_prompt else None,
            )
            
            # Store clarification questions in memory
            self.update_memory(role="assistant", content=f"Before proceeding, I'd like to clarify a few points:\n\n{clarification_response}\n\nPlease provide any clarifications you can, and I'll adjust my approach accordingly.")
            
            # Store clarification in conversation history
            if self.conversation_manager and self.current_conversation_id:
                self.conversation_manager.add_message(
                    self.current_conversation_id,
                    "assistant",
                    f"Before proceeding, I'd like to clarify a few points:\n\n{clarification_response}\n\nPlease provide any clarifications you can, and I'll adjust my approach accordingly."
                )
            
            # Return the clarification questions without proceeding with the task
            return f"Before proceeding, I'd like to clarify a few points:\n\n{clarification_response}\n\nPlease provide any clarifications you can, and I'll adjust my approach accordingly."
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            
            # Store error in conversation history
            if self.conversation_manager and self.current_conversation_id:
                self.conversation_manager.add_message(
                    self.current_conversation_id,
                    "system",
                    f"Error: {str(e)}"
                )
                
            return error_msg

    async def process_clarified_message(self, original_message: str, clarifications: str) -> str:
        """Process the message after receiving clarifications."""
        try:
            # Combine original message with clarifications
            full_context = f"Original request: {original_message}\n\nClarifications provided: {clarifications}"
            
            # Update the agent's memory with the clarified context
            self.update_memory(role="system", content=full_context)
            
            # Store context in conversation history
            if self.conversation_manager and self.current_conversation_id:
                self.conversation_manager.add_message(
                    self.current_conversation_id,
                    "system",
                    full_context
                )
            
            # Analyze if we need to create or improve tools
            if self.tool_creator and "create tool" in original_message.lower():
                result = await self.tool_creator.execute(
                    request=original_message,
                    visualization_required="visualize" in original_message.lower() or "plot" in original_message.lower(),
                    analytics_required="analyze" in original_message.lower() or "report" in original_message.lower()
                )
                if result["status"] == "success":
                    self.completed_tasks.append(f"Created new tool: {result['tool_info']['name']}")
            
            # Run the agent with the clarified context
            response = await self.run()
            
            # Format response with progress and visualization information
            formatted_response = self._format_response_with_progress(response)
            
            # Store assistant response in conversation history
            if self.conversation_manager and self.current_conversation_id:
                self.conversation_manager.add_message(
                    self.current_conversation_id,
                    "assistant",
                    formatted_response,
                    {"visualization_paths": self.visualization_paths.copy() if self.visualization_paths else []}
                )
            
            return formatted_response
            
        except Exception as e:
            error_msg = f"Error processing clarified message: {str(e)}"
            logger.error(error_msg)
            
            # Store error in conversation history
            if self.conversation_manager and self.current_conversation_id:
                self.conversation_manager.add_message(
                    self.current_conversation_id,
                    "system",
                    f"Error: {str(e)}"
                )
                
            return error_msg

    def _format_response_with_progress(self, response: str) -> str:
        """Format response with enhanced progress tracking and metrics."""
        formatted_response = []
        
        # Add current section if available
        if self.current_section:
            formatted_response.append(f"## Working on: {self.current_section}")
        
        # Add main response
        formatted_response.append(response)
        
        # Add completed tasks
        if self.completed_tasks:
            formatted_response.append("\n### Completed Tasks:")
            for task in self.completed_tasks:
                formatted_response.append(f"✓ {task}")
        
        # Add execution metrics if available
        if self.current_metrics and self.current_metrics.end_time:
            execution_time = (self.current_metrics.end_time - self.current_metrics.start_time).total_seconds()
            formatted_response.append(f"\n### Execution Metrics:")
            formatted_response.append(f"- Tool: {self.current_metrics.tool_name}")
            formatted_response.append(f"- Execution Time: {execution_time:.2f} seconds")
            formatted_response.append(f"- Status: {'✓ Success' if self.current_metrics.success else '✗ Failed'}")
            
            if self.current_metrics.error_message:
                formatted_response.append(f"- Error: {self.current_metrics.error_message}")
        
        return "\n\n".join(formatted_response)

    def _filter_disclaimers(self, response: str) -> str:
        """Remove any disclaimer sections from the response."""
        # Common patterns for disclaimer sections
        disclaimer_patterns = [
            r"(?i)## ?Disclaimer.*?(?=##|$)",  # Markdown heading style
            r"(?i)\*\*Disclaimer.*?\*\*.*?(?=\n\n|\Z)",  # Bold text style
            r"(?i)Disclaimer:.*?(?=\n\n|\Z)",  # Simple colon style
            r"(?i)Please note:.*?(?=\n\n|\Z)",  # Note style that often contains disclaimer info
            r"(?i)This (analysis|information|advice) is for informational purposes only.*?(?=\n\n|\Z)",  # Common disclaimer start
            r"(?i)Past performance is not.*?future.*?(?=\n\n|\Z)"  # Common financial disclaimer
        ]
        
        import re
        filtered_response = response
        
        # Apply all patterns
        for pattern in disclaimer_patterns:
            filtered_response = re.sub(pattern, "", filtered_response, flags=re.DOTALL)
        
        # Clean up any extra newlines that might have been left
        filtered_response = re.sub(r'\n{3,}', '\n\n', filtered_response)
        
        return filtered_response.strip()

    async def execute_tool(self, tool_name, args):
        """Execute a tool with the given arguments."""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found."
        
        tool = self.tools[tool_name]
        
        # Adjust file paths in arguments to be in the conversation directory
        if self.conversation_manager and self.current_conversation_id:
            conversation_dir = self.conversation_manager.get_conversation_directory(self.current_conversation_id)
            if conversation_dir:
                for arg, value in args.items():
                    if isinstance(value, str) and ("path" in arg.lower() or "file" in arg.lower() or "dir" in arg.lower()):
                        # If it's a relative path that doesn't already point to the conversation directory
                        if not os.path.isabs(value) and not str(conversation_dir) not in value:
                            args[arg] = str(conversation_dir / value)
        
        # Record start of execution
        self.current_metrics = ToolExecutionMetrics(
            tool_name=tool_name,
            start_time=datetime.now()
        )

        try:
            result = await tool.execute(**args)
            
            # Update metrics
            if self.current_metrics:
                self.current_metrics.end_time = datetime.now()
                self.current_metrics.success = True
                # Check if result is a string or has a specific attribute
                if hasattr(result, 'output'):
                    response = str(result.output)
                    self.current_metrics.response_length = len(response)
                elif isinstance(result, str):
                    response = result
                    self.current_metrics.response_length = len(response)
                elif isinstance(result, dict):
                    response = json.dumps(result)
                    self.current_metrics.response_length = len(response)
                    # Check for visualization paths in result
                    for path_key in ['visualization_path', 'chart_path', 'report_path', 'output_dir', 'file_path']:
                        if path_key in result:
                            path = result.get(path_key)
                            if path and path not in self.visualization_paths:
                                # Adjust path to be within conversation directory if needed
                                if self.conversation_manager:
                                    path = self._ensure_path_in_conversation(path)
                                self.visualization_paths.append(path)
                                self.current_metrics.has_visualization = True
                                break
                else:
                    response = str(result)
                    self.current_metrics.response_length = len(response)
                
                self.execution_history.append(self.current_metrics)
                self.current_metrics = None
            
            return f"Tool {tool_name} executed successfully with result: {result}"
        except Exception as e:
            if self.current_metrics:
                self.current_metrics.end_time = datetime.now()
                self.current_metrics.success = False
                self.current_metrics.error_message = str(e)
                self.execution_history.append(self.current_metrics)
                self.current_metrics = None
            
            return f"Error executing tool {tool_name}: {str(e)}"

    def _ensure_path_in_conversation(self, path: str) -> str:
        """Ensure that the file path is within the conversation directory."""
        if not self.conversation_manager:
            return path
            
        path_obj = Path(path)
        
        # If path is already within conversation directory, return it
        if path_obj.is_relative_to(self.conversation_manager.get_conversation_path()):
            return path
            
        # If path exists, copy it to conversation directory
        if path_obj.exists() and path_obj.is_file():
            new_path = self.conversation_manager.get_conversation_path(path_obj.name)
            shutil.copy2(path, new_path)
            return str(new_path)
            
        # If path doesn't exist, return path within conversation directory
        return str(self.conversation_manager.get_conversation_path(path_obj.name))
        
    async def get_available_conversations(self):
        """Get a list of available conversations."""
        if self.conversation_manager:
            return self.conversation_manager.get_all_conversations()
        return []
        
    async def load_conversation(self, conversation_id):
        """Load a previous conversation."""
        if not self.conversation_manager:
            raise ValueError("Conversation manager not initialized")
            
        # Validate the conversation exists
        conversations = self.conversation_manager.get_all_conversations()
        exists = any(conv["id"] == conversation_id for conv in conversations)
        if not exists:
            raise ValueError(f"Conversation with ID {conversation_id} not found")
            
        # Load the conversation history
        history = self.conversation_manager.load_conversation(conversation_id)
        
        # Clear current memory
        self.memory.clear()
        
        # Load conversation into memory
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                if role == "user":
                    self.memory.add_user_message(content)
                elif role == "assistant":
                    self.memory.add_assistant_message(content)
                elif role == "system":
                    self.memory.add_system_message(content)
        
        # Set as current conversation
        self.current_conversation_id = conversation_id
        
        # Reconfigure tools to use this conversation directory
        self._configure_tools_for_conversation()
        
        # Return the loaded conversation data
        return history
