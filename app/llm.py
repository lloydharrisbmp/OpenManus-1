from typing import Dict, List, Optional, Union

from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Import Google Generative AI
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Import Groq
try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None

from app.config import LLMSettings, config
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)


REASONING_MODELS = ["o1", "o3-mini"]


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url
            self.model_type = llm_config.model_type
            
            if self.api_type == "gemini":
                if genai is None:
                    raise ImportError("Please install google-generativeai package: pip install google-generativeai")
                
                # Configure Gemini with API key
                if not self.api_key:
                    raise ValueError("API key is required for Gemini models")
                
                # Configure Gemini
                genai.configure(api_key=self.api_key)
                
                self.client = genai.GenerativeModel(self.model)
                # Determine model type if not explicitly set
                if not self.model_type:
                    if "flash-thinking" in self.model:
                        self.model_type = "flash-thinking"
                    elif "flash-exp-image" in self.model:
                        self.model_type = "flash-image"
                    elif "pro" in self.model:
                        self.model_type = "pro"
                    else:
                        self.model_type = "flash"
            elif self.api_type == "groq":
                if AsyncGroq is None:
                    raise ImportError("Please install groq package: pip install groq")
                self.client = AsyncGroq(api_key=self.api_key)
            elif self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # If message is already a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # If message is a Message object, convert it to dict
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Handle Gemini models
            if self.api_type == "gemini":
                # Convert messages to Gemini format
                gemini_messages = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    content = msg.get("content", "")
                    gemini_messages.append({"role": role, "parts": [content]})
                
                # Handle different Gemini model types
                if self.model_type == "flash-thinking":
                    # Optimize for detailed reasoning
                    response = await self.client.generate_content_async(
                        gemini_messages[-1]["parts"][0],
                        generation_config={
                            "temperature": temperature or self.temperature,
                            "max_output_tokens": self.max_tokens,
                            "candidate_count": 1,
                        }
                    )
                elif self.model_type == "flash-image":
                    # Handle image generation
                    response = await self.client.generate_content_async(
                        gemini_messages[-1]["parts"][0],
                        generation_config={
                            "temperature": temperature or self.temperature,
                            "max_output_tokens": self.max_tokens,
                        }
                    )
                else:
                    # Pro model with full chat capabilities
                    chat = self.client.start_chat(history=gemini_messages[:-1])
                    response = await chat.send_message_async(
                        gemini_messages[-1]["parts"][0],
                        generation_config={
                            "temperature": temperature or self.temperature,
                            "max_output_tokens": self.max_tokens,
                        }
                    )
                return response.text
            
            # Handle Groq models
            elif self.api_type == "groq":
                params = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature or self.temperature,
                    "stream": stream,
                }
                
                if not stream:
                    response = await self.client.chat.completions.create(**params)
                    return response.choices[0].message.content
                else:
                    response = await self.client.chat.completions.create(**params)
                    collected_messages = []
                    async for chunk in response:
                        chunk_message = chunk.choices[0].delta.content or ""
                        collected_messages.append(chunk_message)
                        print(chunk_message, end="", flush=True)
                    
                    print()  # Newline after streaming
                    full_response = "".join(collected_messages).strip()
                    return full_response
            
            # Handle OpenAI and Azure models
            else:
                params = {
                    "model": self.model,
                    "messages": messages,
                }

                if self.model in REASONING_MODELS:
                    params["max_completion_tokens"] = self.max_tokens
                else:
                    params["max_tokens"] = self.max_tokens
                    params["temperature"] = temperature or self.temperature

                if not stream:
                    # Non-streaming request
                    params["stream"] = False

                    response = await self.client.chat.completions.create(**params)

                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty or invalid response from LLM")
                    return response.choices[0].message.content

                # Streaming request
                params["stream"] = True
                response = await self.client.chat.completions.create(**params)

                collected_messages = []
                async for chunk in response:
                    chunk_message = chunk.choices[0].delta.content or ""
                    collected_messages.append(chunk_message)
                    print(chunk_message, end="", flush=True)

                print()  # Newline after streaming
                full_response = "".join(collected_messages).strip()
                if not full_response:
                    raise ValueError("Empty response from streaming LLM")
                return full_response

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")
            
            # Handle Gemini models for tool calls
            if self.api_type == "gemini":
                try:
                    # Convert OpenAI tool format to Gemini function format
                    gemini_tools = []
                    if tools:
                        for tool in tools:
                            if tool.get("type") == "function":
                                function_def = tool["function"]
                                
                                def process_properties(props):
                                    """Helper function to process parameter properties recursively"""
                                    processed = {}
                                    for name, value in props.items():
                                        if value.get("type") == "object":
                                            if "properties" in value and value["properties"]:
                                                processed[name] = {
                                                    "type": "OBJECT",
                                                    "properties": process_properties(value["properties"])
                                                }
                                            else:
                                                # For empty object properties, provide a minimal valid structure
                                                processed[name] = {
                                                    "type": "STRING",
                                                    "description": value.get("description", "Object value")
                                                }
                                        elif value.get("type") == "array" and "items" in value:
                                            if value["items"].get("type") == "object" and "properties" in value["items"]:
                                                processed[name] = {
                                                    "type": "ARRAY",
                                                    "items": {
                                                        "type": "OBJECT",
                                                        "properties": process_properties(value["items"]["properties"])
                                                    }
                                                }
                                            else:
                                                processed[name] = {
                                                    "type": "ARRAY",
                                                    "items": {
                                                        "type": value["items"].get("type", "STRING").upper()
                                                    }
                                                }
                                        else:
                                            processed[name] = {
                                                "type": value.get("type", "STRING").upper(),
                                                "description": value.get("description", "")
                                            }
                                    return processed
                                
                                # Create tool definition with processed parameters
                                tool_def = {
                                    "function_declarations": [{
                                        "name": function_def["name"],
                                        "description": function_def.get("description", ""),
                                        "parameters": {
                                            "type": "OBJECT",
                                            "properties": process_properties(
                                                function_def.get("parameters", {}).get("properties", {})
                                            )
                                        }
                                    }]
                                }
                                
                                gemini_tools.append(tool_def)
                    
                    # Convert messages to Gemini format
                    gemini_messages = []
                    for msg in messages:
                        role = "user" if msg["role"] == "user" else "model"
                        content = msg.get("content", "")
                        gemini_messages.append({"role": role, "parts": [content]})
                    
                    # Set up function calling
                    if gemini_tools and self.model_type in ["pro", "flash-thinking"]:
                        logger.info(f"Using Gemini model {self.model} with {len(gemini_tools)} tools")
                        chat = self.client.start_chat(history=gemini_messages[:-1])
                        response = await chat.send_message_async(
                            gemini_messages[-1]["parts"][0],
                            tools=gemini_tools,
                            generation_config={
                                "temperature": temperature or self.temperature,
                                "max_output_tokens": self.max_tokens,
                            }
                        )
                        
                        # Convert Gemini response to OpenAI-like format
                        if response.candidates and response.candidates[0].content.parts:
                            function_calls = []
                            for part in response.candidates[0].content.parts:
                                if hasattr(part, 'function_call'):
                                    try:
                                        # Extract function call data
                                        function_name = part.function_call.name or ""
                                        
                                        # Handle arguments that might come as objects or strings
                                        if hasattr(part.function_call, 'args'):
                                            if hasattr(part.function_call.args, 'to_dict'):
                                                # MapComposite object
                                                args = part.function_call.args.to_dict()
                                            else:
                                                # Regular object or string
                                                args = part.function_call.args
                                        else:
                                            args = {}
                                            
                                        function_calls.append({
                                            "id": f"call_{len(function_calls)}",
                                            "type": "function",
                                            "function": {
                                                "name": function_name,
                                                "arguments": args
                                            }
                                        })
                                    except Exception as e:
                                        logger.error(f"Error extracting function call: {e}")
                            
                            return {
                                "role": "assistant",
                                "content": response.text if not function_calls else None,
                                "tool_calls": function_calls if function_calls else None
                            }
                        return {"role": "assistant", "content": response.text}
                    else:
                        # Fall back to regular response if tools not supported
                        logger.info(f"Using Gemini model {self.model} without tools")
                        chat = self.client.start_chat(history=gemini_messages[:-1])
                        response = await chat.send_message_async(
                            gemini_messages[-1]["parts"][0],
                            generation_config={
                                "temperature": temperature or self.temperature,
                                "max_output_tokens": self.max_tokens,
                            }
                        )
                        return {"role": "assistant", "content": response.text}
                except Exception as e:
                    logger.error(f"Gemini API error: {str(e)}")
                    logger.error(f"Error type: {type(e)}")
                    logger.error(f"Error details: {repr(e)}")
                    raise
            
            # Handle Groq models
            elif self.api_type == "groq":
                params = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature or self.temperature,
                    "timeout": timeout,
                    **kwargs,
                }
                
                if tools:
                    params["tools"] = tools
                    params["tool_choice"] = tool_choice
                
                response = await self.client.chat.completions.create(**params)
                return response.choices[0].message
            
            # Handle OpenAI and Azure models
            else:
                # Set up the completion request
                params = {
                    "model": self.model,
                    "messages": messages,
                    "timeout": timeout,
                    **kwargs,
                }
                
                if tools:
                    params["tools"] = tools
                    params["tool_choice"] = tool_choice

                if self.model in REASONING_MODELS:
                    params["max_completion_tokens"] = self.max_tokens
                else:
                    params["max_tokens"] = self.max_tokens
                    params["temperature"] = temperature or self.temperature

                response = await self.client.chat.completions.create(**params)

                # Check if response is valid
                if not response.choices or not response.choices[0].message:
                    print(response)
                    raise ValueError("Invalid or empty response from LLM")

                return response.choices[0].message

        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise

    def get_model_info(self) -> dict:
        """Get detailed information about the current model."""
        model_capabilities = {
            "claude-3-sonnet-20240229": {
                "name": "Claude 3 Sonnet",
                "capabilities": [
                    "advanced reasoning",
                    "code generation and analysis",
                    "complex task planning",
                    "financial analysis",
                    "tool use",
                    "context-aware responses"
                ],
                "selection_reason": "Selected for its superior performance in financial planning tasks, ability to understand complex financial concepts, and reliable tool usage capabilities."
            },
            "gpt-4-turbo-preview": {
                "name": "GPT-4 Turbo",
                "capabilities": [
                    "advanced reasoning",
                    "code generation",
                    "task planning",
                    "financial analysis",
                    "tool use"
                ],
                "selection_reason": "Chosen for its strong general capabilities and reliable performance in financial analysis tasks."
            },
            "gemini-pro": {
                "name": "Gemini Pro",
                "capabilities": [
                    "text generation",
                    "code understanding",
                    "basic task planning",
                    "tool use"
                ],
                "selection_reason": "Used for basic financial planning tasks and general text generation."
            }
        }

        return model_capabilities.get(self.model, {
            "name": self.model,
            "capabilities": ["text generation", "code understanding", "tool use"],
            "selection_reason": f"Default model ({self.model}) for financial planning tasks"
        })
