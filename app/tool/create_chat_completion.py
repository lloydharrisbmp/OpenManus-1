import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError
from pydantic_core import SchemaError

from app.tool.base import BaseTool

logger = logging.getLogger(__name__)

class RetrySettings(BaseModel):
    """Optional retry logic for completion requests."""
    attempts: int = 1
    delay_seconds: float = 1.0

class CreateChatCompletionParameters(BaseModel):
    """Input parameters for CreateChatCompletion."""
    response_schema: Optional[dict] = Field(
        default=None, 
        description="Explicit JSON schema to validate the response"
    )
    response_model: Optional[str] = Field(
        default=None,
        description="Name of a Pydantic model for validating typed responses"
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Template for generating completions, can contain placeholders"
    )
    prompt_variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variables to fill into the prompt_template"
    )
    fallback_prompt: Optional[str] = Field(
        default=None,
        description="Fallback prompt if prompt_template is missing or invalid"
    )
    multi_completions: int = Field(
        default=1,
        description="Number of completions to generate in a single request"
    )
    required_fields: List[str] = Field(
        default_factory=lambda: ["response"],
        description="Names of required fields in the returned object"
    )
    retry_settings: Optional[RetrySettings] = None

    class Config:
        arbitrary_types_allowed = True

class CreateChatCompletion(BaseTool):
    """Enhanced chat completion tool with advanced features."""

    name: str = "create_chat_completion"
    description: str = (
        "Creates a structured completion with specified output formatting, "
        "schema validation, and advanced features."
    )

    parameters: Dict[str, Any] = {
            "type": "object",
            "properties": {
            "response_schema": {
                "type": "object",
                "description": "Explicit JSON schema to validate the response."
            },
            "response_model": {
                    "type": "string",
                "description": "Name of a Pydantic model to validate typed responses."
            },
            "prompt_template": {
                "type": "string",
                "description": "Template for generating completions, can contain placeholders."
            },
            "prompt_variables": {
                "type": "object",
                "description": "Variables to fill into the prompt template"
            },
            "fallback_prompt": {
                        "type": "string",
                "description": "Prompt fallback if the template is missing or invalid."
            },
            "multi_completions": {
                "type": "integer",
                "description": "Number of completions to generate in one request."
            },
            "required_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of required fields in the returned object."
            },
            "retry_settings": {
                "type": "object",
                "properties": {
                    "attempts": {"type": "integer"},
                    "delay_seconds": {"type": "number"}
                }
            }
        }
    }

    # Concurrency control
    concurrency_limit: int = 3
    _sem: asyncio.Semaphore = asyncio.Semaphore(3)

    def __init__(self):
        super().__init__()
        logger.debug("[CreateChatCompletion] Initialized with concurrency limit = 3")

    async def execute(
        self,
        response_schema: Optional[dict] = None,
        response_model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        fallback_prompt: Optional[str] = None,
        multi_completions: int = 1,
        required_fields: Optional[List[str]] = None,
        retry_settings: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute the chat completion with advanced features."""
        params = CreateChatCompletionParameters(
            response_schema=response_schema,
            response_model=response_model,
            prompt_template=prompt_template,
            prompt_variables=prompt_variables or {},
            fallback_prompt=fallback_prompt,
            multi_completions=multi_completions,
            required_fields=required_fields or ["response"],
            retry_settings=(RetrySettings(**retry_settings) if retry_settings else None)
        )

        async with self._sem:
            attempts = params.retry_settings.attempts if params.retry_settings else 1
            delay = params.retry_settings.delay_seconds if params.retry_settings else 1.0

            for attempt_num in range(attempts):
                try:
                    # Generate final prompt text
                    prompt = self._build_prompt(params)
                    logger.info(f"[CreateChatCompletion] Using prompt:\n{prompt}")

                    # Generate completions
                    raw_completions = await self._generate_completions(
                        prompt, count=params.multi_completions
                    )

                    # Validate and convert each completion
                    validated_completions = []
                    for c in raw_completions:
                        validated = self._validate_and_convert_response(c, params)
                        validated_completions.append(validated)

                    return (validated_completions[0] if params.multi_completions == 1 
                           else validated_completions)

                except Exception as e:
                    if attempt_num < attempts - 1:
                        logger.warning(
                            f"[CreateChatCompletion] Retry {attempt_num+1}/{attempts} "
                            f"due to error: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"[CreateChatCompletion] Final attempt failed: {e}",
                            exc_info=True
                        )
                        return {"error": str(e), "success": False}

    def _build_prompt(self, params: CreateChatCompletionParameters) -> str:
        """Build final prompt from template and variables."""
        if params.prompt_template:
            try:
                return params.prompt_template.format(**params.prompt_variables)
            except KeyError as e:
                logger.warning(f"[CreateChatCompletion] Missing placeholder: {e}")
                if params.fallback_prompt:
                    return params.fallback_prompt
        if params.fallback_prompt:
            return params.fallback_prompt
        return "Respond to the user."

    async def _generate_completions(self, prompt: str, count: int = 1) -> List[str]:
        """Generate completions using the configured LLM."""
        try:
            from app.services.llm import get_llm_service
            llm = get_llm_service()
            completions = await llm.generate_completions(prompt, count)
            return completions
        except ImportError:
            # Fallback to mock implementation for testing
            return await self._generate_completions_mock(prompt, count)

    async def _generate_completions_mock(self, prompt: str, count: int = 1) -> List[str]:
        """Mock implementation for testing."""
        logger.debug("[CreateChatCompletion] Using mock completion generator")
        completions = []
        for i in range(count):
            simulated = {
                "response": f"Answer for: {prompt}",
                "extra_field": f"Extra data {i+1}"
            }
            completions.append(json.dumps(simulated))
        return completions

    def _validate_and_convert_response(
        self,
        raw_response: str,
        params: CreateChatCompletionParameters
    ) -> Any:
        """Validate response against schema or model."""
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as e:
            msg = f"Response is not valid JSON: {str(e)}"
            logger.error(msg)
            return {"error": msg, "success": False}

        # Check required fields
        for req in params.required_fields:
            if req not in data:
                msg = f"Missing required field '{req}' in response."
                logger.error(msg)
                return {"error": msg, "success": False}

        # Validate with JSON schema
        if params.response_schema:
            try:
                import jsonschema
                jsonschema.validate(instance=data, schema=params.response_schema)
            except (jsonschema.ValidationError, jsonschema.SchemaError) as e:
                msg = f"JSON schema validation failed: {str(e)}"
                logger.error(msg)
                return {"error": msg, "success": False}

        # Validate with Pydantic model
        if params.response_model:
            model_cls = self._find_registered_model(params.response_model)
            if not model_cls:
                return {
                    "error": f"Response model '{params.response_model}' not found.",
                    "success": False
                }

            try:
                instance = model_cls(**data)
                data = instance.dict()
            except ValidationError as e:
                msg = f"Model validation error: {str(e)}"
                logger.error(msg)
                return {"error": msg, "success": False}

        return data

    def _find_registered_model(self, model_name: str) -> Optional[Type[BaseModel]]:
        """Look up a Pydantic model by name."""
        try:
            from app.models import get_model_by_name
            return get_model_by_name(model_name)
        except ImportError:
            logger.warning(f"Model registry not found. Cannot validate against {model_name}")
            return None
