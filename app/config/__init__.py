from pathlib import Path
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

class LLMSettings(BaseModel):
    """Settings for the LLM configuration."""
    model: str = Field(default="gpt-4-turbo-preview", description="The model to use for LLM calls")
    temperature: float = Field(default=0.7, description="Temperature for response generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in response")
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    organization: Optional[str] = Field(default=None, description="OpenAI organization ID")
    api_type: str = Field(default="openai", description="API type (openai or azure)")
    api_version: Optional[str] = Field(default=None, description="API version for Azure")
    base_url: Optional[str] = Field(default=None, description="Base URL for API")

class ToolConfig(BaseModel):
    """Configuration for a tool."""
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters that the tool accepts")
    required: List[str] = Field(default_factory=list, description="List of required parameters")
    enabled: bool = Field(default=True, description="Whether the tool is enabled")
    async_enabled: bool = Field(default=True, description="Whether async execution is supported")

class Config(BaseModel):
    """Main configuration class."""
    llm: Dict[str, LLMSettings] = Field(
        default_factory=lambda: {
            "default": LLMSettings()
        }
    )
    workspace_dir: Path = Field(default=Path.cwd(), description="Workspace directory")
    tools: Dict[str, ToolConfig] = Field(default_factory=dict, description="Tool configurations")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key for the reasoner")

# Create default config instance
config = Config()
