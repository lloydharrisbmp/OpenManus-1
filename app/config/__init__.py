from pathlib import Path
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
import tomli

class LLMSettings(BaseModel):
    """Settings for the LLM configuration."""
    model: str = Field(default="gpt-4-turbo-preview", description="The model to use for LLM calls")
    temperature: float = Field(default=0.0, description="Temperature for response generation")
    max_tokens: Optional[int] = Field(default=4096, description="Maximum tokens in response")
    api_key: Optional[str] = Field(default=None, description="API key for the model provider")
    organization: Optional[str] = Field(default=None, description="OpenAI organization ID")
    api_type: str = Field(default="openai", description="API type (openai, azure, gemini, or groq)")
    api_version: Optional[str] = Field(default=None, description="API version for Azure")
    base_url: Optional[str] = Field(default="https://api.openai.com/v1", description="Base URL for API")
    model_type: Optional[str] = Field(default=None, description="Model type (pro, flash-thinking, flash-image)")

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

# Load configuration from config.toml if it exists
config_file = Path("config/config.toml")
if config_file.exists():
    with open(config_file, "rb") as f:
        toml_config = tomli.load(f)
        if "llm" in toml_config:
            # Update the default LLM settings with the main [llm] section
            main_llm_config = {k: v for k, v in toml_config["llm"].items() 
                             if not isinstance(v, dict)}
            config.llm["default"] = LLMSettings(**main_llm_config)
            
            # Add any additional LLM configurations from subsections
            for key, value in toml_config["llm"].items():
                if isinstance(value, dict):
                    config.llm[key] = LLMSettings(**value)
