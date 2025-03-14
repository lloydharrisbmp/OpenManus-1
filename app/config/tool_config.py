from pydantic import BaseSettings
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ToolConfig(BaseSettings):
    """Global tool configuration."""
    max_concurrent_tools: int = 5
    default_timeout: float = 30.0
    cache_results: bool = True
    cache_ttl: int = 3600  # 1 hour
    groq_api_key: str = ""
    log_level: str = "INFO"

    # Tool-specific configurations
    tool_configs: Dict[str, Dict[str, Any]] = {
        "google_search": {
            "concurrency_limit": 3,
            "timeout": 10.0,
            "max_results": 20
        },
        "market_data": {
            "cache_ttl": 1800,  # 30 minutes
            "retry_attempts": 3
        }
    }

    class Config:
        env_prefix = "TOOL_"

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tool."""
        return self.tool_configs.get(tool_name, {}) 