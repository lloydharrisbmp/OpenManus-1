import asyncio
from typing import List, Dict, Any

from app.tool.base import BaseTool
from app.config import config
from app.tool.search import WebSearchEngine, BaiduSearchEngine, GoogleSearchEngine, DuckDuckGoSearchEngine


class WebSearchTool(BaseTool):
    """Tool for searching the web."""
    
    name: str = "web_search"
    description: str = "Search the web for information on a specific topic."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to the search engine.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    }
    _search_engine: dict[str, WebSearchEngine] = {
        "google": GoogleSearchEngine(),
        "baidu": BaiduSearchEngine(),
        "duckduckgo": DuckDuckGoSearchEngine(),
    }

    async def execute(self, query: str, num_results: int = 10) -> List[str]:
        """
        Execute a Web search and return a list of URLs.

        Args:
            query (str): The search query to submit to the search engine.
            num_results (int, optional): The number of search results to return. Default is 10.

        Returns:
            List[str]: A list of URLs matching the search query.
        """
        # Run the search in a thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        search_engine = self.get_search_engine()
        links = await loop.run_in_executor(
            None, lambda: list(search_engine.perform_search(query, num_results=num_results))
        )

        return links

    def get_search_engine(self) -> WebSearchEngine:
        """Determines the search engine to use based on the configuration."""
        default_engine = self._search_engine.get("google")
        if not hasattr(config, 'search_config') or config.search_config is None:
            return default_engine
        else:
            if not hasattr(config.search_config, 'engine'):
                return default_engine
            engine = config.search_config.engine.lower()
            return self._search_engine.get(engine, default_engine)
