import asyncio
import logging
import aiohttp
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from googlesearch import search
from app.tool.base import BaseTool

logger = logging.getLogger(__name__)

class GoogleSearchConfig(BaseModel):
    """Configuration for advanced Google searches."""
    query: str = Field(..., description="The search query to submit to Google.")
    num_results: int = Field(10, description="Number of search results to return.")
    lang: Optional[str] = Field(None, description="Language code for the results (e.g. 'en').")
    region: Optional[str] = Field(None, description="Region code for refining the search (e.g. 'us').")
    tbs: Optional[str] = Field(None, description="Time-based search options (e.g. 'qdr:d' for past day).")
    safe: Optional[str] = Field(None, description="Safe search level. Use 'off', 'active', 'strict'.")
    include_sites: List[str] = Field(default_factory=list, description="Restrict search results to these domains.")
    exclude_sites: List[str] = Field(default_factory=list, description="Exclude these domains from the results.")
    fetch_titles: bool = Field(False, description="If true, perform an HTTP request for each link to extract the <title> tag.")
    fetch_snippets: bool = Field(False, description="If true, fetch a short snippet from each page body.")
    concurrency_limit: int = Field(3, description="Maximum number of concurrent HTTP fetches for titles/snippets.")
    timeout: float = Field(10.0, description="HTTP request timeout in seconds for fetching page titles/snippets.")

class GoogleSearch(BaseTool):
    """
    Enhanced Google search tool with advanced parameters and metadata fetching capabilities.
    """
    name: str = "google_search"
    description: str = (
        "Perform a Google search and return a list of relevant links, with advanced options "
        "like language, region, time range, site filtering, and optional page title/snippet fetching."
    )

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "(required) The search query to submit to Google."},
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10
            },
            "lang": {"type": "string", "description": "Language code for the results (e.g. 'en')."},
            "region": {"type": "string", "description": "Region code for refining the search (e.g. 'us')."},
            "tbs": {"type": "string", "description": "Time-based search options (e.g. 'qdr:d' for past day)."},
            "safe": {"type": "string", "description": "Safe search level. Use 'off', 'active', 'strict'."},
            "include_sites": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Restrict search results to these domains."
            },
            "exclude_sites": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exclude these domains from the results."
            },
            "fetch_titles": {
                "type": "boolean",
                "description": "If true, fetch the page title for each link."
            },
            "fetch_snippets": {
                "type": "boolean",
                "description": "If true, fetch a short snippet from each page."
            },
            "concurrency_limit": {
                "type": "integer",
                "description": "Max number of concurrent HTTP fetches for titles/snippets."
            },
            "timeout": {
                "type": "number",
                "description": "HTTP request timeout in seconds for fetching page titles/snippets."
            }
        },
        "required": ["query"]
    }

    async def execute(
        self,
        query: str,
        num_results: int = 10,
        lang: Optional[str] = None,
        region: Optional[str] = None,
        tbs: Optional[str] = None,
        safe: Optional[str] = None,
        include_sites: Optional[List[str]] = None,
        exclude_sites: Optional[List[str]] = None,
        fetch_titles: bool = False,
        fetch_snippets: bool = False,
        concurrency_limit: int = 3,
        timeout: float = 10.0
    ) -> Union[List[Dict[str, str]], Dict[str, Any]]:
        """Execute a Google search with advanced parameters."""
        try:
            cfg = GoogleSearchConfig(
                query=query, 
                num_results=num_results, 
                lang=lang,
                region=region,
                tbs=tbs,
                safe=safe,
                include_sites=include_sites or [],
                exclude_sites=exclude_sites or [],
                fetch_titles=fetch_titles,
                fetch_snippets=fetch_snippets,
                concurrency_limit=concurrency_limit,
                timeout=timeout
            )
            logger.info(f"Executing GoogleSearch with config: {cfg}")

        except Exception as e:
            logger.error(f"Invalid parameters for GoogleSearch: {e}")
            return {"error": str(e), "success": False}

        search_query = self._build_advanced_query(
            cfg.query, cfg.include_sites, cfg.exclude_sites
        )

        loop = asyncio.get_event_loop()
        try:
            raw_links = await loop.run_in_executor(
                None,
                lambda: list(
                    search(
                        search_query,
                        tld=(cfg.region if cfg.region else "com"),
                        lang=cfg.lang or "en",
                        tbs=cfg.tbs,
                        safe=cfg.safe if cfg.safe else "off",
                        num_results=cfg.num_results
                    )
                )
            )
        except Exception as e:
            logger.error(f"Google search failed: {e}", exc_info=True)
            return {"error": str(e), "success": False}

        results = [{"link": link} for link in raw_links]

        if cfg.fetch_titles or cfg.fetch_snippets:
            sem = asyncio.Semaphore(cfg.concurrency_limit)
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._fetch_page_metadata(
                        session,
                        item["link"],
                        fetch_title=cfg.fetch_titles,
                        fetch_snippet=cfg.fetch_snippets,
                        sem=sem,
                        timeout=cfg.timeout
                    )
                    for item in results
                ]
                enriched_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for idx, enriched in enumerate(enriched_results):
                if isinstance(enriched, dict):
                    results[idx].update(enriched)
                else:
                    logger.warning(
                        f"Failed to fetch metadata for link {results[idx]['link']}: {enriched}"
                    )
                    results[idx]["title"] = ""
                    results[idx]["snippet"] = ""
        
        return results

    def _build_advanced_query(
        self, 
        query: str, 
        include_sites: List[str], 
        exclude_sites: List[str]
    ) -> str:
        """Build an advanced query string with site filters."""
        if include_sites:
            site_includes = " OR ".join([f"site:{site}" for site in include_sites])
            query = f"({query}) ({site_includes})"
        for site in exclude_sites:
            query += f" -site:{site}"
        return query

    async def _fetch_page_metadata(
        self,
        session: aiohttp.ClientSession,
        url: str,
        fetch_title: bool,
        fetch_snippet: bool,
        sem: asyncio.Semaphore,
        timeout: float
    ) -> Dict[str, str]:
        """Fetch page title/snippet from a given URL."""
        result = {}
        async with sem:
            try:
                async with session.get(url, timeout=timeout, allow_redirects=True) as resp:
                    if resp.status != 200:
                        logger.warning(f"Non-200 status {resp.status} for {url}")
                        return {"title": "", "snippet": ""}
                    
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        return {"title": "", "snippet": ""}
                    
                    html = await resp.text()
                    if fetch_title:
                        result["title"] = self._extract_title(html)
                    if fetch_snippet:
                        result["snippet"] = self._extract_snippet(html)
            except Exception as e:
                logger.warning(f"Failed to retrieve metadata from {url}: {e}")
                return {"title": "", "snippet": ""}

        return result

    def _extract_title(self, html: str) -> str:
        """Extract the page title from HTML."""
        start_tag = "<title>"
        end_tag = "</title>"
        start_idx = html.lower().find(start_tag)
        if start_idx == -1:
            return ""
        end_idx = html.lower().find(end_tag, start_idx)
        if end_idx == -1:
            return ""
        start_idx += len(start_tag)
        return html[start_idx:end_idx].strip()

    def _extract_snippet(self, html: str, max_chars: int = 200) -> str:
        """Extract a short snippet from the page body."""
        import re
        html_no_scripts = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html_no_styles = re.sub(r"<style.*?>.*?</style>", "", html_no_scripts, flags=re.DOTALL | re.IGNORECASE)
        text_only = re.sub(r"<[^>]+>", " ", html_no_styles)
        text_only = re.sub(r"\s+", " ", text_only).strip()
        return text_only[:max_chars] + ("..." if len(text_only) > max_chars else "")
