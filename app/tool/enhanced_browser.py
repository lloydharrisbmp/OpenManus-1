"""
Enhanced browser control system with advanced features and intelligent automation.
"""
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
import logging
from pathlib import Path
import re
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService

logger = logging.getLogger(__name__)

@dataclass
class ElementInteraction:
    """Records an interaction with a page element."""
    element_id: str
    interaction_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PageAnalysis:
    """Stores analysis of a webpage."""
    url: str
    title: str
    main_content: str
    content_type: str
    interactive_elements: List[Dict[str, Any]]
    forms: List[Dict[str, Any]]
    navigation: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class EnhancedBrowserControl:
    """Enhanced browser control with intelligent automation and analysis."""
    
    def __init__(
        self,
        headless: bool = False,
        intelligent_wait: bool = True,
        auto_retry: bool = True,
        max_retries: int = 3,
        screenshot_dir: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.config = {
            "headless": headless,
            "intelligent_wait": intelligent_wait,
            "auto_retry": auto_retry,
            "max_retries": max_retries
        }
        
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize components
        self.browser: Optional[BrowserUseBrowser] = None
        self.context: Optional[BrowserContext] = None
        self.dom_service: Optional[DomService] = None
        
        # Analysis tools
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
        # State tracking
        self.interaction_history: List[ElementInteraction] = []
        self.page_analyses: Dict[str, PageAnalysis] = {}
        self.form_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.performance_metrics: Dict[str, List[float]] = {
            "page_load_times": [],
            "interaction_times": [],
            "analysis_times": []
        }
    
    async def initialize(self, proxy_settings: Optional[Dict[str, str]] = None) -> None:
        """Initialize the browser with optional proxy settings."""
        browser_config = {
            "headless": self.config["headless"]
        }
        
        if proxy_settings:
            from browser_use.browser.browser import ProxySettings
            browser_config["proxy"] = ProxySettings(**proxy_settings)
        
        self.browser = BrowserUseBrowser(BrowserConfig(**browser_config))
        self.context = await self.browser.new_context(
            config=BrowserContextConfig()
        )
        self.dom_service = DomService(self.context)
    
    async def navigate(self, url: str, wait_for_network: bool = True) -> bool:
        """Navigate to a URL with intelligent waiting."""
        try:
            start_time = datetime.now()
            
            # Perform navigation
            await self.context.navigate_to(url)
            
            if wait_for_network:
                await self._wait_for_network_idle()
            
            # Record performance metric
            load_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics["page_load_times"].append(load_time)
            
            # Analyze page
            await self._analyze_current_page()
            
            return True
            
        except Exception as e:
            logger.error(f"Navigation failed: {str(e)}")
            return False
    
    async def interact(
        self,
        element_index: int,
        action: str,
        value: Optional[str] = None
    ) -> bool:
        """Interact with a page element intelligently."""
        try:
            start_time = datetime.now()
            
            # Get element
            element = await self.context.get_dom_element_by_index(element_index)
            if not element:
                raise ValueError(f"Element with index {element_index} not found")
            
            # Perform interaction
            if action == "click":
                await self._intelligent_click(element)
            elif action == "input":
                if not value:
                    raise ValueError("Value required for input action")
                await self._intelligent_input(element, value)
            else:
                raise ValueError(f"Unknown action: {action}")
            
            # Record interaction
            interaction_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics["interaction_times"].append(interaction_time)
            
            self.interaction_history.append(ElementInteraction(
                element_id=str(element_index),
                interaction_type=action,
                context={"value": value} if value else {}
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Interaction failed: {str(e)}")
            self.interaction_history.append(ElementInteraction(
                element_id=str(element_index),
                interaction_type=action,
                success=False,
                error=str(e)
            ))
            return False
    
    async def analyze_content(
        self,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze page content with specific focus."""
        try:
            start_time = datetime.now()
            
            # Get page content
            html = await self.context.get_page_html()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Analyze based on content type
            analysis = {
                "text_content": main_content,
                "word_count": len(main_content.split()),
                "interactive_elements": await self._find_interactive_elements(),
                "forms": await self._analyze_forms(),
                "links": self._analyze_links(soup),
                "readability_score": self._calculate_readability(main_content)
            }
            
            if content_type == "article":
                analysis.update(await self._analyze_article(soup))
            elif content_type == "form":
                analysis.update(await self._analyze_form_details())
            
            # Record performance
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics["analysis_times"].append(analysis_time)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return {"error": str(e)}
    
    async def smart_form_fill(
        self,
        form_data: Dict[str, str],
        strategy: str = "auto"
    ) -> bool:
        """Intelligently fill form fields."""
        try:
            # Analyze form fields
            form_analysis = await self._analyze_forms()
            
            for field in form_analysis:
                field_value = form_data.get(field["name"])
                if not field_value:
                    continue
                
                # Smart field matching
                if strategy == "auto":
                    matched_field = await self._smart_field_match(field, field_value)
                    if matched_field:
                        await self.interact(
                            matched_field["index"],
                            "input",
                            field_value
                        )
                
            return True
            
        except Exception as e:
            logger.error(f"Form fill failed: {str(e)}")
            return False
    
    async def take_smart_screenshot(
        self,
        element_index: Optional[int] = None,
        full_page: bool = False
    ) -> Optional[str]:
        """Take intelligent screenshots."""
        try:
            if element_index is not None:
                element = await self.context.get_dom_element_by_index(element_index)
                if not element:
                    raise ValueError(f"Element {element_index} not found")
                screenshot = await self.context.take_element_screenshot(element)
            else:
                screenshot = await self.context.take_screenshot(full_page=full_page)
            
            if self.screenshot_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                filepath = self.screenshot_dir / filename
                
                # Save screenshot
                with open(filepath, "wb") as f:
                    f.write(screenshot)
                
                return str(filepath)
            
            return screenshot
            
        except Exception as e:
            logger.error(f"Screenshot failed: {str(e)}")
            return None
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.context:
                await self.context.close()
                self.context = None
                self.dom_service = None
            
            if self.browser:
                await self.browser.close()
                self.browser = None
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    async def _wait_for_network_idle(self, timeout: float = 5.0) -> None:
        """Wait for network activity to settle."""
        try:
            await asyncio.sleep(0.5)  # Initial wait
            
            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < timeout:
                # Check network activity
                if await self._is_network_idle():
                    return
                await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"Network idle check failed: {str(e)}")
    
    async def _is_network_idle(self) -> bool:
        """Check if network is idle."""
        try:
            # This is a placeholder - implement actual network activity checking
            return True
        except Exception:
            return True
    
    async def _intelligent_click(self, element: Any) -> None:
        """Perform intelligent click with retry."""
        max_retries = self.config["max_retries"] if self.config["auto_retry"] else 1
        
        for attempt in range(max_retries):
            try:
                # Ensure element is clickable
                await self._ensure_element_clickable(element)
                
                # Perform click
                await self.context._click_element_node(element)
                return
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(0.5 * (attempt + 1))
    
    async def _intelligent_input(self, element: Any, value: str) -> None:
        """Perform intelligent input with retry."""
        max_retries = self.config["max_retries"] if self.config["auto_retry"] else 1
        
        for attempt in range(max_retries):
            try:
                # Ensure element is interactable
                await self._ensure_element_clickable(element)
                
                # Clear existing value if any
                await self.context.execute_javascript(
                    f"arguments[0].value = '';",
                    element
                )
                
                # Input new value
                await self.context._input_text_element_node(element, value)
                return
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(0.5 * (attempt + 1))
    
    async def _ensure_element_clickable(self, element: Any) -> None:
        """Ensure element is clickable."""
        try:
            # Scroll element into view
            await self.context.execute_javascript(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                element
            )
            
            # Wait for any animations to complete
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.warning(f"Element preparation failed: {str(e)}")
    
    async def _analyze_current_page(self) -> None:
        """Analyze current page and cache results."""
        try:
            url = await self.context.get_current_url()
            title = await self.context.execute_javascript("document.title")
            
            # Get page content
            html = await self.context.get_page_html()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Determine content type
            content_type = self._determine_content_type(soup)
            
            # Analyze interactive elements
            interactive_elements = await self._find_interactive_elements()
            
            # Analyze forms
            forms = await self._analyze_forms()
            
            # Analyze navigation
            navigation = self._analyze_links(soup)
            
            # Extract metadata
            metadata = self._extract_metadata(soup)
            
            # Create analysis
            analysis = PageAnalysis(
                url=url,
                title=title,
                main_content=main_content,
                content_type=content_type,
                interactive_elements=interactive_elements,
                forms=forms,
                navigation=navigation,
                metadata=metadata
            )
            
            # Cache analysis
            self.page_analyses[url] = analysis
            
        except Exception as e:
            logger.error(f"Page analysis failed: {str(e)}")
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Find main content
        main = soup.find('main') or soup.find(id='main') or soup.find(role='main')
        if main:
            return main.get_text(strip=True)
        
        # Fallback to article or largest text block
        article = soup.find('article')
        if article:
            return article.get_text(strip=True)
        
        return soup.get_text(strip=True)
    
    def _determine_content_type(self, soup: BeautifulSoup) -> str:
        """Determine the type of content on the page."""
        if soup.find('form'):
            return "form"
        elif soup.find('article') or soup.find(class_=re.compile(r'article|post|blog')):
            return "article"
        elif soup.find(class_=re.compile(r'product|item')):
            return "product"
        elif soup.find(class_=re.compile(r'search-results')):
            return "search_results"
        return "general"
    
    async def _find_interactive_elements(self) -> List[Dict[str, Any]]:
        """Find and analyze interactive elements."""
        elements = []
        
        try:
            # Get clickable elements
            clickable = await self.context.get_clickable_elements()
            
            for idx, element in enumerate(clickable):
                element_info = {
                    "index": idx,
                    "tag": element.get("tagName", "").lower(),
                    "type": element.get("type", ""),
                    "text": element.get("textContent", "").strip(),
                    "attributes": element
                }
                elements.append(element_info)
            
        except Exception as e:
            logger.error(f"Interactive element analysis failed: {str(e)}")
        
        return elements
    
    async def _analyze_forms(self) -> List[Dict[str, Any]]:
        """Analyze forms on the page."""
        forms = []
        
        try:
            # Find all forms
            form_elements = await self.context.execute_javascript("""
                Array.from(document.forms).map(form => ({
                    id: form.id,
                    name: form.name,
                    method: form.method,
                    action: form.action,
                    fields: Array.from(form.elements).map(field => ({
                        name: field.name,
                        type: field.type,
                        required: field.required,
                        value: field.value,
                        placeholder: field.placeholder
                    }))
                }))
            """)
            
            if isinstance(form_elements, list):
                forms.extend(form_elements)
            
        except Exception as e:
            logger.error(f"Form analysis failed: {str(e)}")
        
        return forms
    
    def _analyze_links(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Analyze navigation and links."""
        links = []
        
        for link in soup.find_all('a', href=True):
            link_info = {
                "text": link.get_text(strip=True),
                "href": link["href"],
                "title": link.get("title", ""),
                "rel": link.get("rel", []),
                "is_navigation": bool(link.find_parent(['nav', 'header', 'footer']))
            }
            links.append(link_info)
        
        return links
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract page metadata."""
        metadata = {}
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            if name:
                metadata[name] = meta.get('content', '')
        
        # Extract schema.org data
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                metadata['schema_org'] = data
            except (json.JSONDecodeError, AttributeError):
                pass
        
        return metadata
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score."""
        try:
            words = text.split()
            sentences = text.split('.')
            
            if not sentences or not words:
                return 0.0
            
            # Simple Flesch Reading Ease score
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            return 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            
        except Exception:
            return 0.0
    
    async def _smart_field_match(
        self,
        field: Dict[str, Any],
        value: str
    ) -> Optional[Dict[str, Any]]:
        """Smart matching of form fields."""
        try:
            # Get all input elements
            elements = await self._find_interactive_elements()
            
            # Filter to likely matches
            candidates = [
                e for e in elements
                if e["tag"] in ["input", "textarea", "select"] and
                (
                    e["attributes"].get("name") == field["name"] or
                    e["attributes"].get("id") == field["name"] or
                    e["attributes"].get("placeholder", "").lower() == field["name"].lower()
                )
            ]
            
            if candidates:
                return candidates[0]
            
        except Exception as e:
            logger.error(f"Field matching failed: {str(e)}")
        
        return None
    
    async def _analyze_article(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze article content."""
        analysis = {
            "author": "",
            "date_published": "",
            "categories": [],
            "tags": []
        }
        
        try:
            # Find author
            author_elem = soup.find(class_=re.compile(r'author|byline'))
            if author_elem:
                analysis["author"] = author_elem.get_text(strip=True)
            
            # Find publication date
            date_elem = soup.find(class_=re.compile(r'date|published|time'))
            if date_elem:
                analysis["date_published"] = date_elem.get_text(strip=True)
            
            # Find categories and tags
            for elem in soup.find_all(class_=re.compile(r'category|tag')):
                text = elem.get_text(strip=True)
                if 'category' in elem.get('class', []):
                    analysis["categories"].append(text)
                else:
                    analysis["tags"].append(text)
            
        except Exception as e:
            logger.error(f"Article analysis failed: {str(e)}")
        
        return analysis
    
    async def _analyze_form_details(self) -> Dict[str, Any]:
        """Analyze form details."""
        analysis = {
            "fields": [],
            "required_fields": [],
            "validation_rules": {},
            "submission_url": "",
            "method": ""
        }
        
        try:
            forms = await self._analyze_forms()
            if not forms:
                return analysis
            
            # Analyze first form
            form = forms[0]
            analysis["submission_url"] = form.get("action", "")
            analysis["method"] = form.get("method", "").upper()
            
            # Analyze fields
            for field in form.get("fields", []):
                analysis["fields"].append(field["name"])
                if field.get("required"):
                    analysis["required_fields"].append(field["name"])
                
                # Extract validation rules
                validation = {}
                if field.get("pattern"):
                    validation["pattern"] = field["pattern"]
                if field.get("minlength"):
                    validation["min_length"] = field["minlength"]
                if field.get("maxlength"):
                    validation["max_length"] = field["maxlength"]
                
                if validation:
                    analysis["validation_rules"][field["name"]] = validation
            
        except Exception as e:
            logger.error(f"Form detail analysis failed: {str(e)}")
        
        return analysis 