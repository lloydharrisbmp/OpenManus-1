import asyncio
import json
import logging
import random
from typing import Optional, List, Dict, Any, Union

import aiohttp
from pydantic import BaseModel, Field, validator
from pydantic_core.core_schema import ValidationInfo

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService

from app.config import config
from app.tool.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)
MAX_LENGTH = 2000

# Model Definitions
class WaitCondition(BaseModel):
    """Configuration for various wait conditions."""
    wait_for_selector: Optional[str] = None
    wait_for_document_ready: bool = False
    wait_for_network_idle: bool = False
    wait_for_js_expression: Optional[str] = None
    timeout_seconds: int = 10

    def needs_waiting(self) -> bool:
        """Check if any wait property is set."""
        return bool(
            self.wait_for_selector or 
            self.wait_for_document_ready or 
            self.wait_for_network_idle or
            self.wait_for_js_expression
        )

class RetrySettings(BaseModel):
    """Configuration for action retry behavior."""
    attempts: int = 1
    delay_seconds: float = 1.0

class BrowserAction(BaseModel):
    """Definition of a single browser action."""
    action: str = Field(..., description="The browser action to perform")
    url: Optional[str] = None
    index: Optional[int] = None
    text: Optional[str] = None
    script: Optional[str] = None
    scroll_amount: Optional[int] = None
    tab_id: Optional[int] = None
    screenshot_mode: Optional[str] = None
    max_base64_length: int = 0
    selector_source: Optional[str] = Field(None, description="For drag_drop: source CSS selector")
    selector_target: Optional[str] = Field(None, description="For drag_drop: target CSS selector")
    file_paths: Optional[List[str]] = Field(None, description="File paths for file_upload action")
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None
    wait_condition: Optional[WaitCondition] = None
    retry_settings: Optional[RetrySettings] = None

    @validator("action")
    def validate_action(cls, val: str):
        supported_actions = {
            "navigate", "click", "input_text", "screenshot", "get_html",
            "get_text", "read_links", "execute_js", "scroll", "switch_tab",
            "new_tab", "close_tab", "refresh", "drag_drop", "file_upload",
            "set_viewport", "generate_pdf", "insert_cookies", "local_storage",
            "collect_performance_metrics"
        }
        if val not in supported_actions:
            raise ValueError(f"Action '{val}' not recognized. Supported: {supported_actions}")
        return val

class ActionSequenceConfig(BaseModel):
    """Configuration for a sequence of browser actions."""
    concurrency_limit: int = 1
    actions: List[BrowserAction] = Field(..., description="Actions to execute.")

class BrowserUseParameters(BaseModel):
    """Parameters for the BrowserUseTool."""
    ephemeral_session: bool = Field(False, description="Create fresh context for each call")
    random_user_agent: bool = Field(False, description="Use random user-agent string")
    persistent_session_id: Optional[str] = Field(None, description="ID for persistent context")
    
    action: Optional[str] = None
    url: Optional[str] = None
    index: Optional[int] = None
    text: Optional[str] = None
    script: Optional[str] = None
    scroll_amount: Optional[int] = None
    tab_id: Optional[int] = None
    screenshot_mode: Optional[str] = None
    max_base64_length: int = 0
    selector_source: Optional[str] = None
    selector_target: Optional[str] = None
    file_paths: Optional[List[str]] = None
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None
    wait_condition: Optional[WaitCondition] = None
    retry_settings: Optional[RetrySettings] = None
    action_sequence: Optional[ActionSequenceConfig] = None

    @validator("action", always=True)
    def single_action_or_sequence(cls, v, values, **kwargs):
        if v and values.get("action_sequence"):
            raise ValueError("Cannot provide both 'action' and 'action_sequence'.")
        if not v and not values.get("action_sequence"):
            raise ValueError("You must provide either 'action' or 'action_sequence'.")
        return v

    @validator("persistent_session_id")
    def check_persistent_session_compat(cls, v, values, **kwargs):
        ephemeral = values.get("ephemeral_session", False)
        if v and ephemeral:
            raise ValueError("Cannot use persistent_session_id with ephemeral_session=True.")
        return v

class BrowserUseTool(BaseTool):
    def __init__(self, **data):
        super().__init__(**data)
        self.ephemeral_browser = None
        self.ephemeral_context = None
        self._persistent_contexts = {}

    async def _get_or_create_context(self, params: BrowserUseParameters) -> BrowserContext:
        """
        Get or create a browser context based on session parameters.
        Handles both ephemeral and persistent sessions.
        """
        # Ephemeral session logic
        if params.ephemeral_session:
            if not self.ephemeral_browser or not self.ephemeral_context:
                self.ephemeral_browser, self.ephemeral_context = await self._create_new_browser_context(
                    random_ua=params.random_user_agent
                )
            return self.ephemeral_context

        # Persistent session logic
        if params.persistent_session_id:
            # Check if we already have a context
            if params.persistent_session_id in self._persistent_contexts:
                context = self._persistent_contexts[params.persistent_session_id]
                if context.is_closed():
                    # Recreate if closed
                    logger.info(f"Recreating closed persistent context '{params.persistent_session_id}'.")
                    context = await self._create_new_persistent_context(
                        params.persistent_session_id,
                        random_ua=params.random_user_agent
                    )
                return context
            else:
                # Create new persistent context
                context = await self._create_new_persistent_context(
                    params.persistent_session_id,
                    random_ua=params.random_user_agent
                )
                return context
        else:
            # No ephemeral, no persistent_session_id => use global context
            if "global" in self._persistent_contexts:
                ctx = self._persistent_contexts["global"]
                if ctx.is_closed():
                    ctx = await self._create_new_persistent_context(
                        "global",
                        random_ua=params.random_user_agent
                    )
            else:
                ctx = await self._create_new_persistent_context(
                    "global",
                    random_ua=params.random_user_agent
                )
            return ctx

    async def _create_new_browser_context(
        self,
        random_ua: bool
    ) -> tuple[BrowserUseBrowser, BrowserContext]:
        """Create a new browser and context for ephemeral sessions."""
        browser_config_kwargs = self._base_browser_config(random_ua)
        browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))
        context_config = self._context_config()
        ctx = await browser.new_context(context_config)
        
        # Set up event listeners if needed
        await self._setup_event_listeners(ctx)
        
        logger.info("[BrowserUseTool] Created ephemeral browser context.")
        return browser, ctx

    async def _create_new_persistent_context(
        self,
        session_id: str,
        random_ua: bool
    ) -> BrowserContext:
        """Create and store a new persistent context."""
        browser_config_kwargs = self._base_browser_config(random_ua)
        browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))
        context_config = self._context_config()
        ctx = await browser.new_context(context_config)
        
        # Set up event listeners
        await self._setup_event_listeners(ctx)
        
        self._persistent_contexts[session_id] = ctx
        logger.info(f"[BrowserUseTool] Created persistent browser context '{session_id}'.")
        return ctx

    def _base_browser_config(self, random_ua: bool) -> Dict[str, Any]:
        """Build base browser configuration."""
        config_kwargs = {"headless": True}  # Default to headless mode
        
        # Apply configuration from app config if available
        if config.browser_config:
            from browser_use.browser.browser import ProxySettings
            if config.browser_config.proxy and config.browser_config.proxy.server:
                config_kwargs["proxy"] = ProxySettings(
                    server=config.browser_config.proxy.server,
                    username=config.browser_config.proxy.username,
                    password=config.browser_config.proxy.password,
                )
            
            # Copy other relevant config attributes
            for attr in [
                "headless", "disable_security", "extra_chromium_args",
                "chrome_instance_path", "wss_url", "cdp_url"
            ]:
                value = getattr(config.browser_config, attr, None)
                if value is not None:
                    config_kwargs[attr] = value

        if random_ua:
            possible_uas = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
            ]
            config_kwargs["user_agent"] = random.choice(possible_uas)

        return config_kwargs

    def _context_config(self) -> BrowserContextConfig:
        """Build context configuration."""
        context_config = BrowserContextConfig()
        if config.browser_config and hasattr(config.browser_config, "new_context_config"):
            cconf = config.browser_config.new_context_config
            if cconf:
                context_config = cconf
        return context_config

    async def _setup_event_listeners(self, context: BrowserContext) -> None:
        """Set up event listeners for the browser context."""
        try:
            # Example: Listen for console messages
            await context.execute_javascript("""
                console.defaultLog = console.log.bind(console);
                console.log = function() {
                    // You might want to send this to your logging system
                    console.defaultLog.apply(console, arguments);
                };
            """)
            
            # Example: Track network requests
            await context.execute_javascript("""
                const observer = new PerformanceObserver((list) => {
                    list.getEntries().forEach((entry) => {
                        if (entry.entryType === 'resource') {
                            console.log('Resource loaded:', entry.name);
                        }
                    });
                });
                observer.observe({ entryTypes: ['resource'] });
            """)
            
        except Exception as e:
            logger.warning(f"Failed to set up event listeners: {e}")

    async def _cleanup_ephemeral(self):
        """Clean up ephemeral browser context and browser."""
        if self.ephemeral_context and not self.ephemeral_context.is_closed():
            await self.ephemeral_context.close()
        self.ephemeral_context = None
        
        if self.ephemeral_browser:
            await self.ephemeral_browser.close()
        self.ephemeral_browser = None
        
        logger.info("[BrowserUseTool] Cleaned up ephemeral session.")

    async def execute(self, params: BrowserUseParameters) -> ToolResult:
        # Implementation of execute method
        pass

    async def cleanup(self):
        # Implementation of cleanup method
        pass

    async def _perform_action(
        self, 
        context: BrowserContext, 
        action: BrowserAction, 
        sem: Optional[asyncio.Semaphore] = None
    ) -> ToolResult:
        """Execute a single browser action with retry and wait logic."""
        if sem is None:
            sem = asyncio.Semaphore(1)

        async with sem:
            # Handle wait conditions first
            if action.wait_condition and action.wait_condition.needs_waiting():
                if not await self._handle_wait_condition(context, action.wait_condition):
                    return ToolResult(error=f"Wait condition failed: {action.wait_condition}")

            # Perform retries if configured
            attempts = action.retry_settings.attempts if action.retry_settings else 1
            delay = action.retry_settings.delay_seconds if action.retry_settings else 1.0

            for attempt_num in range(attempts):
                try:
                    return await self._execute_action(context, action)
                except Exception as e:
                    if attempt_num < attempts - 1:
                        logger.warning(
                            f"[BrowserUseTool] Retry {attempt_num+1}/{attempts} "
                            f"for action '{action.action}' - Error: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"[BrowserUseTool] Action '{action.action}' final failure: {e}")
                        return ToolResult(error=f"Action '{action.action}' failed: {str(e)}")

    async def _handle_wait_condition(self, context: BrowserContext, wait: WaitCondition) -> bool:
        """Handle various wait conditions before executing an action."""
        try:
            if wait.wait_for_selector:
                await context.wait_for_selector(
                    wait.wait_for_selector,
                    timeout_seconds=wait.timeout_seconds
                )
            
            if wait.wait_for_document_ready:
                await context.execute_javascript("""
                    await new Promise(resolve => {
                        if (document.readyState === 'complete') {
                            resolve();
                        } else {
                            document.addEventListener('DOMContentLoaded', resolve);
                        }
                    });
                """, timeout=wait.timeout_seconds)
            
            if wait.wait_for_network_idle:
                await context.execute_javascript("""
                    await new Promise(resolve => {
                        let timeout;
                        const observer = new PerformanceObserver(() => {
                            if (timeout) clearTimeout(timeout);
                            timeout = setTimeout(resolve, 500); // Wait for 500ms of network idle
                        });
                        observer.observe({ entryTypes: ['resource'] });
                        timeout = setTimeout(resolve, 500); // Initial timeout
                    });
                """, timeout=wait.timeout_seconds)
            
            if wait.wait_for_js_expression:
                await context.wait_for_js_expression(
                    wait.wait_for_js_expression,
                    wait.timeout_seconds
                )
            
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Wait condition timed out after {wait.timeout_seconds} seconds")
            return False
        except Exception as e:
            logger.error(f"Wait condition failed: {e}")
            return False

    async def _execute_action(self, context: BrowserContext, action: BrowserAction) -> ToolResult:
        """Execute a specific browser action."""
        match action.action:
            case "navigate":
                if not action.url:
                    return ToolResult(error="URL is required for 'navigate'")
                await context.navigate_to(action.url)
                return ToolResult(output=f"Navigated to {action.url}")

            case "click":
                if action.index is None:
                    return ToolResult(error="Index is required for 'click'")
                element = await context.get_dom_element_by_index(action.index)
                if not element:
                    return ToolResult(error=f"No element at index {action.index}")
                download_path = await context._click_element_node(element)
                out = f"Clicked element at index {action.index}"
                if download_path:
                    out += f" - Downloaded file: {download_path}"
                return ToolResult(output=out)

            case "input_text":
                if action.index is None or action.text is None:
                    return ToolResult(error="Index and text are required for 'input_text'")
                el = await context.get_dom_element_by_index(action.index)
                if not el:
                    return ToolResult(error=f"No element at index {action.index}")
                await context._input_text_element_node(el, action.text)
                return ToolResult(output=f"Typed '{action.text}' into index {action.index}")

            case "drag_drop":
                if not action.selector_source or not action.selector_target:
                    return ToolResult(error="selector_source and selector_target required for 'drag_drop'")
                await context.execute_javascript(f"""
                    const source = document.querySelector('{action.selector_source}');
                    const target = document.querySelector('{action.selector_target}');
                    if (!source || !target) throw new Error('Source or target element not found');
                    
                    // Simulate drag and drop
                    const dragStart = new DragEvent('dragstart');
                    const drop = new DragEvent('drop');
                    source.dispatchEvent(dragStart);
                    target.dispatchEvent(drop);
                """)
                return ToolResult(output=f"Dragged from '{action.selector_source}' to '{action.selector_target}'")

            case "file_upload":
                if action.index is None or not action.file_paths:
                    return ToolResult(error="index and file_paths required for 'file_upload'")
                element = await context.get_dom_element_by_index(action.index)
                if not element:
                    return ToolResult(error=f"No element at index {action.index}")
                await context.upload_files_to_element(element, action.file_paths)
                return ToolResult(output=f"Uploaded files: {', '.join(action.file_paths)}")

            case "set_viewport":
                w = action.viewport_width or 1280
                h = action.viewport_height or 720
                await context.set_viewport_size(width=w, height=h)
                return ToolResult(output=f"Viewport set to {w}x{h}")

            case "generate_pdf":
                pdf_data = await context.generate_pdf()
                encoded = pdf_data.encode("base64") if hasattr(pdf_data, "encode") else "<binary pdf>"
                if action.max_base64_length and len(encoded) > action.max_base64_length:
                    encoded = encoded[:action.max_base64_length] + "..."
                return ToolResult(output="Generated PDF from current page", system=encoded)

            case "insert_cookies":
                cookies = json.loads(action.text or "[]")
                await context.insert_cookies(cookies)
                return ToolResult(output=f"Inserted {len(cookies)} cookies")

            case "local_storage":
                items = json.loads(action.text or "{}")
                for k, v in items.items():
                    await context.execute_javascript(
                        f"localStorage.setItem('{k}', '{v}');"
                    )
                return ToolResult(output=f"Inserted {len(items)} items into localStorage")

            case "collect_performance_metrics":
                metrics = await context.execute_javascript("""
                    const nav = window.performance;
                    const timing = nav.timing;
                    return {
                        navigationStart: timing.navigationStart,
                        loadEventEnd: timing.loadEventEnd,
                        domComplete: timing.domComplete,
                        resources: performance.getEntriesByType('resource').map(r => ({
                            name: r.name,
                            duration: r.duration,
                            type: r.initiatorType
                        }))
                    };
                """)
                return ToolResult(output=json.dumps(metrics))

            case "screenshot":
                mode = action.screenshot_mode or "full_page"
                try:
                    screenshot_data = await self._take_screenshot(context, mode)
                    if action.max_base64_length > 0 and len(screenshot_data) > action.max_base64_length:
                        screenshot_data = screenshot_data[:action.max_base64_length] + "..."
                    return ToolResult(
                        output=f"Screenshot captured ({mode} mode)",
                        system=screenshot_data
                    )
                except Exception as e:
                    return ToolResult(error=f"Screenshot failed: {str(e)}")

            case "get_html":
                try:
                    html = await context.get_page_html()
                    truncated = html[:MAX_LENGTH] + "..." if len(html) > MAX_LENGTH else html
                    return ToolResult(output=truncated)
                except Exception as e:
                    return ToolResult(error=f"Failed to get HTML: {str(e)}")

            case "get_text":
                try:
                    body_text = await context.execute_javascript("""
                        function getVisibleText(node) {
                            if (node.nodeType === Node.TEXT_NODE) {
                                return node.textContent.trim();
                            }
                            if (node.nodeType !== Node.ELEMENT_NODE) {
                                return '';
                            }
                            const style = window.getComputedStyle(node);
                            if (style.display === 'none' || style.visibility === 'hidden') {
                                return '';
                            }
                            let text = '';
                            for (let child of node.childNodes) {
                                text += getVisibleText(child) + ' ';
                            }
                            return text.trim();
                        }
                        return getVisibleText(document.body);
                    """)
                    truncated = body_text[:MAX_LENGTH] + "..." if len(body_text) > MAX_LENGTH else body_text
                    return ToolResult(output=truncated)
                except Exception as e:
                    return ToolResult(error=f"Failed to get text: {str(e)}")

            case "read_links":
                try:
                    links = await context.execute_javascript("""
                        const links = Array.from(document.querySelectorAll('a[href]'))
                            .filter(a => {
                                const rect = a.getBoundingClientRect();
                                return rect.width > 0 && rect.height > 0;
                            })
                            .map(a => ({
                                text: a.innerText.trim(),
                                href: a.href,
                                visible: true,
                                location: {
                                    x: a.getBoundingClientRect().left,
                                    y: a.getBoundingClientRect().top
                                }
                            }));
                        return links;
                    """)
                    return ToolResult(output=json.dumps(links, ensure_ascii=False))
                except Exception as e:
                    return ToolResult(error=f"Failed to read links: {str(e)}")

            case "execute_js":
                if not action.script:
                    return ToolResult(error="script required for 'execute_js'")
                try:
                    result = await context.execute_javascript(action.script)
                    return ToolResult(output=str(result))
                except Exception as e:
                    return ToolResult(error=f"JavaScript execution failed: {str(e)}")

            case "scroll":
                if action.scroll_amount is None:
                    return ToolResult(error="scroll_amount required for 'scroll'")
                try:
                    await context.execute_javascript(f"""
                        window.scrollBy({{
                            top: {action.scroll_amount},
                            behavior: 'smooth'
                        }});
                    """)
                    direction = "down" if action.scroll_amount > 0 else "up"
                    return ToolResult(
                        output=f"Scrolled {direction} by {abs(action.scroll_amount)} pixels"
                    )
                except Exception as e:
                    return ToolResult(error=f"Scroll failed: {str(e)}")

            case "new_tab":
                if not action.url:
                    return ToolResult(error="URL required for 'new_tab'")
                try:
                    await context.create_new_tab(action.url)
                    return ToolResult(output=f"Opened new tab to {action.url}")
                except Exception as e:
                    return ToolResult(error=f"Failed to open new tab: {str(e)}")

            case "switch_tab":
                if action.tab_id is None:
                    return ToolResult(error="tab_id required for 'switch_tab'")
                try:
                    await context.switch_to_tab(action.tab_id)
                    return ToolResult(output=f"Switched to tab {action.tab_id}")
                except Exception as e:
                    return ToolResult(error=f"Failed to switch tab: {str(e)}")

            case "close_tab":
                try:
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")
                except Exception as e:
                    return ToolResult(error=f"Failed to close tab: {str(e)}")

            case "refresh":
                try:
                    await context.refresh_page()
                    return ToolResult(output="Refreshed page")
                except Exception as e:
                    return ToolResult(error=f"Failed to refresh page: {str(e)}")

            case _:
                return ToolResult(error=f"Unknown action: {action.action}")

    async def _take_screenshot(self, context: BrowserContext, mode: str) -> str:
        """Take a screenshot in the specified mode."""
        try:
            match mode:
                case "viewport":
                    return await context.take_screenshot(full_page=False)
                case "element":
                    return await context.take_screenshot(selector="body")
                case _:
                    return await context.take_screenshot(full_page=True)
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            raise

    def _summarize_results(self, results: List[Any]) -> tuple[str, str]:
        """Combine multiple step results into one output."""
        all_outputs = []
        all_errors = []
        
        for r in results:
            if isinstance(r, Exception):
                e_str = f"Unhandled exception: {r}"
                all_errors.append(e_str)
            elif isinstance(r, ToolResult):
                if r.error:
                    all_errors.append(r.error)
                if r.output:
                    all_outputs.append(r.output)
            else:
                all_outputs.append(str(r))
        
        final_output = "\n".join(all_outputs)
        final_error = "; ".join(all_errors) if all_errors else ""
        return final_output, final_error

    async def get_current_state(self) -> ToolResult:
        """Get the current browser state."""
        async with self.lock:
            ctx = self.ephemeral_context or self._persistent_contexts.get("global")
            if not ctx or ctx.is_closed():
                return ToolResult(error="No active browser context.")
            
            try:
                state = await ctx.get_state()
                data = {
                    "url": state.url,
                    "title": state.title,
                    "tabs": [tab.model_dump() for tab in state.tabs],
                    "interactive_elements": state.element_tree.clickable_elements_to_string(),
                }
                return ToolResult(output=json.dumps(data))
            except Exception as e:
                logger.error(f"Failed to get browser state: {e}", exc_info=True)
                return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up all browser contexts."""
        async with self.lock:
            # Clean up ephemeral session
            if self.ephemeral_context and not self.ephemeral_context.is_closed():
                await self.ephemeral_context.close()
            if self.ephemeral_browser:
                await self.ephemeral_browser.close()
            self.ephemeral_context = None
            self.ephemeral_browser = None

            # Clean up persistent sessions
            for sid, ctx in list(self._persistent_contexts.items()):
                if not ctx.is_closed():
                    await ctx.close()
            self._persistent_contexts.clear()

        logger.info("[BrowserUseTool] Cleanup complete.")

    def __del__(self):
        """Attempt final cleanup on object destruction."""
        try:
            if self.ephemeral_context or self._persistent_contexts:
                asyncio.run(self.cleanup())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.cleanup())
            loop.close() 