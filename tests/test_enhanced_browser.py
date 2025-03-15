"""
Tests for the enhanced browser control system.
"""
import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from bs4 import BeautifulSoup

from app.tool.enhanced_browser import (
    EnhancedBrowserControl,
    ElementInteraction,
    PageAnalysis
)

@pytest.fixture
def mock_browser():
    """Create a mock browser instance."""
    browser = AsyncMock()
    browser.close = AsyncMock()
    return browser

@pytest.fixture
def mock_context():
    """Create a mock browser context."""
    context = AsyncMock()
    context.close = AsyncMock()
    context.get_page_html = AsyncMock(return_value="<html><body><main>Test content</main></body></html>")
    context.get_current_url = AsyncMock(return_value="https://example.com")
    context.execute_javascript = AsyncMock(return_value="Test Title")
    context.get_clickable_elements = AsyncMock(return_value=[
        {"tagName": "button", "type": "submit", "textContent": "Submit"}
    ])
    return context

@pytest.fixture
def mock_dom_service():
    """Create a mock DOM service."""
    service = AsyncMock()
    return service

@pytest.fixture
async def browser_control(mock_browser, mock_context, mock_dom_service):
    """Create an EnhancedBrowserControl instance with mocked dependencies."""
    with patch("app.tool.enhanced_browser.BrowserUseBrowser", return_value=mock_browser), \
         patch("app.tool.enhanced_browser.BrowserContext", return_value=mock_context), \
         patch("app.tool.enhanced_browser.DomService", return_value=mock_dom_service):
        
        control = EnhancedBrowserControl(headless=True)
        await control.initialize()
        yield control
        await control.cleanup()

@pytest.mark.asyncio
async def test_initialization(browser_control):
    """Test browser initialization."""
    assert browser_control.browser is not None
    assert browser_control.context is not None
    assert browser_control.dom_service is not None
    assert browser_control.config["headless"] is True

@pytest.mark.asyncio
async def test_navigation(browser_control, mock_context):
    """Test page navigation."""
    # Setup mock
    mock_context.navigate_to = AsyncMock()
    
    # Test navigation
    result = await browser_control.navigate("https://example.com")
    
    assert result is True
    mock_context.navigate_to.assert_called_once_with("https://example.com")
    assert len(browser_control.performance_metrics["page_load_times"]) == 1

@pytest.mark.asyncio
async def test_interaction(browser_control, mock_context):
    """Test element interaction."""
    # Setup mock
    mock_element = AsyncMock()
    mock_context.get_dom_element_by_index = AsyncMock(return_value=mock_element)
    
    # Test click interaction
    result = await browser_control.interact(0, "click")
    
    assert result is True
    assert len(browser_control.interaction_history) == 1
    assert browser_control.interaction_history[0].interaction_type == "click"
    
    # Test input interaction
    result = await browser_control.interact(1, "input", "test value")
    
    assert result is True
    assert len(browser_control.interaction_history) == 2
    assert browser_control.interaction_history[1].interaction_type == "input"

@pytest.mark.asyncio
async def test_content_analysis(browser_control):
    """Test page content analysis."""
    analysis = await browser_control.analyze_content()
    
    assert "text_content" in analysis
    assert "word_count" in analysis
    assert "interactive_elements" in analysis
    assert "forms" in analysis
    assert "links" in analysis
    assert "readability_score" in analysis

@pytest.mark.asyncio
async def test_smart_form_fill(browser_control):
    """Test smart form filling."""
    form_data = {"username": "testuser", "password": "testpass"}
    
    # Setup mock form analysis
    browser_control._analyze_forms = AsyncMock(return_value=[
        {"name": "username", "type": "text"},
        {"name": "password", "type": "password"}
    ])
    
    result = await browser_control.smart_form_fill(form_data)
    assert result is True

@pytest.mark.asyncio
async def test_smart_screenshot(browser_control, mock_context, tmp_path):
    """Test smart screenshot functionality."""
    # Setup screenshot directory
    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    browser_control.screenshot_dir = screenshot_dir
    
    # Mock screenshot data
    mock_screenshot_data = b"fake screenshot data"
    mock_context.take_screenshot = AsyncMock(return_value=mock_screenshot_data)
    
    # Test full page screenshot
    screenshot_path = await browser_control.take_smart_screenshot(full_page=True)
    
    assert screenshot_path is not None
    assert Path(screenshot_path).exists()
    assert Path(screenshot_path).read_bytes() == mock_screenshot_data

@pytest.mark.asyncio
async def test_cleanup(browser_control, mock_browser, mock_context):
    """Test resource cleanup."""
    await browser_control.cleanup()
    
    mock_context.close.assert_called_once()
    mock_browser.close.assert_called_once()
    assert browser_control.context is None
    assert browser_control.browser is None

@pytest.mark.asyncio
async def test_page_analysis(browser_control):
    """Test page analysis functionality."""
    # Setup mock page content
    html_content = """
    <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test description">
        </head>
        <body>
            <main>
                <article>
                    <h1>Test Article</h1>
                    <div class="author">John Doe</div>
                    <div class="date">2024-03-20</div>
                    <div class="content">Test content</div>
                    <div class="tags">
                        <span class="tag">test</span>
                        <span class="tag">example</span>
                    </div>
                </article>
                <form id="test-form">
                    <input type="text" name="username" required>
                    <input type="password" name="password" required>
                    <button type="submit">Submit</button>
                </form>
            </main>
        </body>
    </html>
    """
    browser_control.context.get_page_html = AsyncMock(return_value=html_content)
    
    # Test page analysis
    await browser_control._analyze_current_page()
    
    # Verify analysis results
    current_url = await browser_control.context.get_current_url()
    analysis = browser_control.page_analyses.get(current_url)
    
    assert analysis is not None
    assert isinstance(analysis, PageAnalysis)
    assert "Test content" in analysis.main_content
    assert analysis.content_type in ["article", "form"]
    assert len(analysis.forms) > 0
    assert len(analysis.navigation) >= 0

@pytest.mark.asyncio
async def test_error_handling(browser_control, mock_context):
    """Test error handling in various scenarios."""
    # Test navigation error
    mock_context.navigate_to = AsyncMock(side_effect=Exception("Navigation failed"))
    result = await browser_control.navigate("https://example.com")
    assert result is False
    
    # Test interaction error
    mock_context.get_dom_element_by_index = AsyncMock(side_effect=ValueError("Element not found"))
    result = await browser_control.interact(0, "click")
    assert result is False
    assert not browser_control.interaction_history[-1].success
    
    # Test screenshot error
    mock_context.take_screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))
    result = await browser_control.take_smart_screenshot()
    assert result is None

@pytest.mark.asyncio
async def test_performance_monitoring(browser_control):
    """Test performance monitoring functionality."""
    # Perform some actions to generate metrics
    await browser_control.navigate("https://example.com")
    await browser_control.interact(0, "click")
    await browser_control.analyze_content()
    
    # Check metrics
    assert len(browser_control.performance_metrics["page_load_times"]) > 0
    assert len(browser_control.performance_metrics["interaction_times"]) > 0
    assert len(browser_control.performance_metrics["analysis_times"]) > 0
    
    # Verify metric values are reasonable
    for metric_list in browser_control.performance_metrics.values():
        for value in metric_list:
            assert isinstance(value, float)
            assert value >= 0

@pytest.mark.asyncio
async def test_content_type_detection(browser_control):
    """Test content type detection for different page types."""
    # Test article detection
    html_article = "<html><body><article>Test article</article></body></html>"
    soup_article = BeautifulSoup(html_article, 'html.parser')
    assert browser_control._determine_content_type(soup_article) == "article"
    
    # Test form detection
    html_form = "<html><body><form>Test form</form></body></html>"
    soup_form = BeautifulSoup(html_form, 'html.parser')
    assert browser_control._determine_content_type(soup_form) == "form"
    
    # Test product detection
    html_product = "<html><body><div class='product'>Test product</div></body></html>"
    soup_product = BeautifulSoup(html_product, 'html.parser')
    assert browser_control._determine_content_type(soup_product) == "product"
    
    # Test search results detection
    html_search = "<html><body><div class='search-results'>Test results</div></body></html>"
    soup_search = BeautifulSoup(html_search, 'html.parser')
    assert browser_control._determine_content_type(soup_search) == "search_results"
    
    # Test general content detection
    html_general = "<html><body><div>General content</div></body></html>"
    soup_general = BeautifulSoup(html_general, 'html.parser')
    assert browser_control._determine_content_type(soup_general) == "general" 