"""
Tests for workflow examples combining browser automation with Microsoft services.
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.tool.workflow_examples import AutomatedWorkflows
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.ms_collaboration import SharePointListManager, TeamsMessaging
from app.tool.dataverse_query import DataverseQueryBuilder

@pytest.fixture
def browser_tool():
    """Create a mock browser tool."""
    browser = AsyncMock(spec=BrowserUseTool)
    return browser

@pytest.fixture
def sharepoint_manager():
    """Create a mock SharePoint manager."""
    manager = AsyncMock(spec=SharePointListManager)
    return manager

@pytest.fixture
def teams_messaging():
    """Create a mock Teams messaging."""
    messaging = AsyncMock(spec=TeamsMessaging)
    return messaging

@pytest.fixture
def dataverse_query():
    """Create a mock Dataverse query builder."""
    query = AsyncMock(spec=DataverseQueryBuilder)
    return query

@pytest.fixture
def workflows(browser_tool, sharepoint_manager, teams_messaging):
    """Create an automated workflows instance."""
    return AutomatedWorkflows(browser_tool, sharepoint_manager, teams_messaging)

@pytest.mark.asyncio
async def test_web_form_automation_success(workflows, browser_tool, sharepoint_manager, teams_messaging, dataverse_query):
    """Test successful web form automation workflow."""
    # Mock Dataverse query results
    dataverse_query.execute.return_value = [
        {
            "id": "1",
            "name": "John Doe",
            "email": "john@example.com",
            "company": "Example Corp",
            "message": "Test message"
        }
    ]
    
    # Mock browser interactions
    browser_tool.take_screenshot.return_value = "screenshot.png"
    browser_tool.wait_for_element.return_value = True
    
    # Mock SharePoint and Teams responses
    sharepoint_manager.create_list_item.return_value = {"id": "1"}
    teams_messaging.send_channel_message.return_value = {"id": "msg1"}
    
    # Execute workflow
    success = await workflows.web_form_automation(
        form_url="https://example.com/form",
        dataverse_query=dataverse_query,
        notification_channel="channel1"
    )
    
    assert success is True
    
    # Verify browser interactions
    browser_tool.navigate.assert_called_with("https://example.com/form")
    browser_tool.fill_form.assert_called_with({
        "#name": "John Doe",
        "#email": "john@example.com",
        "#company": "Example Corp",
        "#message": "Test message"
    })
    browser_tool.click_element.assert_called_with("button[type='submit']")
    
    # Verify SharePoint and Teams notifications
    sharepoint_manager.create_list_item.assert_called_once()
    teams_messaging.send_channel_message.assert_called_once()

@pytest.mark.asyncio
async def test_web_form_automation_no_records(workflows, dataverse_query):
    """Test web form automation with no Dataverse records."""
    # Mock empty Dataverse results
    dataverse_query.execute.return_value = []
    
    # Execute workflow
    success = await workflows.web_form_automation(
        form_url="https://example.com/form",
        dataverse_query=dataverse_query
    )
    
    assert success is False

@pytest.mark.asyncio
async def test_data_scraping_workflow_success(workflows, browser_tool, sharepoint_manager, teams_messaging):
    """Test successful data scraping workflow."""
    # Mock browser data extraction
    browser_tool.extract_data.return_value = {
        "title": "Test Product",
        "price": "$99.99",
        "description": "Product description",
        "availability": "In Stock"
    }
    browser_tool.take_screenshot.return_value = "screenshot.png"
    
    # Mock SharePoint response
    sharepoint_manager.create_list_item.return_value = {"id": "1"}
    
    # Execute workflow
    success = await workflows.data_scraping_workflow(
        target_url="https://example.com/product",
        sharepoint_list_name="Products",
        teams_channel="channel1"
    )
    
    assert success is True
    
    # Verify browser interactions
    browser_tool.navigate.assert_called_with("https://example.com/product")
    browser_tool.extract_data.assert_called_once()
    
    # Verify SharePoint and Teams notifications
    sharepoint_manager.create_list_item.assert_called_once()
    teams_messaging.send_channel_message.assert_called_once()

@pytest.mark.asyncio
async def test_automated_testing_workflow_success(workflows, browser_tool, sharepoint_manager, teams_messaging):
    """Test successful automated testing workflow."""
    # Mock browser interactions
    browser_tool.wait_for_element.return_value = True
    browser_tool.get_element_text.return_value = "Expected Text"
    browser_tool.take_screenshot.return_value = "screenshot.png"
    
    # Test cases
    test_cases = [
        {
            "name": "Test Case 1",
            "url": "https://example.com/test1",
            "actions": [
                {"type": "click", "selector": "#button1"},
                {"type": "input", "selector": "#input1", "value": "test"},
                {"type": "wait", "selector": "#result"}
            ],
            "assertions": [
                {"type": "element_present", "selector": "#success"},
                {"type": "element_text", "selector": "#message", "expected_text": "Expected Text"}
            ]
        }
    ]
    
    # Execute workflow
    success = await workflows.automated_testing_workflow(
        test_cases=test_cases,
        teams_channel="channel1"
    )
    
    assert success is True
    
    # Verify browser interactions
    assert browser_tool.navigate.call_count == 1
    assert browser_tool.click_element.call_count == 1
    assert browser_tool.fill_form.call_count == 1
    assert browser_tool.wait_for_element.call_count == 2
    
    # Verify SharePoint and Teams notifications
    sharepoint_manager.create_list_item.assert_called_once()
    teams_messaging.send_channel_message.assert_called_once()

@pytest.mark.asyncio
async def test_automated_testing_workflow_failure(workflows, browser_tool, sharepoint_manager, teams_messaging):
    """Test automated testing workflow with failures."""
    # Mock browser interactions
    browser_tool.wait_for_element.side_effect = Exception("Element not found")
    browser_tool.take_screenshot.return_value = "screenshot.png"
    
    # Test cases
    test_cases = [
        {
            "name": "Failed Test",
            "url": "https://example.com/test",
            "actions": [
                {"type": "click", "selector": "#button1"}
            ],
            "assertions": [
                {"type": "element_present", "selector": "#missing-element"}
            ]
        }
    ]
    
    # Execute workflow
    success = await workflows.automated_testing_workflow(
        test_cases=test_cases,
        teams_channel="channel1"
    )
    
    assert success is True  # Workflow completes despite test failures
    
    # Verify error reporting
    teams_messaging.send_channel_message.assert_called_once()
    call_args = teams_messaging.send_channel_message.call_args[1]
    assert "Failed: ❌ 1" in call_args["content"]
    assert call_args["importance"] == "high"

@pytest.mark.asyncio
async def test_error_handling(workflows, browser_tool, teams_messaging):
    """Test error handling in workflows."""
    # Mock browser error
    browser_tool.navigate.side_effect = Exception("Navigation failed")
    
    # Test form automation
    success = await workflows.web_form_automation(
        form_url="https://example.com/form",
        dataverse_query=AsyncMock(),
        notification_channel="channel1"
    )
    assert success is False
    
    # Verify error notification
    teams_messaging.send_channel_message.assert_called_with(
        content="❌ Form automation failed: Navigation failed",
        importance="high"
    )
    
    # Reset mock
    teams_messaging.send_channel_message.reset_mock()
    
    # Test data scraping
    success = await workflows.data_scraping_workflow(
        target_url="https://example.com/product",
        sharepoint_list_name="Products",
        teams_channel="channel1"
    )
    assert success is False
    
    # Verify error notification
    teams_messaging.send_channel_message.assert_called_with(
        content="❌ Data scraping failed: Navigation failed",
        importance="high"
    ) 