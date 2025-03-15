"""
Tests for SharePoint list and Teams messaging integration.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from msgraph.core import GraphClient

from app.tool.ms_collaboration import (
    SharePointListConfig,
    TeamsConfig,
    SharePointListManager,
    TeamsMessaging
)

@pytest.fixture
def graph_client():
    """Create a mock Graph client."""
    client = AsyncMock(spec=GraphClient)
    return client

@pytest.fixture
def sharepoint_config():
    """Create a SharePoint list configuration."""
    return SharePointListConfig(
        site_id="site123",
        list_id="list456",
        select_fields=["Title", "Description"],
        expand_fields=["Author"]
    )

@pytest.fixture
def teams_config():
    """Create a Teams configuration."""
    return TeamsConfig(
        team_id="team789",
        channel_id="channel012"
    )

@pytest.fixture
def sharepoint_manager(graph_client, sharepoint_config):
    """Create a SharePoint list manager."""
    return SharePointListManager(graph_client, sharepoint_config)

@pytest.fixture
def teams_messaging(graph_client, teams_config):
    """Create a Teams messaging manager."""
    return TeamsMessaging(graph_client, teams_config)

@pytest.mark.asyncio
async def test_get_list_items(sharepoint_manager, graph_client):
    """Test retrieving items from SharePoint list."""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "value": [
            {"id": "1", "fields": {"Title": "Item 1"}},
            {"id": "2", "fields": {"Title": "Item 2"}}
        ]
    }
    graph_client.get.return_value = mock_response
    
    # Test with default parameters
    items = await sharepoint_manager.get_list_items()
    assert len(items) == 2
    assert items[0]["fields"]["Title"] == "Item 1"
    
    # Verify correct path and parameters
    graph_client.get.assert_called_with(
        f"/sites/{sharepoint_manager.config.site_id}/lists/{sharepoint_manager.config.list_id}/items",
        params={
            "$select": "Title,Description",
            "$expand": "Author"
        }
    )
    
    # Test with filter and order
    await sharepoint_manager.get_list_items(
        filter_query="Title eq 'Item 1'",
        order_by="Title desc",
        top=1
    )
    
    graph_client.get.assert_called_with(
        f"/sites/{sharepoint_manager.config.site_id}/lists/{sharepoint_manager.config.list_id}/items",
        params={
            "$select": "Title,Description",
            "$expand": "Author",
            "$filter": "Title eq 'Item 1'",
            "$orderby": "Title desc",
            "$top": "1"
        }
    )

@pytest.mark.asyncio
async def test_create_list_item(sharepoint_manager, graph_client):
    """Test creating an item in SharePoint list."""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "1",
        "fields": {"Title": "New Item"}
    }
    graph_client.post.return_value = mock_response
    
    # Test item creation
    fields = {"Title": "New Item", "Description": "Test item"}
    result = await sharepoint_manager.create_list_item(fields)
    
    assert result["id"] == "1"
    assert result["fields"]["Title"] == "New Item"
    
    # Verify correct path and body
    graph_client.post.assert_called_with(
        f"/sites/{sharepoint_manager.config.site_id}/lists/{sharepoint_manager.config.list_id}/items",
        json={"fields": fields}
    )

@pytest.mark.asyncio
async def test_update_list_item(sharepoint_manager, graph_client):
    """Test updating an item in SharePoint list."""
    # Test item update
    fields = {"Title": "Updated Item"}
    success = await sharepoint_manager.update_list_item("1", fields)
    
    assert success is True
    
    # Verify correct path and body
    graph_client.patch.assert_called_with(
        f"/sites/{sharepoint_manager.config.site_id}/lists/{sharepoint_manager.config.list_id}/items/1",
        json={"fields": fields}
    )

@pytest.mark.asyncio
async def test_delete_list_item(sharepoint_manager, graph_client):
    """Test deleting an item from SharePoint list."""
    # Test item deletion
    success = await sharepoint_manager.delete_list_item("1")
    
    assert success is True
    
    # Verify correct path
    graph_client.delete.assert_called_with(
        f"/sites/{sharepoint_manager.config.site_id}/lists/{sharepoint_manager.config.list_id}/items/1"
    )

@pytest.mark.asyncio
async def test_send_channel_message(teams_messaging, graph_client):
    """Test sending a message to Teams channel."""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "msg1",
        "body": {"content": "Test message"}
    }
    graph_client.post.return_value = mock_response
    
    # Test message sending
    content = "Test message"
    mentions = [{"id": "user1", "name": "User One"}]
    attachments = [{"contentUrl": "https://example.com/file.pdf"}]
    
    result = await teams_messaging.send_channel_message(
        content=content,
        importance="high",
        mentions=mentions,
        attachments=attachments
    )
    
    assert result["id"] == "msg1"
    
    # Verify correct path and body
    graph_client.post.assert_called_with(
        f"/teams/{teams_messaging.config.team_id}/channels/{teams_messaging.config.channel_id}/messages",
        json={
            "body": {
                "content": content,
                "contentType": "html"
            },
            "importance": "high",
            "mentions": mentions,
            "attachments": attachments
        }
    )

@pytest.mark.asyncio
async def test_create_channel_meeting(teams_messaging, graph_client):
    """Test creating a meeting in Teams channel."""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "meeting1",
        "joinUrl": "https://teams.microsoft.com/meet/123"
    }
    graph_client.post.return_value = mock_response
    
    # Test meeting creation
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=1)
    attendees = ["user1@example.com", "user2@example.com"]
    
    result = await teams_messaging.create_channel_meeting(
        subject="Test Meeting",
        start_time=start_time,
        end_time=end_time,
        attendees=attendees,
        content="Meeting agenda"
    )
    
    assert result["id"] == "meeting1"
    
    # Verify correct path and body
    graph_client.post.assert_called_with(
        f"/teams/{teams_messaging.config.team_id}/channels/{teams_messaging.config.channel_id}/meetings",
        json={
            "subject": "Test Meeting",
            "startDateTime": start_time.isoformat(),
            "endDateTime": end_time.isoformat(),
            "joinUrl": None,
            "content": {
                "content": "Meeting agenda",
                "contentType": "html"
            },
            "attendees": [
                {"upn": "user1@example.com"},
                {"upn": "user2@example.com"}
            ]
        }
    )

@pytest.mark.asyncio
async def test_get_channel_messages(teams_messaging, graph_client):
    """Test retrieving messages from Teams channel."""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "value": [
            {"id": "msg1", "content": "Message 1"},
            {"id": "msg2", "content": "Message 2"}
        ]
    }
    graph_client.get.return_value = mock_response
    
    # Test message retrieval
    messages = await teams_messaging.get_channel_messages(
        top=2,
        filter_query="importance eq 'high'"
    )
    
    assert len(messages) == 2
    assert messages[0]["id"] == "msg1"
    
    # Verify correct path and parameters
    graph_client.get.assert_called_with(
        f"/teams/{teams_messaging.config.team_id}/channels/{teams_messaging.config.channel_id}/messages",
        params={
            "$top": "2",
            "$filter": "importance eq 'high'"
        }
    )

@pytest.mark.asyncio
async def test_react_to_message(teams_messaging, graph_client):
    """Test adding a reaction to a Teams message."""
    # Test reaction
    success = await teams_messaging.react_to_message(
        message_id="msg1",
        reaction_type="like"
    )
    
    assert success is True
    
    # Verify correct path and body
    graph_client.post.assert_called_with(
        f"/teams/{teams_messaging.config.team_id}/channels/{teams_messaging.config.channel_id}/messages/msg1/reactions",
        json={"reactionType": "like"}
    )

@pytest.mark.asyncio
async def test_error_handling(sharepoint_manager, teams_messaging, graph_client):
    """Test error handling in SharePoint and Teams operations."""
    # Mock error response
    graph_client.get.side_effect = Exception("API Error")
    graph_client.post.side_effect = Exception("API Error")
    
    # Test SharePoint error handling
    items = await sharepoint_manager.get_list_items()
    assert items == []
    
    result = await sharepoint_manager.create_list_item({"Title": "Test"})
    assert result is None
    
    # Test Teams error handling
    result = await teams_messaging.send_channel_message("Test")
    assert result is None
    
    messages = await teams_messaging.get_channel_messages()
    assert messages == [] 