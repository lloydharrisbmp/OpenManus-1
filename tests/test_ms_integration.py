"""
Tests for Microsoft Dataverse and Office 365 integration.
"""
import pytest
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from app.tool.ms_integration import (
    MSIntegration,
    MSAuthConfig,
    DataverseConfig
)

@pytest.fixture
def auth_config():
    """Create test authentication configuration."""
    return MSAuthConfig(
        tenant_id="test-tenant",
        client_id="test-client",
        client_secret="test-secret"
    )

@pytest.fixture
def dataverse_config():
    """Create test Dataverse configuration."""
    return DataverseConfig(
        environment_url="https://test.crm.dynamics.com",
        table_name="test_table",
        select_fields=["id", "name", "description"],
        filter_query="status eq 'active'"
    )

@pytest.fixture
def mock_graph_client():
    """Create mock Graph client."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.close = AsyncMock()
    return client

@pytest.fixture
def mock_dataverse_client():
    """Create mock Dataverse client."""
    client = MagicMock()
    client.get_records = MagicMock()
    client.create_record = MagicMock()
    client.update_record = MagicMock()
    client.close = MagicMock()
    return client

@pytest.fixture
async def ms_integration(auth_config, dataverse_config, mock_graph_client, mock_dataverse_client):
    """Create MSIntegration instance with mocked dependencies."""
    with patch("app.tool.ms_integration.GraphClient", return_value=mock_graph_client), \
         patch("app.tool.ms_integration.DataverseClient", return_value=mock_dataverse_client):
        
        integration = MSIntegration(auth_config, dataverse_config)
        await integration.initialize()
        yield integration
        await integration.cleanup()

@pytest.mark.asyncio
async def test_initialization(ms_integration, mock_graph_client, mock_dataverse_client):
    """Test Microsoft integration initialization."""
    assert ms_integration.graph_client == mock_graph_client
    assert ms_integration.dataverse_client == mock_dataverse_client
    assert ms_integration.auth_config is not None
    assert ms_integration.dataverse_config is not None

@pytest.mark.asyncio
async def test_get_dataverse_records(ms_integration, mock_dataverse_client):
    """Test retrieving records from Dataverse."""
    # Setup mock response
    mock_records = {
        "value": [
            {"id": "1", "name": "Test 1"},
            {"id": "2", "name": "Test 2"}
        ]
    }
    mock_dataverse_client.get_records.return_value = mock_records
    
    # Test with default config
    records = await ms_integration.get_dataverse_records()
    assert len(records) == 2
    assert records[0]["name"] == "Test 1"
    
    # Test with custom parameters
    records = await ms_integration.get_dataverse_records(
        table_name="custom_table",
        select_fields=["id", "name"],
        filter_query="name eq 'Test 1'"
    )
    assert len(records) == 2
    mock_dataverse_client.get_records.assert_called_with(
        "custom_table",
        "id,name",
        "$filter=name eq 'Test 1'"
    )

@pytest.mark.asyncio
async def test_create_dataverse_record(ms_integration, mock_dataverse_client):
    """Test creating a record in Dataverse."""
    # Setup mock response
    mock_dataverse_client.create_record.return_value = {"id": "new-record-id"}
    
    # Test record creation
    record_data = {"name": "New Record", "description": "Test description"}
    record_id = await ms_integration.create_dataverse_record("test_table", record_data)
    
    assert record_id == "new-record-id"
    mock_dataverse_client.create_record.assert_called_with("test_table", record_data)

@pytest.mark.asyncio
async def test_update_dataverse_record(ms_integration, mock_dataverse_client):
    """Test updating a record in Dataverse."""
    # Test record update
    record_data = {"name": "Updated Record"}
    result = await ms_integration.update_dataverse_record(
        "test_table",
        "record-id",
        record_data
    )
    
    assert result is True
    mock_dataverse_client.update_record.assert_called_with(
        "test_table",
        "record-id",
        record_data
    )

@pytest.mark.asyncio
async def test_get_office_files(ms_integration, mock_graph_client):
    """Test retrieving files from OneDrive/SharePoint."""
    # Setup mock response
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "value": [
            {"name": "file1.txt", "id": "1"},
            {"name": "file2.txt", "id": "2"}
        ]
    }
    mock_graph_client.get.return_value = mock_response
    
    # Test with default parameters
    files = await ms_integration.get_office_files()
    assert len(files) == 2
    assert files[0]["name"] == "file1.txt"
    
    # Test with custom parameters
    files = await ms_integration.get_office_files(
        drive_id="test-drive",
        folder_path="test-folder"
    )
    assert len(files) == 2
    mock_graph_client.get.assert_called_with(
        "/drives/test-drive/root:/test-folder:/children"
    )

@pytest.mark.asyncio
async def test_upload_office_file(ms_integration, mock_graph_client, tmp_path):
    """Test uploading a file to OneDrive/SharePoint."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    # Setup mock response
    mock_response = AsyncMock()
    mock_response.json.return_value = {"id": "uploaded-file-id"}
    mock_graph_client.put.return_value = mock_response
    
    # Test file upload
    result = await ms_integration.upload_office_file(
        test_file,
        "test-folder",
        "test-drive"
    )
    
    assert result["id"] == "uploaded-file-id"
    mock_graph_client.put.assert_called_once()

@pytest.mark.asyncio
async def test_get_outlook_messages(ms_integration, mock_graph_client):
    """Test retrieving Outlook messages."""
    # Setup mock response
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "value": [
            {
                "id": "1",
                "subject": "Test Email 1",
                "receivedDateTime": "2024-03-20T10:00:00Z"
            },
            {
                "id": "2",
                "subject": "Test Email 2",
                "receivedDateTime": "2024-03-20T11:00:00Z"
            }
        ]
    }
    mock_graph_client.get.return_value = mock_response
    
    # Test with default parameters
    messages = await ms_integration.get_outlook_messages()
    assert len(messages) == 2
    assert messages[0]["subject"] == "Test Email 1"
    
    # Test with custom parameters
    messages = await ms_integration.get_outlook_messages(
        folder_id="test-folder",
        filter_query="subject eq 'Test'",
        top=5
    )
    assert len(messages) == 2
    mock_graph_client.get.assert_called_with(
        "/me/mailFolders/test-folder/messages",
        params={"$top": "5", "$filter": "subject eq 'Test'"}
    )

@pytest.mark.asyncio
async def test_send_outlook_message(ms_integration, mock_graph_client):
    """Test sending Outlook messages."""
    # Setup mock response
    mock_response = AsyncMock()
    mock_response.status_code = 202
    mock_response.headers = {"message-id": "test-message-id"}
    mock_graph_client.post.return_value = mock_response
    
    # Test sending message
    result = await ms_integration.send_outlook_message(
        subject="Test Subject",
        body="<p>Test body</p>",
        to_recipients=["test@example.com"],
        cc_recipients=["cc@example.com"]
    )
    
    assert result is True
    mock_graph_client.post.assert_called_once()
    call_args = mock_graph_client.post.call_args[1]
    assert "message" in call_args["json"]
    assert call_args["json"]["message"]["subject"] == "Test Subject"

@pytest.mark.asyncio
async def test_error_handling(ms_integration, mock_graph_client, mock_dataverse_client):
    """Test error handling in various scenarios."""
    # Test Dataverse error
    mock_dataverse_client.get_records.side_effect = Exception("Dataverse error")
    records = await ms_integration.get_dataverse_records()
    assert records == []
    
    # Test Graph API error
    mock_graph_client.get.side_effect = Exception("Graph API error")
    messages = await ms_integration.get_outlook_messages()
    assert messages == []
    
    # Test file not found error
    result = await ms_integration.upload_office_file(
        "nonexistent.txt",
        "test-folder"
    )
    assert result is None

@pytest.mark.asyncio
async def test_cleanup(ms_integration, mock_graph_client, mock_dataverse_client):
    """Test cleanup of resources."""
    await ms_integration.cleanup()
    
    assert len(ms_integration._token_cache) == 0
    assert len(ms_integration._data_cache) == 0
    mock_graph_client.close.assert_called_once()
    mock_dataverse_client.close.assert_called_once() 