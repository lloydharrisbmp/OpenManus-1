"""
SharePoint list and Teams messaging integration.
"""
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import asyncio
from pathlib import Path

from msgraph.core import GraphClient

logger = logging.getLogger(__name__)

@dataclass
class SharePointListConfig:
    """SharePoint list configuration."""
    site_id: str
    list_id: str
    select_fields: List[str] = field(default_factory=list)
    expand_fields: List[str] = field(default_factory=list)

@dataclass
class TeamsConfig:
    """Teams configuration."""
    team_id: str
    channel_id: Optional[str] = None

class SharePointListManager:
    """Manager for SharePoint list operations."""
    
    def __init__(self, graph_client: GraphClient, config: SharePointListConfig):
        self.graph_client = graph_client
        self.config = config
    
    async def get_list_items(
        self,
        filter_query: Optional[str] = None,
        order_by: Optional[str] = None,
        top: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve items from SharePoint list."""
        try:
            # Build path and parameters
            path = f"/sites/{self.config.site_id}/lists/{self.config.list_id}/items"
            
            params = {}
            if self.config.select_fields:
                params["$select"] = ",".join(self.config.select_fields)
            if self.config.expand_fields:
                params["$expand"] = ",".join(self.config.expand_fields)
            if filter_query:
                params["$filter"] = filter_query
            if order_by:
                params["$orderby"] = order_by
            if top:
                params["$top"] = str(top)
            
            # Execute request
            response = await self.graph_client.get(path, params=params)
            return response.json().get("value", [])
            
        except Exception as e:
            logger.error(f"Failed to retrieve list items: {str(e)}")
            return []
    
    async def create_list_item(
        self,
        fields: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a new item in SharePoint list."""
        try:
            path = f"/sites/{self.config.site_id}/lists/{self.config.list_id}/items"
            
            # Prepare request body
            body = {
                "fields": fields
            }
            
            # Execute request
            response = await self.graph_client.post(path, json=body)
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to create list item: {str(e)}")
            return None
    
    async def update_list_item(
        self,
        item_id: str,
        fields: Dict[str, Any]
    ) -> bool:
        """Update an existing item in SharePoint list."""
        try:
            path = f"/sites/{self.config.site_id}/lists/{self.config.list_id}/items/{item_id}"
            
            # Prepare request body
            body = {
                "fields": fields
            }
            
            # Execute request
            await self.graph_client.patch(path, json=body)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update list item: {str(e)}")
            return False
    
    async def delete_list_item(self, item_id: str) -> bool:
        """Delete an item from SharePoint list."""
        try:
            path = f"/sites/{self.config.site_id}/lists/{self.config.list_id}/items/{item_id}"
            
            # Execute request
            await self.graph_client.delete(path)
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete list item: {str(e)}")
            return False

class TeamsMessaging:
    """Manager for Teams messaging operations."""
    
    def __init__(self, graph_client: GraphClient, config: TeamsConfig):
        self.graph_client = graph_client
        self.config = config
    
    async def send_channel_message(
        self,
        content: str,
        importance: str = "normal",
        mentions: Optional[List[Dict[str, str]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Send a message to a Teams channel."""
        try:
            if not self.config.channel_id:
                raise ValueError("Channel ID not configured")
            
            path = f"/teams/{self.config.team_id}/channels/{self.config.channel_id}/messages"
            
            # Prepare message
            message = {
                "body": {
                    "content": content,
                    "contentType": "html"
                },
                "importance": importance
            }
            
            # Add mentions if provided
            if mentions:
                message["mentions"] = mentions
            
            # Add attachments if provided
            if attachments:
                message["attachments"] = attachments
            
            # Execute request
            response = await self.graph_client.post(path, json=message)
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to send channel message: {str(e)}")
            return None
    
    async def send_chat_message(
        self,
        chat_id: str,
        content: str,
        importance: str = "normal",
        mentions: Optional[List[Dict[str, str]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Send a message to a Teams chat."""
        try:
            path = f"/chats/{chat_id}/messages"
            
            # Prepare message
            message = {
                "body": {
                    "content": content,
                    "contentType": "html"
                },
                "importance": importance
            }
            
            # Add mentions if provided
            if mentions:
                message["mentions"] = mentions
            
            # Add attachments if provided
            if attachments:
                message["attachments"] = attachments
            
            # Execute request
            response = await self.graph_client.post(path, json=message)
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to send chat message: {str(e)}")
            return None
    
    async def create_channel_meeting(
        self,
        subject: str,
        start_time: datetime,
        end_time: datetime,
        attendees: Optional[List[str]] = None,
        content: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a meeting in a Teams channel."""
        try:
            if not self.config.channel_id:
                raise ValueError("Channel ID not configured")
            
            path = f"/teams/{self.config.team_id}/channels/{self.config.channel_id}/meetings"
            
            # Prepare meeting
            meeting = {
                "subject": subject,
                "startDateTime": start_time.isoformat(),
                "endDateTime": end_time.isoformat(),
                "joinUrl": None  # Will be generated by Teams
            }
            
            if content:
                meeting["content"] = {
                    "content": content,
                    "contentType": "html"
                }
            
            if attendees:
                meeting["attendees"] = [
                    {"upn": attendee} for attendee in attendees
                ]
            
            # Execute request
            response = await self.graph_client.post(path, json=meeting)
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to create channel meeting: {str(e)}")
            return None
    
    async def get_channel_messages(
        self,
        top: int = 50,
        filter_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve messages from a Teams channel."""
        try:
            if not self.config.channel_id:
                raise ValueError("Channel ID not configured")
            
            path = f"/teams/{self.config.team_id}/channels/{self.config.channel_id}/messages"
            
            params = {
                "$top": str(top)
            }
            if filter_query:
                params["$filter"] = filter_query
            
            # Execute request
            response = await self.graph_client.get(path, params=params)
            return response.json().get("value", [])
            
        except Exception as e:
            logger.error(f"Failed to retrieve channel messages: {str(e)}")
            return []
    
    async def react_to_message(
        self,
        message_id: str,
        reaction_type: str
    ) -> bool:
        """Add a reaction to a Teams message."""
        try:
            if not self.config.channel_id:
                raise ValueError("Channel ID not configured")
            
            path = f"/teams/{self.config.team_id}/channels/{self.config.channel_id}/messages/{message_id}/reactions"
            
            # Prepare reaction
            reaction = {
                "reactionType": reaction_type
            }
            
            # Execute request
            await self.graph_client.post(path, json=reaction)
            return True
            
        except Exception as e:
            logger.error(f"Failed to add reaction: {str(e)}")
            return False 