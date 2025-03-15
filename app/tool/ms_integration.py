"""
Microsoft Dataverse and Office 365 integration module.
"""
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import asyncio
from pathlib import Path

from msal import ConfidentialClientApplication
from msgraph.core import GraphClient
from azure.identity import ClientSecretCredential
from dataverse.client import Client as DataverseClient

logger = logging.getLogger(__name__)

@dataclass
class MSAuthConfig:
    """Microsoft authentication configuration."""
    tenant_id: str
    client_id: str
    client_secret: str
    scopes: List[str] = field(default_factory=lambda: [
        "https://graph.microsoft.com/.default",
        "https://orgname.crm.dynamics.com/.default"
    ])

@dataclass
class DataverseConfig:
    """Dataverse configuration."""
    environment_url: str
    table_name: str
    select_fields: List[str] = field(default_factory=list)
    filter_query: Optional[str] = None

class MSIntegration:
    """Microsoft integration for Dataverse and Office 365."""
    
    def __init__(
        self,
        auth_config: MSAuthConfig,
        dataverse_config: Optional[DataverseConfig] = None,
        cache_dir: Optional[str] = None
    ):
        self.auth_config = auth_config
        self.dataverse_config = dataverse_config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize clients
        self.graph_client: Optional[GraphClient] = None
        self.dataverse_client: Optional[DataverseClient] = None
        
        # Cache for access tokens and data
        self._token_cache: Dict[str, Any] = {}
        self._data_cache: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize Microsoft clients."""
        try:
            # Initialize Graph client
            credential = ClientSecretCredential(
                tenant_id=self.auth_config.tenant_id,
                client_id=self.auth_config.client_id,
                client_secret=self.auth_config.client_secret
            )
            self.graph_client = GraphClient(credential=credential)
            
            # Initialize Dataverse client if configured
            if self.dataverse_config:
                self.dataverse_client = DataverseClient(
                    self.dataverse_config.environment_url,
                    self.auth_config.client_id,
                    self.auth_config.client_secret
                )
            
            logger.info("Microsoft integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Microsoft integration: {str(e)}")
            raise
    
    async def get_dataverse_records(
        self,
        table_name: Optional[str] = None,
        select_fields: Optional[List[str]] = None,
        filter_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve records from Dataverse table."""
        try:
            if not self.dataverse_client:
                raise ValueError("Dataverse client not initialized")
            
            # Use provided values or fall back to config
            table = table_name or self.dataverse_config.table_name
            fields = select_fields or self.dataverse_config.select_fields
            query = filter_query or self.dataverse_config.filter_query
            
            # Build query
            select_clause = ",".join(fields) if fields else "*"
            filter_clause = f"$filter={query}" if query else ""
            
            # Execute query
            response = await asyncio.to_thread(
                self.dataverse_client.get_records,
                table,
                select_clause,
                filter_clause
            )
            
            return response.get("value", [])
            
        except Exception as e:
            logger.error(f"Failed to retrieve Dataverse records: {str(e)}")
            return []
    
    async def create_dataverse_record(
        self,
        table_name: str,
        record_data: Dict[str, Any]
    ) -> Optional[str]:
        """Create a new record in Dataverse table."""
        try:
            if not self.dataverse_client:
                raise ValueError("Dataverse client not initialized")
            
            response = await asyncio.to_thread(
                self.dataverse_client.create_record,
                table_name,
                record_data
            )
            
            return response.get("id")
            
        except Exception as e:
            logger.error(f"Failed to create Dataverse record: {str(e)}")
            return None
    
    async def update_dataverse_record(
        self,
        table_name: str,
        record_id: str,
        record_data: Dict[str, Any]
    ) -> bool:
        """Update an existing record in Dataverse table."""
        try:
            if not self.dataverse_client:
                raise ValueError("Dataverse client not initialized")
            
            await asyncio.to_thread(
                self.dataverse_client.update_record,
                table_name,
                record_id,
                record_data
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update Dataverse record: {str(e)}")
            return False
    
    async def get_office_files(
        self,
        drive_id: Optional[str] = None,
        folder_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve files from OneDrive or SharePoint."""
        try:
            if not self.graph_client:
                raise ValueError("Graph client not initialized")
            
            # Build path
            path = f"/drives/{drive_id}" if drive_id else "/me/drive"
            if folder_path:
                path += f"/root:/{folder_path}:/children"
            else:
                path += "/root/children"
            
            # Execute request
            response = await self.graph_client.get(path)
            return response.json().get("value", [])
            
        except Exception as e:
            logger.error(f"Failed to retrieve Office files: {str(e)}")
            return []
    
    async def upload_office_file(
        self,
        file_path: Union[str, Path],
        target_path: str,
        drive_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Upload a file to OneDrive or SharePoint."""
        try:
            if not self.graph_client:
                raise ValueError("Graph client not initialized")
            
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Build path
            path = f"/drives/{drive_id}" if drive_id else "/me/drive"
            path += f"/root:/{target_path}/{file_path.name}:/content"
            
            # Upload file
            with open(file_path, "rb") as f:
                response = await self.graph_client.put(path, data=f)
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to upload Office file: {str(e)}")
            return None
    
    async def get_outlook_messages(
        self,
        folder_id: Optional[str] = None,
        filter_query: Optional[str] = None,
        top: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve messages from Outlook."""
        try:
            if not self.graph_client:
                raise ValueError("Graph client not initialized")
            
            # Build path and query
            path = "/me/messages"
            if folder_id:
                path = f"/me/mailFolders/{folder_id}/messages"
            
            params = {
                "$top": str(top)
            }
            if filter_query:
                params["$filter"] = filter_query
            
            # Execute request
            response = await self.graph_client.get(path, params=params)
            return response.json().get("value", [])
            
        except Exception as e:
            logger.error(f"Failed to retrieve Outlook messages: {str(e)}")
            return []
    
    async def send_outlook_message(
        self,
        subject: str,
        body: str,
        to_recipients: List[str],
        cc_recipients: Optional[List[str]] = None,
        attachments: Optional[List[Union[str, Path]]] = None
    ) -> bool:
        """Send an email message through Outlook."""
        try:
            if not self.graph_client:
                raise ValueError("Graph client not initialized")
            
            # Prepare message
            message = {
                "subject": subject,
                "body": {
                    "contentType": "HTML",
                    "content": body
                },
                "toRecipients": [
                    {"emailAddress": {"address": email}}
                    for email in to_recipients
                ]
            }
            
            if cc_recipients:
                message["ccRecipients"] = [
                    {"emailAddress": {"address": email}}
                    for email in cc_recipients
                ]
            
            # Send message
            response = await self.graph_client.post(
                "/me/sendMail",
                json={"message": message}
            )
            
            # Handle attachments if present
            if attachments and response.status_code == 202:
                message_id = response.headers.get("message-id")
                if message_id:
                    await self._add_attachments(message_id, attachments)
            
            return response.status_code in (202, 204)
            
        except Exception as e:
            logger.error(f"Failed to send Outlook message: {str(e)}")
            return False
    
    async def _add_attachments(
        self,
        message_id: str,
        attachments: List[Union[str, Path]]
    ) -> None:
        """Add attachments to a draft message."""
        try:
            for attachment in attachments:
                file_path = Path(attachment)
                if not file_path.exists():
                    logger.warning(f"Attachment not found: {file_path}")
                    continue
                
                with open(file_path, "rb") as f:
                    content = f.read()
                    
                attachment_data = {
                    "@odata.type": "#microsoft.graph.fileAttachment",
                    "name": file_path.name,
                    "contentBytes": content
                }
                
                await self.graph_client.post(
                    f"/me/messages/{message_id}/attachments",
                    json=attachment_data
                )
                
        except Exception as e:
            logger.error(f"Failed to add attachments: {str(e)}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear caches
            self._token_cache.clear()
            self._data_cache.clear()
            
            # Close clients
            if self.dataverse_client:
                await asyncio.to_thread(self.dataverse_client.close)
            
            if self.graph_client:
                await self.graph_client.close()
            
            logger.info("Microsoft integration cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Microsoft integration: {str(e)}") 