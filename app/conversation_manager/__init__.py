import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid


class ConversationManager:
    """Manages conversations and their related files for the Financial Planning Agent."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the conversation manager.
        
        Args:
            base_dir: Optional base directory for client documents. If not provided,
                     will use 'client_documents' in the current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path("client_documents")
        self.conversations_dir = self.base_dir / "conversations"
        
        # Create directories if they don't exist
        self.base_dir.mkdir(exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)
    
    def start_new_conversation(self, title: Optional[str] = None) -> str:
        """Start a new conversation.
        
        Args:
            title: Optional title for the conversation.
            
        Returns:
            The ID of the new conversation.
        """
        timestamp = datetime.now().isoformat()
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        # Create conversation directory
        conversation_dir = self.conversations_dir / conversation_id
        conversation_dir.mkdir(exist_ok=True)
        
        # Initialize conversation metadata
        metadata = {
            "id": conversation_id,
            "title": title or "Financial Planning Session",
            "created_at": timestamp,
            "last_updated": timestamp,
            "message_count": 0
        }
        
        # Save metadata
        with open(conversation_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Initialize empty conversation history
        with open(conversation_dir / "conversation.json", "w") as f:
            json.dump([], f, indent=2)
        
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a message to a conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            role: The role of the message sender (user, assistant, system).
            content: The content of the message.
            metadata: Optional metadata for the message.
            
        Returns:
            True if successful, False otherwise.
        """
        conversation_dir = self.conversations_dir / conversation_id
        if not conversation_dir.exists():
            return False
        
        # Update metadata
        metadata_file = conversation_dir / "metadata.json"
        try:
            with open(metadata_file, "r") as f:
                metadata_data = json.load(f)
            
            metadata_data["last_updated"] = datetime.now().isoformat()
            metadata_data["message_count"] = metadata_data.get("message_count", 0) + 1
            
            with open(metadata_file, "w") as f:
                json.dump(metadata_data, f, indent=2)
        except Exception as e:
            print(f"Error updating metadata: {e}")
            return False
        
        # Add message to conversation history
        conversation_file = conversation_dir / "conversation.json"
        try:
            with open(conversation_file, "r") as f:
                conversation_data = json.load(f)
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            conversation_data.append(message)
            
            with open(conversation_file, "w") as f:
                json.dump(conversation_data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error adding message: {e}")
            return False
    
    def get_file_path(self, conversation_id: str, filename: str) -> Path:
        """Get the path for saving a file in a conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            filename: The name of the file.
            
        Returns:
            The path to save the file.
        """
        conversation_dir = self.conversations_dir / conversation_id
        if not conversation_dir.exists():
            conversation_dir.mkdir(exist_ok=True)
        
        return conversation_dir / filename
    
    def get_conversation_path(self, filename: Optional[str] = None) -> Path:
        """Get the path for the current conversation.
        
        Args:
            filename: Optional filename to append to the path.
            
        Returns:
            The path to the conversation directory or a file within it.
        """
        # Find the most recent conversation
        conversations = self.get_all_conversations()
        if not conversations:
            # Create a new conversation if none exists
            conversation_id = self.start_new_conversation()
        else:
            conversation_id = conversations[0]["id"]
            
        conversation_dir = self.conversations_dir / conversation_id
        
        if filename:
            return conversation_dir / filename
        
        return conversation_dir
    
    def get_conversation_directory(self, conversation_id: str) -> Optional[Path]:
        """Get the directory for a conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            The path to the conversation directory, or None if it doesn't exist.
        """
        conversation_dir = self.conversations_dir / conversation_id
        return conversation_dir if conversation_dir.exists() else None
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversations sorted by last updated time (newest first).
        
        Returns:
            A list of dictionaries with conversation metadata.
        """
        conversations = []
        
        if not self.conversations_dir.exists():
            return []
        
        for conversation_dir in self.conversations_dir.iterdir():
            if not conversation_dir.is_dir():
                continue
            
            metadata_file = conversation_dir / "metadata.json"
            if not metadata_file.exists():
                continue
            
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                conversations.append(metadata)
            except Exception:
                continue
        
        # Sort by last_updated (newest first)
        return sorted(conversations, key=lambda x: x.get("last_updated", ""), reverse=True)
    
    def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load a conversation's history.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            A list of dictionaries with message data.
        """
        conversation_dir = self.conversations_dir / conversation_id
        conversation_file = conversation_dir / "conversation.json"
        
        if not conversation_file.exists():
            return []
        
        try:
            with open(conversation_file, "r") as f:
                return json.load(f)
        except Exception:
            return []
    
    def get_conversation_files(self, conversation_id: Optional[str] = None) -> List[Path]:
        """Get all files in a conversation except for metadata and conversation JSON.
        
        Args:
            conversation_id: Optional ID of the conversation. If not provided, uses the current conversation.
            
        Returns:
            A list of Path objects for files in the conversation.
        """
        if not conversation_id:
            # Find the most recent conversation
            conversations = self.get_all_conversations()
            if not conversations:
                return []
            conversation_id = conversations[0]["id"]
        
        conversation_dir = self.conversations_dir / conversation_id
        if not conversation_dir.exists():
            return []
        
        # Get all files except metadata.json and conversation.json
        return [
            f for f in conversation_dir.iterdir() 
            if f.is_file() and f.name not in ["metadata.json", "conversation.json"]
        ]
    
    def rename_conversation(self, conversation_id: str, new_title: str) -> bool:
        """Rename a conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            new_title: The new title for the conversation.
            
        Returns:
            True if successful, False otherwise.
        """
        conversation_dir = self.conversations_dir / conversation_id
        metadata_file = conversation_dir / "metadata.json"
        
        if not metadata_file.exists():
            return False
        
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            metadata["title"] = new_title
            metadata["last_updated"] = datetime.now().isoformat()
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception:
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its files.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            True if successful, False otherwise.
        """
        conversation_dir = self.conversations_dir / conversation_id
        
        if not conversation_dir.exists():
            return False
        
        try:
            shutil.rmtree(conversation_dir)
            return True
        except Exception:
            return False
