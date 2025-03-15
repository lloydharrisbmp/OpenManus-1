import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

class ConversationManager:
    """Manages conversation history and related files."""
    
    def __init__(self, base_dir: str = "client_documents"):
        """
        Initialize the conversation manager.
        
        Args:
            base_dir: Base directory for all client documents
        """
        self.base_dir = Path(base_dir)
        self.conversations_dir = self.base_dir / "conversations"
        self.current_conversation_id = None
        self.current_conversation_dir = None
        self._ensure_directories_exist()
    
    def _ensure_directories_exist(self):
        """Ensure that the necessary directories exist."""
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.conversations_dir.mkdir(exist_ok=True, parents=True)
    
    def start_new_conversation(self, title: Optional[str] = None) -> str:
        """
        Start a new conversation.
        
        Args:
            title: Optional title for the conversation
            
        Returns:
            str: The ID of the new conversation
        """
        # Generate conversation ID based on timestamp
        timestamp = datetime.datetime.now()
        conversation_id = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # If title is provided, use it; otherwise use the timestamp
        display_title = title or f"Conversation {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Create conversation directory
        conversation_dir = self.conversations_dir / conversation_id
        conversation_dir.mkdir(exist_ok=True)
        
        # Create metadata file
        metadata = {
            "id": conversation_id,
            "title": display_title,
            "created_at": timestamp.isoformat(),
            "last_updated": timestamp.isoformat(),
            "message_count": 0
        }
        
        with open(conversation_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create empty conversation history file
        with open(conversation_dir / "history.json", "w") as f:
            json.dump([], f, indent=2)
        
        # Set as current conversation
        self.current_conversation_id = conversation_id
        self.current_conversation_dir = conversation_dir
        
        return conversation_id
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the current conversation.
        
        Args:
            role: The role of the message sender (user, assistant, system)
            content: The message content
        """
        if not self.current_conversation_id:
            raise ValueError("No active conversation. Call start_new_conversation first.")
        
        # Load existing history
        history_file = self.current_conversation_dir / "history.json"
        with open(history_file, "r") as f:
            history = json.load(f)
        
        # Add new message
        timestamp = datetime.datetime.now().isoformat()
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        history.append(message)
        
        # Update history file
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        
        # Update metadata
        metadata_file = self.current_conversation_dir / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        metadata["last_updated"] = timestamp
        metadata["message_count"] = len(history)
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_conversation_path(self, filename: Optional[str] = None) -> Path:
        """
        Get the path to save a file in the current conversation directory.
        
        Args:
            filename: Optional filename. If not provided, returns the directory path.
            
        Returns:
            Path: The path to save the file
        """
        if not self.current_conversation_dir:
            raise ValueError("No active conversation. Call start_new_conversation first.")
        
        if filename:
            return self.current_conversation_dir / filename
        else:
            return self.current_conversation_dir
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """
        Get all conversations, sorted by last updated time (newest first).
        
        Returns:
            List of conversation metadata
        """
        conversations = []
        
        for conversation_dir in self.conversations_dir.iterdir():
            if conversation_dir.is_dir():
                metadata_file = conversation_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        conversations.append(metadata)
        
        # Sort by last_updated, newest first
        return sorted(conversations, key=lambda x: x.get("last_updated", ""), reverse=True)
    
    def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Load a specific conversation history.
        
        Args:
            conversation_id: The ID of the conversation to load
            
        Returns:
            List of messages in the conversation
        """
        conversation_dir = self.conversations_dir / conversation_id
        history_file = conversation_dir / "history.json"
        
        if not history_file.exists():
            raise ValueError(f"Conversation {conversation_id} not found")
        
        with open(history_file, "r") as f:
            history = json.load(f)
        
        # Set as current conversation
        self.current_conversation_id = conversation_id
        self.current_conversation_dir = conversation_dir
        
        return history
    
    def get_conversation_files(self, conversation_id: Optional[str] = None) -> List[Path]:
        """
        Get a list of files in the conversation directory (excluding metadata and history).
        
        Args:
            conversation_id: The ID of the conversation. If not provided, uses current conversation.
            
        Returns:
            List of file paths
        """
        if conversation_id:
            conversation_dir = self.conversations_dir / conversation_id
        elif self.current_conversation_dir:
            conversation_dir = self.current_conversation_dir
        else:
            raise ValueError("No conversation specified or active")
        
        if not conversation_dir.exists():
            raise ValueError(f"Conversation directory {conversation_dir} not found")
        
        # Get all files except metadata.json and history.json
        return [
            f for f in conversation_dir.iterdir() 
            if f.is_file() and f.name not in ["metadata.json", "history.json"]
        ]
    
    def rename_conversation(self, conversation_id: str, new_title: str) -> None:
        """
        Rename a conversation.
        
        Args:
            conversation_id: The ID of the conversation to rename
            new_title: The new title for the conversation
        """
        conversation_dir = self.conversations_dir / conversation_id
        metadata_file = conversation_dir / "metadata.json"
        
        if not metadata_file.exists():
            raise ValueError(f"Conversation {conversation_id} not found")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        metadata["title"] = new_title
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation and all its files.
        
        Args:
            conversation_id: The ID of the conversation to delete
        """
        conversation_dir = self.conversations_dir / conversation_id
        
        if not conversation_dir.exists():
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Delete the entire directory
        shutil.rmtree(conversation_dir)
        
        # If this was the current conversation, reset current conversation
        if self.current_conversation_id == conversation_id:
            self.current_conversation_id = None
            self.current_conversation_dir = None 