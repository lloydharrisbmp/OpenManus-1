import os
from pathlib import Path
from typing import Optional

class FileOrganizer:
    """Handles file organization and directory management for generated files."""
    
    def __init__(self):
        self.base_dir = Path("client_documents")
        self.dirs = {
            "markdown": self.base_dir / "markdown",
            "text": self.base_dir / "text",
            "reports": self.base_dir / "reports",
            "websites": self.base_dir / "websites"
        }
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, file_type: str, filename: str) -> Path:
        """Get the appropriate path for a file based on its type."""
        if file_type not in self.dirs:
            raise ValueError(f"Unknown file type: {file_type}")
        
        return self.dirs[file_type] / filename
    
    @staticmethod
    def determine_file_type(filename: str) -> str:
        """Determine the appropriate directory based on file extension."""
        ext = Path(filename).suffix.lower()
        if ext in ['.md', '.markdown']:
            return 'markdown'
        elif ext in ['.txt', '.text']:
            return 'text'
        elif ext in ['.html', '.htm']:
            return 'websites'
        else:
            return 'reports'  # Default to reports for other file types

# Create a singleton instance
file_organizer = FileOrganizer() 