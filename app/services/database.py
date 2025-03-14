"""
Enhanced Database Service with async support, concurrency control, and improved error handling.
"""

import os
import json
import logging
import asyncio
import aiofiles
import aiosqlite
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass

@dataclass
class DatabaseResult:
    """Structured result object for database operations."""
    success: bool
    value: Optional[Any] = None
    error: Optional[str] = None
    source: Optional[str] = None
    duration_ms: Optional[float] = None
    partial_success: bool = False
    details: Optional[Dict] = None

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle dates and custom objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class EnhancedDatabaseService:
    """
    Enhanced database service with async support, concurrency control,
    and improved error handling.
    """

    def __init__(self):
        self.db_path = Path("database")
        self.sqlite_path = self.db_path / "financial_planning.db"
        self.semaphore = asyncio.Semaphore(3)  # Limit concurrent operations
        self.cleanup_lock = asyncio.Lock()
        self.cleanup_interval = 3600  # 1 hour
        self.max_entries = 10000
        self._setup_logging()
        self._ensure_directories()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("EnhancedDatabaseService")

    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        (self.db_path / "archive").mkdir(exist_ok=True)

    async def initialize_db(self) -> DatabaseResult:
        """Initialize the SQLite database with required tables."""
        try:
            async with aiosqlite.connect(self.sqlite_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS clients (
                        id TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        client_id TEXT,
                        data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (client_id) REFERENCES clients(id)
                    )
                """)
                await db.commit()
            return DatabaseResult(success=True)
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return DatabaseResult(success=False, error=str(e))

    async def save_json(self, collection: str, data: Dict) -> DatabaseResult:
        """Save JSON data to a file with concurrency control."""
        start_time = datetime.now()
        try:
            async with self.semaphore:
                file_path = self.db_path / f"{collection}.json"
                
                # Read existing data
                existing_data = []
                if file_path.exists():
                    async with aiofiles.open(file_path, 'r') as f:
                        content = await f.read()
                        if content:
                            existing_data = json.loads(content)

                # Add new data with timestamp
                data['timestamp'] = datetime.now().isoformat()
                existing_data.append(data)

                # Write back to file
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(json.dumps(existing_data, cls=JSONEncoder, indent=2))

                duration = (datetime.now() - start_time).total_seconds() * 1000
                return DatabaseResult(
                    success=True,
                    value=data,
                    duration_ms=duration,
                    source=str(file_path)
                )

        except Exception as e:
            self.logger.error(f"Error saving JSON data: {e}")
            return DatabaseResult(
                success=False,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    async def load_json(self, collection: str) -> DatabaseResult:
        """Load JSON data from a file with error handling."""
        start_time = datetime.now()
        try:
            async with self.semaphore:
                file_path = self.db_path / f"{collection}.json"
                if not file_path.exists():
                    return DatabaseResult(
                        success=False,
                        error=f"Collection {collection} not found",
                        duration_ms=0
                    )

                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content) if content else []

                duration = (datetime.now() - start_time).total_seconds() * 1000
                return DatabaseResult(
                    success=True,
                    value=data,
                    duration_ms=duration,
                    source=str(file_path)
                )

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return DatabaseResult(
                success=False,
                error=f"Invalid JSON format: {str(e)}",
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        except Exception as e:
            self.logger.error(f"Error loading JSON data: {e}")
            return DatabaseResult(
                success=False,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    async def delete_document(self, collection: str, doc_id: str) -> DatabaseResult:
        """Delete a document from a collection."""
        start_time = datetime.now()
        try:
            async with self.semaphore:
                file_path = self.db_path / f"{collection}.json"
                if not file_path.exists():
                    return DatabaseResult(
                        success=False,
                        error=f"Collection {collection} not found",
                        duration_ms=0
                    )

                # Read existing data
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content) if content else []

                # Filter out the document to delete
                filtered_data = [doc for doc in data if doc.get('id') != doc_id]
                
                if len(filtered_data) == len(data):
                    return DatabaseResult(
                        success=False,
                        error=f"Document {doc_id} not found",
                        duration_ms=(datetime.now() - start_time).total_seconds() * 1000
                    )

                # Write back filtered data
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(json.dumps(filtered_data, cls=JSONEncoder, indent=2))

                duration = (datetime.now() - start_time).total_seconds() * 1000
                return DatabaseResult(
                    success=True,
                    duration_ms=duration,
                    source=str(file_path)
                )

        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return DatabaseResult(
                success=False,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    async def save_client_data(self, client_id: str, data: Dict) -> DatabaseResult:
        """Save client data to SQLite database."""
        start_time = datetime.now()
        try:
            async with self.semaphore:
                async with aiosqlite.connect(self.sqlite_path) as db:
                    await db.execute("""
                        INSERT OR REPLACE INTO clients (id, data, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """, (client_id, json.dumps(data, cls=JSONEncoder)))
                    await db.commit()

                duration = (datetime.now() - start_time).total_seconds() * 1000
                return DatabaseResult(
                    success=True,
                    value={"client_id": client_id},
                    duration_ms=duration
                )

        except Exception as e:
            self.logger.error(f"Error saving client data: {e}")
            return DatabaseResult(
                success=False,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    async def get_client_data(self, client_id: str) -> DatabaseResult:
        """Retrieve client data from SQLite database."""
        start_time = datetime.now()
        try:
            async with self.semaphore:
                async with aiosqlite.connect(self.sqlite_path) as db:
                    db.row_factory = aiosqlite.Row
                    async with db.execute(
                        "SELECT * FROM clients WHERE id = ?",
                        (client_id,)
                    ) as cursor:
                        row = await cursor.fetchone()
                        if not row:
                            return DatabaseResult(
                                success=False,
                                error=f"Client {client_id} not found",
                                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
                            )
                        
                        data = json.loads(row['data'])
                        data['created_at'] = row['created_at']
                        data['updated_at'] = row['updated_at']

                        duration = (datetime.now() - start_time).total_seconds() * 1000
                        return DatabaseResult(
                            success=True,
                            value=data,
                            duration_ms=duration
                        )

        except Exception as e:
            self.logger.error(f"Error retrieving client data: {e}")
            return DatabaseResult(
                success=False,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    async def cleanup_old_data(self) -> DatabaseResult:
        """Clean up old data and archive if necessary."""
        if not await self.cleanup_lock.acquire():
            return DatabaseResult(
                success=False,
                error="Cleanup already in progress"
            )

        try:
            start_time = datetime.now()
            threshold_date = datetime.now() - timedelta(days=90)
            
            async with aiosqlite.connect(self.sqlite_path) as db:
                # Archive old client data
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM clients WHERE updated_at < ?",
                    (threshold_date.isoformat(),)
                ) as cursor:
                    old_clients = await cursor.fetchall()
                    
                    for client in old_clients:
                        # Archive client data
                        archive_path = self.db_path / "archive" / f"client_{client['id']}_{datetime.now().strftime('%Y%m%d')}.json"
                        async with aiofiles.open(archive_path, 'w') as f:
                            await f.write(json.dumps({
                                'data': json.loads(client['data']),
                                'created_at': client['created_at'],
                                'updated_at': client['updated_at']
                            }, cls=JSONEncoder, indent=2))
                
                # Delete archived data
                await db.execute(
                    "DELETE FROM clients WHERE updated_at < ?",
                    (threshold_date.isoformat(),)
                )
                await db.commit()

            duration = (datetime.now() - start_time).total_seconds() * 1000
            return DatabaseResult(
                success=True,
                duration_ms=duration,
                details={'archived_clients': len(old_clients)}
            )

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return DatabaseResult(
                success=False,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        finally:
            self.cleanup_lock.release()

    async def get_stats(self) -> DatabaseResult:
        """Get database statistics."""
        start_time = datetime.now()
        try:
            stats = {
                'collections': {},
                'clients': {
                    'total': 0,
                    'active': 0
                },
                'storage': {
                    'total_size': 0,
                    'archive_size': 0
                }
            }

            # Collect JSON collection stats
            for file in self.db_path.glob('*.json'):
                stats['collections'][file.stem] = {
                    'size': file.stat().st_size,
                    'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                }

            # Collect SQLite stats
            async with aiosqlite.connect(self.sqlite_path) as db:
                async with db.execute("SELECT COUNT(*) FROM clients") as cursor:
                    stats['clients']['total'] = (await cursor.fetchone())[0]
                
                async with db.execute(
                    "SELECT COUNT(*) FROM clients WHERE updated_at >= ?",
                    ((datetime.now() - timedelta(days=30)).isoformat(),)
                ) as cursor:
                    stats['clients']['active'] = (await cursor.fetchone())[0]

            # Calculate storage stats
            stats['storage']['total_size'] = sum(
                f.stat().st_size for f in self.db_path.rglob('*') if f.is_file()
            )
            stats['storage']['archive_size'] = sum(
                f.stat().st_size for f in (self.db_path / "archive").rglob('*') if f.is_file()
            )

            duration = (datetime.now() - start_time).total_seconds() * 1000
            return DatabaseResult(
                success=True,
                value=stats,
                duration_ms=duration
            )

        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return DatabaseResult(
                success=False,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

# Global instance
db_service = EnhancedDatabaseService() 