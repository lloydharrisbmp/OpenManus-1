"""
Enhanced database service for financial planning application.

This module provides an abstraction layer for database operations with:
- Asynchronous file I/O with concurrency control
- Partial success/failure tracking
- Enhanced error handling and logging
- Data integrity validation
- Automatic archiving of old data
"""

import os
import json
import uuid
import time
import asyncio
import aiofiles
import aiosqlite
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Union, TypeVar
from pathlib import Path

from app.logger import logger
from app.exceptions import DocumentProcessingError

T = TypeVar('T')

class DatabaseResult:
    """
    Structured result object for database operations with partial success tracking.
    """
    def __init__(
        self,
        success: bool,
        value: Any = None,
        error: Optional[str] = None,
        source: str = "unknown",  # "sqlite", "file", "memory"
        duration_ms: Optional[float] = None,
        partial_success: bool = False,
        details: Dict[str, Any] = None
    ):
        self.success = success
        self.value = value
        self.error = error
        self.source = source
        self.duration_ms = duration_ms
        self.partial_success = partial_success
        self.details = details or {}

    def __bool__(self):
        return self.success

class JSONEncoder(json.JSONEncoder):
    """Extended JSON encoder that handles dates and custom objects."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)

class EnhancedDatabaseService:
    """
    Enhanced database service for storing client and financial data.
    
    Features:
    - Asynchronous file I/O for scalability
    - Concurrency control for parallel operations
    - Partial success/failure tracking
    - Enhanced error handling and logging
    - Automatic archiving of old data
    """
    
    def __init__(
        self,
        db_path: str = "database",
        concurrency_limit: int = 3,
        cleanup_interval: int = 3600,  # 1 hour
        max_entries_per_collection: int = 10000
    ):
        """
        Initialize the database service.
        
        Args:
            db_path: Path to the database directory or file
            concurrency_limit: Maximum concurrent file/DB operations
            cleanup_interval: Seconds between automatic cleanups
            max_entries_per_collection: Maximum entries per collection before cleanup
        """
        self.db_path = db_path
        self.sqlite_path = os.path.join(db_path, "financial_planning.db")
        self.cleanup_interval = cleanup_interval
        self.max_entries_per_collection = max_entries_per_collection
        
        # Concurrency control
        self._file_sem = asyncio.Semaphore(concurrency_limit)
        self._sqlite_sem = asyncio.Semaphore(concurrency_limit)
        self._cleanup_lock = asyncio.Lock()
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Create subdirectories for different data types
        for subdir in ["clients", "portfolios", "reports", "analysis", "archive"]:
            os.makedirs(os.path.join(db_path, subdir), exist_ok=True)
        
        # Track last cleanup time
        self.last_cleanup = datetime.now()
        
        logger.info(f"[DatabaseService] Initialized with db_path='{db_path}', concurrency_limit={concurrency_limit}")

    async def initialize(self):
        """Initialize SQLite database with required tables."""
        await self._init_sqlite()

    async def _init_sqlite(self):
        """Initialize SQLite database with required tables asynchronously."""
        try:
            async with self._sqlite_sem:
                async with aiosqlite.connect(self.sqlite_path) as db:
                    await db.execute('''
                    CREATE TABLE IF NOT EXISTS clients (
                        client_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT,
                        risk_profile TEXT,
                        data TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    )
                    ''')
                    
                    await db.execute('''
                    CREATE TABLE IF NOT EXISTS portfolios (
                        portfolio_id TEXT PRIMARY KEY,
                        client_id TEXT,
                        name TEXT NOT NULL,
                        entity_type TEXT,
                        risk_profile TEXT,
                        data TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        FOREIGN KEY (client_id) REFERENCES clients (client_id)
                    )
                    ''')
                    
                    await db.execute('''
                    CREATE TABLE IF NOT EXISTS goals (
                        goal_id TEXT PRIMARY KEY,
                        client_id TEXT,
                        goal_type TEXT,
                        description TEXT,
                        target_amount REAL,
                        timeline_years INTEGER,
                        priority INTEGER,
                        current_progress REAL,
                        created_at TEXT,
                        updated_at TEXT,
                        FOREIGN KEY (client_id) REFERENCES clients (client_id)
                    )
                    ''')
                    
                    await db.execute('''
                    CREATE TABLE IF NOT EXISTS reports (
                        report_id TEXT PRIMARY KEY,
                        client_id TEXT,
                        title TEXT,
                        report_type TEXT,
                        file_path TEXT,
                        format TEXT,
                        created_at TEXT,
                        FOREIGN KEY (client_id) REFERENCES clients (client_id)
                    )
                    ''')
                    
                    await db.commit()
            
            logger.debug("[DatabaseService] SQLite tables initialized successfully")
        except Exception as e:
            logger.error(f"[DatabaseService] SQLite initialization error: {e}", exc_info=True)

    async def save_json(self, data: Dict, collection: str, doc_id: Optional[str] = None) -> DatabaseResult:
        """
        Save data as a JSON file asynchronously.
        
        Args:
            data: Data to save
            collection: Collection/directory name
            doc_id: Document ID (generated if not provided)
            
        Returns:
            DatabaseResult with operation outcome
        """
        start_time = datetime.now()
        doc_id = doc_id or str(uuid.uuid4())
        file_path = os.path.join(self.db_path, collection, f"{doc_id}.json")
        
        try:
            async with self._file_sem:
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(json.dumps(data, cls=JSONEncoder, indent=2))
            
            # Check if cleanup is needed
            await self._maybe_cleanup(collection)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return DatabaseResult(
                success=True,
                value=doc_id,
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[DatabaseService] Error saving JSON to {file_path}: {e}", exc_info=True)
            return DatabaseResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def load_json(self, collection: str, doc_id: str) -> DatabaseResult:
        """
        Load data from a JSON file asynchronously.
        
        Args:
            collection: Collection/directory name
            doc_id: Document ID
            
        Returns:
            DatabaseResult with loaded data
        """
        start_time = datetime.now()
        file_path = os.path.join(self.db_path, collection, f"{doc_id}.json")
        
        if not os.path.exists(file_path):
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return DatabaseResult(
                success=False,
                error="Document not found",
                source="file",
                duration_ms=duration
            )
        
        try:
            async with self._file_sem:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return DatabaseResult(
                success=True,
                value=data,
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[DatabaseService] Error loading JSON from {file_path}: {e}", exc_info=True)
            return DatabaseResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def delete_document(self, collection: str, doc_id: str) -> DatabaseResult:
        """
        Delete a document asynchronously.
        
        Args:
            collection: Collection/directory name
            doc_id: Document ID
            
        Returns:
            DatabaseResult indicating success/failure
        """
        start_time = datetime.now()
        file_path = os.path.join(self.db_path, collection, f"{doc_id}.json")
        
        if not os.path.exists(file_path):
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return DatabaseResult(
                success=False,
                error="Document not found",
                source="file",
                duration_ms=duration
            )
        
        try:
            async with self._file_sem:
                os.remove(file_path)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return DatabaseResult(
                success=True,
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[DatabaseService] Error deleting {file_path}: {e}", exc_info=True)
            return DatabaseResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def save_client(self, client_data: Dict) -> DatabaseResult:
        """
        Save client data to SQLite asynchronously.
        
        Args:
            client_data: Client data
            
        Returns:
            DatabaseResult with client ID
        """
        start_time = datetime.now()
        client_id = client_data.get('client_id', str(uuid.uuid4()))
        client_data['client_id'] = client_id
        now = datetime.now().isoformat()
        
        try:
            async with self._sqlite_sem:
                async with aiosqlite.connect(self.sqlite_path) as db:
                    # Check if client exists
                    async with db.execute(
                        "SELECT client_id FROM clients WHERE client_id = ?",
                        (client_id,)
                    ) as cursor:
                        exists = await cursor.fetchone() is not None
                    
                    if exists:
                        # Update
                        await db.execute(
                            """
                            UPDATE clients 
                            SET name = ?, email = ?, risk_profile = ?, data = ?, updated_at = ? 
                            WHERE client_id = ?
                            """,
                            (
                                client_data.get('name', ''),
                                client_data.get('email'),
                                client_data.get('risk_profile', 'moderate'),
                                json.dumps(client_data, cls=JSONEncoder),
                                now,
                                client_id
                            )
                        )
                    else:
                        # Insert
                        await db.execute(
                            """
                            INSERT INTO clients 
                            (client_id, name, email, risk_profile, data, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                client_id,
                                client_data.get('name', ''),
                                client_data.get('email'),
                                client_data.get('risk_profile', 'moderate'),
                                json.dumps(client_data, cls=JSONEncoder),
                                now,
                                now
                            )
                        )
                    
                    await db.commit()
            
            # Also save as JSON for backward compatibility
            json_result = await self.save_json(client_data, "clients", client_id)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return DatabaseResult(
                success=True,
                value=client_id,
                source="sqlite",
                duration_ms=duration,
                partial_success=not json_result.success,
                details={
                    "json_error": json_result.error if not json_result.success else None
                }
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[DatabaseService] SQLite error saving client: {e}", exc_info=True)
            
            # Fall back to JSON
            json_result = await self.save_json(client_data, "clients", client_id)
            if json_result.success:
                return DatabaseResult(
                    success=True,
                    value=client_id,
                    source="file",
                    duration_ms=duration,
                    partial_success=True,
                    details={"sqlite_error": str(e)}
                )
            else:
                return DatabaseResult(
                    success=False,
                    error=f"Both SQLite and JSON storage failed. SQLite: {str(e)}, JSON: {json_result.error}",
                    source="both",
                    duration_ms=duration
                )

    async def get_client(self, client_id: str) -> DatabaseResult:
        """
        Get client data from SQLite asynchronously.
        
        Args:
            client_id: Client ID
            
        Returns:
            DatabaseResult with client data
        """
        start_time = datetime.now()
        
        try:
            async with self._sqlite_sem:
                async with aiosqlite.connect(self.sqlite_path) as db:
                    async with db.execute(
                        "SELECT data FROM clients WHERE client_id = ?",
                        (client_id,)
                    ) as cursor:
                        row = await cursor.fetchone()
            
            if row:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                return DatabaseResult(
                    success=True,
                    value=json.loads(row[0]),
                    source="sqlite",
                    duration_ms=duration
                )
            
            # Try JSON fallback
            json_result = await self.load_json("clients", client_id)
            if json_result.success:
                return DatabaseResult(
                    success=True,
                    value=json_result.value,
                    source="file",
                    duration_ms=json_result.duration_ms,
                    partial_success=True,
                    details={"note": "Retrieved from JSON fallback"}
                )
            else:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                return DatabaseResult(
                    success=False,
                    error="Client not found in either SQLite or JSON storage",
                    source="both",
                    duration_ms=duration
                )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[DatabaseService] Error getting client {client_id}: {e}", exc_info=True)
            
            # Try JSON fallback
            json_result = await self.load_json("clients", client_id)
            if json_result.success:
                return DatabaseResult(
                    success=True,
                    value=json_result.value,
                    source="file",
                    duration_ms=duration,
                    partial_success=True,
                    details={"sqlite_error": str(e)}
                )
            else:
                return DatabaseResult(
                    success=False,
                    error=f"Both SQLite and JSON retrieval failed. SQLite: {str(e)}, JSON: {json_result.error}",
                    source="both",
                    duration_ms=duration
                )

    async def _maybe_cleanup(self, collection: str = None) -> None:
        """Check if cleanup is needed and run it if necessary."""
        now = datetime.now()
        if (now - self.last_cleanup).total_seconds() > self.cleanup_interval:
            if self._cleanup_lock.locked():
                return
            
            async with self._cleanup_lock:
                if (now - self.last_cleanup).total_seconds() > self.cleanup_interval:
                    await self.cleanup_old_data(collection)
                    self.last_cleanup = now

    async def cleanup_old_data(self, collection: str = None, days_threshold: int = 90) -> DatabaseResult:
        """
        Archive old data asynchronously.
        
        Args:
            collection: Optional collection to clean up (all if None)
            days_threshold: Number of days after which data is considered old
            
        Returns:
            DatabaseResult with cleanup statistics
        """
        start_time = datetime.now()
        archived_files = []
        errors = []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            collections = [collection] if collection else ["clients", "portfolios", "reports"]
            
            for col in collections:
                col_path = os.path.join(self.db_path, col)
                if not os.path.isdir(col_path):
                    continue
                
                files = os.listdir(col_path)
                tasks = []
                
                for fname in files:
                    if fname.endswith('.json'):
                        tasks.append(self._archive_if_old(col_path, fname, cutoff_date))
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, tuple):
                            success, info = result
                            if success:
                                archived_files.append(info)
                            else:
                                errors.append(info)
                        elif isinstance(result, Exception):
                            errors.append(str(result))
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return DatabaseResult(
                success=True,
                value={"archived": archived_files, "errors": errors},
                source="cleanup",
                duration_ms=duration,
                partial_success=bool(errors)
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[DatabaseService] Error during cleanup: {e}", exc_info=True)
            return DatabaseResult(
                success=False,
                error=str(e),
                source="cleanup",
                duration_ms=duration
            )

    async def _archive_if_old(self, dir_path: str, fname: str, cutoff_date: datetime) -> tuple[bool, str]:
        """Archive a single file if it's older than the cutoff date."""
        try:
            file_path = os.path.join(dir_path, fname)
            collection = os.path.basename(dir_path)
            
            async with self._file_sem:
                async with aiofiles.open(file_path, 'r') as f:
                    data = json.loads(await f.read())
                
                # Check updated_at or created_at
                date_str = data.get('updated_at') or data.get('created_at')
                if not date_str:
                    return True, f"Skipped {fname} (no date information)"
                
                file_date = datetime.fromisoformat(date_str)
                if file_date < cutoff_date:
                    archive_path = os.path.join(
                        self.db_path,
                        "archive",
                        collection,
                        file_date.strftime('%Y-%m'),
                        fname
                    )
                    
                    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                    os.rename(file_path, archive_path)
                    
                    return True, f"Archived {file_path} to {archive_path}"
                
                return True, f"Skipped {fname} (not old enough)"
        except Exception as e:
            return False, f"Error archiving {fname}: {str(e)}"

    async def get_stats(self) -> DatabaseResult:
        """
        Get database statistics.
        
        Returns:
            DatabaseResult with statistics about the database
        """
        start_time = datetime.now()
        try:
            stats = {
                "collections": {},
                "sqlite_tables": {},
                "archive": {
                    "total_files": 0,
                    "size_bytes": 0
                },
                "performance": {
                    "avg_query_time_ms": 0,
                    "total_archived": 0
                }
            }
            
            # Collect file stats
            for collection in ["clients", "portfolios", "reports"]:
                col_path = os.path.join(self.db_path, collection)
                if os.path.exists(col_path):
                    files = [f for f in os.listdir(col_path) if f.endswith('.json')]
                    stats["collections"][collection] = {
                        "total_files": len(files),
                        "size_bytes": sum(os.path.getsize(os.path.join(col_path, f)) for f in files)
                    }
            
            # Collect SQLite stats
            async with self._sqlite_sem:
                async with aiosqlite.connect(self.sqlite_path) as db:
                    for table in ["clients", "portfolios", "reports", "goals"]:
                        async with db.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
                            count = await cursor.fetchone()
                            stats["sqlite_tables"][table] = {
                                "total_rows": count[0] if count else 0
                            }
            
            # Collect archive stats
            archive_path = os.path.join(self.db_path, "archive")
            if os.path.exists(archive_path):
                for root, _, files in os.walk(archive_path):
                    stats["archive"]["total_files"] += len(files)
                    stats["archive"]["size_bytes"] += sum(
                        os.path.getsize(os.path.join(root, f))
                        for f in files
                    )
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return DatabaseResult(
                success=True,
                value=stats,
                source="stats",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[DatabaseService] Error getting stats: {e}", exc_info=True)
            return DatabaseResult(
                success=False,
                error=str(e),
                source="stats",
                duration_ms=duration
            )

# Global instance - needs to be initialized before use
enhanced_db_service = EnhancedDatabaseService() 