"""
Enhanced caching service for the financial planning application.

This module provides advanced caching functionality with asynchronous concurrency,
partial success/failure tracking, robust error handling, extended logging, 
and improved flexibility in caching design.
"""

import os
import json
import pickle
import hashlib
import time
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable, TypeVar, List, Union, Tuple
from functools import wraps
from pathlib import Path
import logging
from dataclasses import dataclass

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from app.logger import logger

# Type variables for better type hinting
T = TypeVar('T')
K = TypeVar('K')

@dataclass
class CacheResult:
    """Structured result object for cache operations."""
    success: bool
    value: Optional[Any] = None
    error: Optional[str] = None
    ttl: Optional[int] = None
    hit: bool = False
    source: str = "cache"

class CacheService:
    """
    Advanced cache service with TTL support and memory management.
    Features:
    - TTL (Time To Live) support
    - Memory usage monitoring
    - Async operations
    - Cache statistics
    - Automatic cleanup
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        cleanup_interval: int = 300
    ):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
        self._setup_logging()
        self._start_cleanup_task()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("CacheService")

    def _start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired entries."""
        while True:
            try:
                await self._cleanup_expired()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        now = datetime.now()
        expired_keys = [
            key for key, value in self.cache.items()
            if value['expires_at'] <= now
        ]
        
        for key in expired_keys:
            await self.delete(key)
            self.stats['evictions'] += 1

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> CacheResult:
        """
        Set a value in the cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds (optional)
            
        Returns:
            CacheResult object
        """
        try:
            # Check cache size
            if len(self.cache) >= self.max_size:
                await self._evict_oldest()

            # Calculate expiration
            ttl = ttl if ttl is not None else self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)

            # Store value with metadata
            self.cache[key] = {
                'value': value,
                'created_at': datetime.now(),
                'expires_at': expires_at,
                'ttl': ttl
            }

            self.stats['size'] = len(self.cache)
            return CacheResult(success=True, value=value, ttl=ttl)

        except Exception as e:
            self.logger.error(f"Error setting cache value: {e}")
            return CacheResult(success=False, error=str(e))

    async def get(self, key: str) -> CacheResult:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            CacheResult object
        """
        try:
            if key not in self.cache:
                self.stats['misses'] += 1
                return CacheResult(success=False, error="Key not found", hit=False)

            entry = self.cache[key]
            
            # Check if expired
            if entry['expires_at'] <= datetime.now():
                await self.delete(key)
                self.stats['misses'] += 1
                return CacheResult(success=False, error="Key expired", hit=False)

            self.stats['hits'] += 1
            return CacheResult(
                success=True,
                value=entry['value'],
                ttl=entry['ttl'],
                hit=True
            )

        except Exception as e:
            self.logger.error(f"Error getting cache value: {e}")
            return CacheResult(success=False, error=str(e))

    async def delete(self, key: str) -> CacheResult:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            CacheResult object
        """
        try:
            if key in self.cache:
                del self.cache[key]
                self.stats['size'] = len(self.cache)
                return CacheResult(success=True)
            return CacheResult(success=False, error="Key not found")

        except Exception as e:
            self.logger.error(f"Error deleting cache value: {e}")
            return CacheResult(success=False, error=str(e))

    async def clear(self) -> CacheResult:
        """Clear all values from the cache."""
        try:
            self.cache.clear()
            self.stats['size'] = 0
            return CacheResult(success=True)

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return CacheResult(success=False, error=str(e))

    async def _evict_oldest(self) -> None:
        """Evict the oldest entry from the cache."""
        if not self.cache:
            return

        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]['created_at']
        )
        await self.delete(oldest_key)
        self.stats['evictions'] += 1

    async def get_stats(self) -> CacheResult:
        """Get cache statistics."""
        try:
            stats = {
                **self.stats,
                'hit_ratio': (
                    self.stats['hits'] /
                    (self.stats['hits'] + self.stats['misses'])
                    if (self.stats['hits'] + self.stats['misses']) > 0
                    else 0
                ),
                'memory_usage': len(json.dumps(self.cache))
            }
            return CacheResult(success=True, value=stats)

        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return CacheResult(success=False, error=str(e))

    async def get_keys(self) -> CacheResult:
        """Get all cache keys."""
        try:
            return CacheResult(success=True, value=list(self.cache.keys()))
        except Exception as e:
            self.logger.error(f"Error getting cache keys: {e}")
            return CacheResult(success=False, error=str(e))

    async def get_ttl(self, key: str) -> CacheResult:
        """
        Get the remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            CacheResult object with remaining TTL in seconds
        """
        try:
            if key not in self.cache:
                return CacheResult(success=False, error="Key not found")

            entry = self.cache[key]
            remaining = (entry['expires_at'] - datetime.now()).total_seconds()
            
            if remaining <= 0:
                await self.delete(key)
                return CacheResult(success=False, error="Key expired")

            return CacheResult(success=True, value=int(remaining))

        except Exception as e:
            self.logger.error(f"Error getting TTL: {e}")
            return CacheResult(success=False, error=str(e))

    async def update_ttl(self, key: str, ttl: int) -> CacheResult:
        """
        Update the TTL for a key.
        
        Args:
            key: Cache key
            ttl: New TTL in seconds
            
        Returns:
            CacheResult object
        """
        try:
            if key not in self.cache:
                return CacheResult(success=False, error="Key not found")

            entry = self.cache[key]
            entry['ttl'] = ttl
            entry['expires_at'] = datetime.now() + timedelta(seconds=ttl)
            
            return CacheResult(success=True, ttl=ttl)

        except Exception as e:
            self.logger.error(f"Error updating TTL: {e}")
            return CacheResult(success=False, error=str(e))


# Create a global instance for use throughout the application
cache_service = CacheService() 