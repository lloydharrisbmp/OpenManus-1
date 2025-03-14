"""
Caching service for the financial planning application.

This module provides caching functionality to improve performance by storing
expensive operation results such as financial data requests or complex calculations.
"""

import os
import json
import pickle
import hashlib
from typing import Any, Dict, Optional, Callable, TypeVar, List, Union
import time
from datetime import datetime, timedelta
from functools import wraps
import asyncio
from pathlib import Path

from app.logger import logger

# Type variables for better type hinting
T = TypeVar('T')
K = TypeVar('K')

class CacheService:
    """
    Service for caching expensive operations like financial data retrieval.
    
    Supports in-memory and file-based caching with configurable expiration.
    """
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        """
        Initialize the cache service.
        
        Args:
            cache_dir: Directory to store cached files
            default_ttl: Default time-to-live for cache entries in seconds (1 hour default)
        """
        self.in_memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create subdirectories for different types of data
        for subdir in ["financial", "market", "reports", "portfolio"]:
            os.makedirs(os.path.join(self.cache_dir, subdir), exist_ok=True)
    
    def _get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key from function arguments.
        
        Args:
            prefix: Prefix for the cache key (usually function name)
            *args, **kwargs: Function arguments
            
        Returns:
            A unique cache key as string
        """
        # Convert args and kwargs to a stable string representation
        key_parts = [prefix]
        
        for arg in args:
            if isinstance(arg, (list, dict)):
                key_parts.append(str(sorted(str(arg).items())) if isinstance(arg, dict) else str(sorted(arg)))
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (list, dict)):
                key_parts.append(f"{k}:{str(sorted(str(v).items())) if isinstance(v, dict) else str(sorted(v))}")
            else:
                key_parts.append(f"{k}:{v}")
        
        # Generate a hash of the key parts
        key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
        return f"{prefix}_{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the in-memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key in self.in_memory_cache:
            entry = self.in_memory_cache[key]
            # Check if entry has expired
            if entry['expiry'] > time.time():
                return entry['value']
            else:
                # Remove expired entry
                del self.in_memory_cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the in-memory cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds, uses default_ttl if None
        """
        expiry = time.time() + (ttl if ttl is not None else self.default_ttl)
        self.in_memory_cache[key] = {
            'value': value,
            'expiry': expiry
        }
    
    def file_get(self, key: str, category: str = "financial") -> Optional[Any]:
        """
        Get a value from the file cache.
        
        Args:
            key: Cache key
            category: Category for organizing cache files
            
        Returns:
            Cached value or None if not found or expired
        """
        file_path = os.path.join(self.cache_dir, category, f"{key}.pkl")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check if entry has expired
                if entry['expiry'] > time.time():
                    return entry['value']
                else:
                    # Remove expired entry
                    os.remove(file_path)
            except (pickle.PickleError, OSError, EOFError) as e:
                logger.warning(f"Error reading cache file {file_path}: {e}")
        return None
    
    def file_set(self, key: str, value: Any, category: str = "financial", ttl: Optional[int] = None) -> None:
        """
        Set a value in the file cache.
        
        Args:
            key: Cache key
            value: Value to cache
            category: Category for organizing cache files
            ttl: Time-to-live in seconds, uses default_ttl if None
        """
        file_path = os.path.join(self.cache_dir, category, f"{key}.pkl")
        try:
            expiry = time.time() + (ttl if ttl is not None else self.default_ttl)
            entry = {
                'value': value,
                'expiry': expiry
            }
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
        except (pickle.PickleError, OSError) as e:
            logger.warning(f"Error writing to cache file {file_path}: {e}")
    
    def clear(self, category: Optional[str] = None) -> None:
        """
        Clear the cache.
        
        Args:
            category: If provided, only clear this category, otherwise clear all
        """
        # Clear in-memory cache
        self.in_memory_cache = {}
        
        # Clear file cache
        if category:
            dir_path = os.path.join(self.cache_dir, category)
            if os.path.exists(dir_path):
                for file_name in os.listdir(dir_path):
                    if file_name.endswith('.pkl'):
                        os.remove(os.path.join(dir_path, file_name))
        else:
            for category_dir in os.listdir(self.cache_dir):
                category_path = os.path.join(self.cache_dir, category_dir)
                if os.path.isdir(category_path):
                    for file_name in os.listdir(category_path):
                        if file_name.endswith('.pkl'):
                            os.remove(os.path.join(category_path, file_name))
    
    def cached(self, ttl: Optional[int] = None, category: str = "financial", use_file_cache: bool = False):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time-to-live in seconds
            category: Category for organizing cache files
            use_file_cache: Whether to use file-based cache
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._get_cache_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                if use_file_cache:
                    result = self.file_get(cache_key, category)
                else:
                    result = self.get(cache_key)
                
                if result is not None:
                    return result
                
                # Call the function and cache the result
                result = func(*args, **kwargs)
                
                if use_file_cache:
                    self.file_set(cache_key, result, category, ttl)
                else:
                    self.set(cache_key, result, ttl)
                
                return result
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._get_cache_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                if use_file_cache:
                    result = self.file_get(cache_key, category)
                else:
                    result = self.get(cache_key)
                
                if result is not None:
                    return result
                
                # Call the function and cache the result
                result = await func(*args, **kwargs)
                
                if use_file_cache:
                    self.file_set(cache_key, result, category, ttl)
                else:
                    self.set(cache_key, result, ttl)
                
                return result
            
            # Return appropriate wrapper based on whether the function is async
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return wrapper
        
        return decorator


# Create a global instance for use throughout the application
cache_service = CacheService() 