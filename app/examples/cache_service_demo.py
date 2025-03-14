#!/usr/bin/env python3
"""
Demonstration of the Enhanced CacheService capabilities.

This script showcases the advanced features of the CacheService including:
- Asynchronous concurrency with file operations
- Partial success/failure tracking
- Advanced TTL formats
- Cache cleanup and statistics
"""

import os
import sys
import time
import asyncio
import json
from datetime import datetime, timedelta
from pprint import pprint

# Add the parent directory to the path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.cache_service import CacheService
from app.logger import logger

# Configure more verbose logging for demonstration
logger.level("INFO")

async def demo_basic_caching():
    """Demonstrate basic in-memory and file caching."""
    print("\n=== Basic Caching Demo ===")
    
    # Create a custom cache service for this demo
    cache = CacheService(
        cache_dir="demo_cache",
        default_ttl=60,  # 60 seconds default TTL
        concurrency_limit=5
    )
    
    # Clear any existing data
    cache.clear()
    
    # In-memory caching demo
    print("\n1. In-memory Caching:")
    
    # Set a value
    key = "demo_key"
    value = {"name": "John", "age": 30, "data": [1, 2, 3, 4, 5]}
    result = cache.set(key, value)
    print(f"  Set result: {result.success} (duration: {result.duration_ms:.2f}ms)")
    
    # Get the value back
    result = cache.get(key)
    print(f"  Get result: {result.success} (duration: {result.duration_ms:.2f}ms)")
    print(f"  Retrieved value: {result.value}")
    
    # File caching demo
    print("\n2. File-based Caching:")
    
    # Set a value in file cache
    key = "file_demo_key"
    value = {"timestamp": datetime.now().isoformat(), "complex_data": [{"id": i, "value": f"item_{i}"} for i in range(5)]}
    result = cache.file_set(key, value, category="reports")
    print(f"  File set result: {result.success} (duration: {result.duration_ms:.2f}ms)")
    
    # Get the value back
    result = cache.file_get(key, category="reports")
    print(f"  File get result: {result.success} (duration: {result.duration_ms:.2f}ms)")
    print(f"  Retrieved value: {result.value}")
    
    # Check cache statistics
    stats = cache.get_stats()
    print("\nCache Statistics:")
    pprint(stats)
    
    return cache

async def demo_advanced_ttl():
    """Demonstrate advanced TTL handling with different formats."""
    print("\n=== Advanced TTL Demo ===")
    
    cache = CacheService(cache_dir="demo_cache")
    
    # 1. Integer TTL (seconds)
    print("\n1. Integer TTL (5 seconds):")
    result = cache.set("int_ttl", "This expires in 5 seconds", ttl=5)
    print(f"  Set with integer TTL: {result.success}")
    
    # 2. Timedelta TTL
    print("\n2. Timedelta TTL (10 seconds):")
    result = cache.set("delta_ttl", "This expires in 10 seconds", ttl=timedelta(seconds=10))
    print(f"  Set with timedelta TTL: {result.success}")
    
    # 3. Absolute datetime TTL
    future_time = datetime.now() + timedelta(seconds=15)
    print(f"\n3. Absolute Datetime TTL ({future_time.isoformat()}):")
    result = cache.set("dt_ttl", "This expires at a specific time", ttl=future_time)
    print(f"  Set with datetime TTL: {result.success}")
    
    # Verify all three values exist
    for key in ["int_ttl", "delta_ttl", "dt_ttl"]:
        result = cache.get(key)
        print(f"  Value for {key}: {result.value if result.success else 'Not found or expired'}")
    
    # Wait for the first one to expire
    print("\nWaiting 6 seconds for the first entry to expire...")
    await asyncio.sleep(6)
    
    # Check again
    print("\nAfter 6 seconds:")
    for key in ["int_ttl", "delta_ttl", "dt_ttl"]:
        result = cache.get(key)
        print(f"  Value for {key}: {result.value if result.success else 'Not found or expired'}")
    
    # Wait for all to expire
    print("\nWaiting 10 more seconds for all entries to expire...")
    await asyncio.sleep(10)
    
    # Check again
    print("\nAfter 16 seconds total:")
    for key in ["int_ttl", "delta_ttl", "dt_ttl"]:
        result = cache.get(key)
        print(f"  Value for {key}: {result.value if result.success else 'Not found or expired'}")
    
    return cache

async def demo_async_concurrency():
    """Demonstrate async concurrency with multiple file operations."""
    print("\n=== Async Concurrency Demo ===")
    
    # Create a cache with limited concurrency
    cache = CacheService(
        cache_dir="demo_cache",
        concurrency_limit=3,  # Only 3 concurrent file operations
    )
    
    # Function to simulate an expensive operation
    async def expensive_operation(index):
        await asyncio.sleep(1)  # Simulate work
        return f"Result for operation {index}"
    
    # Create a list of tasks to execute concurrently
    tasks = []
    
    # Perform 10 concurrent cache reads/writes
    print("\nPerforming 10 concurrent cache operations (with limit=3):")
    start_time = time.time()
    
    for i in range(10):
        key = f"concurrent_key_{i}"
        # This would normally be a cache.get() followed by expensive_operation() if not found
        # For demo purposes, we'll skip the get and just set new values
        result = await expensive_operation(i)
        
        # Use async file operations
        tasks.append(cache.file_set_async(key, result, category="concurrent_demo"))
    
    # Wait for all operations to complete
    results = await asyncio.gather(*tasks)
    
    # Summarize results
    duration = time.time() - start_time
    success_count = sum(1 for r in results if r.success)
    
    print(f"  Completed {len(tasks)} operations in {duration:.2f} seconds")
    print(f"  Successful operations: {success_count}/{len(tasks)}")
    print(f"  Average operation time: {sum(r.duration_ms for r in results) / len(results):.2f}ms")
    
    # Read back some values to verify
    print("\nReading back values:")
    for i in range(0, 10, 2):  # Just check a few
        key = f"concurrent_key_{i}"
        result = await cache.file_get_async(key, category="concurrent_demo")
        print(f"  Key {key}: {result.value if result.success else 'Failed'}")
    
    return cache

async def demo_error_handling():
    """Demonstrate error handling and partial success tracking."""
    print("\n=== Error Handling Demo ===")
    
    cache = CacheService(cache_dir="demo_cache")
    
    # Create a complex object that might fail to pickle
    class UnpickleableObject:
        def __init__(self, name):
            self.name = name
            
        # Prevent pickling
        def __getstate__(self):
            raise Exception("This object cannot be pickled!")
    
    print("\n1. Handling errors with in-memory cache:")
    try:
        # This should work fine with in-memory cache
        obj = UnpickleableObject("test")
        result = cache.set("unpickleable", obj)
        print(f"  Set result: {result.success} (error: {result.error})")
        
        # Reading should work since in-memory doesn't require pickling
        result = cache.get("unpickleable")
        print(f"  Get result: {result.success}")
        print(f"  Retrieved object name: {result.value.name if result.success else 'N/A'}")
    except Exception as e:
        print(f"  Unexpected error: {e}")
    
    print("\n2. Handling errors with file cache:")
    try:
        # This should fail when trying to pickle for file storage
        obj = UnpickleableObject("test2")
        result = cache.file_set("unpickleable_file", obj, category="error_demo")
        print(f"  File set result: {result.success} (error: {result.error})")
    except Exception as e:
        print(f"  Unexpected error: {e}")
    
    # Try with a valid object after a failure
    valid_obj = {"name": "This can be pickled", "data": [1, 2, 3]}
    result = cache.file_set("valid_file", valid_obj, category="error_demo")
    print(f"  Set valid object result: {result.success}")
    
    # Check cache statistics
    stats = cache.get_stats()
    print("\nCache Statistics (note the error count):")
    pprint(stats)
    
    return cache

async def demo_cleanup():
    """Demonstrate automatic cache cleanup functionality."""
    print("\n=== Cache Cleanup Demo ===")
    
    # Create a cache service with a low maximum entries limit and frequent cleanup
    cache = CacheService(
        cache_dir="demo_cache",
        max_entries_per_category=5,  # Only keep 5 entries per category
        cleanup_interval=1,  # Run cleanup every second for demo purposes
    )
    
    print("\n1. Creating many cache entries:")
    category = "cleanup_demo"
    
    # Create more entries than the maximum
    for i in range(10):
        key = f"cleanup_key_{i}"
        ttl = 30 if i < 3 else 3600  # First 3 with short TTL, rest with long
        value = {"index": i, "data": f"This is test data for entry {i}"}
        result = cache.file_set(key, value, category=category)
        print(f"  Set {key}: {result.success}")
    
    # List files before cleanup
    dir_path = os.path.join(cache.cache_dir, category)
    files_before = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    print(f"\nFiles before cleanup: {len(files_before)}")
    
    # Force an immediate cleanup
    print("\nRunning manual cleanup:")
    cleanup_result = await cache.cleanup_expired_files()
    print(f"  Cleanup result: {cleanup_result}")
    
    # List files after cleanup
    files_after = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    print(f"\nFiles after cleanup: {len(files_after)}")
    print(f"  Files removed: {len(files_before) - len(files_after)}")
    
    # Wait for short TTL entries to expire
    print("\nWaiting for short TTL entries to expire (5 seconds)...")
    await asyncio.sleep(5)
    
    # Force another cleanup
    print("\nRunning another cleanup:")
    cleanup_result = await cache.cleanup_expired_files()
    print(f"  Cleanup result: {cleanup_result}")
    
    # List files after second cleanup
    files_after_second = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    print(f"\nFiles after second cleanup: {len(files_after_second)}")
    print(f"  Files removed in second cleanup: {len(files_after) - len(files_after_second)}")
    
    return cache

@CacheService().cached(ttl=60, category="demo")
def demo_cached_function(a, b):
    """A function using the cached decorator."""
    print(f"  Computing expensive result for {a} + {b}...")
    time.sleep(1)  # Simulate expensive computation
    return a + b

@CacheService().cached(ttl=60, category="demo", use_file_cache=True)
async def demo_cached_async_function(a, b):
    """An async function using the cached decorator."""
    print(f"  Computing expensive async result for {a} * {b}...")
    await asyncio.sleep(1)  # Simulate expensive async computation
    return a * b

async def demo_decorators():
    """Demonstrate the cached decorator for functions."""
    print("\n=== Decorator Usage Demo ===")
    
    print("\n1. Cached synchronous function:")
    # First call - should execute the function
    print("First call:")
    result1 = demo_cached_function(5, 10)
    print(f"  Result: {result1}")
    
    # Second call - should use cache
    print("\nSecond call (should be cached):")
    result2 = demo_cached_function(5, 10)
    print(f"  Result: {result2}")
    
    # Different arguments - should execute the function
    print("\nDifferent arguments (should execute function):")
    result3 = demo_cached_function(7, 3)
    print(f"  Result: {result3}")
    
    print("\n2. Cached asynchronous function:")
    # First call - should execute the function
    print("First call:")
    result1 = await demo_cached_async_function(4, 5)
    print(f"  Result: {result1}")
    
    # Second call - should use cache
    print("\nSecond call (should be cached):")
    result2 = await demo_cached_async_function(4, 5)
    print(f"  Result: {result2}")
    
    # Different arguments - should execute the function
    print("\nDifferent arguments (should execute function):")
    result3 = await demo_cached_async_function(6, 7)
    print(f"  Result: {result3}")

async def main():
    """Run all the demo functions sequentially."""
    print("=== Enhanced CacheService Demonstration ===\n")
    print("This script showcases the advanced features of the CacheService")
    
    try:
        await demo_basic_caching()
        await demo_advanced_ttl()
        await demo_async_concurrency()
        await demo_error_handling()
        await demo_cleanup()
        await demo_decorators()
        
        print("\n=== Demo Summary ===")
        print("The enhanced CacheService successfully demonstrated:")
        print("1. Basic in-memory and file-based caching with CacheResult objects")
        print("2. Advanced TTL handling (integers, timedeltas, and absolute datetimes)")
        print("3. Asynchronous concurrency with controlled limits")
        print("4. Robust error handling and partial success tracking")
        print("5. Automatic cache cleanup and entry management")
        print("6. Decorator-based caching for regular and async functions")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up demo files
        print("\nCleaning up demo files...")
        cleanup_cache = CacheService(cache_dir="demo_cache")
        cleanup_cache.clear()
        try:
            import shutil
            shutil.rmtree("demo_cache")
            print("Demo cache directory removed.")
        except Exception as e:
            print(f"Note: Could not remove demo directory: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 