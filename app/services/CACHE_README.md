# Enhanced CacheService

## Overview

The Enhanced CacheService is a high-performance, concurrent caching solution for the financial planning application, designed to improve performance by efficiently storing and retrieving expensive operation results such as financial data requests or complex calculations.

This enhanced version adds several powerful features:

1. **Asynchronous Concurrency**: Non-blocking file I/O operations with controlled concurrency limits
2. **Partial Success/Failure Tracking**: Structured result objects with detailed error information
3. **Advanced TTL Options**: Support for different Time-To-Live formats including timedeltas and absolute datetimes
4. **Robust Error Handling**: Graceful degradation with detailed error reporting
5. **Automatic Cleanup**: Built-in garbage collection for expired cache entries
6. **Performance Monitoring**: Comprehensive statistics tracking for cache hits/misses

## Key Features

### 1. Asynchronous File I/O with Concurrency Control

The CacheService uses `aiofiles` for non-blocking file operations and manages concurrent access with a semaphore:

```python
async def file_get_async(self, key: str, category: str = "financial") -> CacheResult:
    async with self._io_sem:  # Limit concurrent file operations
        # Async file I/O logic
```

This ensures the cache can handle many concurrent operations without blocking the event loop, making it ideal for high-throughput applications.

### 2. Structured Result Tracking

Instead of returning raw values or None, the enhanced service returns `CacheResult` objects that provide detailed information about the operation:

```python
result = cache.get("my_key")
if result.success:
    value = result.value
else:
    error = result.error
    duration_ms = result.duration_ms  # Performance metrics
```

This makes error handling more robust and enables better debugging of cache-related issues.

### 3. Advanced TTL Formats

The service supports multiple TTL formats for more flexible cache expiration policies:

```python
# Seconds (integer)
cache.set("key1", value, ttl=60)  # Expire in 60 seconds

# Timedelta objects
cache.set("key2", value, ttl=timedelta(hours=1))  # Expire in 1 hour

# Absolute datetime
tomorrow = datetime.now() + timedelta(days=1)
cache.set("key3", value, ttl=tomorrow)  # Expire at specific time
```

### 4. Automatic Cache Cleanup

The service includes a smart cleanup mechanism to prevent cache bloat:

- Periodic cleanup of expired entries
- Limiting category sizes to prevent unbounded growth
- Least-recently-used (LRU) eviction when categories exceed limits

```python
# Configure cleanup behavior
cache = CacheService(
    max_entries_per_category=1000,  # Limit entries per category
    cleanup_interval=3600  # Check every hour
)
```

### 5. Comprehensive Statistics

Track cache performance with built-in statistics:

```python
stats = cache.get_stats()
print(f"Memory hit rate: {stats['memory_cache']['hit_rate']:.2%}")
print(f"File hit rate: {stats['file_cache']['hit_rate']:.2%}")
```

## Usage Examples

### Basic Usage

```python
from app.services.cache_service import cache_service

# In-memory caching
result = cache_service.get("my_key")
if not result.success:
    # Cache miss - compute value
    value = expensive_calculation()
    cache_service.set("my_key", value, ttl=3600)  # 1 hour TTL
else:
    value = result.value

# File-based caching
result = cache_service.file_get("complex_data", category="financial")
if result.success:
    process_data(result.value)
```

### Async Usage

```python
async def fetch_financial_data(ticker):
    key = f"financial_{ticker}"
    
    # Try to get from cache first
    result = await cache_service.file_get_async(key, category="market_data")
    if result.success:
        return result.value
        
    # Cache miss - fetch from external source
    data = await fetch_from_api(ticker)
    
    # Store in cache asynchronously
    await cache_service.file_set_async(key, data, category="market_data", ttl=3600)
    return data
```

### Decorator Usage

```python
# Sync function with in-memory caching
@cache_service.cached(ttl=timedelta(minutes=30), category="calculations")
def calculate_portfolio_returns(portfolio_id):
    # Expensive calculation...
    return result

# Async function with file-based caching
@cache_service.cached(ttl=3600, category="api_responses", use_file_cache=True, async_file_io=True)
async def fetch_market_data(ticker):
    # Expensive API call...
    return data
```

### Error Handling

```python
result = cache_service.file_set("key", complex_data)
if not result.success:
    logger.error(f"Failed to cache data: {result.error}")
    # Use fallback strategy or continue without caching
```

## Implementation Details

### CacheResult Class

The `CacheResult` class provides structured information about cache operations:

```python
class CacheResult:
    def __init__(self, 
                 success: bool, 
                 value: Any = None, 
                 error: Optional[Exception] = None, 
                 source: str = "unknown",
                 duration_ms: Optional[float] = None):
        self.success = success
        self.value = value
        self.error = error
        self.source = source  # "memory", "file", "function"
        self.duration_ms = duration_ms
    
    def __bool__(self):
        return self.success
```

This allows for more expressive code like:

```python
if result:  # Evaluates to True if result.success is True
    process(result.value)
else:
    handle_error(result.error)
```

### Concurrency Control

The service uses asyncio primitives to ensure safe concurrent access:

- `asyncio.Semaphore` limits the number of concurrent file operations
- `asyncio.Lock` prevents multiple simultaneous cleanup operations

### TTL Parsing

The `_parse_ttl` method handles different TTL formats:

```python
def _parse_ttl(self, ttl: Optional[Union[int, timedelta, datetime]] = None) -> float:
    if ttl is None:
        return time.time() + self.default_ttl
    
    if isinstance(ttl, int):
        return time.time() + ttl
    elif isinstance(ttl, timedelta):
        return time.time() + ttl.total_seconds()
    elif isinstance(ttl, datetime):
        return ttl.timestamp()
    else:
        logger.warning(f"Unknown TTL format: {type(ttl)}, using default")
        return time.time() + self.default_ttl
```

## Performance Considerations

- **Memory vs. File Cache**: In-memory caching is significantly faster but doesn't persist across application restarts
- **Concurrency Limit**: Default is 3, but can be adjusted based on system resources
- **Cleanup Interval**: Balance between keeping the cache tidy and the overhead of cleanup operations
- **Maximum Entries**: Consider memory constraints when setting `max_entries_per_category`

## Demo Script

A comprehensive demo script is available at `app/examples/cache_service_demo.py`, which showcases:

1. Basic caching operations
2. Advanced TTL handling
3. Asynchronous concurrency
4. Error handling
5. Cache cleanup
6. Decorator usage

Run the demo with:

```bash
python app/examples/cache_service_demo.py
```

## Future Enhancements

Potential improvements for future versions:

1. **Distributed Caching**: Integration with Redis or Memcached for multi-server deployments
2. **Cache Preloading**: Ability to preload commonly accessed data on startup
3. **Compression**: Option to compress large cached values
4. **Advanced Eviction Policies**: Additional policies like FIFO, LFU, etc.
5. **Observability**: Integration with application monitoring systems

## Requirements

- Python 3.7+
- `aiofiles` library (optional, but recommended for async file I/O) 