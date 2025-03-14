#!/usr/bin/env python3
"""
Demo script for the enhanced database service.

This script demonstrates key features including:
- Asynchronous operations with concurrency control
- Partial success/failure tracking
- Enhanced error handling
- Data integrity validation
- Automatic archiving
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from app.services.enhanced_database import enhanced_db_service, DatabaseResult

async def demo_basic_operations():
    """Demonstrate basic CRUD operations with the enhanced service."""
    print("\n=== Basic Operations Demo ===")
    
    # Create a test client
    client_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "risk_profile": "moderate",
        "created_at": datetime.now().isoformat()
    }
    
    result = await enhanced_db_service.save_client(client_data)
    print(f"Save client result: success={result.success}, source={result.source}, duration={result.duration_ms:.2f}ms")
    
    if result.success:
        client_id = result.value
        # Retrieve the client
        get_result = await enhanced_db_service.get_client(client_id)
        print(f"Get client result: success={get_result.success}, source={get_result.source}, duration={get_result.duration_ms:.2f}ms")
        
        if get_result.success:
            print(f"Retrieved client: {get_result.value['name']}")

async def demo_concurrency():
    """Demonstrate concurrent operations with proper handling."""
    print("\n=== Concurrency Demo ===")
    
    async def save_client(index: int) -> DatabaseResult:
        client_data = {
            "name": f"Test Client {index}",
            "email": f"client{index}@example.com",
            "created_at": datetime.now().isoformat()
        }
        return await enhanced_db_service.save_client(client_data)
    
    # Create multiple clients concurrently
    tasks = [save_client(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    successes = sum(1 for r in results if r.success)
    avg_duration = sum(r.duration_ms for r in results) / len(results)
    
    print(f"Concurrent operations: {successes}/5 successful")
    print(f"Average operation duration: {avg_duration:.2f}ms")

async def demo_error_handling():
    """Demonstrate error handling and partial success tracking."""
    print("\n=== Error Handling Demo ===")
    
    # Try to get a non-existent client
    result = await enhanced_db_service.get_client("non_existent_id")
    print(f"Get non-existent client: success={result.success}, error='{result.error}'")
    
    # Try to save invalid data
    invalid_data = {
        "name": 123,  # Invalid type
        "email": "invalid@email",  # Invalid email
        "created_at": "invalid_date"  # Invalid date
    }
    
    result = await enhanced_db_service.save_client(invalid_data)
    print(f"Save invalid data: success={result.success}, partial_success={result.partial_success}")
    if result.error:
        print(f"Error details: {result.error}")

async def demo_cleanup():
    """Demonstrate automatic cleanup and archiving."""
    print("\n=== Cleanup Demo ===")
    
    # Create some old test data
    old_date = (datetime.now() - timedelta(days=100)).isoformat()
    old_client = {
        "name": "Old Client",
        "email": "old@example.com",
        "created_at": old_date,
        "updated_at": old_date
    }
    
    save_result = await enhanced_db_service.save_client(old_client)
    if save_result.success:
        # Trigger cleanup
        cleanup_result = await enhanced_db_service.cleanup_old_data(days_threshold=30)
        print(f"Cleanup result: success={cleanup_result.success}")
        if cleanup_result.success:
            stats = cleanup_result.value
            print(f"Archived files: {len(stats['archived'])}")
            print(f"Errors: {len(stats['errors'])}")

async def demo_stats():
    """Demonstrate database statistics collection."""
    print("\n=== Statistics Demo ===")
    
    stats_result = await enhanced_db_service.get_stats()
    if stats_result.success:
        stats = stats_result.value
        print("\nDatabase Statistics:")
        print(f"Collections: {list(stats['collections'].keys())}")
        print(f"Total SQLite rows: {sum(t['total_rows'] for t in stats['sqlite_tables'].values())}")
        print(f"Archived files: {stats['archive']['total_files']}")
        print(f"Archive size: {stats['archive']['size_bytes'] / 1024:.2f} KB")

async def main():
    """Run all demos sequentially."""
    print("Enhanced Database Service Demo")
    print("=" * 40)
    
    # Initialize the database service
    print("\nInitializing database service...")
    await enhanced_db_service.initialize()
    
    demos = [
        demo_basic_operations(),
        demo_concurrency(),
        demo_error_handling(),
        demo_cleanup(),
        demo_stats()
    ]
    
    for demo in demos:
        await demo
        print("\n" + "-" * 40)

if __name__ == "__main__":
    asyncio.run(main()) 