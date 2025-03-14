#!/usr/bin/env python3

"""
Demo script for the Enhanced ComplianceService.

This script demonstrates the key features of the enhanced ComplianceService including:
- Asynchronous file I/O with concurrency control
- Partial success/failure tracking
- Advanced TTL handling
- Automatic cleanup
- Comprehensive statistics
"""

import os
import sys
import json
import asyncio
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.services.compliance_service import (
    ComplianceService,
    ComplianceResult,
    ComplianceRule,
    ComplianceCheck,
    ComplianceReport
)
from loguru import logger

async def demo_basic_caching():
    """Demonstrate basic in-memory and file caching."""
    logger.info("=== Demonstrating Basic Caching ===")
    
    service = ComplianceService(data_dir="demo_compliance", concurrency_limit=3)
    
    # Create a compliance check
    check_result = await service.create_compliance_check_async(
        tenant_id="demo_tenant",
        rule_ids=["ASIC_FSG_001"],  # This rule was created in _initialize_default_rules
        performed_by="demo_user",
        status="pending"
    )
    
    if check_result.success:
        logger.info(f"Created compliance check: {check_result.value}")
        logger.info(f"Operation took {check_result.duration_ms:.2f}ms")
    else:
        logger.error(f"Failed to create check: {check_result.error}")

async def demo_advanced_ttl():
    """Demonstrate advanced TTL handling."""
    logger.info("\n=== Demonstrating Advanced TTL Handling ===")
    
    service = ComplianceService(data_dir="demo_compliance")
    
    # Create checks with different TTL formats
    checks = []
    ttl_formats = [
        (30, "30 seconds"),
        (timedelta(minutes=5), "5 minutes"),
        (datetime.now() + timedelta(hours=1), "1 hour from now")
    ]
    
    for ttl, desc in ttl_formats:
        result = await service.create_compliance_check_async(
            tenant_id="demo_tenant",
            rule_ids=["ATO_SMSF_001"],
            performed_by="demo_user",
            status="completed",
            expiry_date=ttl
        )
        if result.success:
            checks.append((result.value, desc))
            logger.info(f"Created check with TTL {desc}: {result.value}")

async def demo_async_concurrency():
    """Demonstrate async concurrency with multiple file operations."""
    logger.info("\n=== Demonstrating Async Concurrency ===")
    
    service = ComplianceService(data_dir="demo_compliance", concurrency_limit=3)
    
    # Simulate multiple concurrent operations
    async def simulate_expensive_operation(i: int) -> ComplianceResult:
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate varying load
        return await service.create_compliance_check_async(
            tenant_id=f"demo_tenant_{i}",
            rule_ids=["AFCA_001"],
            performed_by="demo_user",
            status="pending"
        )
    
    # Run 10 operations concurrently
    tasks = [simulate_expensive_operation(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    successes = sum(1 for r in results if r.success)
    failures = sum(1 for r in results if not r.success)
    avg_duration = sum(r.duration_ms for r in results) / len(results)
    
    logger.info(f"Completed {len(results)} concurrent operations:")
    logger.info(f"Successes: {successes}")
    logger.info(f"Failures: {failures}")
    logger.info(f"Average duration: {avg_duration:.2f}ms")

async def demo_error_handling():
    """Demonstrate error handling and partial success tracking."""
    logger.info("\n=== Demonstrating Error Handling ===")
    
    service = ComplianceService(data_dir="demo_compliance")
    
    # Create a check with valid and invalid data
    check_result = await service.create_compliance_check_async(
        tenant_id="demo_tenant",
        rule_ids=["PRIVACY_001"],
        performed_by="demo_user",
        status="completed",
        results={
            "PRIVACY_001": {
                "compliant": True,
                "items": [
                    {"id": 1, "compliant": True},
                    {"id": 2, "compliant": False},
                    {"id": "invalid", "compliant": None}  # This will cause partial success
                ]
            }
        }
    )
    
    if check_result.success:
        logger.info("Check created successfully")
        if check_result.partial_success:
            logger.warning("Some items had validation errors")
            logger.warning(f"Details: {check_result.details}")
    else:
        logger.error(f"Failed to create check: {check_result.error}")

async def demo_cleanup():
    """Demonstrate automatic cleanup functionality."""
    logger.info("\n=== Demonstrating Cleanup ===")
    
    service = ComplianceService(
        data_dir="demo_compliance",
        cleanup_interval=5  # Set short interval for demo
    )
    
    # Create multiple checks
    old_date = datetime.now() - timedelta(days=100)
    
    # Create some old checks
    for i in range(5):
        await service.create_compliance_check_async(
            tenant_id="demo_tenant",
            rule_ids=["AML_CTF_001"],
            performed_by="demo_user",
            status="completed",
            performed_at=old_date
        )
    
    # Run cleanup
    cleanup_result = await service.cleanup_old_data(days_threshold=30)
    
    if cleanup_result.success:
        logger.info("Cleanup completed successfully")
        logger.info(f"Archived files: {len(cleanup_result.value['archived'])}")
        if cleanup_result.value['errors']:
            logger.warning(f"Some errors occurred: {cleanup_result.value['errors']}")
    else:
        logger.error(f"Cleanup failed: {cleanup_result.error}")

@service.cached(ttl=300)  # Cache for 5 minutes
def demo_cached_function(param: str) -> Dict[str, Any]:
    """Example of a cached synchronous function."""
    return {"result": f"Processed {param}", "timestamp": datetime.now().isoformat()}

@service.cached(ttl=300)
async def demo_cached_async_function(param: str) -> Dict[str, Any]:
    """Example of a cached asynchronous function."""
    await asyncio.sleep(1)  # Simulate async work
    return {"result": f"Async processed {param}", "timestamp": datetime.now().isoformat()}

async def demo_decorators():
    """Demonstrate the usage of the cached decorator."""
    logger.info("\n=== Demonstrating Cached Decorators ===")
    
    # Test sync function
    result1 = demo_cached_function("test1")
    logger.info(f"First call result: {result1}")
    
    result2 = demo_cached_function("test1")  # Should return cached result
    logger.info(f"Second call result (should be cached): {result2}")
    
    # Test async function
    result3 = await demo_cached_async_function("test2")
    logger.info(f"First async call result: {result3}")
    
    result4 = await demo_cached_async_function("test2")  # Should return cached result
    logger.info(f"Second async call result (should be cached): {result4}")

async def main():
    """Run all demos sequentially."""
    logger.info("Starting ComplianceService Demo")
    
    # Ensure clean state
    if os.path.exists("demo_compliance"):
        import shutil
        shutil.rmtree("demo_compliance")
    
    # Run demos
    await demo_basic_caching()
    await demo_advanced_ttl()
    await demo_async_concurrency()
    await demo_error_handling()
    await demo_cleanup()
    await demo_decorators()
    
    logger.info("\nDemo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 