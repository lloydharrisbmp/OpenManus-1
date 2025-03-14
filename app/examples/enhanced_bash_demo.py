#!/usr/bin/env python3
"""
Demo script for the enhanced bash tool.

This script demonstrates key features including:
- Multiple concurrent sessions
- Background command execution
- Partial success/failure tracking
- Enhanced error handling and logging
- Session management
"""

import asyncio
import time
from typing import List

from app.tool.base import CLIResult
from app.tool.bash import Bash

async def demo_basic_operations(bash: Bash):
    """Demonstrate basic command execution."""
    print("\n=== Basic Operations Demo ===")
    
    result = await bash.execute("echo 'Hello, World!'")
    print(f"Simple command result: {result.output}")
    print(f"Duration: {result.duration_ms:.2f}ms")
    
    # Multiple commands in sequence
    commands = [
        "pwd",
        "ls -la",
        "date"
    ]
    
    for cmd in commands:
        result = await bash.execute(cmd)
        print(f"\nCommand: {cmd}")
        print(f"Output: {result.output}")
        print(f"Duration: {result.duration_ms:.2f}ms")

async def demo_concurrent_sessions(bash: Bash):
    """Demonstrate concurrent command execution."""
    print("\n=== Concurrent Sessions Demo ===")
    
    async def run_command(session_id: str, command: str) -> CLIResult:
        return await bash.execute(command, session_id=session_id)
    
    # Run multiple commands concurrently
    commands = [
        ("session1", "sleep 2; echo 'Session 1 done'"),
        ("session2", "sleep 1; echo 'Session 2 done'"),
        ("session3", "echo 'Session 3 done'")
    ]
    
    tasks = [run_command(sid, cmd) for sid, cmd in commands]
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    print(f"\nRan {len(commands)} commands concurrently in {total_time:.2f}s")
    for (sid, cmd), result in zip(commands, results):
        print(f"\nSession {sid} ({cmd}):")
        print(f"Output: {result.output}")
        print(f"Duration: {result.duration_ms:.2f}ms")

async def demo_background_execution(bash: Bash):
    """Demonstrate background command execution and log retrieval."""
    print("\n=== Background Execution Demo ===")
    
    session_id = "background_demo"
    
    # Start a long-running command in the background
    result = await bash.execute(
        "for i in {1..5}; do echo \"Step $i\"; sleep 1; done",
        session_id=session_id,
        background=True
    )
    print(f"Background command started: {result.output}")
    
    # Wait a bit and check logs
    await asyncio.sleep(3)
    logs = await bash.execute("", session_id=session_id, get_logs=True)
    print("\nPartial logs after 3 seconds:")
    print(logs.output)
    
    # Wait for completion and get final logs
    await asyncio.sleep(3)
    logs = await bash.execute("", session_id=session_id, get_logs=True)
    print("\nFinal logs:")
    print(logs.output)

async def demo_error_handling(bash: Bash):
    """Demonstrate error handling and partial success."""
    print("\n=== Error Handling Demo ===")
    
    # Command that doesn't exist
    result = await bash.execute("nonexistent_command")
    print(f"\nInvalid command error: {result.error}")
    print(f"Return code: {result.returncode}")
    
    # Command that times out
    try:
        result = await bash.execute("sleep 180")  # Should timeout after 120s
        print("This shouldn't be reached")
    except Exception as e:
        print(f"\nTimeout handling: {str(e)}")
    
    # Command with partial success
    result = await bash.execute("echo 'Starting'; nonexistent_command; echo 'Done'")
    print(f"\nPartial success command:")
    print(f"Output: {result.output}")
    print(f"Error: {result.error}")
    print(f"Partial success: {result.partial_success}")

async def demo_session_management(bash: Bash):
    """Demonstrate session management features."""
    print("\n=== Session Management Demo ===")
    
    session_id = "managed_session"
    
    # Start a session
    result = await bash.execute("echo 'Session started'", session_id=session_id)
    print(f"Initial command: {result.output}")
    
    # Use the same session
    result = await bash.execute("pwd", session_id=session_id)
    print(f"Working directory: {result.output}")
    
    # Restart the session
    result = await bash.execute(
        "echo 'Fresh session'",
        session_id=session_id,
        restart=True
    )
    print(f"\nAfter restart: {result.output}")
    
    # Clean up
    bash.cleanup()
    print("\nAll sessions cleaned up")

async def main():
    """Run all demos sequentially."""
    print("Enhanced Bash Tool Demo")
    print("=" * 40)
    
    # Create bash tool with concurrency limit of 3
    bash = Bash(concurrency_limit=3)
    
    demos = [
        demo_basic_operations(bash),
        demo_concurrent_sessions(bash),
        demo_background_execution(bash),
        demo_error_handling(bash),
        demo_session_management(bash)
    ]
    
    for demo in demos:
        await demo
        print("\n" + "-" * 40)
    
    # Final cleanup
    bash.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 