#!/usr/bin/env python3
"""
Simple demo script for the enhanced bash tool.
"""

import asyncio
import os
import signal
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ToolResult:
    """Base class for tool execution results."""
    system: Optional[str] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class CLIResult(ToolResult):
    """Result from a CLI command execution."""
    output: str = ""
    error: str = ""
    returncode: int = 0
    duration_ms: Optional[float] = None
    partial_success: bool = False

class BashSession:
    """A single interactive bash shell session."""

    def __init__(self, session_id: str, timeout: float = 120.0):
        self.session_id = session_id
        self._timeout = timeout
        self._started = False
        self._process: Optional[asyncio.subprocess.Process] = None
        self._output_delay = 0.1  # seconds to wait before reading buffers

    async def start(self):
        """Start the bash session."""
        if self._started:
            return
        
        self._process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        self._started = True

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            raise RuntimeError(f"Session {self.session_id} has not started.")
        
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass  # Process already terminated

    async def run(self, command: str) -> CLIResult:
        """Execute a command in the bash shell."""
        if not self._started:
            raise RuntimeError(f"Session {self.session_id} has not started.")
        
        if self._process is None or self._process.returncode is not None:
            return CLIResult(
                output="",
                error=f"bash session {self.session_id} is no longer active",
                returncode=-2
            )

        start_time = time.time()
        
        try:
            assert self._process.stdin
            self._process.stdin.write(command.encode() + b"\n")
            await self._process.stdin.drain()
            
            # Wait briefly for output
            await asyncio.sleep(self._output_delay)
            
            # Read output
            assert self._process.stdout and self._process.stderr
            stdout = await self._process.stdout.read()
            stderr = await self._process.stderr.read()
            
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return CLIResult(
                output=stdout.decode().strip(),
                error=stderr.decode().strip(),
                returncode=self._process.returncode or 0,
                duration_ms=duration
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return CLIResult(
                output="",
                error=str(e),
                returncode=-1,
                duration_ms=duration
            )

async def main():
    """Run a simple demo with the enhanced bash tool."""
    print("Enhanced Bash Tool Demo")
    print("=" * 40)
    
    # Create a bash session
    session = BashSession("demo_session")
    await session.start()
    
    try:
        # Run some basic commands
        commands = [
            "echo 'Hello, World!'",
            "pwd",
            "ls -la",
            "date"
        ]
        
        for cmd in commands:
            print(f"\nExecuting: {cmd}")
            result = await session.run(cmd)
            print(f"Output: {result.output}")
            if result.error:
                print(f"Error: {result.error}")
            print(f"Duration: {result.duration_ms:.2f}ms")
        
        # Try a command that doesn't exist
        print("\nTrying invalid command:")
        result = await session.run("nonexistent_command")
        print(f"Error: {result.error}")
        
        # Try a command with partial success
        print("\nTrying command with partial success:")
        result = await session.run("echo 'Starting'; nonexistent_cmd; echo 'Done'")
        print(f"Output: {result.output}")
        print(f"Error: {result.error}")
    
    finally:
        # Clean up
        session.stop()
        print("\nSession cleaned up")

if __name__ == "__main__":
    asyncio.run(main()) 