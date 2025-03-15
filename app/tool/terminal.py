import asyncio
import os
import re
import shlex
import shutil
from typing import Optional, Union, Dict, Any

from app.tool.base import BaseTool, CLIResult


class TerminalTool(BaseTool):
    """Tool for running terminal commands."""
    
    name: str = "terminal"
    description: str = "Execute a terminal command and return the output"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "(required) The CLI command to execute. This should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions.",
            }
        },
        "required": ["command"],
    }
    process: Optional[asyncio.subprocess.Process] = None
    current_path: str = os.getcwd()
    lock: asyncio.Lock = asyncio.Lock()

    async def execute(self, command: str) -> CLIResult:
        """
        Execute a terminal command asynchronously with persistent context.

        Args:
            command: The command to execute.

        Returns:
            CLIResult: The result of the command execution.
        """
        async with self.lock:
            # Handle cd commands specially to maintain directory state
            if command.strip().startswith("cd "):
                return await self._handle_cd_command(command)

            # Sanitize the command
            command = self._sanitize_command(command)

            try:
                # Create subprocess
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.current_path,
                    shell=True,
                )

                # Initialize result here to ensure it's defined
                result = CLIResult(exit_code=-1, output="Command timed out")
                
                try:
                    # Wait for the process to complete with timeout
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=600
                    )
                    
                    # Decode the output
                    stdout_str = stdout.decode("utf-8", errors="replace")
                    stderr_str = stderr.decode("utf-8", errors="replace")
                    
                    # Create the result
                    result = CLIResult(
                        exit_code=process.returncode,
                        output=stdout_str + (
                            f"\nError: {stderr_str}" if stderr_str.strip() else ""
                        ),
                    )
                except asyncio.TimeoutError:
                    # Handle timeout by killing the process
                    process.kill()
                    result = CLIResult(exit_code=-1, output="Command timed out")
                
                return result
            
            except Exception as e:
                return CLIResult(exit_code=-1, output=f"Error executing command: {str(e)}")

    async def execute_in_env(self, env_name: str, command: str) -> CLIResult:
        """
        Execute a terminal command asynchronously within a specified Conda environment.

        Args:
            env_name (str): The name of the Conda environment.
            command (str): The terminal command to execute within the environment.

        Returns:
            str: The output, and error of the command execution.
        """
        sanitized_command = self._sanitize_command(command)

        # Construct the command to run within the Conda environment
        # Using 'conda run -n env_name command' to execute without activating
        conda_command = f"conda run -n {shlex.quote(env_name)} {sanitized_command}"

        return await self.execute(conda_command)

    async def _handle_cd_command(self, command: str) -> CLIResult:
        """
        Handle 'cd' commands to change the current path.

        Args:
            command (str): The 'cd' command to process.

        Returns:
            TerminalOutput: The result of the 'cd' command.
        """
        try:
            parts = shlex.split(command)
            if len(parts) < 2:
                new_path = os.path.expanduser("~")
            else:
                new_path = os.path.expanduser(parts[1])

            # Handle relative paths
            if not os.path.isabs(new_path):
                new_path = os.path.join(self.current_path, new_path)

            new_path = os.path.abspath(new_path)

            if os.path.isdir(new_path):
                self.current_path = new_path
                return CLIResult(
                    output=f"Changed directory to {self.current_path}", error=""
                )
            else:
                return CLIResult(output="", error=f"No such directory: {new_path}")
        except Exception as e:
            return CLIResult(output="", error=str(e))

    @staticmethod
    def _sanitize_command(command: str) -> str:
        """
        Sanitize the command for safe execution.

        Args:
            command (str): The command to sanitize.

        Returns:
            str: The sanitized command.
        """
        # Example sanitization: restrict certain dangerous commands
        dangerous_commands = ["rm", "sudo", "shutdown", "reboot"]
        try:
            parts = shlex.split(command)
            if any(cmd in dangerous_commands for cmd in parts):
                raise ValueError("Use of dangerous commands is restricted.")
        except Exception:
            # If shlex.split fails, try basic string comparison
            if any(cmd in command for cmd in dangerous_commands):
                raise ValueError("Use of dangerous commands is restricted.")

        # Additional sanitization logic can be added here
        return command

    async def close(self):
        """Close the persistent shell process if it exists."""
        async with self.lock:
            if self.process:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                finally:
                    self.process = None

    async def __aenter__(self):
        """Enter the asynchronous context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the asynchronous context manager and close the process."""
        await self.close()
