import os
import inspect
import importlib.util
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import Field

import ast
from app.tool.base import BaseTool
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.bash import Bash
from app.logger import logger

class ToolCreatorTool(BaseTool):
    """Meta-tool for creating and improving tools based on user requirements."""

    name: str = "tool_creator"
    description: str = "Analyzes user requests to create new tools or improve existing ones"
    parameters: dict = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "User request describing the tool they need"
            },
            "improve_existing": {
                "type": "boolean",
                "description": "Whether to improve an existing tool (true) or create a new one (false)"
            },
            "tool_name": {
                "type": "string",
                "description": "Name of the tool to improve (if improve_existing is true)"
            }
        },
        "required": ["request"]
    }
    
    tool_templates_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "tool_templates")
    editor: StrReplaceEditor = Field(default_factory=StrReplaceEditor)
    bash: Bash = Field(default_factory=Bash)
    
    def __init__(self):
        super().__init__()
        # Create templates directory if it doesn't exist
        if not self.tool_templates_dir.exists():
            self.tool_templates_dir.mkdir(parents=True)
            self._create_default_templates()

    def _create_default_templates(self):
        """Create default tool templates."""
        basic_tool_template = """import os
from typing import Dict, Any, Optional
from pydantic import Field

from app.tool.base import BaseTool

class {ToolName}(BaseTool):
    \"\"\"Tool for {tool_description}\"\"\"

    name: str = "{tool_name}"
    description: str = "{tool_description}"
    parameters: dict = {parameters}

    # Add any additional fields needed by the tool
    {additional_fields}

    async def execute(
        self,
        {parameters_args}
    ) -> Dict[str, Any]:
        \"\"\"Execute the tool with the given parameters.\"\"\"
        try:
            # Implement the tool's functionality here
            {tool_implementation}
            
            return {
                "status": "success",
                "message": "Operation completed successfully",
                # Add other result fields as needed
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
"""
        with open(self.tool_templates_dir / "basic_tool.py.template", "w") as f:
            f.write(basic_tool_template)

    def _get_existing_tools(self) -> Dict[str, str]:
        """Get a dictionary of existing tools with their file paths."""
        tools = {}
        tool_dir = Path(__file__).parent
        
        for file_path in tool_dir.glob("*.py"):
            if file_path.name in ["__init__.py", "base.py", "tool_collection.py"]:
                continue
                
            with open(file_path, "r") as f:
                content = f.read()
                
            # Parse the file to find tool classes
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if it's a Tool class (inherits from BaseTool)
                        if any(base.id == "BaseTool" for base in node.bases if isinstance(base, ast.Name)):
                            # Find the name attribute if it exists
                            for stmt in node.body:
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if isinstance(target, ast.Name) and target.id == "name":
                                            if isinstance(stmt.value, ast.Str):
                                                tools[stmt.value.s] = str(file_path)
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                
        return tools

    def _analyze_user_request(self, request: str) -> Dict[str, Any]:
        """Analyze the user request to determine tool requirements."""
        # This would ideally use AI/LLM to analyze the request in a real implementation
        # For now, we'll use a simplified approach based on keywords
        
        tool_info = {
            "name": "",
            "description": "",
            "parameters": {},
            "functionality": []
        }
        
        # Extract potential functionality from the request
        if "website" in request.lower() or "html" in request.lower():
            tool_info["functionality"].append("website_generation")
        
        if "search" in request.lower() or "find" in request.lower():
            tool_info["functionality"].append("search")
        
        if "analyze" in request.lower() or "report" in request.lower():
            tool_info["functionality"].append("analysis")
            
        if "finance" in request.lower() or "financial" in request.lower():
            tool_info["functionality"].append("financial")
            
        if "document" in request.lower() or "pdf" in request.lower():
            tool_info["functionality"].append("document_processing")
        
        # Extract potential name from the request
        words = request.lower().split()
        for i, word in enumerate(words):
            if word in ["tool", "generator", "creator", "analyzer", "processor"]:
                if i > 0:
                    tool_info["name"] = f"{words[i-1]}_{word}"
                    break
        
        # Default name if none found
        if not tool_info["name"] and tool_info["functionality"]:
            tool_info["name"] = f"{tool_info['functionality'][0]}_tool"
        
        # Create a description based on the request
        tool_info["description"] = f"Tool for {request.split('.')[0].lower()}"
        
        return tool_info

    def _generate_tool_code(self, tool_info: Dict[str, Any]) -> str:
        """Generate tool code from the template and tool info."""
        with open(self.tool_templates_dir / "basic_tool.py.template", "r") as f:
            template = f.read()
        
        # Format tool name for class
        tool_name_parts = tool_info["name"].split("_")
        class_name = "".join(part.capitalize() for part in tool_name_parts) + "Tool"
        
        # Basic parameter structure
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The input query or content to process"
                }
            },
            "required": ["query"]
        }
        
        # Add custom parameters based on functionality
        if "website_generation" in tool_info["functionality"]:
            parameters["properties"]["title"] = {
                "type": "string", 
                "description": "Title for the generated website"
            }
            parameters["properties"]["theme"] = {
                "type": "string", 
                "description": "Optional theme for the website"
            }
            parameters["required"].append("title")
        
        if "search" in tool_info["functionality"]:
            parameters["properties"]["max_results"] = {
                "type": "integer", 
                "description": "Maximum number of results to return"
            }
        
        # Format parameter arguments for the execute method
        param_args = []
        for param in parameters["required"]:
            param_args.append(f"{param}: str")
        
        for param, details in parameters["properties"].items():
            if param not in parameters["required"]:
                param_type = "str"
                if details["type"] == "integer":
                    param_type = "int"
                elif details["type"] == "boolean":
                    param_type = "bool"
                param_args.append(f"{param}: Optional[{param_type}] = None")
        
        param_args_str = ",\n        ".join(param_args)
        
        # Generate implementation code based on functionality
        implementation = []
        if "website_generation" in tool_info["functionality"]:
            implementation.append("# Generate a website from the input content")
            implementation.append("output_dir = f\"client_documents/websites/{title.lower().replace(' ', '_')}\"")
            implementation.append("os.makedirs(output_dir, exist_ok=True)")
            implementation.append("index_path = os.path.join(output_dir, 'index.html')")
            implementation.append("with open(index_path, 'w') as f:")
            implementation.append("    f.write(f\"<html><head><title>{title}</title></head><body>{query}</body></html>\")")
        
        if "search" in tool_info["functionality"]:
            implementation.append("# Implement search functionality")
            implementation.append("results = []")
            implementation.append("# Add search logic here")
            implementation.append("results.append({\"title\": \"Example result\", \"content\": \"This is a placeholder result.\"})")
        
        implementation_str = "\n            ".join(implementation) if implementation else "pass  # Implement tool functionality"
        
        # Additional fields based on functionality
        additional_fields = []
        if "website_generation" in tool_info["functionality"]:
            additional_fields.append("output_dir: str = Field(default=\"client_documents/websites\")")
        
        if "search" in tool_info["functionality"]:
            additional_fields.append("default_max_results: int = Field(default=10)")
        
        additional_fields_str = "\n    ".join(additional_fields)
        
        # Format the template
        tool_code = template.format(
            ToolName=class_name,
            tool_description=tool_info["description"],
            tool_name=tool_info["name"],
            parameters=json.dumps(parameters, indent=4),
            additional_fields=additional_fields_str,
            parameters_args=param_args_str,
            tool_implementation=implementation_str
        )
        
        return tool_code

    def _analyze_existing_tool(self, tool_name: str) -> Dict[str, Any]:
        """Analyze an existing tool for improvement opportunities."""
        tools = self._get_existing_tools()
        if tool_name not in tools:
            return {
                "found": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        file_path = tools[tool_name]
        with open(file_path, "r") as f:
            content = f.read()
        
        improvement_ideas = []
        
        # Simple code analysis to find improvement opportunities
        if "error" not in content.lower():
            improvement_ideas.append("Add better error handling with specific error types")
        
        if "progress" not in content.lower():
            improvement_ideas.append("Add progress reporting for long-running operations")
        
        if "async" not in content.lower():
            improvement_ideas.append("Consider making the tool asynchronous for better performance")
        
        return {
            "found": True,
            "file_path": file_path,
            "improvement_ideas": improvement_ideas
        }

    def _improve_tool(self, tool_name: str, request: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Improve an existing tool based on analysis and request."""
        if not analysis["found"]:
            return {
                "status": "error",
                "message": analysis["error"]
            }
        
        file_path = analysis["file_path"]
        with open(file_path, "r") as f:
            original_code = f.read()
        
        # For the initial implementation, let's focus on simpler improvements
        # We could use more sophisticated techniques (like AST transformation) for deeper changes
        
        improved_code = original_code
        
        # Add better error handling if suggested
        if "Add better error handling" in str(analysis["improvement_ideas"]):
            if "try:" in improved_code and "except Exception as e:" in improved_code:
                # Replace generic exception with more specific error handling
                improved_code = improved_code.replace(
                    "except Exception as e:", 
                    "except ValueError as e:\n            # Handle value errors\n            return {\"status\": \"error\", \"message\": f\"Invalid value: {str(e)}\"}\n        except IOError as e:\n            # Handle I/O errors\n            return {\"status\": \"error\", \"message\": f\"I/O error: {str(e)}\"}\n        except Exception as e:"
                )
        
        # Add progress reporting if suggested
        if "Add progress reporting" in str(analysis["improvement_ideas"]):
            if "async def execute" in improved_code:
                progress_code = "\n        # Report progress\n        progress = 0\n        # TODO: Update progress as the task progresses\n        # You could emit progress updates if this tool is used in a long-running operation\n"
                improved_code = improved_code.replace("async def execute", f"async def execute{progress_code}")
        
        # Write the improved code back to the file
        with open(file_path, "w") as f:
            f.write(improved_code)
        
        return {
            "status": "success",
            "message": f"Tool '{tool_name}' improved successfully",
            "improvements": analysis["improvement_ideas"]
        }

    async def execute(
        self,
        request: str,
        improve_existing: bool = False,
        tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the tool creator based on user request."""
        try:
            if improve_existing and tool_name:
                # Analyze and improve an existing tool
                analysis = self._analyze_existing_tool(tool_name)
                if not analysis["found"]:
                    return {
                        "status": "error",
                        "message": analysis["error"]
                    }
                
                result = self._improve_tool(tool_name, request, analysis)
                return result
            else:
                # Create a new tool based on the request
                tool_info = self._analyze_user_request(request)
                
                # Generate tool code
                tool_code = self._generate_tool_code(tool_info)
                
                # Determine filename
                filename = f"{tool_info['name']}.py"
                file_path = Path(__file__).parent / filename
                
                # Check if tool already exists
                if file_path.exists():
                    return {
                        "status": "error",
                        "message": f"Tool with name '{tool_info['name']}' already exists at {file_path}"
                    }
                
                # Write the tool code to file
                with open(file_path, "w") as f:
                    f.write(tool_code)
                
                # Update __init__.py to include the new tool
                init_path = Path(__file__).parent / "__init__.py"
                with open(init_path, "r") as f:
                    init_content = f.read()
                
                # Format class name
                tool_name_parts = tool_info["name"].split("_")
                class_name = "".join(part.capitalize() for part in tool_name_parts) + "Tool"
                
                # Add import statement
                import_statement = f"from app.tool.{tool_info['name']} import {class_name}"
                if import_statement not in init_content:
                    # Find the last import line
                    lines = init_content.split("\n")
                    last_import_idx = 0
                    for i, line in enumerate(lines):
                        if line.startswith("from") or line.startswith("import"):
                            last_import_idx = i
                    
                    # Insert new import after the last import
                    lines.insert(last_import_idx + 1, import_statement)
                    
                    # Add to __all__ list
                    all_start_idx = init_content.find("__all__ = [")
                    if all_start_idx != -1:
                        all_end_idx = init_content.find("]", all_start_idx)
                        if all_end_idx != -1:
                            # Check if there are any items in the list
                            if all_end_idx - all_start_idx > 11:  # 11 is the length of "__all__ = ["
                                # Add a comma after the last item
                                lines = "\n".join(lines)
                                all_list = lines[all_start_idx:all_end_idx]
                                last_item_end = all_list.rstrip().rfind('"')
                                if last_item_end != -1:
                                    lines = lines[:all_start_idx + last_item_end + 1] + ",\n    \"" + class_name + "\"" + lines[all_end_idx:]
                            else:
                                # Empty list, add the first item
                                lines = lines[:all_end_idx] + f'"{class_name}"' + lines[all_end_idx:]
                    
                    # Write updated content
                    with open(init_path, "w") as f:
                        f.write(lines if isinstance(lines, str) else "\n".join(lines))
                
                return {
                    "status": "success",
                    "message": f"Tool '{tool_info['name']}' created successfully",
                    "file_path": str(file_path),
                    "tool_info": tool_info,
                    "note": "Remember to restart the application or reload the tools to use the new tool"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating/improving tool: {str(e)}"
            } 