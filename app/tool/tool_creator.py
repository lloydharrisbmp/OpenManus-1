import os
import inspect
import importlib.util
import json
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import Field, BaseModel
from jinja2 import Environment, FileSystemLoader, select_autoescape

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from app.tool.base import BaseTool
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.bash import Bash
from app.logger import logger

class ToolUsageMetrics(BaseModel):
    """Tracks usage metrics for tools to enable continuous improvement."""
    calls: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    last_used: datetime = Field(default_factory=datetime.now)
    error_counts: Dict[str, int] = Field(default_factory=dict)
    improvement_history: List[Dict[str, Any]] = Field(default_factory=list)

class ToolAnalytics(BaseModel):
    """Analytics for tool performance and usage patterns."""
    usage_patterns: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)
    visualization_capabilities: Dict[str, List[str]] = Field(default_factory=dict)

class ToolCreatorTool(BaseTool):
    """Advanced meta-tool for creating, improving, and managing tools with continuous learning capabilities."""

    name: str = "tool_creator"
    description: str = "Creates, improves, and manages tools with advanced analytics and continuous learning"
    parameters: dict = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "User request describing the tool they need or improvement desired"
            },
            "improve_existing": {
                "type": "boolean",
                "description": "Whether to improve an existing tool (true) or create a new one (false)"
            },
            "tool_name": {
                "type": "string",
                "description": "Name of the tool to improve (if improve_existing is true)"
            },
            "visualization_required": {
                "type": "boolean",
                "description": "Whether the tool requires visualization capabilities"
            },
            "analytics_required": {
                "type": "boolean",
                "description": "Whether the tool requires analytics capabilities"
            }
        },
        "required": ["request"]
    }
    
    tool_templates_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "tool_templates")
    editor: StrReplaceEditor = Field(default_factory=StrReplaceEditor)
    bash: Bash = Field(default_factory=Bash)
    
    # New fields for enhanced functionality
    tool_metrics: Dict[str, ToolUsageMetrics] = Field(default_factory=dict)
    tool_analytics: ToolAnalytics = Field(default_factory=ToolAnalytics)
    metrics_file: Path = Field(default_factory=lambda: Path(__file__).parent / "tool_metrics.json")
    jinja_env: Environment = Field(default=None)
    
    def __init__(self):
        super().__init__()
        self._initialize_directories()
        self._load_metrics()
        
        # Initialize Jinja2 Environment for template rendering
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.tool_templates_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        self._initialize_templates()

    def _initialize_directories(self):
        """Initialize necessary directories for tool management."""
        dirs = [
            self.tool_templates_dir,
            self.tool_templates_dir / "visualization",
            self.tool_templates_dir / "analytics",
            self.tool_templates_dir / "ml",
            Path(__file__).parent / "tool_tests"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _load_metrics(self):
        """Load tool usage metrics from file."""
        if self.metrics_file.exists():
            try:
                with self._read_file(self.metrics_file) as data:
                    for tool_name, metrics in json.loads(data).items():
                        self.tool_metrics[tool_name] = ToolUsageMetrics(**metrics)
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")

    def _save_metrics(self):
        """Save tool usage metrics to file."""
        try:
            data = {name: metrics.dict() for name, metrics in self.tool_metrics.items()}
            self._write_file(self.metrics_file, json.dumps(data, default=str, indent=2))
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def _read_file(self, file_path: Path) -> str:
        """Helper method to read a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def _write_file(self, file_path: Path, content: str) -> None:
        """Helper method to write content to a file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise

    def _initialize_templates(self):
        """Initialize comprehensive tool templates using Jinja2."""
        templates = {
            "basic_tool.py.template": self._create_basic_template(),
            "visualization_tool.py.template": self._create_visualization_template(),
            "analytics_tool.py.template": self._create_analytics_template(),
            "ml_tool.py.template": self._create_ml_template()
        }
        
        for name, content in templates.items():
            path = self.tool_templates_dir / name
            if not path.exists():
                self._write_file(path, content)

    def _create_basic_template(self) -> str:
        """Create basic tool template."""
        return '''import os
from typing import Dict, Any, Optional
from pydantic import Field

from app.tool.base import BaseTool

class {{ ToolName }}(BaseTool):
    """
    Tool for {{ tool_description }}
    """

    name: str = "{{ tool_name }}"
    description: str = "{{ tool_description }}"
    parameters: dict = {{ parameters }}

    # Additional fields
    {{ additional_fields }}

    async def execute(
        self,
        {{ parameters_args }}
    ) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.
        """
        try:
            # Report progress
            progress = 0.0
            self._update_progress("Starting execution", progress)
            
            {{ tool_implementation | indent(12) }}
            
            self._update_progress("Execution completed", 1.0)
            return {
                "status": "success",
                "message": "Operation completed successfully",
            }
        except ValueError as e:
            # Handle value errors
            return {
                "status": "error",
                "message": f"Invalid value: {str(e)}"
            }
        except IOError as e:
            # Handle I/O errors
            return {
                "status": "error",
                "message": f"I/O error: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
            
    def _update_progress(self, message: str, percentage: float):
        """Update progress for long-running operations."""
        # This method can be overridden to report progress
        pass
'''

    def _create_visualization_template(self) -> str:
        """Create template for visualization-capable tools."""
        return '''import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pydantic import Field

from app.tool.base import BaseTool

class {{ ToolName }}(BaseTool):
    """
    Tool for {{ tool_description }} with visualization capabilities
    """
    
    name: str = "{{ tool_name }}"
    description: str = "{{ tool_description }}"
    parameters: dict = {{ parameters }}
    
    # Visualization settings
    plt_style: str = Field(default="seaborn")
    figure_size: Tuple[int, int] = Field(default=(10, 6))
    output_dir: str = Field(default="client_documents/visualizations")
    
    def setup_visualization(self):
        """Configure visualization settings."""
        plt.style.use(self.plt_style)
        plt.figure(figsize=self.figure_size)
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_visualization(self, filename: str = None):
        """Save current visualization to file."""
        if filename is None:
            filename = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        return output_path
    
    def _update_progress(self, message: str, percentage: float):
        """Update progress for long-running operations."""
        # This method can be overridden to report progress
        pass
    
    async def execute(
        self,
        {{ parameters_args }}
    ) -> Dict[str, Any]:
        """
        Execute the tool with visualization capabilities.
        """
        try:
            # Report progress
            self._update_progress("Setting up visualization", 0.1)
            self.setup_visualization()
            
            # Tool-specific visualization implementation
            {{ tool_implementation | indent(12) }}
            
            # Save the visualization
            self._update_progress("Saving visualization", 0.9)
            output_path = self.save_visualization()
            
            self._update_progress("Execution completed", 1.0)
            return {
                "status": "success",
                "message": "Visualization generated successfully",
                "visualization_path": output_path
            }
        except ValueError as e:
            # Handle value errors
            return {
                "status": "error",
                "message": f"Invalid value: {str(e)}"
            }
        except IOError as e:
            # Handle I/O errors
            return {
                "status": "error",
                "message": f"I/O error: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
'''

    def _create_analytics_template(self) -> str:
        """Create template for analytics-capable tools."""
        return '''import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import Field
from datetime import datetime

from app.tool.base import BaseTool

class {{ ToolName }}(BaseTool):
    """
    Tool for {{ tool_description }} with analytics capabilities
    """
    
    name: str = "{{ tool_name }}"
    description: str = "{{ tool_description }}"
    parameters: dict = {{ parameters }}
    
    # Analytics settings
    analysis_methods: List[str] = Field(default_factory=lambda: ["basic", "advanced"])
    output_dir: str = Field(default="client_documents/analytics")
    
    def perform_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform comprehensive data analysis."""
        results = {}
        
        if "basic" in self.analysis_methods:
            results["basic"] = self._basic_analysis(data)
        if "advanced" in self.analysis_methods:
            results["advanced"] = self._advanced_analysis(data)
            
        return results
    
    def _basic_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform basic statistical analysis."""
        # Basic analysis implementation
        if isinstance(data, pd.DataFrame):
            return {
                "summary": data.describe().to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "column_types": {col: str(dtype) for col, dtype in data.dtypes.items()}
            }
        return {"message": "Basic analysis requires pandas DataFrame"}
    
    def _advanced_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform advanced statistical analysis."""
        # Advanced analysis implementation
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=['number']).columns
            results = {}
            
            if len(numeric_cols) > 1:
                # Correlation analysis
                results["correlation"] = data[numeric_cols].corr().to_dict()
                
            return results
        return {"message": "Advanced analysis requires pandas DataFrame"}
    
    def _update_progress(self, message: str, percentage: float):
        """Update progress for long-running operations."""
        # This method can be overridden to report progress
        pass
    
    async def execute(
        self,
        {{ parameters_args }}
    ) -> Dict[str, Any]:
        """
        Execute the tool with analytics capabilities.
        """
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Report progress
            self._update_progress("Starting analysis", 0.1)
            
            # Tool-specific analytics implementation
            {{ tool_implementation | indent(12) }}
            
            self._update_progress("Execution completed", 1.0)
            return {
                "status": "success",
                "message": "Analysis completed successfully",
                "results": results
            }
        except ValueError as e:
            # Handle value errors
            return {
                "status": "error",
                "message": f"Invalid value: {str(e)}"
            }
        except IOError as e:
            # Handle I/O errors
            return {
                "status": "error",
                "message": f"I/O error: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
'''

    def _create_ml_template(self) -> str:
        """Create template for machine learning-capable tools."""
        return '''import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, List, Optional
from pydantic import Field
from datetime import datetime
from sklearn.base import BaseEstimator

from app.tool.base import BaseTool

class {{ ToolName }}(BaseTool):
    """
    Tool for {{ tool_description }} with machine learning capabilities
    """
    
    name: str = "{{ tool_name }}"
    description: str = "{{ tool_description }}"
    parameters: dict = {{ parameters }}
    
    # ML settings
    model: Optional[BaseEstimator] = None
    model_params: Dict[str, Any] = Field(default_factory=dict)
    output_dir: str = Field(default="client_documents/models")
    
    def train_model(self, X: np.ndarray, y: np.ndarray):
        """Train the machine learning model."""
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def save_model(self, filename: str = None):
        """Save the trained model to file."""
        if self.model is None:
            raise ValueError("No model to save")
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        if filename is None:
            filename = f"{self.name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        return output_path
    
    def load_model(self, filepath: str):
        """Load a trained model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
    
    def _update_progress(self, message: str, percentage: float):
        """Update progress for long-running operations."""
        # This method can be overridden to report progress
        pass
    
    async def execute(
        self,
        {{ parameters_args }}
    ) -> Dict[str, Any]:
        """
        Execute the tool with ML capabilities.
        """
        try:
            # Report progress
            self._update_progress("Initializing ML operation", 0.1)
            
            # Tool-specific ML implementation
            {{ tool_implementation | indent(12) }}
            
            self._update_progress("Execution completed", 1.0)
            return {
                "status": "success",
                "message": "ML operation completed successfully",
                "results": results
            }
        except ValueError as e:
            # Handle value errors
            return {
                "status": "error",
                "message": f"Invalid value: {str(e)}"
            }
        except IOError as e:
            # Handle I/O errors
            return {
                "status": "error",
                "message": f"I/O error: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
'''

    def _get_existing_tools(self) -> Dict[str, str]:
        """
        Get a dictionary of existing tools with their file paths.
        Uses AST parsing to detect classes that inherit from BaseTool.
        """
        tools = {}
        tool_dir = Path(__file__).parent
        
        for file_path in tool_dir.glob("*.py"):
            if file_path.name in ["__init__.py", "base.py", "tool_collection.py"]:
                continue
                
            try:
                content = self._read_file(file_path)
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if the class inherits from BaseTool (supports multiple inheritance)
                        inherits_base = any(
                            (isinstance(base, ast.Name) and base.id == "BaseTool") or
                            (isinstance(base, ast.Attribute) and base.attr == "BaseTool")
                            for base in node.bases
                        )
                        
                        if inherits_base:
                            # Extract the name attribute from the class definition
                            for stmt in node.body:
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if isinstance(target, ast.Name) and target.id == "name":
                                            if isinstance(stmt.value, ast.Str) or isinstance(stmt.value, ast.Constant):
                                                name_value = stmt.value.s if hasattr(stmt.value, 's') else stmt.value.value
                                                tools[name_value] = str(file_path)
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                
        return tools

    def _analyze_user_request(self, request: str) -> Dict[str, Any]:
        """
        Analyze the user request to determine tool requirements.
        Uses a comprehensive approach to extract functionality needs.
        """
        tool_info = {
            "name": "",
            "description": "",
            "parameters": {},
            "functionality": [],
            "class_name": ""
        }
        
        # Extract potential functionality from the request
        functionality_keywords = {
            "website_generation": ["website", "html", "web", "page"],
            "search": ["search", "find", "lookup", "query"],
            "analysis": ["analyze", "analyse", "report", "statistics", "metrics"],
            "financial": ["finance", "financial", "money", "investment", "stock", "market"],
            "document_processing": ["document", "pdf", "text", "file", "parse"],
            "visualization": ["visualize", "plot", "graph", "chart", "display"],
            "machine_learning": ["predict", "model", "train", "ml", "ai", "classify", "regression"]
        }
        
        lower_req = request.lower()
        for func, keywords in functionality_keywords.items():
            if any(keyword in lower_req for keyword in keywords):
                tool_info["functionality"].append(func)
        
        # Extract potential name from the request
        words = lower_req.split()
        for i, word in enumerate(words):
            if word in ["tool", "generator", "creator", "analyzer", "processor", "visualizer"]:
                if i > 0:
                    tool_info["name"] = f"{words[i-1]}_{word}"
                    break
        
        # Default name if none found
        if not tool_info["name"] and tool_info["functionality"]:
            tool_info["name"] = f"{tool_info['functionality'][0]}_tool"
        
        # Create a description based on the request
        tool_info["description"] = f"Tool for {request.split('.')[0].lower()}"
        
        # Format class name
        tool_name_parts = tool_info["name"].split("_")
        tool_info["class_name"] = "".join(part.capitalize() for part in tool_name_parts) + "Tool"
        
        return tool_info

    def _analyze_tool_requirements(self, request: str) -> Dict[str, Any]:
        """Advanced analysis of tool requirements from user request."""
        requirements = {
            "visualization_needed": False,
            "analytics_needed": False,
            "ml_needed": False,
            "complexity": "basic",
            "suggested_template": "basic_tool.py.template",
            "required_imports": set(),
            "suggested_parameters": {},
            "suggested_methods": []
        }
        
        # Analyze visualization needs
        viz_keywords = {"plot", "graph", "chart", "visualize", "display", "show", "draw"}
        if any(keyword in request.lower() for keyword in viz_keywords):
            requirements["visualization_needed"] = True
            requirements["suggested_template"] = "visualization_tool.py.template"
            requirements["required_imports"].update(["matplotlib.pyplot", "seaborn"])
            
        # Analyze analytics needs
        analytics_keywords = {"analyze", "calculate", "compute", "statistics", "metrics"}
        if any(keyword in request.lower() for keyword in analytics_keywords):
            requirements["analytics_needed"] = True
            if not requirements["visualization_needed"]:
                requirements["suggested_template"] = "analytics_tool.py.template"
            requirements["required_imports"].update(["pandas", "numpy"])
            
        # Analyze ML needs
        ml_keywords = {"predict", "classify", "cluster", "train", "model"}
        if any(keyword in request.lower() for keyword in ml_keywords):
            requirements["ml_needed"] = True
            requirements["suggested_template"] = "ml_tool.py.template"
            requirements["required_imports"].update(["sklearn"])
            
        # Determine complexity
        if sum([requirements["visualization_needed"], 
                requirements["analytics_needed"],
                requirements["ml_needed"]]) > 1:
            requirements["complexity"] = "advanced"
            
        return requirements

    def _generate_tool_code(self, tool_info: Dict[str, Any]) -> str:
        """
        Generate tool code using Jinja2 templates based on tool requirements.
        """
        # Select appropriate template based on requirements
        template_name = "basic_tool.py.template"
        if tool_info.get("visualization_needed"):
            template_name = "visualization_tool.py.template"
        elif tool_info.get("analytics_needed"):
            template_name = "analytics_tool.py.template"
        elif tool_info.get("ml_needed"):
            template_name = "ml_tool.py.template"
        
        # Define parameters structure
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
        
        if "visualization" in tool_info["functionality"] or tool_info.get("visualization_needed"):
            parameters["properties"]["output_format"] = {
                "type": "string",
                "description": "Format for the visualization output (png, pdf, svg)"
            }
            parameters["properties"]["title"] = {
                "type": "string",
                "description": "Title for the visualization"
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
            implementation.extend([
                "# Generate a website from the input content",
                "output_dir = f\"client_documents/websites/{title.lower().replace(' ', '_')}\"",
                "os.makedirs(output_dir, exist_ok=True)",
                "index_path = os.path.join(output_dir, 'index.html')",
                "with open(index_path, 'w') as f:",
                "    f.write(f\"<html><head><title>{title}</title></head><body>{query}</body></html>\")",
                "",
                "self._update_progress(\"Website generated\", 0.8)",
                "",
                "return {",
                "    \"status\": \"success\",",
                "    \"message\": \"Website generated successfully\",",
                "    \"file_url\": f\"file://{os.path.abspath(index_path)}\",",
                "    \"index_file\": index_path",
                "}"
            ])
        
        if "search" in tool_info["functionality"]:
            implementation.extend([
                "# Implement search functionality",
                "results = []",
                "max_to_return = max_results if max_results is not None else 10",
                "",
                "# Add search logic here",
                "results.append({\"title\": \"Example result\", \"content\": \"This is a placeholder result.\"})",
                "",
                "self._update_progress(\"Search completed\", 0.8)",
                "",
                "return {",
                "    \"status\": \"success\",",
                "    \"message\": f\"Found {len(results)} results\",",
                "    \"results\": results[:max_to_return]",
                "}"
            ])
        
        if "visualization" in tool_info["functionality"] or tool_info.get("visualization_needed"):
            implementation.extend([
                "# Create visualization",
                "plt.figure(figsize=self.figure_size)",
                "# Example: create a simple plot (replace with actual visualization code)",
                "x = np.linspace(0, 10, 100)",
                "y = np.sin(x)",
                "plt.plot(x, y)",
                "plt.title(title if title else 'Visualization')",
                "plt.xlabel('X axis')",
                "plt.ylabel('Y axis')",
                "",
                "# Save the visualization",
                "format_ext = output_format if output_format else 'png'",
                "filename = f\"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_ext}\"",
                "output_path = self.save_visualization(filename)",
                "",
                "return {",
                "    \"status\": \"success\",",
                "    \"message\": \"Visualization created successfully\",",
                "    \"visualization_path\": output_path,",
                "    \"file_url\": f\"file://{os.path.abspath(output_path)}\"",
                "}"
            ])
        
        if not implementation:
            implementation = ["pass  # Implement tool functionality"]
        
        implementation_str = "\n".join(implementation)
        
        # Additional fields based on functionality
        additional_fields = []
        if "website_generation" in tool_info["functionality"]:
            additional_fields.append('output_dir: str = Field(default="client_documents/websites")')
        
        if "search" in tool_info["functionality"]:
            additional_fields.append("default_max_results: int = Field(default=10)")
        
        additional_fields_str = "\n    ".join(additional_fields)
        
        # Render the tool code using the Jinja2 template
        template = self.jinja_env.get_template(template_name)
        tool_code = template.render(
            ToolName=tool_info["class_name"],
            tool_description=tool_info["description"],
            tool_name=tool_info["name"],
            parameters=json.dumps(parameters, indent=4),
            additional_fields=additional_fields_str,
            parameters_args=param_args_str,
            tool_implementation=implementation_str
        )
        
        return tool_code

    def _generate_tool_tests(self, tool_info: Dict[str, Any]) -> str:
        """Generate comprehensive test cases for the tool."""
        test_template = '''import pytest
import os
from {tool_module} import {tool_class}

def test_{tool_name}_initialization():
    """Test tool initialization."""
    tool = {tool_class}()
    assert tool.name == "{tool_name}"
    assert tool.description
    
def test_{tool_name}_parameters():
    """Test tool parameters validation."""
    tool = {tool_class}()
    params = tool.parameters
    assert isinstance(params, dict)
    assert "properties" in params
    assert "required" in params
    
def test_{tool_name}_execution():
    """Test tool execution."""
    tool = {tool_class}()
    # Add basic execution test
    result = await tool.execute(query="test")
    assert isinstance(result, dict)
    assert "status" in result
    
{additional_tests}
'''
        additional_tests = []
        
        if tool_info.get("visualization_needed"):
            additional_tests.append('''
@pytest.mark.asyncio
async def test_{tool_name}_visualization():
    """Test visualization capabilities."""
    tool = {tool_class}()
    result = await tool.execute(
        query="test visualization",
        title="Test Plot",
        output_format="png"
    )
    assert result["status"] == "success"
    assert "visualization_path" in result
    assert os.path.exists(result["visualization_path"])
''')
            
        if tool_info.get("analytics_needed"):
            additional_tests.append('''
@pytest.mark.asyncio
async def test_{tool_name}_analytics():
    """Test analytics capabilities."""
    tool = {tool_class}()
    result = await tool.execute(
        query="test analytics",
        analysis_type="basic"
    )
    assert result["status"] == "success"
    assert "results" in result
''')
            
        return test_template.format(
            tool_module=f"app.tool.{tool_info['name']}",
            tool_class=tool_info['class_name'],
            tool_name=tool_info['name'],
            additional_tests="\n".join(additional_tests)
        )

    def _update_init_file(self, tool_info: Dict[str, Any]) -> None:
        """Update __init__.py with new tool imports and registrations."""
        init_path = Path(__file__).parent / "__init__.py"
        
        try:
            content = self._read_file(init_path)
            
            # Add import statement
            import_statement = f"from app.tool.{tool_info['name']} import {tool_info['class_name']}"
            if import_statement not in content:
                lines = content.split("\n")
                
                # Find last import
                last_import_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith("from") or line.startswith("import"):
                        last_import_idx = i
                        
                # Insert new import
                lines.insert(last_import_idx + 1, import_statement)
                
                # Update __all__
                all_idx = content.find("__all__")
                if all_idx != -1:
                    bracket_idx = content.find("[", all_idx)
                    if bracket_idx != -1:
                        close_bracket_idx = content.find("]", bracket_idx)
                        if close_bracket_idx != -1:
                            current_all = content[bracket_idx+1:close_bracket_idx]
                            if current_all.strip():
                                new_all = current_all.rstrip() + f',\n    "{tool_info["class_name"]}"'
                            else:
                                new_all = f'"{tool_info["class_name"]}"'
                            content = content[:bracket_idx+1] + new_all + content[close_bracket_idx:]
                            
                # Write updated content
                self._write_file(init_path, "\n".join(lines))
                
        except Exception as e:
            logger.error(f"Error updating __init__.py: {e}")
            raise

    def _update_tool_metrics(self, tool_name: str, execution_time: float, success: bool, error_type: Optional[str] = None):
        """Update usage metrics for a tool."""
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = ToolUsageMetrics()
            
        metrics = self.tool_metrics[tool_name]
        metrics.calls += 1
        metrics.last_used = datetime.now()
        
        # Update success rate
        if success:
            metrics.success_rate = ((metrics.success_rate * (metrics.calls - 1)) + 1) / metrics.calls
        else:
            metrics.success_rate = (metrics.success_rate * (metrics.calls - 1)) / metrics.calls
            if error_type:
                metrics.error_counts[error_type] = metrics.error_counts.get(error_type, 0) + 1
                
        # Update average response time
        metrics.avg_response_time = ((metrics.avg_response_time * (metrics.calls - 1)) + execution_time) / metrics.calls
        
        self._save_metrics()

    def _suggest_improvements(self, tool_name: str) -> List[Dict[str, Any]]:
        """Suggest improvements based on tool usage metrics."""
        if tool_name not in self.tool_metrics:
            return []
            
        metrics = self.tool_metrics[tool_name]
        suggestions = []
        
        # Analyze error patterns
        if metrics.error_counts:
            most_common_error = max(metrics.error_counts.items(), key=lambda x: x[1])
            suggestions.append({
                "type": "error_handling",
                "description": f"Add specific handling for {most_common_error[0]} errors",
                "priority": "high" if most_common_error[1] > 5 else "medium"
            })
            
        # Analyze performance
        if metrics.avg_response_time > 1.0:  # More than 1 second
            suggestions.append({
                "type": "performance",
                "description": "Optimize tool execution for better performance",
                "priority": "high"
            })
            
        # Analyze success rate
        if metrics.success_rate < 0.95:  # Less than 95% success
            suggestions.append({
                "type": "reliability",
                "description": "Improve tool reliability and error handling",
                "priority": "high"
            })
            
        return suggestions

    async def execute(
        self,
        request: str,
        improve_existing: bool = False,
        tool_name: Optional[str] = None,
        visualization_required: bool = False,
        analytics_required: bool = False
    ) -> Dict[str, Any]:
        """Execute the enhanced tool creator."""
        start_time = datetime.now()
        try:
            if improve_existing and tool_name:
                # Analyze and improve existing tool
                analysis = self._analyze_existing_tool(tool_name)
                if not analysis["found"]:
                    return {"status": "error", "message": analysis["error"]}
                
                # Get improvement suggestions
                suggestions = self._suggest_improvements(tool_name)
                analysis["improvement_ideas"].extend([s["description"] for s in suggestions])
                
                # Improve the tool
                result = self._improve_tool(tool_name, request, analysis)
                
                # Update metrics
                execution_time = (datetime.now() - start_time).total_seconds()
                self._update_tool_metrics(tool_name, execution_time, True)
                
                return result
            else:
                # Analyze requirements for new tool
                requirements = self._analyze_tool_requirements(request)
                
                # Override requirements if specified
                if visualization_required:
                    requirements["visualization_needed"] = True
                if analytics_required:
                    requirements["analytics_needed"] = True
                
                # Create tool info
                tool_info = self._analyze_user_request(request)
                tool_info.update(requirements)
                
                # Generate tool code
                tool_code = self._generate_tool_code(tool_info)
                
                # Generate test cases
                test_code = self._generate_tool_tests(tool_info)
                
                # Save tool code
                filename = f"{tool_info['name']}.py"
                file_path = Path(__file__).parent / filename
                test_path = Path(__file__).parent / "tool_tests" / f"test_{filename}"
                
                self._write_file(file_path, tool_code)
                self._write_file(test_path, test_code)
                
                # Update __init__.py
                self._update_init_file(tool_info)
                
                # Initialize metrics for new tool
                execution_time = (datetime.now() - start_time).total_seconds()
                self._update_tool_metrics(tool_info["name"], execution_time, True)
                
                return {
                    "status": "success",
                    "message": f"Tool '{tool_info['name']}' created successfully",
                    "file_path": str(file_path),
                    "test_path": str(test_path),
                    "tool_info": tool_info,
                    "requirements": requirements,
                    "note": "Remember to restart the application or reload the tools to use the new tool"
                }
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            if tool_name:
                self._update_tool_metrics(tool_name, execution_time, False, str(type(e).__name__))
            return {
                "status": "error",
                "message": f"Error creating/improving tool: {str(e)}"
            } 