"""
Enhanced Report Generator Tool for creating professional financial and analytical reports.
Supports multiple formats, templating, and data visualization.
"""

import os
import json
import logging
from typing import Dict, List, Union, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, Template
import pdfkit
import markdown2
from .base import BaseTool

class ReportGeneratorTool(BaseTool):
    """
    Advanced Report Generator Tool for creating professional reports with data visualization
    and multiple output formats (PDF, HTML, Markdown).
    """

    def __init__(self):
        super().__init__()
        self.name = "ReportGeneratorTool"
        self.description = "Generates professional reports with data visualization"
        self.template_dir = Path(__file__).parent / "report_templates"
        self.output_dir = Path("generated_reports")
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._setup_directories()
        self._setup_logging()

    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different report types
        (self.output_dir / "pdf").mkdir(exist_ok=True)
        (self.output_dir / "html").mkdir(exist_ok=True)
        (self.output_dir / "markdown").mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)

    def _setup_logging(self) -> None:
        """Configure logging for the tool."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.name)

    def _validate_data(self, data: Dict) -> bool:
        """
        Validate the report data structure.
        
        Args:
            data: Dictionary containing report data
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValueError: If data is invalid
        """
        required_fields = ['title', 'sections']
        if not all(field in data for field in required_fields):
            missing = [f for f in required_fields if f not in data]
            raise ValueError(f"Missing required fields in data: {missing}")
            
        if not isinstance(data['sections'], list):
            raise ValueError("'sections' must be a list of section configurations")
            
        for section in data['sections']:
            if 'title' not in section or 'content' not in section:
                raise ValueError("Each section must have 'title' and 'content' fields")
                
        return True

    def _create_visualization(self, data: Dict, viz_type: str) -> str:
        """
        Create data visualization based on the specified type.
        
        Args:
            data: Dictionary containing visualization data
            viz_type: Type of visualization to create
            
        Returns:
            str: Path to the generated visualization image
        """
        plt.figure(figsize=(10, 6))
        
        if viz_type == 'line':
            df = pd.DataFrame(data['data'])
            sns.lineplot(data=df, x=data['x'], y=data['y'])
        elif viz_type == 'bar':
            df = pd.DataFrame(data['data'])
            sns.barplot(data=df, x=data['x'], y=data['y'])
        elif viz_type == 'pie':
            plt.pie(data['values'], labels=data['labels'], autopct='%1.1f%%')
        elif viz_type == 'scatter':
            df = pd.DataFrame(data['data'])
            sns.scatterplot(data=df, x=data['x'], y=data['y'])
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
            
        plt.title(data.get('title', ''))
        plt.xlabel(data.get('xlabel', ''))
        plt.ylabel(data.get('ylabel', ''))
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.output_dir / "images" / f"viz_{timestamp}.png"
        plt.savefig(image_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(image_path)

    def _process_markdown(self, content: str) -> str:
        """
        Process markdown content to HTML.
        
        Args:
            content: Markdown content string
            
        Returns:
            str: Processed HTML content
        """
        return markdown2.markdown(
            content,
            extras=['tables', 'fenced-code-blocks', 'footnotes']
        )

    async def generate_report(
        self,
        data: Dict,
        output_format: str = 'pdf',
        template_name: str = 'default.html'
    ) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Generate a report based on the provided data and format.
        
        Args:
            data: Report data dictionary containing:
                 - title: Report title
                 - sections: List of section configurations
                 - visualizations: Optional list of visualization configs
            output_format: Output format ('pdf', 'html', or 'markdown')
            template_name: Template name to use
            
        Returns:
            Dict containing:
                - success: Boolean indicating success
                - output_path: Path to generated report
                - visualizations: List of generated visualization paths
        """
        try:
            self._validate_data(data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            generated_vizs = []
            
            # Process visualizations if present
            if 'visualizations' in data:
                for viz_config in data['visualizations']:
                    viz_path = self._create_visualization(
                        viz_config['data'],
                        viz_config['type']
                    )
                    generated_vizs.append(viz_path)
                    viz_config['image_path'] = viz_path
            
            # Get template and render content
            template = self.env.get_template(template_name)
            html_content = template.render(
                report=data,
                timestamp=timestamp,
                generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Generate output based on format
            if output_format == 'pdf':
                output_path = self.output_dir / "pdf" / f"report_{timestamp}.pdf"
                pdfkit.from_string(html_content, str(output_path))
            elif output_format == 'html':
                output_path = self.output_dir / "html" / f"report_{timestamp}.html"
                with open(output_path, 'w') as f:
                    f.write(html_content)
            elif output_format == 'markdown':
                output_path = self.output_dir / "markdown" / f"report_{timestamp}.md"
                # Convert HTML to Markdown (simplified version)
                markdown_content = html_content  # You might want to use html2text here
                with open(output_path, 'w') as f:
                    f.write(markdown_content)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'visualizations': generated_vizs
            }
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'partial_success': bool(generated_vizs),
                'visualizations': generated_vizs
            }

    async def create_template(self, template_name: str, template_content: str) -> Dict[str, Union[bool, str]]:
        """
        Create a new report template.
        
        Args:
            template_name: Name of the template file
            template_content: Template content
            
        Returns:
            Dict containing success status and template path or error message
        """
        try:
            template_path = self.template_dir / template_name
            with open(template_path, 'w') as f:
                f.write(template_content)
                
            return {
                'success': True,
                'template_path': str(template_path)
            }
        except Exception as e:
            self.logger.error(f"Template creation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def validate_template(self, template_content: str) -> Dict[str, Union[bool, str]]:
        """
        Validate a template's syntax.
        
        Args:
            template_content: Template content to validate
            
        Returns:
            Dict containing validation results
        """
        try:
            env = Environment()
            env.parse(template_content)
            
            return {
                'success': True,
                'message': 'Template syntax is valid'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            } 