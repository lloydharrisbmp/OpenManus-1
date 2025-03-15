"""
Enhanced Website Generator Tool with multi-page support, advanced templating, and SEO features.
This tool provides functionality to generate modern, responsive websites with advanced features.
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, Template
from bs4 import BeautifulSoup
import sass
from .base import BaseTool

class WebsiteGeneratorTool(BaseTool):
    """
    Enhanced Website Generator Tool for creating modern, responsive websites with advanced features.
    Supports multi-page layouts, SEO optimization, and advanced templating.
    """

    name: str = "website_generator"
    description: str = "Generates a simple HTML website with CSS styling"
    parameters: Dict[str, Any] = {
        "template_dir": Path(__file__).parent / "website_templates",
        "output_dir": Path("generated_websites"),
        "env": Environment(
            loader=FileSystemLoader(str(Path(__file__).parent / "website_templates")),
            trim_blocks=True,
            lstrip_blocks=True
        )
    }

    def __init__(self):
        super().__init__()
        self._setup_directories()
        self._setup_logging()

    async def execute(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the website generation with the provided configuration.
        
        Args:
            config: Dictionary containing website configuration
            
        Returns:
            Dict[str, Any]: Result of the website generation
        """
        return self.generate_website(config)

    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create assets directories
        (self.template_dir / "css").mkdir(exist_ok=True)
        (self.template_dir / "js").mkdir(exist_ok=True)
        (self.template_dir / "images").mkdir(exist_ok=True)

    def _setup_logging(self) -> None:
        """Configure logging for the tool."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.name)

    def _validate_config(self, config: Dict) -> bool:
        """
        Validate the website configuration.
        
        Args:
            config: Dictionary containing website configuration
            
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['title', 'description', 'pages']
        if not all(field in config for field in required_fields):
            missing = [f for f in required_fields if f not in config]
            raise ValueError(f"Missing required fields in config: {missing}")
            
        if not isinstance(config['pages'], list):
            raise ValueError("'pages' must be a list of page configurations")
            
        for page in config['pages']:
            if 'name' not in page or 'content' not in page:
                raise ValueError("Each page must have 'name' and 'content' fields")
                
        return True

    def _generate_seo_meta(self, page_config: Dict) -> Dict:
        """
        Generate SEO meta tags for a page.
        
        Args:
            page_config: Page configuration dictionary
            
        Returns:
            Dict: Dictionary containing SEO meta tags
        """
        return {
            'title': page_config.get('title', ''),
            'description': page_config.get('description', ''),
            'keywords': page_config.get('keywords', ''),
            'author': page_config.get('author', ''),
            'robots': page_config.get('robots', 'index, follow'),
            'og_title': page_config.get('og_title', page_config.get('title', '')),
            'og_description': page_config.get('og_description', page_config.get('description', '')),
            'og_image': page_config.get('og_image', ''),
            'twitter_card': page_config.get('twitter_card', 'summary_large_image'),
        }

    def _process_scss(self, scss_content: str) -> str:
        """
        Process SCSS content into CSS.
        
        Args:
            scss_content: SCSS content string
            
        Returns:
            str: Processed CSS content
        """
        try:
            return sass.compile(string=scss_content)
        except sass.CompileError as e:
            self.logger.error(f"SCSS compilation error: {e}")
            raise

    def _optimize_html(self, html_content: str) -> str:
        """
        Optimize HTML content.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            str: Optimized HTML content
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
            comment.extract()
            
        # Minify HTML
        return str(soup).replace('\n', '').replace('  ', '')

    def _create_sitemap(self, pages: List[Dict], base_url: str) -> str:
        """
        Generate XML sitemap for the website.
        
        Args:
            pages: List of page configurations
            base_url: Base URL of the website
            
        Returns:
            str: XML sitemap content
        """
        sitemap = ['<?xml version="1.0" encoding="UTF-8"?>']
        sitemap.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
        
        for page in pages:
            sitemap.append('  <url>')
            page_url = f"{base_url.rstrip('/')}/{page['name']}.html"
            sitemap.append(f"    <loc>{page_url}</loc>")
            sitemap.append(f"    <lastmod>{datetime.now().strftime('%Y-%m-%d')}</lastmod>")
            sitemap.append('    <changefreq>weekly</changefreq>')
            sitemap.append('    <priority>0.8</priority>')
            sitemap.append('  </url>')
            
        sitemap.append('</urlset>')
        return '\n'.join(sitemap)

    def generate_website(self, config: Dict) -> Dict[str, Union[bool, str, List[str]]]:
        """Generate a complete website based on the provided configuration."""
        try:
            # Validate configuration
            if not self._validate_config(config):
                return {"success": False, "message": "Invalid configuration"}
            
            # Create output directory
            site_name = config.get("site_name", "generated_site")
            output_dir = self.output_dir / site_name
            
            # Check if directory exists and create it if needed
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                
            # Create directories for assets
            assets_dir = output_dir / "assets"
            css_dir = assets_dir / "css"
            js_dir = assets_dir / "js"
            images_dir = assets_dir / "images"
            
            # Create all needed directories
            for directory in [assets_dir, css_dir, js_dir, images_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Copy assets
            shutil.copytree(
                self.template_dir / "css",
                css_dir,
                dirs_exist_ok=True
            )
            shutil.copytree(
                self.template_dir / "js",
                js_dir,
                dirs_exist_ok=True
            )
            shutil.copytree(
                self.template_dir / "images",
                images_dir,
                dirs_exist_ok=True
            )
            
            # Process SCSS if exists
            scss_file = self.template_dir / "css" / "styles.scss"
            if scss_file.exists():
                with open(scss_file) as f:
                    css_content = self._process_scss(f.read())
                with open(css_dir / "styles.css", 'w') as f:
                    f.write(css_content)
            
            # Generate pages
            template_name = config.get('template', 'modern.html')
            template = self.env.get_template(template_name)
            generated_pages = []
            
            for page in config['pages']:
                try:
                    # Generate SEO meta tags
                    seo_meta = self._generate_seo_meta(page)
                    
                    # Render page
                    html_content = template.render(
                        page=page,
                        config=config,
                        seo=seo_meta
                    )
                    
                    # Optimize HTML
                    optimized_html = self._optimize_html(html_content)
                    
                    # Save page
                    output_file = output_dir / f"{page['name']}.html"
                    with open(output_file, 'w') as f:
                        f.write(optimized_html)
                        
                    generated_pages.append(str(output_file))
                    
                except Exception as e:
                    self.logger.error(f"Error generating page {page['name']}: {e}")
                    return {
                        'success': False,
                        'error': f"Error generating page {page['name']}: {str(e)}",
                        'partial_success': True,
                        'generated_pages': generated_pages
                    }
            
            # Generate sitemap if base_url provided
            if 'base_url' in config:
                sitemap_content = self._create_sitemap(config['pages'], config['base_url'])
                with open(output_dir / 'sitemap.xml', 'w') as f:
                    f.write(sitemap_content)
            
            return {
                'success': True,
                'output_dir': str(output_dir),
                'generated_pages': generated_pages,
                'sitemap': 'sitemap.xml' if 'base_url' in config else None
            }
            
        except Exception as e:
            self.logger.error(f"Website generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'partial_success': bool(generated_pages),
                'generated_pages': generated_pages
            }

    async def create_template(self, template_name: str, template_content: str) -> Dict[str, Union[bool, str]]:
        """
        Create a new template in the templates directory.
        
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

    async def validate_template(self, template_content: str) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate a template's syntax.
        
        Args:
            template_content: Template content to validate
            
        Returns:
            Dict containing validation results
        """
        try:
            # Create a temporary Environment for validation
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
