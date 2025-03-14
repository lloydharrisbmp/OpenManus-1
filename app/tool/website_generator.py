import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
from jinja2 import Environment, FileSystemLoader, Template
from pydantic import Field
import markdown
from slugify import slugify

from app.tool.base import BaseTool
from app.tool.file_organizer import file_organizer

class WebsiteGeneratorTool(BaseTool):
    """Tool for generating static websites from content."""

    name: str = "website_generator"
    description: str = "Generates a static website from provided content with modern styling and responsive design"
    parameters: dict = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Website title"
            },
            "content": {
                "type": "string",
                "description": "Main content in markdown format"
            },
            "theme": {
                "type": "string",
                "description": "Optional theme name (default: 'modern')"
            },
            "output_dir": {
                "type": "string",
                "description": "Optional output directory name (default: based on title)"
            }
        },
        "required": ["title", "content"]
    }
    templates_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "website_templates")

    def __init__(self):
        super().__init__()
        # Create templates directory if it doesn't exist
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True)
            self._create_default_templates()

    def _create_default_templates(self):
        """Create default website templates."""
        modern_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --text-color: #333;
            --light-bg: #f8f9fa;
            --border-color: #dee2e6;
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--light-bg);
        }

        .navbar {
            background-color: var(--primary-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .navbar-brand {
            color: white !important;
            font-weight: 600;
        }

        .container {
            max-width: 1200px;
            margin: 40px auto;
            padding: 0;
        }

        .content-wrapper {
            background-color: white;
            box-shadow: 0 0 30px rgba(0,0,0,0.1);
            border-radius: 12px;
            overflow: hidden;
        }

        .sidebar {
            background-color: var(--light-bg);
            padding: 20px;
            border-right: 1px solid var(--border-color);
        }

        .main-content {
            padding: 40px;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-color);
            margin-bottom: 1rem;
            font-weight: 600;
        }

        h1 {
            font-size: 2.5rem;
            border-bottom: 3px solid var(--secondary-color);
            padding-bottom: 10px;
            margin-bottom: 30px;
        }

        h2 {
            font-size: 1.8rem;
            margin-top: 2rem;
        }

        h3 {
            font-size: 1.4rem;
            color: var(--secondary-color);
        }

        p {
            margin-bottom: 1.2rem;
            font-size: 1.1rem;
        }

        .nav-link {
            color: var(--primary-color);
            padding: 8px 16px;
            border-radius: 6px;
            margin-bottom: 5px;
        }

        .nav-link:hover {
            background-color: var(--secondary-color);
            color: white;
        }

        .nav-link.active {
            background-color: var(--secondary-color);
            color: white;
        }

        ul:not(.nav), ol:not(.nav) {
            padding-left: 1.5rem;
            margin-bottom: 1.5rem;
        }

        li:not(.nav-item) {
            margin-bottom: 0.5rem;
        }

        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 4px;
            color: var(--accent-color);
        }

        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            margin-bottom: 1.5rem;
        }

        blockquote {
            border-left: 4px solid var(--secondary-color);
            padding-left: 1rem;
            margin-left: 0;
            color: #666;
        }

        table {
            width: 100%;
            margin-bottom: 1.5rem;
            border-collapse: collapse;
        }

        th, td {
            padding: 12px;
            border: 1px solid var(--border-color);
        }

        th {
            background-color: var(--light-bg);
        }

        .footer {
            margin-top: 60px;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid var(--border-color);
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .container {
                margin: 20px auto;
            }
            
            .main-content {
                padding: 20px;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .sidebar {
                margin-bottom: 20px;
                border-right: none;
                border-bottom: 1px solid var(--border-color);
            }
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">{{ title }}</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
        </div>
    </nav>

    <div class="container">
        <div class="content-wrapper">
            <div class="row g-0">
                <div class="col-md-3 sidebar">
                    <nav class="nav flex-column" id="tableOfContents">
                        <!-- Table of contents will be dynamically generated -->
                    </nav>
                </div>
                <div class="col-md-9 main-content">
                    <h1>{{ title }}</h1>
                    <div class="content">
                        {{ content | safe }}
                    </div>
                    <div class="footer">
                        <p>Generated by OpenManus Financial Planning AI</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Generate table of contents
        document.addEventListener('DOMContentLoaded', function() {
            const toc = document.getElementById('tableOfContents');
            const headings = document.querySelectorAll('.main-content h2');
            
            headings.forEach((heading, index) => {
                // Create an ID for the heading if it doesn't have one
                if (!heading.id) {
                    heading.id = 'section-' + index;
                }
                
                // Create the navigation link
                const link = document.createElement('a');
                link.href = '#' + heading.id;
                link.className = 'nav-link';
                link.textContent = heading.textContent;
                
                // Add click handler for smooth scrolling
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    heading.scrollIntoView({ behavior: 'smooth' });
                });
                
                toc.appendChild(link);
            });
            
            // Highlight active section on scroll
            window.addEventListener('scroll', function() {
                const scrollPosition = window.scrollY;
                
                headings.forEach((heading) => {
                    const section = heading.getBoundingClientRect();
                    const link = toc.querySelector(`a[href="#${heading.id}"]`);
                    
                    if (section.top <= 100 && section.bottom >= 100) {
                        link.classList.add('active');
                    } else {
                        link.classList.remove('active');
                    }
                });
            });
        });
    </script>
</body>
</html>
"""
        with open(self.templates_dir / "modern.html", "w") as f:
            f.write(modern_template)

    async def execute(
        self,
        title: str,
        content: str,
        theme: str = "modern",
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a static website from the provided content."""
        if not output_dir:
            output_dir = slugify(title)
        
        # Get the appropriate directory from FileOrganizer
        base_path = file_organizer.get_path("websites", output_dir)
        os.makedirs(base_path, exist_ok=True)

        # Set up Jinja2 environment
        env = Environment(loader=FileSystemLoader(str(self.templates_dir)))
        template = env.get_template(f"{theme}.html")

        # Convert markdown to HTML with extended features
        html_content = markdown.markdown(
            content,
            extensions=[
                'extra',
                'codehilite',
                'tables',
                'toc',
                'fenced_code',
                'sane_lists'
            ]
        )

        # Generate the HTML
        html = template.render(
            title=title,
            content=html_content
        )

        # Write the HTML file
        output_file = base_path / "index.html"
        with open(output_file, "w") as f:
            f.write(html)

        # Create a response with the correct URL
        relative_path = os.path.relpath(str(output_file), "client_documents")
        website_url = f"http://localhost:8000/generated/{relative_path}"
        
        return {
            "status": "success",
            "message": f"""✅ WEBSITE GENERATED SUCCESSFULLY!

📱 VIEW YOUR WEBSITE:
→ {website_url}

Click the link above or find your website in the "Generated Documents" sidebar.
Look for the "{os.path.basename(output_file)}" file with the 🌐 icon.""",
            "output_dir": str(base_path),
            "index_file": str(output_file),
            "website_url": website_url
        } 