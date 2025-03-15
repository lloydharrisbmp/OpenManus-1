from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.tool.base import BaseTool
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from app.tool.file_organizer import file_organizer


class MarketAnalysisTool(BaseTool):
    """Tool for analyzing market data and performing stock research."""

    name: str = "market_analysis"
    description: str = "Analyzes market data, stock performance, and financial metrics"
    parameters: dict = {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of stock symbols to analyze",
            },
            "period": {
                "type": "string",
                "description": "Time period for analysis (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of metrics to analyze (price, volume, pe_ratio, market_cap, etc.)",
            }
        },
        "required": ["symbols"]
    }

    async def execute(self, symbols: List[str], period: str = "1y", metrics: List[str] = None) -> Dict:
        """Execute market analysis for given symbols and metrics."""
        try:
            results = {}
            for symbol in symbols:
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                
                analysis = {
                    "current_price": data['Close'][-1],
                    "price_change": (data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100,
                    "volume": data['Volume'][-1],
                    "market_cap": stock.info.get('marketCap'),
                    "pe_ratio": stock.info.get('trailingPE'),
                }
                results[symbol] = analysis
                
            return {"observation": str(results), "success": True}
        except Exception as e:
            return {"observation": f"Error in market analysis: {str(e)}", "success": False}


class AustralianMarketAnalysisTool(BaseTool):
    """Tool for analyzing Australian market data and ASX-listed securities."""

    name: str = "aus_market_analysis"
    description: str = "Analyzes ASX market data, stock performance, and Australian market metrics"
    parameters: dict = {
        "type": "object",
        "properties": {
            "asx_symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of ASX stock symbols to analyze (e.g., 'BHP.AX', 'CBA.AX')",
            },
            "period": {
                "type": "string",
                "description": "Time period for analysis (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
            },
            "include_indices": {
                "type": "boolean",
                "description": "Whether to include relevant ASX indices (e.g., XJO.AX for ASX 200)",
            }
        },
        "required": ["asx_symbols"]
    }

    async def execute(self, asx_symbols: List[str], period: str = "1y", include_indices: bool = True) -> Dict:
        """Execute Australian market analysis."""
        try:
            results = {}
            
            # Add ASX 200 index if requested
            if include_indices:
                asx_symbols = ['^AXJO'] + asx_symbols
            
            for symbol in asx_symbols:
                # Ensure .AX suffix for ASX stocks
                if not symbol.startswith('^') and not symbol.endswith('.AX'):
                    symbol = f"{symbol}.AX"
                
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                
                analysis = {
                    "current_price": data['Close'][-1],
                    "price_change": (data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100,
                    "volume": data['Volume'][-1],
                    "market_cap": stock.info.get('marketCap'),
                    "pe_ratio": stock.info.get('trailingPE'),
                    "dividend_yield": stock.info.get('dividendYield', 0) * 100 if stock.info.get('dividendYield') else None,
                    "franking_credits": "Data not available",  # Would need additional data source
                }
                results[symbol] = analysis
            
            return {"observation": str(results), "success": True}
        except Exception as e:
            return {"observation": f"Error in Australian market analysis: {str(e)}", "success": False}


class TaxOptimizationTool(BaseTool):
    """Tool for Australian tax optimization strategies."""

    name: str = "tax_optimization"
    description: str = "Analyzes and recommends tax optimization strategies for Australian clients"
    parameters: dict = {
        "type": "object",
        "properties": {
            "entity_type": {
                "type": "string",
                "description": "Type of entity (individual, company, trust, smsf)",
            },
            "income_streams": {
                "type": "object",
                "description": "Dictionary of income streams and amounts",
            },
            "tax_year": {
                "type": "string",
                "description": "Australian tax year (e.g., '2023-2024')",
            }
        },
        "required": ["entity_type", "income_streams"]
    }

    async def execute(self, entity_type: str, income_streams: Dict, tax_year: str = None) -> Dict:
        """Execute tax optimization analysis."""
        try:
            # Tax rates for 2023-2024
            individual_tax_rates = {
                18200: 0,
                45000: 0.19,
                120000: 0.325,
                180000: 0.37,
                float('inf'): 0.45
            }
            
            company_tax_rate = 0.30  # Standard company tax rate
            small_business_tax_rate = 0.25  # For base rate entities
            
            strategies = {
                "individual": [
                    "Consider salary sacrifice into superannuation",
                    "Review negative gearing opportunities",
                    "Maximize deductible expenses",
                    "Consider timing of income recognition",
                ],
                "company": [
                    "Review eligibility for small business tax rate",
                    "Consider dividend distribution timing",
                    "Review capital investment opportunities for instant asset write-off",
                ],
                "trust": [
                    "Review trust distribution strategy",
                    "Consider streaming of different income types",
                    "Review timing of distributions",
                ],
                "smsf": [
                    "Review pension strategy",
                    "Consider contribution timing",
                    "Review asset allocation for tax efficiency",
                ]
            }
            
            return {
                "observation": f"Tax optimization strategies for {entity_type}:\n" +
                             "\n".join(f"- {s}" for s in strategies.get(entity_type.lower(), [])),
                "success": True
            }
        except Exception as e:
            return {"observation": f"Error in tax optimization: {str(e)}", "success": False}


class PortfolioOptimizationTool(BaseTool):
    """Tool for portfolio optimization and strategy development."""

    name: str = "portfolio_optimization"
    description: str = "Optimizes investment portfolios based on Australian market conditions and risk-return preferences"
    parameters: dict = {
        "type": "object",
        "properties": {
            "assets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of assets in the portfolio",
            },
            "risk_tolerance": {
                "type": "string",
                "description": "Risk tolerance level (conservative, moderate, aggressive)",
            },
            "investment_horizon": {
                "type": "string",
                "description": "Investment time horizon in years",
            },
            "tax_entity": {
                "type": "string",
                "description": "Entity type for tax purposes (individual, super, company)",
            }
        },
        "required": ["assets", "risk_tolerance"]
    }

    async def execute(self, assets: List[str], risk_tolerance: str, investment_horizon: str = "5", tax_entity: str = "individual") -> Dict:
        """Execute portfolio optimization based on given parameters."""
        try:
            # Enhanced allocation logic with Australian focus
            allocations = {
                "conservative": {
                    "australian_equities": 0.20,
                    "international_equities": 0.10,
                    "australian_fixed_income": 0.35,
                    "international_fixed_income": 0.15,
                    "cash": 0.15,
                    "property": 0.05
                },
                "moderate": {
                    "australian_equities": 0.35,
                    "international_equities": 0.25,
                    "australian_fixed_income": 0.20,
                    "international_fixed_income": 0.10,
                    "cash": 0.05,
                    "property": 0.05
                },
                "aggressive": {
                    "australian_equities": 0.45,
                    "international_equities": 0.35,
                    "australian_fixed_income": 0.10,
                    "international_fixed_income": 0.05,
                    "cash": 0.02,
                    "property": 0.03
                }
            }
            
            allocation = allocations.get(risk_tolerance.lower(), allocations["moderate"])
            
            # Add tax considerations
            tax_notes = {
                "individual": "Consider franking credits for Australian equities",
                "super": "Consider concessional tax treatment in super environment",
                "company": "Consider corporate tax implications and dividend strategies"
            }
            
            return {
                "observation": (f"Recommended portfolio allocation for {tax_entity}:\n" +
                              "\n".join(f"- {k}: {v*100:.1f}%" for k, v in allocation.items()) +
                              f"\n\nTax consideration: {tax_notes.get(tax_entity, '')}"),
                "success": True
            }
        except Exception as e:
            return {"observation": f"Error in portfolio optimization: {str(e)}", "success": False}


class ReportGeneratorTool(BaseTool):
    """Tool for generating financial reports and documentation."""

    name: str = "report_generator"
    description: str = "Generates financial reports, documentation, and analysis summaries"
    parameters: dict = {
        "type": "object",
        "properties": {
            "report_type": {
                "type": "string",
                "description": "Type of report to generate (e.g., portfolio_review, tax_strategy, estate_plan)",
            },
            "title": {
                "type": "string",
                "description": "Title of the report",
            },
            "content": {
                "type": "string",
                "description": "Main content of the report",
            },
            "format": {
                "type": "string",
                "description": "Output format (markdown, text)",
                "default": "markdown"
            }
        },
        "required": ["report_type", "title", "content"]
    }

    async def execute(
        self,
        report_type: str,
        title: str,
        content: str,
        format: str = "markdown"
    ) -> Dict:
        """Generate a report with the specified content."""
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = ".md" if format == "markdown" else ".txt"
            filename = f"{report_type}_{timestamp}{extension}"
            
            # Get appropriate file path using FileOrganizer
            file_type = 'markdown' if format == 'markdown' else 'text'
            file_path = file_organizer.get_path(file_type, filename)
            
            # Format content
            formatted_content = f"""# {title}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{content}
"""
            
            # Write the report
            with open(file_path, 'w') as f:
                f.write(formatted_content)
            
            return {
                "observation": f"Report generated successfully at {file_path}",
                "success": True,
                "file_path": str(file_path)
            }
            
        except Exception as e:
            return {
                "observation": f"Error generating report: {str(e)}",
                "success": False
            } 