from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.tool.base import BaseTool
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


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


class PortfolioOptimizationTool(BaseTool):
    """Tool for portfolio optimization and strategy development."""

    name: str = "portfolio_optimization"
    description: str = "Optimizes investment portfolios based on risk-return preferences"
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
            }
        },
        "required": ["assets", "risk_tolerance"]
    }

    async def execute(self, assets: List[str], risk_tolerance: str, investment_horizon: str = "5") -> Dict:
        """Execute portfolio optimization based on given parameters."""
        try:
            # Simplified portfolio optimization logic
            # In a real implementation, this would use more sophisticated optimization algorithms
            allocation = {}
            if risk_tolerance == "conservative":
                allocation = {"stocks": 0.3, "bonds": 0.5, "cash": 0.2}
            elif risk_tolerance == "moderate":
                allocation = {"stocks": 0.6, "bonds": 0.3, "cash": 0.1}
            else:  # aggressive
                allocation = {"stocks": 0.8, "bonds": 0.15, "cash": 0.05}
            
            return {
                "observation": f"Recommended portfolio allocation: {allocation}",
                "success": True
            }
        except Exception as e:
            return {"observation": f"Error in portfolio optimization: {str(e)}", "success": False}


class ReportGeneratorTool(BaseTool):
    """Tool for generating client-facing financial reports."""

    name: str = "report_generator"
    description: str = "Generates comprehensive financial reports for clients"
    parameters: dict = {
        "type": "object",
        "properties": {
            "report_type": {
                "type": "string",
                "description": "Type of report to generate (portfolio_review, financial_plan, tax_strategy, estate_plan)",
            },
            "client_id": {
                "type": "string",
                "description": "Client identifier",
            },
            "period": {
                "type": "string",
                "description": "Report period (e.g., Q1_2024, 2023, YTD)",
            }
        },
        "required": ["report_type", "client_id"]
    }

    async def execute(self, report_type: str, client_id: str, period: str = None) -> Dict:
        """Generate a client-facing report."""
        try:
            report_templates = {
                "portfolio_review": "Comprehensive Portfolio Review\n" +
                                 "===========================\n" +
                                 "Client ID: {client_id}\n" +
                                 "Period: {period}\n\n" +
                                 "1. Portfolio Performance\n" +
                                 "2. Asset Allocation\n" +
                                 "3. Risk Analysis\n" +
                                 "4. Recommendations\n",
                
                "financial_plan": "Strategic Financial Plan\n" +
                                "=====================\n" +
                                "Client ID: {client_id}\n" +
                                "Period: {period}\n\n" +
                                "1. Goals and Objectives\n" +
                                "2. Current Situation\n" +
                                "3. Strategy Recommendations\n" +
                                "4. Implementation Timeline\n",
                
                "tax_strategy": "Tax Optimization Strategy\n" +
                              "=====================\n" +
                              "Client ID: {client_id}\n" +
                              "Period: {period}\n\n" +
                              "1. Current Tax Position\n" +
                              "2. Optimization Opportunities\n" +
                              "3. Implementation Steps\n" +
                              "4. Projected Outcomes\n"
            }
            
            template = report_templates.get(report_type, "Custom Report Template")
            report = template.format(client_id=client_id, period=period or "Current")
            
            return {"observation": f"Generated report:\n\n{report}", "success": True}
        except Exception as e:
            return {"observation": f"Error generating report: {str(e)}", "success": False} 