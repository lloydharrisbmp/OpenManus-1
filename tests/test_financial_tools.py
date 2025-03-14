"""
Unit tests for financial tools.

This module contains tests for the financial analysis tools, ensuring that they 
work correctly and handle errors appropriately.
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.tool.financial_tools import (
    MarketAnalysisTool,
    AustralianMarketAnalysisTool,
    TaxOptimizationTool,
    PortfolioOptimizationTool,
    ReportGeneratorTool
)
from app.exceptions import FinancialDataError


class TestMarketAnalysisTool(unittest.TestCase):
    """Test cases for the MarketAnalysisTool."""

    @patch('yfinance.Ticker')
    def test_market_analysis_successful(self, mock_ticker):
        """Test that market analysis works with valid data."""
        # Mock the yfinance Ticker behavior
        mock_stock = MagicMock()
        mock_ticker.return_value = mock_stock
        
        # Create dummy historical data
        mock_history = pd.DataFrame({
            'Close': [100, 105, 110],
            'Volume': [1000, 2000, 3000]
        })
        mock_stock.history.return_value = mock_history
        
        # Mock stock info
        mock_stock.info = {
            'marketCap': 1000000,
            'trailingPE': 15.5
        }
        
        # Create the tool and run it
        tool = MarketAnalysisTool()
        result = asyncio.run(tool.execute(symbols=['AAPL']))
        
        # Verify the result contains expected data
        self.assertIn('AAPL', result)
        self.assertEqual(result['AAPL']['current_price'], 110)
        self.assertEqual(result['AAPL']['volume'], 3000)
        self.assertEqual(result['AAPL']['market_cap'], 1000000)
        self.assertEqual(result['AAPL']['pe_ratio'], 15.5)
    
    @patch('yfinance.Ticker')
    def test_market_analysis_handles_missing_data(self, mock_ticker):
        """Test that the tool handles missing stock data gracefully."""
        # Mock the yfinance Ticker behavior with missing data
        mock_stock = MagicMock()
        mock_ticker.return_value = mock_stock
        
        # Create dummy historical data
        mock_history = pd.DataFrame({
            'Close': [100, 105, 110],
            'Volume': [1000, 2000, 3000]
        })
        mock_stock.history.return_value = mock_history
        
        # Mock stock info with missing fields
        mock_stock.info = {
            'marketCap': None,
            # PE ratio missing
        }
        
        # Create the tool and run it
        tool = MarketAnalysisTool()
        result = asyncio.run(tool.execute(symbols=['UNKNOWN']))
        
        # Verify the result handles missing data
        self.assertIn('UNKNOWN', result)
        self.assertEqual(result['UNKNOWN']['market_cap'], None)
        self.assertEqual(result['UNKNOWN']['pe_ratio'], None)
    
    @patch('yfinance.Ticker')
    def test_market_analysis_with_multiple_symbols(self, mock_ticker):
        """Test that market analysis works with multiple stock symbols."""
        # Mock the yfinance Ticker behavior
        mock_stock = MagicMock()
        mock_ticker.return_value = mock_stock
        
        # Create dummy historical data
        mock_history = pd.DataFrame({
            'Close': [100, 105, 110],
            'Volume': [1000, 2000, 3000]
        })
        mock_stock.history.return_value = mock_history
        
        # Mock stock info
        mock_stock.info = {
            'marketCap': 1000000,
            'trailingPE': 15.5
        }
        
        # Create the tool and run it
        tool = MarketAnalysisTool()
        result = asyncio.run(tool.execute(symbols=['AAPL', 'MSFT']))
        
        # Verify both symbols are in the result
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)


class TestTaxOptimizationTool(unittest.TestCase):
    """Test cases for the TaxOptimizationTool."""
    
    def test_tax_optimization_individual(self):
        """Test tax optimization for individual entities."""
        tool = TaxOptimizationTool()
        income_streams = {
            "salary": 150000,
            "dividends": 20000,
            "rental_income": 30000
        }
        
        result = asyncio.run(tool.execute(
            entity_type="individual",
            income_streams=income_streams,
            tax_year="2023-2024"
        ))
        
        # Verify result structure
        self.assertIn('tax_summary', result)
        self.assertIn('optimization_strategies', result)
        self.assertIn('recommended_actions', result)
    
    def test_tax_optimization_company(self):
        """Test tax optimization for company entities."""
        tool = TaxOptimizationTool()
        income_streams = {
            "business_income": 500000,
            "investments": 50000
        }
        
        result = asyncio.run(tool.execute(
            entity_type="company",
            income_streams=income_streams,
            tax_year="2023-2024"
        ))
        
        # Verify result structure and company-specific advice
        self.assertIn('tax_summary', result)
        self.assertIn('company_tax_rate', result['tax_summary'])


class TestPortfolioOptimizationTool(unittest.TestCase):
    """Test cases for the PortfolioOptimizationTool."""
    
    @patch('app.tool.financial_tools.PortfolioOptimizationTool._get_historical_data')
    def test_portfolio_optimization(self, mock_get_historical):
        """Test portfolio optimization with mock data."""
        # Create mock historical return data
        mock_returns = pd.DataFrame({
            'ASX:VAS': [0.01, 0.02, -0.01, 0.03],
            'ASX:VGS': [0.02, 0.01, 0.02, 0.01],
            'ASX:VAF': [0.005, 0.003, 0.004, 0.002]
        })
        mock_get_historical.return_value = mock_returns
        
        tool = PortfolioOptimizationTool()
        result = asyncio.run(tool.execute(
            assets=['ASX:VAS', 'ASX:VGS', 'ASX:VAF'],
            risk_tolerance='moderate',
            investment_horizon='5',
            tax_entity='individual'
        ))
        
        # Verify result structure
        self.assertIn('optimal_portfolio', result)
        self.assertIn('expected_return', result)
        self.assertIn('expected_risk', result)
        self.assertIn('asset_allocation', result)
        self.assertIn('visualization_path', result)
        
        # Check asset allocation adds up to approximately 100%
        total_allocation = sum(result['asset_allocation'].values())
        self.assertAlmostEqual(total_allocation, 100.0, delta=1.0)


class TestReportGeneratorTool(unittest.TestCase):
    """Test cases for the ReportGeneratorTool."""
    
    def test_report_generation_markdown(self):
        """Test report generation in markdown format."""
        tool = ReportGeneratorTool()
        result = asyncio.run(tool.execute(
            report_type="investment_strategy",
            title="Investment Strategy 2024",
            content="# Strategy Overview\n\nThis is a test strategy document.",
            format="markdown"
        ))
        
        # Verify result
        self.assertIn('file_path', result)
        self.assertTrue(result['file_path'].endswith('.md'))
        self.assertIn('success', result)
        self.assertTrue(result['success'])
    
    def test_report_generation_pdf(self):
        """Test report generation in PDF format."""
        tool = ReportGeneratorTool()
        result = asyncio.run(tool.execute(
            report_type="financial_plan",
            title="Comprehensive Financial Plan",
            content="This is a test financial plan document.",
            format="pdf"
        ))
        
        # Verify result
        self.assertIn('file_path', result)
        self.assertTrue(result['file_path'].endswith('.pdf'))
        self.assertIn('success', result)
        self.assertTrue(result['success'])


if __name__ == '__main__':
    unittest.main() 