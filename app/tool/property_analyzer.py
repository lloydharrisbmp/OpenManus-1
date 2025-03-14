"""
Property Analyzer Tool for analyzing real estate investments and property data.
Supports property valuation, investment analysis, and market trends.
"""

import os
import json
import logging
from typing import Dict, List, Union, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .base import BaseTool

class PropertyAnalyzerTool(BaseTool):
    """
    Advanced Property Analyzer Tool for real estate investment analysis.
    Features include property valuation, ROI calculation, and market analysis.
    """

    def __init__(self):
        super().__init__()
        self.name = "PropertyAnalyzerTool"
        self.description = "Analyzes real estate investments and property data"
        self.data_dir = Path("property_data")
        self.output_dir = Path("property_analysis")
        self._setup_directories()
        self._setup_logging()

    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)

    def _setup_logging(self) -> None:
        """Configure logging for the tool."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.name)

    async def analyze_property_value(
        self,
        property_data: Dict,
        comparable_properties: List[Dict]
    ) -> Dict[str, Union[bool, Dict]]:
        """
        Analyze property value using comparable properties and market data.
        
        Args:
            property_data: Dictionary containing property details
            comparable_properties: List of comparable property data
            
        Returns:
            Dict containing valuation results
        """
        try:
            # Convert comparable properties to DataFrame
            comp_df = pd.DataFrame(comparable_properties)
            
            # Calculate price per square foot
            comp_df['price_per_sqft'] = comp_df['sale_price'] / comp_df['square_feet']
            
            # Calculate basic statistics
            stats = {
                'mean_price': comp_df['sale_price'].mean(),
                'median_price': comp_df['sale_price'].median(),
                'mean_price_per_sqft': comp_df['price_per_sqft'].mean(),
                'median_price_per_sqft': comp_df['price_per_sqft'].median()
            }
            
            # Estimate property value
            estimated_value = property_data['square_feet'] * stats['mean_price_per_sqft']
            
            # Calculate confidence interval
            confidence_interval = np.percentile(comp_df['price_per_sqft'], [25, 75])
            value_range = {
                'low': property_data['square_feet'] * confidence_interval[0],
                'high': property_data['square_feet'] * confidence_interval[1]
            }
            
            return {
                'success': True,
                'valuation': {
                    'estimated_value': estimated_value,
                    'value_range': value_range,
                    'comparable_stats': stats
                }
            }

        except Exception as e:
            self.logger.error(f"Error analyzing property value: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def calculate_roi(
        self,
        property_data: Dict,
        investment_params: Dict
    ) -> Dict[str, Union[bool, Dict]]:
        """
        Calculate return on investment for a property.
        
        Args:
            property_data: Dictionary containing property details
            investment_params: Dictionary containing investment parameters
            
        Returns:
            Dict containing ROI analysis results
        """
        try:
            # Calculate costs
            purchase_price = property_data['price']
            down_payment = purchase_price * investment_params['down_payment_percentage']
            loan_amount = purchase_price - down_payment
            
            # Calculate monthly costs
            monthly_mortgage = self._calculate_mortgage_payment(
                loan_amount,
                investment_params['interest_rate'],
                investment_params['loan_term_years']
            )
            
            monthly_expenses = {
                'mortgage': monthly_mortgage,
                'property_tax': investment_params['annual_property_tax'] / 12,
                'insurance': investment_params['annual_insurance'] / 12,
                'maintenance': investment_params['monthly_maintenance'],
                'utilities': investment_params.get('monthly_utilities', 0),
                'property_management': investment_params.get('monthly_property_management', 0)
            }
            
            total_monthly_expenses = sum(monthly_expenses.values())
            
            # Calculate income
            monthly_income = {
                'rental_income': investment_params['monthly_rental_income'],
                'other_income': investment_params.get('monthly_other_income', 0)
            }
            
            total_monthly_income = sum(monthly_income.values())
            
            # Calculate cash flow
            monthly_cash_flow = total_monthly_income - total_monthly_expenses
            annual_cash_flow = monthly_cash_flow * 12
            
            # Calculate ROI metrics
            initial_investment = down_payment + investment_params.get('closing_costs', 0)
            cash_on_cash_roi = (annual_cash_flow / initial_investment) * 100
            
            # Calculate appreciation
            future_value = purchase_price * (1 + investment_params['annual_appreciation_rate']) ** investment_params['holding_period_years']
            total_appreciation = future_value - purchase_price
            
            # Calculate total return
            total_return = {
                'appreciation': total_appreciation,
                'cash_flow': annual_cash_flow * investment_params['holding_period_years'],
                'tax_benefits': investment_params.get('annual_tax_benefits', 0) * investment_params['holding_period_years']
            }
            
            total_roi = (sum(total_return.values()) / initial_investment) * 100
            
            return {
                'success': True,
                'roi_analysis': {
                    'monthly_expenses': monthly_expenses,
                    'monthly_income': monthly_income,
                    'monthly_cash_flow': monthly_cash_flow,
                    'annual_cash_flow': annual_cash_flow,
                    'cash_on_cash_roi': cash_on_cash_roi,
                    'total_return': total_return,
                    'total_roi': total_roi,
                    'initial_investment': initial_investment
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating ROI: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_mortgage_payment(
        self,
        loan_amount: float,
        annual_rate: float,
        term_years: int
    ) -> float:
        """Calculate monthly mortgage payment."""
        monthly_rate = annual_rate / 12 / 100
        num_payments = term_years * 12
        
        return loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)

    async def analyze_market_trends(
        self,
        location: str,
        historical_data: List[Dict]
    ) -> Dict[str, Union[bool, Dict]]:
        """
        Analyze real estate market trends for a location.
        
        Args:
            location: Location identifier
            historical_data: List of historical market data
            
        Returns:
            Dict containing market trend analysis
        """
        try:
            # Convert historical data to DataFrame
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Calculate key metrics
            price_trends = {
                'current_median_price': df['median_price'].iloc[-1],
                'price_change_1y': self._calculate_change(df['median_price'], 12),
                'price_change_5y': self._calculate_change(df['median_price'], 60),
                'price_change_10y': self._calculate_change(df['median_price'], 120)
            }
            
            # Calculate market indicators
            market_indicators = {
                'inventory_level': df['inventory'].iloc[-1],
                'days_on_market': df['days_on_market'].iloc[-1],
                'price_to_rent_ratio': df['median_price'].iloc[-1] / (df['median_rent'].iloc[-1] * 12)
            }
            
            # Predict future trends
            future_trends = self._predict_future_trends(df)
            
            return {
                'success': True,
                'market_analysis': {
                    'price_trends': price_trends,
                    'market_indicators': market_indicators,
                    'future_trends': future_trends
                }
            }

        except Exception as e:
            self.logger.error(f"Error analyzing market trends: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_change(self, series: pd.Series, periods: int) -> float:
        """Calculate percentage change over specified periods."""
        if len(series) >= periods:
            return ((series.iloc[-1] / series.iloc[-periods]) - 1) * 100
        return 0

    def _predict_future_trends(self, df: pd.DataFrame) -> Dict:
        """Predict future market trends using linear regression."""
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['median_price'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_periods = 12
        future_X = np.arange(len(df), len(df) + future_periods).reshape(-1, 1)
        future_prices = model.predict(future_X)
        
        return {
            'predicted_prices': future_prices.tolist(),
            'predicted_growth_rate': (model.coef_[0] / df['median_price'].mean()) * 100
        }

    async def generate_analysis_charts(
        self,
        data: Dict,
        chart_types: List[str]
    ) -> Dict[str, Union[bool, List[str]]]:
        """
        Generate analysis charts and visualizations.
        
        Args:
            data: Dictionary containing analysis data
            chart_types: List of chart types to generate
            
        Returns:
            Dict containing paths to generated charts
        """
        try:
            chart_paths = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for chart_type in chart_types:
                plt.figure(figsize=(12, 6))
                
                if chart_type == 'price_trends':
                    self._create_price_trends_chart(data['historical_prices'])
                elif chart_type == 'roi_analysis':
                    self._create_roi_chart(data['roi_analysis'])
                elif chart_type == 'market_indicators':
                    self._create_market_indicators_chart(data['market_indicators'])
                
                chart_path = self.output_dir / "charts" / f"{chart_type}_{timestamp}.png"
                plt.savefig(chart_path)
                plt.close()
                chart_paths.append(str(chart_path))
            
            return {
                'success': True,
                'chart_paths': chart_paths
            }

        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _create_price_trends_chart(self, price_data: pd.DataFrame):
        """Create price trends chart."""
        sns.lineplot(data=price_data)
        plt.title('Property Price Trends')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)

    def _create_roi_chart(self, roi_data: Dict):
        """Create ROI analysis chart."""
        labels = list(roi_data['monthly_expenses'].keys())
        values = list(roi_data['monthly_expenses'].values())
        
        plt.pie(values, labels=labels, autopct='%1.1f%%')
        plt.title('Monthly Expenses Breakdown')

    def _create_market_indicators_chart(self, indicator_data: Dict):
        """Create market indicators chart."""
        plt.bar(indicator_data.keys(), indicator_data.values())
        plt.title('Market Indicators')
        plt.xticks(rotation=45)

    async def generate_property_report(
        self,
        property_data: Dict,
        analysis_results: Dict,
        output_format: str = 'html'
    ) -> Dict[str, Union[bool, str]]:
        """
        Generate a comprehensive property analysis report.
        
        Args:
            property_data: Dictionary containing property details
            analysis_results: Dictionary containing analysis results
            output_format: Output format ('html' or 'pdf')
            
        Returns:
            Dict containing report path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / "reports" / f"property_analysis_{timestamp}.{output_format}"
            
            # Create report content
            report_content = f"""
            <h1>Property Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Property Details</h2>
            <p>Address: {property_data['address']}</p>
            <p>Square Feet: {property_data['square_feet']}</p>
            <p>Price: ${property_data['price']:,.2f}</p>
            
            <h2>Valuation Analysis</h2>
            <p>Estimated Value: ${analysis_results['valuation']['estimated_value']:,.2f}</p>
            <p>Value Range: ${analysis_results['valuation']['value_range']['low']:,.2f} - ${analysis_results['valuation']['value_range']['high']:,.2f}</p>
            
            <h2>ROI Analysis</h2>
            <p>Monthly Cash Flow: ${analysis_results['roi_analysis']['monthly_cash_flow']:,.2f}</p>
            <p>Cash on Cash ROI: {analysis_results['roi_analysis']['cash_on_cash_roi']:.2f}%</p>
            <p>Total ROI: {analysis_results['roi_analysis']['total_roi']:.2f}%</p>
            
            <h2>Market Analysis</h2>
            <img src="{analysis_results['charts']['price_trends']}" alt="Price Trends">
            <img src="{analysis_results['charts']['market_indicators']}" alt="Market Indicators">
            """
            
            # Save report
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            return {
                'success': True,
                'report_path': str(report_file)
            }

        except Exception as e:
            self.logger.error(f"Error generating property report: {e}")
            return {
                'success': False,
                'error': str(e)
            } 