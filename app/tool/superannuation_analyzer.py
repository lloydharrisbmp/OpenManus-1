"""
Superannuation Analyzer Tool for analyzing retirement and superannuation investments.
Supports retirement planning, contribution strategies, and investment analysis.
"""

import os
import json
import logging
from typing import Dict, List, Union, Optional
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .base import BaseTool

class SuperannuationAnalyzerTool(BaseTool):
    """
    Advanced Superannuation Analyzer Tool for retirement planning and analysis.
    Features include contribution analysis, investment strategies, and retirement projections.
    """

    def __init__(self):
        super().__init__()
        self.name = "SuperannuationAnalyzerTool"
        self.description = "Analyzes retirement and superannuation investments"
        self.data_dir = Path("superannuation_data")
        self.output_dir = Path("superannuation_analysis")
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

    async def analyze_current_position(
        self,
        personal_data: Dict,
        super_data: Dict
    ) -> Dict[str, Union[bool, Dict]]:
        """
        Analyze current superannuation position.
        
        Args:
            personal_data: Dictionary containing personal details
            super_data: Dictionary containing superannuation details
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Calculate age and years to retirement
            birth_date = datetime.strptime(personal_data['birth_date'], '%Y-%m-%d').date()
            today = date.today()
            current_age = (today - birth_date).days / 365.25
            years_to_retirement = max(0, personal_data['retirement_age'] - current_age)
            
            # Analyze current balance and contributions
            current_position = {
                'current_balance': super_data['current_balance'],
                'employer_contributions': super_data['annual_employer_contributions'],
                'personal_contributions': super_data.get('annual_personal_contributions', 0),
                'investment_returns': super_data.get('annual_investment_returns', 0),
                'fees': super_data.get('annual_fees', 0)
            }
            
            # Calculate key metrics
            metrics = {
                'total_annual_contributions': (
                    current_position['employer_contributions'] +
                    current_position['personal_contributions']
                ),
                'net_annual_growth': (
                    current_position['employer_contributions'] +
                    current_position['personal_contributions'] +
                    current_position['investment_returns'] -
                    current_position['fees']
                ),
                'years_to_retirement': years_to_retirement,
                'current_age': current_age
            }
            
            return {
                'success': True,
                'analysis': {
                    'current_position': current_position,
                    'metrics': metrics
                }
            }

        except Exception as e:
            self.logger.error(f"Error analyzing current position: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def project_retirement_balance(
        self,
        current_data: Dict,
        assumptions: Dict
    ) -> Dict[str, Union[bool, Dict]]:
        """
        Project retirement balance based on current position and assumptions.
        
        Args:
            current_data: Dictionary containing current position data
            assumptions: Dictionary containing projection assumptions
            
        Returns:
            Dict containing projection results
        """
        try:
            years = int(current_data['metrics']['years_to_retirement'])
            current_balance = current_data['current_position']['current_balance']
            
            # Initialize projection arrays
            balances = np.zeros(years + 1)
            contributions = np.zeros(years + 1)
            returns = np.zeros(years + 1)
            fees = np.zeros(years + 1)
            
            balances[0] = current_balance
            
            # Calculate yearly projections
            for year in range(1, years + 1):
                # Calculate contributions with inflation adjustment
                employer_contribution = (
                    current_data['current_position']['employer_contributions'] *
                    (1 + assumptions['wage_growth_rate']) ** year
                )
                personal_contribution = (
                    current_data['current_position']['personal_contributions'] *
                    (1 + assumptions['wage_growth_rate']) ** year
                )
                
                contributions[year] = employer_contribution + personal_contribution
                
                # Calculate investment returns
                returns[year] = balances[year-1] * assumptions['investment_return_rate']
                
                # Calculate fees
                fees[year] = (
                    balances[year-1] * assumptions['fee_rate'] +
                    assumptions['fixed_fees']
                )
                
                # Update balance
                balances[year] = (
                    balances[year-1] +
                    contributions[year] +
                    returns[year] -
                    fees[year]
                )
            
            # Create projection summary
            projection_summary = {
                'final_balance': balances[-1],
                'total_contributions': np.sum(contributions),
                'total_returns': np.sum(returns),
                'total_fees': np.sum(fees),
                'yearly_projections': {
                    'years': list(range(years + 1)),
                    'balances': balances.tolist(),
                    'contributions': contributions.tolist(),
                    'returns': returns.tolist(),
                    'fees': fees.tolist()
                }
            }
            
            return {
                'success': True,
                'projections': projection_summary
            }

        except Exception as e:
            self.logger.error(f"Error projecting retirement balance: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def analyze_contribution_strategies(
        self,
        current_data: Dict,
        income_data: Dict
    ) -> Dict[str, Union[bool, Dict]]:
        """
        Analyze different contribution strategies.
        
        Args:
            current_data: Dictionary containing current position data
            income_data: Dictionary containing income details
            
        Returns:
            Dict containing strategy analysis results
        """
        try:
            gross_income = income_data['annual_gross_income']
            tax_rate = income_data['marginal_tax_rate']
            
            # Analyze different contribution strategies
            strategies = {
                'current': {
                    'personal_contributions': current_data['current_position']['personal_contributions'],
                    'tax_savings': current_data['current_position']['personal_contributions'] * tax_rate
                },
                'maximum_concessional': {
                    'personal_contributions': min(
                        25000 - current_data['current_position']['employer_contributions'],
                        gross_income * 0.10
                    ),
                    'tax_savings': 0  # Will be calculated below
                },
                'salary_sacrifice': {
                    'personal_contributions': gross_income * 0.05,
                    'tax_savings': 0  # Will be calculated below
                }
            }
            
            # Calculate tax savings for each strategy
            for strategy in ['maximum_concessional', 'salary_sacrifice']:
                strategies[strategy]['tax_savings'] = strategies[strategy]['personal_contributions'] * tax_rate
            
            # Calculate impact on retirement
            strategy_impacts = {}
            for strategy_name, strategy in strategies.items():
                additional_contribution = strategy['personal_contributions'] - strategies['current']['personal_contributions']
                
                # Simple projection of additional contributions over retirement period
                years = int(current_data['metrics']['years_to_retirement'])
                compound_factor = (1 + 0.07) ** years  # Assuming 7% annual return
                
                impact = additional_contribution * ((compound_factor - 1) / 0.07)
                strategy_impacts[strategy_name] = {
                    'additional_retirement_balance': impact,
                    'annual_tax_savings': strategy['tax_savings']
                }
            
            return {
                'success': True,
                'strategies': {
                    'contribution_options': strategies,
                    'retirement_impacts': strategy_impacts
                }
            }

        except Exception as e:
            self.logger.error(f"Error analyzing contribution strategies: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def analyze_investment_mix(
        self,
        current_allocation: Dict,
        risk_profile: str
    ) -> Dict[str, Union[bool, Dict]]:
        """
        Analyze investment mix and provide recommendations.
        
        Args:
            current_allocation: Dictionary containing current investment allocation
            risk_profile: Risk profile category
            
        Returns:
            Dict containing investment analysis results
        """
        try:
            # Define target allocations for different risk profiles
            target_allocations = {
                'conservative': {
                    'cash': 0.20,
                    'fixed_interest': 0.40,
                    'property': 0.10,
                    'australian_shares': 0.15,
                    'international_shares': 0.15
                },
                'balanced': {
                    'cash': 0.10,
                    'fixed_interest': 0.30,
                    'property': 0.10,
                    'australian_shares': 0.25,
                    'international_shares': 0.25
                },
                'growth': {
                    'cash': 0.05,
                    'fixed_interest': 0.15,
                    'property': 0.10,
                    'australian_shares': 0.35,
                    'international_shares': 0.35
                }
            }
            
            # Calculate differences from target allocation
            target = target_allocations[risk_profile]
            differences = {}
            for asset_class in target:
                current = current_allocation.get(asset_class, 0)
                differences[asset_class] = target[asset_class] - current
            
            # Generate rebalancing recommendations
            recommendations = []
            for asset_class, difference in differences.items():
                if abs(difference) >= 0.05:  # 5% threshold for rebalancing
                    action = 'increase' if difference > 0 else 'decrease'
                    recommendations.append({
                        'asset_class': asset_class,
                        'action': action,
                        'amount': abs(difference),
                        'priority': 'high' if abs(difference) >= 0.10 else 'medium'
                    })
            
            return {
                'success': True,
                'analysis': {
                    'current_allocation': current_allocation,
                    'target_allocation': target,
                    'differences': differences,
                    'recommendations': recommendations
                }
            }

        except Exception as e:
            self.logger.error(f"Error analyzing investment mix: {e}")
            return {
                'success': False,
                'error': str(e)
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
                
                if chart_type == 'balance_projection':
                    self._create_balance_projection_chart(data['projections'])
                elif chart_type == 'contribution_analysis':
                    self._create_contribution_chart(data['strategies'])
                elif chart_type == 'investment_mix':
                    self._create_investment_mix_chart(data['investment_analysis'])
                
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

    def _create_balance_projection_chart(self, projection_data: Dict):
        """Create balance projection chart."""
        years = projection_data['yearly_projections']['years']
        balances = projection_data['yearly_projections']['balances']
        
        plt.plot(years, balances)
        plt.title('Projected Superannuation Balance')
        plt.xlabel('Years')
        plt.ylabel('Balance ($)')
        plt.grid(True)

    def _create_contribution_chart(self, strategy_data: Dict):
        """Create contribution strategy comparison chart."""
        strategies = strategy_data['contribution_options']
        names = list(strategies.keys())
        contributions = [s['personal_contributions'] for s in strategies.values()]
        tax_savings = [s['tax_savings'] for s in strategies.values()]
        
        x = np.arange(len(names))
        width = 0.35
        
        plt.bar(x - width/2, contributions, width, label='Contributions')
        plt.bar(x + width/2, tax_savings, width, label='Tax Savings')
        plt.title('Contribution Strategy Comparison')
        plt.xlabel('Strategy')
        plt.ylabel('Amount ($)')
        plt.xticks(x, names)
        plt.legend()

    def _create_investment_mix_chart(self, investment_data: Dict):
        """Create investment mix comparison chart."""
        current = investment_data['current_allocation']
        target = investment_data['target_allocation']
        
        labels = list(target.keys())
        current_values = [current.get(label, 0) for label in labels]
        target_values = [target[label] for label in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, current_values, width, label='Current')
        plt.bar(x + width/2, target_values, width, label='Target')
        plt.title('Investment Mix Comparison')
        plt.xlabel('Asset Class')
        plt.ylabel('Allocation (%)')
        plt.xticks(x, labels, rotation=45)
        plt.legend()

    async def generate_super_report(
        self,
        personal_data: Dict,
        analysis_results: Dict,
        output_format: str = 'html'
    ) -> Dict[str, Union[bool, str]]:
        """
        Generate a comprehensive superannuation analysis report.
        
        Args:
            personal_data: Dictionary containing personal details
            analysis_results: Dictionary containing analysis results
            output_format: Output format ('html' or 'pdf')
            
        Returns:
            Dict containing report path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / "reports" / f"super_analysis_{timestamp}.{output_format}"
            
            # Create report content
            report_content = f"""
            <h1>Superannuation Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Personal Details</h2>
            <p>Current Age: {analysis_results['current_position']['metrics']['current_age']:.1f} years</p>
            <p>Years to Retirement: {analysis_results['current_position']['metrics']['years_to_retirement']:.1f}</p>
            
            <h2>Current Position</h2>
            <p>Current Balance: ${analysis_results['current_position']['current_position']['current_balance']:,.2f}</p>
            <p>Annual Contributions: ${analysis_results['current_position']['metrics']['total_annual_contributions']:,.2f}</p>
            
            <h2>Retirement Projections</h2>
            <img src="{analysis_results['charts']['balance_projection']}" alt="Balance Projection">
            <p>Projected Retirement Balance: ${analysis_results['projections']['final_balance']:,.2f}</p>
            
            <h2>Contribution Strategies</h2>
            <img src="{analysis_results['charts']['contribution_analysis']}" alt="Contribution Analysis">
            
            <h2>Investment Mix</h2>
            <img src="{analysis_results['charts']['investment_mix']}" alt="Investment Mix">
            """
            
            # Save report
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            return {
                'success': True,
                'report_path': str(report_file)
            }

        except Exception as e:
            self.logger.error(f"Error generating superannuation report: {e}")
            return {
                'success': False,
                'error': str(e)
            } 