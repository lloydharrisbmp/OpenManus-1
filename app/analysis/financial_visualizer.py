import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

class FinancialVisualizer:
    """Specialized financial visualizations for planning analysis."""
    
    def __init__(self, output_dir: str = "client_documents/financial_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_financial_visualizations(self, plan_data: Dict[str, Any]) -> List[str]:
        """Generate all financial visualizations for a plan."""
        viz_paths = []
        
        # Wealth Growth Projection
        viz_paths.append(self._create_wealth_projection(plan_data))
        
        # Asset Allocation Sunburst
        viz_paths.append(self._create_asset_allocation(plan_data))
        
        # Risk-Return Analysis
        viz_paths.append(self._create_risk_return_analysis(plan_data))
        
        # Cash Flow Waterfall
        viz_paths.append(self._create_cashflow_waterfall(plan_data))
        
        # Tax Impact Analysis
        viz_paths.append(self._create_tax_impact_analysis(plan_data))
        
        # Retirement Scenario Analysis
        viz_paths.append(self._create_retirement_scenarios(plan_data))
        
        # SMSF Performance Dashboard
        viz_paths.append(self._create_smsf_dashboard(plan_data))
        
        # Estate Planning Network
        viz_paths.append(self._create_estate_network(plan_data))
        
        return [p for p in viz_paths if p]

    def _create_wealth_projection(self, plan_data: Dict[str, Any]) -> str:
        """Create interactive wealth projection visualization."""
        # Generate time series data
        years = range(30)
        scenarios = {
            "Conservative": 0.05,
            "Moderate": 0.07,
            "Aggressive": 0.09
        }
        
        initial_wealth = plan_data.get("current_wealth", 100000)
        annual_contribution = plan_data.get("annual_contribution", 10000)
        
        fig = go.Figure()
        
        for scenario, rate in scenarios.items():
            wealth = []
            for year in years:
                projected_wealth = initial_wealth * (1 + rate) ** year
                projected_wealth += annual_contribution * ((1 + rate) ** year - 1) / rate
                wealth.append(projected_wealth)
            
            fig.add_trace(go.Scatter(
                x=list(years),
                y=wealth,
                name=scenario,
                mode='lines',
                hovertemplate="Year %{x}<br>Wealth: $%{y:,.0f}"
            ))
        
        fig.update_layout(
            title="Wealth Projection Scenarios",
            xaxis_title="Years",
            yaxis_title="Projected Wealth ($)",
            yaxis_tickformat="$,.0f",
            showlegend=True
        )
        
        output_path = self.output_dir / f"wealth_projection_{datetime.now():%Y%m%d_%H%M%S}.html"
        fig.write_html(str(output_path))
        return str(output_path)

    def _create_asset_allocation(self, plan_data: Dict[str, Any]) -> str:
        """Create interactive asset allocation sunburst chart."""
        fig = go.Figure(go.Sunburst(
            labels=[
                "Total Portfolio",
                "Equities", "Fixed Income", "Real Estate", "Cash",
                "Domestic Equities", "International Equities",
                "Government Bonds", "Corporate Bonds",
                "Residential", "Commercial",
                "High Interest Savings", "Term Deposits"
            ],
            parents=[
                "",
                "Total Portfolio", "Total Portfolio", "Total Portfolio", "Total Portfolio",
                "Equities", "Equities",
                "Fixed Income", "Fixed Income",
                "Real Estate", "Real Estate",
                "Cash", "Cash"
            ],
            values=[
                100,
                40, 30, 20, 10,
                25, 15,
                15, 15,
                12, 8,
                6, 4
            ],
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="Current Asset Allocation",
            width=800,
            height=800
        )
        
        output_path = self.output_dir / f"asset_allocation_{datetime.now():%Y%m%d_%H%M%S}.html"
        fig.write_html(str(output_path))
        return str(output_path)

    def _create_risk_return_analysis(self, plan_data: Dict[str, Any]) -> str:
        """Create risk-return analysis visualization."""
        # Sample portfolio data
        portfolios = {
            "Current": {"risk": 12, "return": 7, "size": 100},
            "Proposed": {"risk": 10, "return": 8, "size": 100},
            "Conservative": {"risk": 6, "return": 4, "size": 80},
            "Moderate": {"risk": 10, "return": 6, "size": 90},
            "Aggressive": {"risk": 15, "return": 9, "size": 70}
        }
        
        fig = go.Figure()
        
        # Efficient frontier curve
        risk_range = np.linspace(4, 18, 100)
        returns = 2 + (risk_range/4) + np.exp(risk_range/40 - 1)
        fig.add_trace(go.Scatter(
            x=risk_range,
            y=returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='rgba(0,0,0,0.3)', dash='dot')
        ))
        
        # Portfolio points
        for name, data in portfolios.items():
            fig.add_trace(go.Scatter(
                x=[data["risk"]],
                y=[data["return"]],
                mode='markers',
                name=name,
                marker=dict(size=data["size"]),
                hovertemplate=f"{name}<br>Risk: %{{x}}%<br>Return: %{{y}}%"
            ))
        
        fig.update_layout(
            title="Risk-Return Analysis",
            xaxis_title="Risk (%)",
            yaxis_title="Expected Return (%)",
            showlegend=True
        )
        
        output_path = self.output_dir / f"risk_return_{datetime.now():%Y%m%d_%H%M%S}.html"
        fig.write_html(str(output_path))
        return str(output_path)

    def _create_cashflow_waterfall(self, plan_data: Dict[str, Any]) -> str:
        """Create cash flow waterfall chart."""
        fig = go.Figure(go.Waterfall(
            name="Cash Flow",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", 
                    "relative", "relative", "total"],
            x=["Income", "Investment Returns", "Tax", "Living Expenses", 
               "Insurance", "Investment Contributions", "Net Position"],
            textposition="outside",
            text=["+80,000", "+20,000", "-30,000", "-40,000", 
                  "-5,000", "-15,000", ""],
            y=[80000, 20000, -30000, -40000, -5000, -15000, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Annual Cash Flow Analysis",
            showlegend=True,
            yaxis_title="Amount ($)",
            yaxis_tickformat="$,.0f"
        )
        
        output_path = self.output_dir / f"cashflow_{datetime.now():%Y%m%d_%H%M%S}.html"
        fig.write_html(str(output_path))
        return str(output_path)

    def _create_tax_impact_analysis(self, plan_data: Dict[str, Any]) -> str:
        """Create tax impact analysis visualization."""
        # Create subplots for different tax aspects
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Income Tax Breakdown", "Capital Gains Impact",
                          "Tax Deductions", "After-Tax Returns")
        )
        
        # Income Tax Breakdown
        fig.add_trace(
            go.Pie(labels=["Tax Paid", "Net Income"],
                  values=[30000, 70000],
                  name="Income Distribution"),
            row=1, col=1
        )
        
        # Capital Gains Impact
        fig.add_trace(
            go.Bar(x=["Current Strategy", "Optimized Strategy"],
                  y=[15000, 10000],
                  name="Capital Gains Tax"),
            row=1, col=2
        )
        
        # Tax Deductions
        fig.add_trace(
            go.Bar(x=["Work Related", "Investment", "Super Contributions", "Other"],
                  y=[5000, 8000, 15000, 2000],
                  name="Deductions"),
            row=2, col=1
        )
        
        # After-Tax Returns
        fig.add_trace(
            go.Scatter(x=list(range(5)),
                      y=[100, 105, 112, 118, 125],
                      name="After-Tax Value"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Tax Impact Analysis")
        
        output_path = self.output_dir / f"tax_impact_{datetime.now():%Y%m%d_%H%M%S}.html"
        fig.write_html(str(output_path))
        return str(output_path)

    def _create_retirement_scenarios(self, plan_data: Dict[str, Any]) -> str:
        """Create retirement scenario analysis visualization."""
        years = list(range(30))
        scenarios = {
            "Early Retirement (60)": {
                "savings": [2000000 * (1.05 ** year) - 80000 * year for year in years],
                "color": "green"
            },
            "Standard Retirement (65)": {
                "savings": [2000000 * (1.05 ** year) - 70000 * year for year in years],
                "color": "blue"
            },
            "Late Retirement (70)": {
                "savings": [2000000 * (1.05 ** year) - 60000 * year for year in years],
                "color": "red"
            }
        }
        
        fig = go.Figure()
        
        for scenario, data in scenarios.items():
            fig.add_trace(go.Scatter(
                x=years,
                y=data["savings"],
                name=scenario,
                line=dict(color=data["color"]),
                hovertemplate="Year %{x}<br>Savings: $%{y:,.0f}"
            ))
        
        fig.update_layout(
            title="Retirement Scenario Analysis",
            xaxis_title="Years in Retirement",
            yaxis_title="Remaining Savings ($)",
            yaxis_tickformat="$,.0f",
            showlegend=True
        )
        
        output_path = self.output_dir / f"retirement_scenarios_{datetime.now():%Y%m%d_%H%M%S}.html"
        fig.write_html(str(output_path))
        return str(output_path)

    def _create_smsf_dashboard(self, plan_data: Dict[str, Any]) -> str:
        """Create SMSF performance dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("SMSF Asset Allocation", "Performance vs Benchmark",
                          "Contribution History", "Fee Analysis")
        )
        
        # SMSF Asset Allocation
        fig.add_trace(
            go.Pie(labels=["Australian Shares", "International Shares", 
                          "Property", "Fixed Interest", "Cash"],
                  values=[40, 30, 15, 10, 5],
                  name="Asset Allocation"),
            row=1, col=1
        )
        
        # Performance vs Benchmark
        fig.add_trace(
            go.Scatter(x=list(range(5)),
                      y=[100, 108, 112, 122, 128],
                      name="SMSF Performance"),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(range(5)),
                      y=[100, 106, 110, 115, 120],
                      name="Benchmark"),
            row=1, col=2
        )
        
        # Contribution History
        fig.add_trace(
            go.Bar(x=["2019", "2020", "2021", "2022", "2023"],
                  y=[25000, 25000, 27500, 27500, 27500],
                  name="Contributions"),
            row=2, col=1
        )
        
        # Fee Analysis
        fig.add_trace(
            go.Bar(x=["Admin", "Investment", "Advice", "Other"],
                  y=[2000, 3000, 1500, 500],
                  name="Fees"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="SMSF Dashboard")
        
        output_path = self.output_dir / f"smsf_dashboard_{datetime.now():%Y%m%d_%H%M%S}.html"
        fig.write_html(str(output_path))
        return str(output_path)

    def _create_estate_network(self, plan_data: Dict[str, Any]) -> str:
        """Create estate planning network visualization."""
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes for different estate components
        components = {
            "Estate": {"size": 40, "color": "#1f77b4"},
            "Primary Residence": {"size": 30, "color": "#2ca02c"},
            "Investment Properties": {"size": 25, "color": "#2ca02c"},
            "Superannuation": {"size": 35, "color": "#ff7f0e"},
            "Investments": {"size": 30, "color": "#d62728"},
            "Life Insurance": {"size": 20, "color": "#9467bd"},
            "Beneficiary 1": {"size": 25, "color": "#8c564b"},
            "Beneficiary 2": {"size": 25, "color": "#8c564b"},
            "Trust": {"size": 20, "color": "#e377c2"}
        }
        
        for name, attrs in components.items():
            G.add_node(name, **attrs)
        
        # Add edges
        edges = [
            ("Estate", "Primary Residence"),
            ("Estate", "Investment Properties"),
            ("Estate", "Superannuation"),
            ("Estate", "Investments"),
            ("Estate", "Life Insurance"),
            ("Estate", "Trust"),
            ("Trust", "Beneficiary 1"),
            ("Trust", "Beneficiary 2"),
            ("Life Insurance", "Beneficiary 1"),
            ("Superannuation", "Beneficiary 2")
        ]
        G.add_edges_from(edges)
        
        # Create plotly figure
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_colors.append(G.nodes[node]["color"])
            node_sizes.append(G.nodes[node]["size"])
            node_text.append(node)
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))
            
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="bottom center",
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line_width=2)))
                
        fig.update_layout(
            title="Estate Planning Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        output_path = self.output_dir / f"estate_network_{datetime.now():%Y%m%d_%H%M%S}.html"
        fig.write_html(str(output_path))
        return str(output_path) 