from typing import Dict, List, Any, Optional
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

class FinancialPredictor:
    """Advanced financial prediction and analysis."""
    
    def analyze_financial_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive financial metric analysis."""
        return {
            "wealth_projections": self._analyze_wealth_projections(data),
            "risk_metrics": self._analyze_risk_metrics(data),
            "retirement_analysis": self._analyze_retirement_scenarios(data),
            "tax_efficiency": self._analyze_tax_efficiency(data),
            "portfolio_optimization": self._analyze_portfolio_optimization(data),
            "market_correlation": self._analyze_market_correlation(data)
        }

    def _analyze_wealth_projections(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and project wealth growth scenarios."""
        initial_wealth = data.get("current_wealth", 100000)
        annual_savings = data.get("annual_savings", 10000)
        time_horizon = data.get("time_horizon", 30)
        
        # Monte Carlo simulation
        n_simulations = 1000
        returns_mean = 0.07
        returns_std = 0.15
        
        simulations = []
        for _ in range(n_simulations):
            wealth = initial_wealth
            yearly_returns = np.random.normal(returns_mean, returns_std, time_horizon)
            yearly_wealth = [wealth]
            
            for year in range(time_horizon):
                wealth = wealth * (1 + yearly_returns[year]) + annual_savings
                yearly_wealth.append(wealth)
            
            simulations.append(yearly_wealth)
        
        simulations = np.array(simulations)
        
        percentiles = {
            "optimistic": np.percentile(simulations, 90, axis=0),
            "expected": np.percentile(simulations, 50, axis=0),
            "conservative": np.percentile(simulations, 10, axis=0)
        }
        
        return {
            "projections": percentiles,
            "confidence_interval": {
                "lower": np.percentile(simulations[:, -1], 5),
                "upper": np.percentile(simulations[:, -1], 95)
            },
            "probability_of_target": self._calculate_target_probability(
                simulations[:, -1], 
                data.get("wealth_target", initial_wealth * 2)
            )
        }

    def _analyze_risk_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced risk metric analysis."""
        returns = np.array(data.get("historical_returns", []))
        if len(returns) == 0:
            return {}
        
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        var_95 = np.percentile(returns, 5)  # 95% VaR
        cvar_95 = returns[returns <= var_95].mean()  # 95% CVaR
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return {
            "volatility": volatility,
            "value_at_risk_95": var_95,
            "conditional_var_95": cvar_95,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": self._calculate_sharpe_ratio(returns),
            "sortino_ratio": self._calculate_sortino_ratio(returns),
            "risk_assessment": self._assess_risk_level(volatility, max_drawdown)
        }

    def _analyze_retirement_scenarios(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze retirement scenarios and sustainability."""
        current_age = data.get("current_age", 40)
        retirement_age = data.get("retirement_age", 65)
        life_expectancy = data.get("life_expectancy", 90)
        current_savings = data.get("current_savings", 500000)
        annual_contribution = data.get("annual_contribution", 25000)
        desired_income = data.get("desired_income", 80000)
        
        scenarios = []
        for market_condition in ["poor", "average", "good"]:
            if market_condition == "poor":
                return_rate = 0.04
            elif market_condition == "average":
                return_rate = 0.06
            else:
                return_rate = 0.08
                
            years_to_retirement = retirement_age - current_age
            retirement_duration = life_expectancy - retirement_age
            
            # Accumulation phase
            retirement_savings = current_savings
            for _ in range(years_to_retirement):
                retirement_savings = (retirement_savings * (1 + return_rate) + 
                                   annual_contribution)
            
            # Withdrawal phase
            remaining_savings = []
            current_savings = retirement_savings
            for _ in range(retirement_duration):
                current_savings = (current_savings * (1 + return_rate) - 
                                 desired_income)
                remaining_savings.append(current_savings)
            
            scenarios.append({
                "market_condition": market_condition,
                "retirement_savings": retirement_savings,
                "remaining_savings": remaining_savings,
                "sustainable": min(remaining_savings) > 0
            })
        
        return {
            "scenarios": scenarios,
            "recommended_adjustments": self._get_retirement_recommendations(scenarios),
            "success_probability": self._calculate_retirement_success_probability(scenarios)
        }

    def _analyze_tax_efficiency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tax efficiency and optimization opportunities."""
        income = data.get("annual_income", 100000)
        deductions = data.get("deductions", [])
        investments = data.get("investments", [])
        
        # Calculate effective tax rate
        tax_paid = self._calculate_tax(income, deductions)
        effective_tax_rate = tax_paid / income
        
        # Analyze investment tax efficiency
        investment_tax_impact = self._analyze_investment_tax_impact(investments)
        
        # Generate optimization recommendations
        recommendations = self._generate_tax_recommendations(
            income, deductions, investments
        )
        
        return {
            "current_tax_efficiency": {
                "effective_tax_rate": effective_tax_rate,
                "tax_paid": tax_paid,
                "potential_savings": self._calculate_potential_tax_savings(
                    income, deductions, investments
                )
            },
            "investment_tax_impact": investment_tax_impact,
            "recommendations": recommendations,
            "optimization_potential": self._calculate_tax_optimization_potential(
                income, deductions, investments
            )
        }

    def _analyze_portfolio_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced portfolio optimization analysis."""
        returns = np.array(data.get("asset_returns", []))
        if len(returns) == 0:
            return {}
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(returns.T)
        
        # Calculate optimal weights using mean-variance optimization
        optimal_weights = self._calculate_optimal_weights(returns)
        
        # Calculate efficient frontier
        efficient_frontier = self._calculate_efficient_frontier(returns)
        
        return {
            "optimal_allocation": {
                "weights": optimal_weights,
                "expected_return": np.sum(optimal_weights * np.mean(returns, axis=0)),
                "expected_risk": np.sqrt(
                    optimal_weights.T @ np.cov(returns.T) @ optimal_weights
                )
            },
            "efficient_frontier": efficient_frontier,
            "correlation_matrix": correlation_matrix.tolist(),
            "diversification_score": self._calculate_diversification_score(
                optimal_weights, correlation_matrix
            ),
            "rebalancing_recommendations": self._generate_rebalancing_recommendations(
                data.get("current_weights", []), optimal_weights
            )
        }

    def _analyze_market_correlation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market correlations and factor exposures."""
        returns = np.array(data.get("returns", []))
        market_returns = np.array(data.get("market_returns", []))
        if len(returns) == 0 or len(market_returns) == 0:
            return {}
        
        # Calculate beta and alpha
        beta = self._calculate_beta(returns, market_returns)
        alpha = self._calculate_alpha(returns, market_returns, beta)
        
        # Calculate factor exposures
        factor_exposures = self._calculate_factor_exposures(
            returns, data.get("factor_returns", {})
        )
        
        return {
            "market_metrics": {
                "beta": beta,
                "alpha": alpha,
                "r_squared": self._calculate_r_squared(returns, market_returns)
            },
            "factor_exposures": factor_exposures,
            "correlation_analysis": self._analyze_correlation_patterns(
                returns, market_returns
            ),
            "regime_analysis": self._analyze_market_regimes(returns, market_returns)
        }

    def _calculate_target_probability(self, final_values: np.ndarray, 
                                   target: float) -> float:
        """Calculate probability of reaching target value."""
        return np.mean(final_values >= target)

    def _calculate_sharpe_ratio(self, returns: np.ndarray, 
                              risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    def _calculate_sortino_ratio(self, returns: np.ndarray, 
                               risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        return np.sqrt(252) * np.mean(excess_returns) / downside_std if downside_std > 0 else np.inf

    def _assess_risk_level(self, volatility: float, 
                          max_drawdown: float) -> str:
        """Assess overall risk level."""
        risk_score = volatility * 0.7 + abs(max_drawdown) * 0.3
        if risk_score < 0.1:
            return "Very Low"
        elif risk_score < 0.15:
            return "Low"
        elif risk_score < 0.25:
            return "Medium"
        elif risk_score < 0.35:
            return "High"
        else:
            return "Very High" 