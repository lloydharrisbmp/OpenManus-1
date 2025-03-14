"""
Portfolio Management Service for tracking and analyzing investment portfolios.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class PortfolioResult:
    """Structured result object for portfolio operations."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    source: str = "portfolio"
    timestamp: datetime = datetime.now()

class PortfolioService:
    """
    Service for managing and analyzing investment portfolios.
    Features:
    - Portfolio tracking and valuation
    - Performance analysis
    - Asset allocation management
    - Risk analysis
    - Rebalancing recommendations
    """

    def __init__(self, market_data_service=None):
        self.market_data_service = market_data_service
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("PortfolioService")

    def create_portfolio(
        self,
        name: str,
        initial_cash: float,
        currency: str = "AUD",
        target_allocation: Optional[Dict[str, float]] = None
    ) -> PortfolioResult:
        """
        Create a new portfolio.
        
        Args:
            name: Portfolio name
            initial_cash: Initial cash balance
            currency: Base currency
            target_allocation: Target asset allocation
            
        Returns:
            PortfolioResult object
        """
        try:
            portfolio = {
                "name": name,
                "created_at": datetime.now().isoformat(),
                "currency": currency,
                "cash": initial_cash,
                "holdings": {},
                "transactions": [],
                "target_allocation": target_allocation or {},
                "performance": {
                    "initial_value": initial_cash,
                    "current_value": initial_cash,
                    "total_return": 0.0,
                    "total_return_pct": 0.0
                }
            }
            
            return PortfolioResult(success=True, data=portfolio)

        except Exception as e:
            self.logger.error(f"Error creating portfolio: {e}")
            return PortfolioResult(success=False, error=str(e))

    def add_transaction(
        self,
        portfolio: Dict,
        symbol: str,
        transaction_type: str,
        quantity: float,
        price: float,
        date: datetime,
        fees: float = 0.0
    ) -> PortfolioResult:
        """
        Add a transaction to the portfolio.
        
        Args:
            portfolio: Portfolio data
            symbol: Asset symbol
            transaction_type: buy/sell
            quantity: Number of units
            price: Price per unit
            date: Transaction date
            fees: Transaction fees
            
        Returns:
            PortfolioResult object
        """
        try:
            transaction = {
                "id": len(portfolio["transactions"]) + 1,
                "date": date.isoformat(),
                "symbol": symbol,
                "type": transaction_type.lower(),
                "quantity": quantity,
                "price": price,
                "fees": fees,
                "total": (quantity * price) + fees
            }
            
            # Update cash balance
            if transaction_type.lower() == "buy":
                if portfolio["cash"] < transaction["total"]:
                    return PortfolioResult(
                        success=False,
                        error="Insufficient funds"
                    )
                portfolio["cash"] -= transaction["total"]
            else:  # sell
                portfolio["cash"] += transaction["total"] - fees
            
            # Update holdings
            if symbol not in portfolio["holdings"]:
                portfolio["holdings"][symbol] = {
                    "quantity": 0,
                    "cost_basis": 0,
                    "last_price": price
                }
            
            holding = portfolio["holdings"][symbol]
            if transaction_type.lower() == "buy":
                # Update cost basis
                total_cost = (holding["quantity"] * holding["cost_basis"]) + (quantity * price)
                total_quantity = holding["quantity"] + quantity
                holding["cost_basis"] = total_cost / total_quantity if total_quantity > 0 else 0
                holding["quantity"] += quantity
            else:
                holding["quantity"] -= quantity
                if holding["quantity"] <= 0:
                    del portfolio["holdings"][symbol]
            
            portfolio["transactions"].append(transaction)
            
            return PortfolioResult(
                success=True,
                data={"portfolio": portfolio, "transaction": transaction}
            )

        except Exception as e:
            self.logger.error(f"Error adding transaction: {e}")
            return PortfolioResult(success=False, error=str(e))

    async def update_portfolio_prices(
        self,
        portfolio: Dict
    ) -> PortfolioResult:
        """
        Update current prices for all holdings.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            PortfolioResult object
        """
        try:
            if not self.market_data_service:
                return PortfolioResult(
                    success=False,
                    error="Market data service not available"
                )
            
            total_value = portfolio["cash"]
            
            for symbol, holding in portfolio["holdings"].items():
                price_result = await self.market_data_service.get_stock_price(symbol)
                if price_result.success:
                    holding["last_price"] = price_result.data["price"]
                    holding["last_updated"] = datetime.now().isoformat()
                    total_value += holding["quantity"] * holding["last_price"]
            
            # Update performance metrics
            portfolio["performance"]["current_value"] = total_value
            portfolio["performance"]["total_return"] = (
                total_value - portfolio["performance"]["initial_value"]
            )
            portfolio["performance"]["total_return_pct"] = (
                (total_value / portfolio["performance"]["initial_value"]) - 1
            ) * 100
            
            return PortfolioResult(success=True, data=portfolio)

        except Exception as e:
            self.logger.error(f"Error updating prices: {e}")
            return PortfolioResult(success=False, error=str(e))

    def calculate_returns(
        self,
        portfolio: Dict,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PortfolioResult:
        """
        Calculate portfolio returns over a period.
        
        Args:
            portfolio: Portfolio data
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            PortfolioResult object with return metrics
        """
        try:
            if not start_date:
                start_date = datetime.fromisoformat(portfolio["created_at"])
            if not end_date:
                end_date = datetime.now()
            
            # Filter transactions in date range
            transactions = [
                t for t in portfolio["transactions"]
                if start_date <= datetime.fromisoformat(t["date"]) <= end_date
            ]
            
            # Calculate metrics
            period_days = (end_date - start_date).days
            current_value = portfolio["performance"]["current_value"]
            
            # Calculate cash flows for XIRR
            cashflows = []
            for t in transactions:
                amount = -t["total"] if t["type"] == "buy" else t["total"]
                cashflows.append({
                    "date": datetime.fromisoformat(t["date"]),
                    "amount": amount
                })
            
            # Add final portfolio value
            cashflows.append({
                "date": end_date,
                "amount": current_value
            })
            
            # Calculate XIRR
            try:
                xirr = self._calculate_xirr(cashflows)
            except:
                xirr = None
            
            returns = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "period_days": period_days,
                "initial_value": portfolio["performance"]["initial_value"],
                "current_value": current_value,
                "absolute_return": portfolio["performance"]["total_return"],
                "return_pct": portfolio["performance"]["total_return_pct"],
                "annualized_return": xirr * 100 if xirr else None
            }
            
            return PortfolioResult(success=True, data=returns)

        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            return PortfolioResult(success=False, error=str(e))

    def _calculate_xirr(self, cashflows: List[Dict]) -> float:
        """
        Calculate the Internal Rate of Return (XIRR).
        
        Args:
            cashflows: List of cash flows with dates and amounts
            
        Returns:
            XIRR as decimal
        """
        def xnpv(rate: float) -> float:
            """Calculate Net Present Value."""
            first_date = cashflows[0]["date"]
            return sum(
                cf["amount"] / (1 + rate) ** ((cf["date"] - first_date).days / 365)
                for cf in cashflows
            )
        
        def xirr_objective(rate: float) -> float:
            """Objective function for solving XIRR."""
            return xnpv(rate)
        
        # Use Newton's method to find the rate that makes NPV = 0
        rate = 0.1  # Initial guess
        for _ in range(100):  # Max iterations
            npv = xnpv(rate)
            if abs(npv) < 0.000001:
                break
                
            # Approximate derivative
            delta = rate * 0.0001
            derivative = (xnpv(rate + delta) - npv) / delta
            
            # Newton step
            rate = rate - npv / derivative
            
            if rate < -0.999999:  # Prevent division by zero
                return -0.999999
        
        return rate

    def analyze_asset_allocation(
        self,
        portfolio: Dict
    ) -> PortfolioResult:
        """
        Analyze current asset allocation vs target.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            PortfolioResult object with allocation analysis
        """
        try:
            total_value = portfolio["performance"]["current_value"]
            current_allocation = {}
            
            # Calculate current allocation
            for symbol, holding in portfolio["holdings"].items():
                value = holding["quantity"] * holding["last_price"]
                current_allocation[symbol] = (value / total_value) * 100
            
            # Add cash allocation
            current_allocation["CASH"] = (portfolio["cash"] / total_value) * 100
            
            # Compare with target allocation
            allocation_diff = {}
            for asset, target in portfolio["target_allocation"].items():
                current = current_allocation.get(asset, 0)
                allocation_diff[asset] = current - target
            
            analysis = {
                "total_value": total_value,
                "current_allocation": current_allocation,
                "target_allocation": portfolio["target_allocation"],
                "allocation_diff": allocation_diff
            }
            
            return PortfolioResult(success=True, data=analysis)

        except Exception as e:
            self.logger.error(f"Error analyzing allocation: {e}")
            return PortfolioResult(success=False, error=str(e))

    def get_rebalancing_recommendations(
        self,
        portfolio: Dict,
        threshold: float = 5.0
    ) -> PortfolioResult:
        """
        Get recommendations for portfolio rebalancing.
        
        Args:
            portfolio: Portfolio data
            threshold: Rebalancing threshold percentage
            
        Returns:
            PortfolioResult object with recommendations
        """
        try:
            allocation_result = self.analyze_asset_allocation(portfolio)
            if not allocation_result.success:
                return allocation_result
            
            analysis = allocation_result.data
            recommendations = []
            
            for asset, diff in analysis["allocation_diff"].items():
                if abs(diff) > threshold:
                    if diff > 0:
                        action = "SELL"
                        reason = "Overweight"
                    else:
                        action = "BUY"
                        reason = "Underweight"
                    
                    target_value = (
                        analysis["total_value"] *
                        portfolio["target_allocation"][asset] / 100
                    )
                    
                    current_value = (
                        analysis["total_value"] *
                        analysis["current_allocation"].get(asset, 0) / 100
                    )
                    
                    trade_value = abs(target_value - current_value)
                    
                    recommendations.append({
                        "asset": asset,
                        "action": action,
                        "reason": reason,
                        "current_allocation": analysis["current_allocation"].get(asset, 0),
                        "target_allocation": portfolio["target_allocation"][asset],
                        "difference": diff,
                        "trade_value": trade_value
                    })
            
            return PortfolioResult(success=True, data=recommendations)

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return PortfolioResult(success=False, error=str(e))

    async def calculate_risk_metrics(
        self,
        portfolio: Dict,
        risk_free_rate: float = 0.02
    ) -> PortfolioResult:
        """
        Calculate portfolio risk metrics.
        
        Args:
            portfolio: Portfolio data
            risk_free_rate: Annual risk-free rate
            
        Returns:
            PortfolioResult object with risk metrics
        """
        try:
            # Get daily returns for holdings
            returns_data = {}
            total_value = portfolio["performance"]["current_value"]
            
            for symbol, holding in portfolio["holdings"].items():
                weight = (holding["quantity"] * holding["last_price"]) / total_value
                if self.market_data_service:
                    # Get historical data for past year
                    start_date = datetime.now() - timedelta(days=365)
                    hist_result = await self.market_data_service.get_historical_data(
                        symbol, start_date
                    )
                    if hist_result.success:
                        returns_data[symbol] = {
                            "returns": hist_result.data["close"].pct_change(),
                            "weight": weight
                        }
            
            if not returns_data:
                return PortfolioResult(
                    success=False,
                    error="No historical data available"
                )
            
            # Calculate portfolio returns
            portfolio_returns = pd.Series(0.0, index=next(iter(returns_data.values()))["returns"].index)
            for symbol, data in returns_data.items():
                portfolio_returns += data["returns"] * data["weight"]
            
            # Calculate metrics
            metrics = {
                "volatility": portfolio_returns.std() * np.sqrt(252) * 100,  # Annualized
                "sharpe_ratio": (
                    (portfolio_returns.mean() * 252 - risk_free_rate) /
                    (portfolio_returns.std() * np.sqrt(252))
                ),
                "max_drawdown": (
                    (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax())
                    .min() * 100
                ),
                "var_95": portfolio_returns.quantile(0.05) * 100,
                "skewness": portfolio_returns.skew(),
                "kurtosis": portfolio_returns.kurtosis()
            }
            
            return PortfolioResult(success=True, data=metrics)

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return PortfolioResult(success=False, error=str(e))

    async def generate_portfolio_report(
        self,
        portfolio: Dict,
        include_recommendations: bool = True
    ) -> PortfolioResult:
        """
        Generate a comprehensive portfolio report.
        
        Args:
            portfolio: Portfolio data
            include_recommendations: Whether to include rebalancing recommendations
            
        Returns:
            PortfolioResult object with report
        """
        try:
            # Get various analyses
            returns_result = self.calculate_returns(portfolio)
            allocation_result = self.analyze_asset_allocation(portfolio)
            risk_result = await self.calculate_risk_metrics(portfolio)
            
            recommendations = None
            if include_recommendations:
                rebalance_result = self.get_rebalancing_recommendations(portfolio)
                if rebalance_result.success:
                    recommendations = rebalance_result.data
            
            # Compile report
            report = {
                "portfolio_summary": {
                    "name": portfolio["name"],
                    "created_at": portfolio["created_at"],
                    "currency": portfolio["currency"],
                    "total_value": portfolio["performance"]["current_value"],
                    "cash_balance": portfolio["cash"],
                    "number_of_holdings": len(portfolio["holdings"])
                },
                "performance": returns_result.data if returns_result.success else None,
                "allocation": allocation_result.data if allocation_result.success else None,
                "risk_metrics": risk_result.data if risk_result.success else None,
                "holdings_detail": portfolio["holdings"],
                "recent_transactions": portfolio["transactions"][-5:],
                "rebalancing_recommendations": recommendations
            }
            
            return PortfolioResult(success=True, data=report)

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return PortfolioResult(success=False, error=str(e)) 