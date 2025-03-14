"""
Financial Planning Service for retirement planning, goal tracking, and financial projections.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class PlanningResult:
    """Structured result object for financial planning operations."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    source: str = "planning"
    timestamp: datetime = datetime.now()

class FinancialPlanningService:
    """
    Service for financial planning and analysis.
    Features:
    - Retirement planning
    - Goal tracking and projections
    - Cash flow analysis
    - Tax planning
    - Insurance needs analysis
    """

    def __init__(self, market_data_service=None, portfolio_service=None):
        self.market_data_service = market_data_service
        self.portfolio_service = portfolio_service
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("FinancialPlanningService")

    def create_financial_plan(
        self,
        personal_info: Dict,
        financial_goals: List[Dict],
        current_assets: Dict,
        income_sources: List[Dict],
        expenses: List[Dict]
    ) -> PlanningResult:
        """
        Create a comprehensive financial plan.
        
        Args:
            personal_info: Personal information
            financial_goals: List of financial goals
            current_assets: Current asset details
            income_sources: Income sources
            expenses: Regular expenses
            
        Returns:
            PlanningResult object
        """
        try:
            plan = {
                "created_at": datetime.now().isoformat(),
                "personal_info": personal_info,
                "financial_goals": financial_goals,
                "current_assets": current_assets,
                "income_sources": income_sources,
                "expenses": expenses,
                "net_worth": self._calculate_net_worth(current_assets),
                "monthly_cash_flow": self._calculate_cash_flow(income_sources, expenses)
            }
            
            return PlanningResult(success=True, data=plan)

        except Exception as e:
            self.logger.error(f"Error creating financial plan: {e}")
            return PlanningResult(success=False, error=str(e))

    def _calculate_net_worth(self, assets: Dict) -> float:
        """Calculate total net worth from assets and liabilities."""
        total_assets = sum(
            asset["value"] for asset in assets.get("assets", [])
        )
        total_liabilities = sum(
            liability["value"] for liability in assets.get("liabilities", [])
        )
        return total_assets - total_liabilities

    def _calculate_cash_flow(
        self,
        income_sources: List[Dict],
        expenses: List[Dict]
    ) -> Dict:
        """Calculate monthly cash flow."""
        total_income = sum(source["monthly_amount"] for source in income_sources)
        total_expenses = sum(expense["monthly_amount"] for expense in expenses)
        
        return {
            "total_income": total_income,
            "total_expenses": total_expenses,
            "net_cash_flow": total_income - total_expenses
        }

    def analyze_retirement_needs(
        self,
        current_age: int,
        retirement_age: int,
        life_expectancy: int,
        current_savings: float,
        monthly_savings: float,
        monthly_expenses: float,
        inflation_rate: float = 0.025,
        investment_return: float = 0.06,
        social_security: Optional[float] = None
    ) -> PlanningResult:
        """
        Analyze retirement needs and funding.
        
        Args:
            current_age: Current age
            retirement_age: Target retirement age
            life_expectancy: Expected life span
            current_savings: Current retirement savings
            monthly_savings: Monthly savings amount
            monthly_expenses: Current monthly expenses
            inflation_rate: Expected inflation rate
            investment_return: Expected investment return
            social_security: Expected monthly social security benefit
            
        Returns:
            PlanningResult object with retirement analysis
        """
        try:
            # Calculate time periods
            years_to_retirement = retirement_age - current_age
            retirement_years = life_expectancy - retirement_age
            
            # Project retirement expenses
            retirement_expenses = monthly_expenses * 12 * (1 + inflation_rate) ** years_to_retirement
            
            # Calculate total retirement needs
            total_needs = 0
            annual_expenses = retirement_expenses
            for year in range(retirement_years):
                total_needs += annual_expenses
                annual_expenses *= (1 + inflation_rate)
            
            # Project retirement savings
            projected_savings = current_savings
            annual_savings = monthly_savings * 12
            
            for year in range(years_to_retirement):
                projected_savings *= (1 + investment_return)
                projected_savings += annual_savings
                annual_savings *= (1 + inflation_rate)
            
            # Calculate social security benefits if provided
            ss_benefits = 0
            if social_security:
                monthly_ss = social_security * (1 + inflation_rate) ** years_to_retirement
                annual_ss = monthly_ss * 12
                for year in range(retirement_years):
                    ss_benefits += annual_ss
                    annual_ss *= (1 + inflation_rate)
            
            # Analyze results
            total_funding = projected_savings + ss_benefits
            funding_gap = total_needs - total_funding
            funding_ratio = (total_funding / total_needs) * 100 if total_needs > 0 else 0
            
            analysis = {
                "retirement_age": retirement_age,
                "years_to_retirement": years_to_retirement,
                "retirement_years": retirement_years,
                "total_needs": total_needs,
                "projected_savings": projected_savings,
                "social_security_benefits": ss_benefits,
                "total_funding": total_funding,
                "funding_gap": funding_gap,
                "funding_ratio": funding_ratio,
                "monthly_retirement_expenses": retirement_expenses / 12,
                "assumptions": {
                    "inflation_rate": inflation_rate,
                    "investment_return": investment_return,
                    "life_expectancy": life_expectancy
                }
            }
            
            return PlanningResult(success=True, data=analysis)

        except Exception as e:
            self.logger.error(f"Error analyzing retirement needs: {e}")
            return PlanningResult(success=False, error=str(e))

    def track_financial_goals(
        self,
        goals: List[Dict],
        current_savings: Dict[str, float],
        monthly_contributions: Dict[str, float]
    ) -> PlanningResult:
        """
        Track progress towards financial goals.
        
        Args:
            goals: List of financial goals
            current_savings: Current savings for each goal
            monthly_contributions: Monthly contributions to each goal
            
        Returns:
            PlanningResult object with goal tracking analysis
        """
        try:
            tracking_results = []
            
            for goal in goals:
                goal_id = goal["id"]
                target_amount = goal["target_amount"]
                target_date = datetime.fromisoformat(goal["target_date"])
                current_amount = current_savings.get(goal_id, 0)
                monthly_contribution = monthly_contributions.get(goal_id, 0)
                
                # Calculate time remaining
                months_remaining = (target_date - datetime.now()).days / 30.44  # Average days per month
                
                # Project final amount
                projected_amount = self._project_savings(
                    current_amount,
                    monthly_contribution,
                    months_remaining,
                    goal.get("expected_return", 0.04) / 12  # Monthly return
                )
                
                # Calculate required monthly savings
                required_monthly = self._calculate_required_savings(
                    target_amount,
                    current_amount,
                    months_remaining,
                    goal.get("expected_return", 0.04) / 12
                )
                
                # Analyze progress
                progress_pct = (current_amount / target_amount) * 100
                on_track = projected_amount >= target_amount
                
                tracking_results.append({
                    "goal_id": goal_id,
                    "goal_name": goal["name"],
                    "target_amount": target_amount,
                    "target_date": goal["target_date"],
                    "current_amount": current_amount,
                    "progress_pct": progress_pct,
                    "projected_amount": projected_amount,
                    "monthly_contribution": monthly_contribution,
                    "required_monthly": required_monthly,
                    "months_remaining": months_remaining,
                    "on_track": on_track,
                    "funding_gap": target_amount - projected_amount
                })
            
            return PlanningResult(success=True, data=tracking_results)

        except Exception as e:
            self.logger.error(f"Error tracking goals: {e}")
            return PlanningResult(success=False, error=str(e))

    def _project_savings(
        self,
        current_amount: float,
        monthly_contribution: float,
        months: float,
        monthly_return: float
    ) -> float:
        """Project future savings with regular contributions."""
        future_value = current_amount * (1 + monthly_return) ** months
        
        if monthly_contribution > 0 and monthly_return > 0:
            # Future value of annuity formula
            fv_contributions = monthly_contribution * (
                ((1 + monthly_return) ** months - 1) / monthly_return
            )
            future_value += fv_contributions
        else:
            future_value += monthly_contribution * months
        
        return future_value

    def _calculate_required_savings(
        self,
        target_amount: float,
        current_amount: float,
        months: float,
        monthly_return: float
    ) -> float:
        """Calculate required monthly savings to reach target."""
        if months <= 0:
            return float('inf')
        
        if monthly_return == 0:
            return (target_amount - current_amount) / months
        
        # Solve for PMT in future value formula
        future_value_factor = (1 + monthly_return) ** months
        required_monthly = (
            (target_amount - current_amount * future_value_factor) *
            monthly_return / (future_value_factor - 1)
        )
        
        return max(0, required_monthly)

    def analyze_insurance_needs(
        self,
        personal_info: Dict,
        financial_info: Dict,
        current_coverage: Dict
    ) -> PlanningResult:
        """
        Analyze insurance needs.
        
        Args:
            personal_info: Personal information
            financial_info: Financial details
            current_coverage: Current insurance coverage
            
        Returns:
            PlanningResult object with insurance needs analysis
        """
        try:
            # Life insurance needs analysis
            life_insurance_needs = self._analyze_life_insurance_needs(
                financial_info["annual_income"],
                financial_info["total_debt"],
                financial_info.get("education_needs", 0),
                financial_info.get("funeral_costs", 25000),
                personal_info.get("years_to_retirement", 20)
            )
            
            # Disability insurance needs
            disability_needs = self._analyze_disability_insurance_needs(
                financial_info["annual_income"],
                financial_info["monthly_expenses"],
                personal_info.get("years_to_retirement", 20)
            )
            
            # Health insurance analysis
            health_insurance = self._analyze_health_insurance_needs(
                personal_info["age"],
                personal_info.get("health_conditions", []),
                financial_info["monthly_expenses"]
            )
            
            # Compare with current coverage
            analysis = {
                "life_insurance": {
                    "recommended": life_insurance_needs,
                    "current": current_coverage.get("life", 0),
                    "gap": life_insurance_needs - current_coverage.get("life", 0)
                },
                "disability_insurance": {
                    "recommended": disability_needs,
                    "current": current_coverage.get("disability", 0),
                    "gap": disability_needs - current_coverage.get("disability", 0)
                },
                "health_insurance": health_insurance
            }
            
            return PlanningResult(success=True, data=analysis)

        except Exception as e:
            self.logger.error(f"Error analyzing insurance needs: {e}")
            return PlanningResult(success=False, error=str(e))

    def _analyze_life_insurance_needs(
        self,
        annual_income: float,
        total_debt: float,
        education_needs: float,
        funeral_costs: float,
        years_needed: int
    ) -> float:
        """Calculate life insurance needs."""
        # Income replacement (70% of income for specified years)
        income_replacement = (annual_income * 0.7) * years_needed
        
        # Add other needs
        total_needs = (
            income_replacement +
            total_debt +
            education_needs +
            funeral_costs
        )
        
        return total_needs

    def _analyze_disability_insurance_needs(
        self,
        annual_income: float,
        monthly_expenses: float,
        years_needed: int
    ) -> float:
        """Calculate disability insurance needs."""
        # Aim to replace 60% of income
        monthly_benefit = (annual_income * 0.6) / 12
        
        # Ensure it covers essential expenses
        monthly_benefit = max(monthly_benefit, monthly_expenses)
        
        return monthly_benefit

    def _analyze_health_insurance_needs(
        self,
        age: int,
        health_conditions: List[str],
        monthly_expenses: float
    ) -> Dict:
        """Analyze health insurance needs."""
        # Basic recommendations
        recommended_deductible = min(monthly_expenses * 3, 5000)
        
        # Adjust based on age and health conditions
        risk_level = "low"
        if age > 50 or health_conditions:
            risk_level = "high"
            recommended_deductible = min(recommended_deductible, 2000)
        elif age > 30:
            risk_level = "medium"
        
        return {
            "risk_level": risk_level,
            "recommended_deductible": recommended_deductible,
            "hsa_eligible": recommended_deductible >= 1400,
            "recommended_coverage": {
                "deductible": recommended_deductible,
                "out_of_pocket_max": recommended_deductible * 2,
                "prescription_coverage": risk_level == "high",
                "dental": True,
                "vision": True
            }
        }

    def generate_financial_projections(
        self,
        current_finances: Dict,
        assumptions: Dict,
        projection_years: int = 30
    ) -> PlanningResult:
        """
        Generate long-term financial projections.
        
        Args:
            current_finances: Current financial state
            assumptions: Growth and inflation assumptions
            projection_years: Number of years to project
            
        Returns:
            PlanningResult object with financial projections
        """
        try:
            # Initialize projection arrays
            years = range(projection_years + 1)
            income = np.zeros(projection_years + 1)
            expenses = np.zeros(projection_years + 1)
            savings = np.zeros(projection_years + 1)
            investments = np.zeros(projection_years + 1)
            net_worth = np.zeros(projection_years + 1)
            
            # Set initial values
            income[0] = current_finances["annual_income"]
            expenses[0] = current_finances["annual_expenses"]
            savings[0] = current_finances["current_savings"]
            investments[0] = current_finances["current_investments"]
            net_worth[0] = current_finances["net_worth"]
            
            # Project future values
            for year in range(1, projection_years + 1):
                # Income growth
                income[year] = income[year-1] * (1 + assumptions["income_growth"])
                
                # Expense growth
                expenses[year] = expenses[year-1] * (1 + assumptions["inflation"])
                
                # Investment returns
                investment_return = investments[year-1] * assumptions["investment_return"]
                
                # New savings
                new_savings = (income[year] - expenses[year]) * assumptions["savings_rate"]
                
                # Update positions
                savings[year] = savings[year-1] + new_savings
                investments[year] = (
                    investments[year-1] +
                    investment_return +
                    (savings[year-1] * assumptions["investment_allocation"])
                )
                savings[year] -= (savings[year-1] * assumptions["investment_allocation"])
                
                # Net worth
                net_worth[year] = savings[year] + investments[year]
            
            # Create projection data
            projections = pd.DataFrame({
                "year": years,
                "income": income,
                "expenses": expenses,
                "savings": savings,
                "investments": investments,
                "net_worth": net_worth
            })
            
            # Calculate key metrics
            analysis = {
                "projections": projections.to_dict(orient="records"),
                "summary": {
                    "ending_net_worth": net_worth[-1],
                    "total_investment_growth": investments[-1] - investments[0],
                    "average_savings_rate": (
                        (savings + investments).diff().mean() / income.mean()
                    ),
                    "wealth_multiple": net_worth[-1] / income[-1]
                }
            }
            
            return PlanningResult(success=True, data=analysis)

        except Exception as e:
            self.logger.error(f"Error generating projections: {e}")
            return PlanningResult(success=False, error=str(e))

    def generate_planning_report(
        self,
        plan: Dict,
        include_projections: bool = True
    ) -> PlanningResult:
        """
        Generate a comprehensive financial planning report.
        
        Args:
            plan: Financial plan data
            include_projections: Whether to include financial projections
            
        Returns:
            PlanningResult object with report
        """
        try:
            # Analyze retirement needs
            retirement_analysis = self.analyze_retirement_needs(
                plan["personal_info"]["age"],
                plan["personal_info"].get("retirement_age", 65),
                plan["personal_info"].get("life_expectancy", 85),
                plan["current_assets"].get("retirement_savings", 0),
                plan["monthly_cash_flow"]["net_cash_flow"] * plan.get("savings_rate", 0.2),
                sum(e["monthly_amount"] for e in plan["expenses"]),
                social_security=plan.get("social_security_benefit")
            )
            
            # Track financial goals
            goals_analysis = self.track_financial_goals(
                plan["financial_goals"],
                {g["id"]: g.get("current_amount", 0) for g in plan["financial_goals"]},
                {g["id"]: g.get("monthly_contribution", 0) for g in plan["financial_goals"]}
            )
            
            # Insurance needs analysis
            insurance_analysis = self.analyze_insurance_needs(
                plan["personal_info"],
                {
                    "annual_income": sum(s["monthly_amount"] * 12 for s in plan["income_sources"]),
                    "total_debt": sum(l["value"] for l in plan["current_assets"].get("liabilities", [])),
                    "monthly_expenses": sum(e["monthly_amount"] for e in plan["expenses"])
                },
                plan.get("current_insurance", {})
            )
            
            # Financial projections
            projections = None
            if include_projections:
                projections_analysis = self.generate_financial_projections(
                    {
                        "annual_income": sum(s["monthly_amount"] * 12 for s in plan["income_sources"]),
                        "annual_expenses": sum(e["monthly_amount"] * 12 for e in plan["expenses"]),
                        "current_savings": plan["current_assets"].get("cash", 0),
                        "current_investments": sum(
                            a["value"] for a in plan["current_assets"].get("assets", [])
                            if a["type"] in ["stocks", "bonds", "mutual_funds"]
                        ),
                        "net_worth": plan["net_worth"]
                    },
                    {
                        "income_growth": 0.03,
                        "inflation": 0.025,
                        "investment_return": 0.06,
                        "savings_rate": 0.2,
                        "investment_allocation": 0.8
                    }
                )
                if projections_analysis.success:
                    projections = projections_analysis.data
            
            # Compile report
            report = {
                "summary": {
                    "name": plan["personal_info"]["name"],
                    "age": plan["personal_info"]["age"],
                    "net_worth": plan["net_worth"],
                    "monthly_cash_flow": plan["monthly_cash_flow"],
                    "total_assets": sum(a["value"] for a in plan["current_assets"].get("assets", [])),
                    "total_liabilities": sum(l["value"] for l in plan["current_assets"].get("liabilities", []))
                },
                "retirement_analysis": retirement_analysis.data if retirement_analysis.success else None,
                "goals_tracking": goals_analysis.data if goals_analysis.success else None,
                "insurance_analysis": insurance_analysis.data if insurance_analysis.success else None,
                "financial_projections": projections,
                "recommendations": self._generate_recommendations(
                    retirement_analysis.data if retirement_analysis.success else None,
                    goals_analysis.data if goals_analysis.success else None,
                    insurance_analysis.data if insurance_analysis.success else None
                )
            }
            
            return PlanningResult(success=True, data=report)

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return PlanningResult(success=False, error=str(e))

    def _generate_recommendations(
        self,
        retirement_analysis: Optional[Dict],
        goals_analysis: Optional[List[Dict]],
        insurance_analysis: Optional[Dict]
    ) -> List[Dict]:
        """Generate prioritized recommendations based on analyses."""
        recommendations = []
        
        # Retirement recommendations
        if retirement_analysis:
            if retirement_analysis["funding_ratio"] < 75:
                recommendations.append({
                    "priority": "high",
                    "category": "retirement",
                    "recommendation": "Increase retirement savings",
                    "details": (
                        f"Current funding ratio is {retirement_analysis['funding_ratio']:.1f}%. "
                        f"Consider increasing monthly savings by "
                        f"${-retirement_analysis['funding_gap']/retirement_analysis['years_to_retirement']/12:,.2f}"
                    )
                })
        
        # Goals recommendations
        if goals_analysis:
            for goal in goals_analysis:
                if not goal["on_track"]:
                    recommendations.append({
                        "priority": "medium",
                        "category": "goals",
                        "recommendation": f"Increase savings for {goal['goal_name']}",
                        "details": (
                            f"Currently {goal['progress_pct']:.1f}% funded. "
                            f"Consider increasing monthly contribution by "
                            f"${goal['required_monthly'] - goal['monthly_contribution']:,.2f}"
                        )
                    })
        
        # Insurance recommendations
        if insurance_analysis:
            for category, analysis in insurance_analysis.items():
                if category != "health_insurance" and analysis.get("gap", 0) > 0:
                    recommendations.append({
                        "priority": "high",
                        "category": "insurance",
                        "recommendation": f"Increase {category} coverage",
                        "details": (
                            f"Current coverage gap of ${analysis['gap']:,.2f}. "
                            f"Consider increasing coverage to ${analysis['recommended']:,.2f}"
                        )
                    })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order[x["priority"]])
        
        return recommendations 