"""
Tax Planning Service for tax calculations, optimization strategies, and tax reporting.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class TaxResult:
    """Structured result object for tax planning operations."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    source: str = "tax"
    timestamp: datetime = datetime.now()

class TaxPlanningService:
    """
    Service for tax planning and analysis.
    Features:
    - Tax calculations
    - Tax optimization strategies
    - Deduction analysis
    - Tax reporting
    - Capital gains planning
    """

    def __init__(self, portfolio_service=None):
        self.portfolio_service = portfolio_service
        self._setup_logging()
        self._load_tax_rates()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TaxPlanningService")

    def _load_tax_rates(self) -> None:
        """Load current tax rates and thresholds."""
        # 2024 Australian tax rates
        self.tax_rates = {
            "individual": [
                {"threshold": 18200, "rate": 0.0, "base": 0},
                {"threshold": 45000, "rate": 0.19, "base": 0},
                {"threshold": 120000, "rate": 0.325, "base": 5092},
                {"threshold": 180000, "rate": 0.37, "base": 29467},
                {"threshold": float('inf'), "rate": 0.45, "base": 51667}
            ],
            "medicare_levy": 0.02,
            "capital_gains": {
                "individual": 0.5,  # 50% discount for assets held > 12 months
                "super": 0.333  # 33.33% discount for super funds
            }
        }

    def calculate_income_tax(
        self,
        taxable_income: float,
        tax_year: int = 2024
    ) -> TaxResult:
        """
        Calculate income tax.
        
        Args:
            taxable_income: Annual taxable income
            tax_year: Tax year
            
        Returns:
            TaxResult object with tax calculation
        """
        try:
            # Find applicable tax bracket
            tax_payable = 0
            previous_threshold = 0
            
            for bracket in self.tax_rates["individual"]:
                if taxable_income > previous_threshold:
                    taxable_in_bracket = min(
                        taxable_income - previous_threshold,
                        bracket["threshold"] - previous_threshold
                    )
                    tax_payable = bracket["base"] + (taxable_in_bracket * bracket["rate"])
                    
                    if taxable_income <= bracket["threshold"]:
                        break
                    
                previous_threshold = bracket["threshold"]
            
            # Calculate Medicare levy
            medicare_levy = taxable_income * self.tax_rates["medicare_levy"]
            
            # Calculate effective tax rate
            effective_rate = (
                (tax_payable + medicare_levy) / taxable_income
                if taxable_income > 0 else 0
            )
            
            result = {
                "taxable_income": taxable_income,
                "tax_payable": tax_payable,
                "medicare_levy": medicare_levy,
                "total_tax": tax_payable + medicare_levy,
                "effective_rate": effective_rate,
                "tax_year": tax_year,
                "take_home": taxable_income - (tax_payable + medicare_levy)
            }
            
            return TaxResult(success=True, data=result)

        except Exception as e:
            self.logger.error(f"Error calculating income tax: {e}")
            return TaxResult(success=False, error=str(e))

    def analyze_deductions(
        self,
        income_details: Dict,
        expenses: List[Dict]
    ) -> TaxResult:
        """
        Analyze potential tax deductions.
        
        Args:
            income_details: Income information
            expenses: List of expenses
            
        Returns:
            TaxResult object with deduction analysis
        """
        try:
            deductions = {
                "work_related": [],
                "investment": [],
                "other": []
            }
            
            total_deductions = 0
            
            for expense in expenses:
                category = expense["category"]
                amount = expense["amount"]
                
                if self._is_deductible(expense):
                    if category in ["work_equipment", "work_travel", "work_training"]:
                        deductions["work_related"].append(expense)
                    elif category in ["investment_interest", "property_expenses"]:
                        deductions["investment"].append(expense)
                    else:
                        deductions["other"].append(expense)
                    
                    total_deductions += amount
            
            # Calculate tax savings
            current_tax = self.calculate_income_tax(
                income_details["taxable_income"]
            )
            tax_with_deductions = self.calculate_income_tax(
                income_details["taxable_income"] - total_deductions
            )
            
            if not current_tax.success or not tax_with_deductions.success:
                return TaxResult(
                    success=False,
                    error="Error calculating tax impact"
                )
            
            tax_savings = (
                current_tax.data["total_tax"] -
                tax_with_deductions.data["total_tax"]
            )
            
            analysis = {
                "total_deductions": total_deductions,
                "deductions_by_category": {
                    "work_related": sum(d["amount"] for d in deductions["work_related"]),
                    "investment": sum(d["amount"] for d in deductions["investment"]),
                    "other": sum(d["amount"] for d in deductions["other"])
                },
                "tax_savings": tax_savings,
                "detailed_deductions": deductions,
                "recommendations": self._generate_deduction_recommendations(
                    deductions,
                    income_details
                )
            }
            
            return TaxResult(success=True, data=analysis)

        except Exception as e:
            self.logger.error(f"Error analyzing deductions: {e}")
            return TaxResult(success=False, error=str(e))

    def _is_deductible(self, expense: Dict) -> bool:
        """Check if an expense is tax deductible."""
        deductible_categories = {
            "work_equipment",
            "work_travel",
            "work_training",
            "investment_interest",
            "property_expenses",
            "donations",
            "self_education",
            "home_office"
        }
        
        return (
            expense["category"] in deductible_categories and
            expense.get("purpose") == "work_related"
        )

    def calculate_capital_gains(
        self,
        transactions: List[Dict],
        holding_period: int  # days
    ) -> TaxResult:
        """
        Calculate capital gains tax.
        
        Args:
            transactions: List of buy/sell transactions
            holding_period: Number of days held
            
        Returns:
            TaxResult object with CGT calculation
        """
        try:
            total_proceeds = sum(
                t["amount"] for t in transactions
                if t["type"] == "sell"
            )
            total_cost = sum(
                t["amount"] for t in transactions
                if t["type"] == "buy"
            )
            
            capital_gain = total_proceeds - total_cost
            
            # Apply CGT discount if eligible
            discount = 0
            if holding_period > 365:  # 12 months
                discount = capital_gain * self.tax_rates["capital_gains"]["individual"]
            
            taxable_gain = capital_gain - discount
            
            result = {
                "total_proceeds": total_proceeds,
                "total_cost": total_cost,
                "capital_gain": capital_gain,
                "cgt_discount": discount,
                "taxable_gain": taxable_gain,
                "holding_period_days": holding_period,
                "discount_eligible": holding_period > 365
            }
            
            return TaxResult(success=True, data=result)

        except Exception as e:
            self.logger.error(f"Error calculating capital gains: {e}")
            return TaxResult(success=False, error=str(e))

    def optimize_tax_strategy(
        self,
        income_details: Dict,
        investments: Dict,
        super_info: Dict
    ) -> TaxResult:
        """
        Generate tax optimization strategies.
        
        Args:
            income_details: Income information
            investments: Investment portfolio details
            super_info: Superannuation information
            
        Returns:
            TaxResult object with optimization strategies
        """
        try:
            strategies = []
            
            # Analyze super contributions
            concessional_cap = 27500  # 2024 cap
            current_contributions = super_info.get("employer_contributions", 0)
            remaining_cap = concessional_cap - current_contributions
            
            if remaining_cap > 0:
                tax_saving = remaining_cap * (
                    self._get_marginal_rate(income_details["taxable_income"]) - 0.15
                )
                if tax_saving > 0:
                    strategies.append({
                        "type": "super_contribution",
                        "action": "Make additional concessional contributions",
                        "amount": remaining_cap,
                        "tax_saving": tax_saving,
                        "priority": "high" if tax_saving > 1000 else "medium"
                    })
            
            # Analyze investment structure
            if income_details["taxable_income"] > 120000:  # High tax bracket
                strategies.append({
                    "type": "investment_structure",
                    "action": "Consider investment bond for long-term investments",
                    "reason": "Cap tax rate at 30% vs marginal rate of 37-45%",
                    "priority": "medium"
                })
            
            # Analyze capital gains
            unrealized_gains = investments.get("unrealized_gains", 0)
            if unrealized_gains > 0:
                strategies.append({
                    "type": "capital_gains",
                    "action": "Review timing of asset sales",
                    "details": (
                        "Consider spreading sales across tax years or offsetting with losses"
                    ),
                    "priority": "high" if unrealized_gains > 10000 else "medium"
                })
            
            # Analyze deductions
            if income_details.get("work_from_home", False):
                strategies.append({
                    "type": "deductions",
                    "action": "Track home office expenses",
                    "details": "Can claim $0.52 per hour or actual costs method",
                    "priority": "medium"
                })
            
            return TaxResult(success=True, data={
                "strategies": strategies,
                "potential_savings": sum(
                    s.get("tax_saving", 0) for s in strategies
                )
            })

        except Exception as e:
            self.logger.error(f"Error optimizing tax strategy: {e}")
            return TaxResult(success=False, error=str(e))

    def _get_marginal_rate(self, taxable_income: float) -> float:
        """Get marginal tax rate for income level."""
        for bracket in reversed(self.tax_rates["individual"]):
            if taxable_income > bracket["threshold"]:
                return bracket["rate"]
        return 0.0

    def _generate_deduction_recommendations(
        self,
        deductions: Dict,
        income_details: Dict
    ) -> List[Dict]:
        """Generate recommendations for maximizing deductions."""
        recommendations = []
        
        # Work-related expenses
        if income_details.get("employment_type") == "employee":
            if not any(d["category"] == "work_training" for d in deductions["work_related"]):
                recommendations.append({
                    "type": "work_related",
                    "action": "Consider work-related training opportunities",
                    "details": "Professional development costs may be tax deductible"
                })
        
        # Investment deductions
        if deductions["investment"]:
            recommendations.append({
                "type": "investment",
                "action": "Keep detailed investment expense records",
                "details": "Include interest, management fees, and property expenses"
            })
        
        # Home office
        if income_details.get("work_from_home", False):
            has_home_office = any(
                d["category"] == "home_office"
                for d in deductions["work_related"]
            )
            if not has_home_office:
                recommendations.append({
                    "type": "home_office",
                    "action": "Track home office hours and expenses",
                    "details": "Can claim using fixed rate or actual cost methods"
                })
        
        return recommendations

    def generate_tax_report(
        self,
        tax_year: int,
        income_details: Dict,
        deductions: Dict,
        capital_gains: Dict
    ) -> TaxResult:
        """
        Generate comprehensive tax report.
        
        Args:
            tax_year: Tax year
            income_details: Income information
            deductions: Deduction details
            capital_gains: Capital gains information
            
        Returns:
            TaxResult object with tax report
        """
        try:
            # Calculate total tax
            taxable_income = (
                income_details["assessable_income"] -
                deductions["total_deductions"] +
                capital_gains["taxable_gain"]
            )
            
            tax_result = self.calculate_income_tax(taxable_income, tax_year)
            if not tax_result.success:
                return tax_result
            
            # Generate report
            report = {
                "tax_year": tax_year,
                "summary": {
                    "assessable_income": income_details["assessable_income"],
                    "total_deductions": deductions["total_deductions"],
                    "taxable_income": taxable_income,
                    "tax_payable": tax_result.data["tax_payable"],
                    "medicare_levy": tax_result.data["medicare_levy"],
                    "total_tax": tax_result.data["total_tax"],
                    "effective_rate": tax_result.data["effective_rate"]
                },
                "income_details": {
                    "salary_wages": income_details.get("salary_wages", 0),
                    "investment_income": income_details.get("investment_income", 0),
                    "business_income": income_details.get("business_income", 0),
                    "other_income": income_details.get("other_income", 0)
                },
                "deductions_summary": {
                    "work_related": deductions["deductions_by_category"]["work_related"],
                    "investment": deductions["deductions_by_category"]["investment"],
                    "other": deductions["deductions_by_category"]["other"]
                },
                "capital_gains_summary": {
                    "total_gain": capital_gains["capital_gain"],
                    "discount": capital_gains["cgt_discount"],
                    "taxable_gain": capital_gains["taxable_gain"]
                },
                "tax_offsets": income_details.get("tax_offsets", {}),
                "medicare_details": {
                    "levy": tax_result.data["medicare_levy"],
                    "surcharge": self._calculate_medicare_surcharge(
                        taxable_income,
                        income_details.get("private_health", False)
                    )
                }
            }
            
            return TaxResult(success=True, data=report)

        except Exception as e:
            self.logger.error(f"Error generating tax report: {e}")
            return TaxResult(success=False, error=str(e))

    def _calculate_medicare_surcharge(
        self,
        taxable_income: float,
        has_private_health: bool
    ) -> float:
        """Calculate Medicare Levy Surcharge if applicable."""
        if has_private_health:
            return 0.0
            
        if taxable_income <= 90000:
            return 0.0
        elif taxable_income <= 105000:
            return taxable_income * 0.01
        elif taxable_income <= 140000:
            return taxable_income * 0.0125
        else:
            return taxable_income * 0.015 