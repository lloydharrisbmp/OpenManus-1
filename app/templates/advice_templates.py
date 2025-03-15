"""
Template management system for financial advice statements with advanced features.
Provides a structured way to store, retrieve, and combine different sections
of financial advice templates with advanced sectioning logic.
"""
from typing import Dict, List, Optional, Any, Set, Callable, Union, Pattern
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from abc import ABC, abstractmethod

class TemplateCategory(Enum):
    """Categories for different types of advice templates."""
    INVESTMENT_STRATEGY = "investment_strategy"
    RETIREMENT_PLANNING = "retirement_planning"
    RISK_MANAGEMENT = "risk_management"
    ESTATE_PLANNING = "estate_planning"
    SUPERANNUATION = "superannuation"
    INSURANCE = "insurance"
    TAX_STRATEGY = "tax_strategy"
    CASH_FLOW = "cash_flow"
    DEBT_MANAGEMENT = "debt_management"
    GENERAL_ADVICE = "general_advice"

class ConditionOperator(Enum):
    """Operators for condition evaluation."""
    EQUALS = "=="
    GREATER = ">"
    LESS = "<"
    IN = "in"
    CONTAINS = "contains"
    BETWEEN = "between"
    MATCHES = "matches"  # Regex matching
    EXISTS = "exists"

class BaseCondition(ABC):
    """Base class for all condition types."""
    @abstractmethod
    def evaluate(self, parameters: Dict[str, Any], client_context: Dict[str, Any]) -> bool:
        pass

@dataclass
class SimpleCondition(BaseCondition):
    """Simple field comparison condition."""
    field: str
    operator: ConditionOperator
    value: Any
    source: str = "parameters"

    def evaluate(self, parameters: Dict[str, Any], client_context: Dict[str, Any]) -> bool:
        source_data = parameters if self.source == "parameters" else client_context
        if self.operator == ConditionOperator.EXISTS:
            return self.field in source_data
            
        if self.field not in source_data:
            return False
            
        actual_value = source_data[self.field]
        
        if self.operator == ConditionOperator.EQUALS:
            return actual_value == self.value
        elif self.operator == ConditionOperator.GREATER:
            return actual_value > self.value
        elif self.operator == ConditionOperator.LESS:
            return actual_value < self.value
        elif self.operator == ConditionOperator.IN:
            return actual_value in self.value
        elif self.operator == ConditionOperator.CONTAINS:
            return self.value in actual_value
        elif self.operator == ConditionOperator.BETWEEN:
            min_val, max_val = self.value
            return min_val <= actual_value <= max_val
        elif self.operator == ConditionOperator.MATCHES:
            return bool(re.match(self.value, str(actual_value)))
        return False

@dataclass
class CompositeCondition(BaseCondition):
    """Composite condition for AND/OR logic."""
    conditions: List[BaseCondition]
    operator: str = "AND"  # "AND" or "OR"

    def evaluate(self, parameters: Dict[str, Any], client_context: Dict[str, Any]) -> bool:
        if self.operator == "AND":
            return all(
                condition.evaluate(parameters, client_context)
                for condition in self.conditions
            )
        else:  # OR
            return any(
                condition.evaluate(parameters, client_context)
                for condition in self.conditions
            )

@dataclass
class NotCondition(BaseCondition):
    """Negates another condition."""
    condition: BaseCondition

    def evaluate(self, parameters: Dict[str, Any], client_context: Dict[str, Any]) -> bool:
        return not self.condition.evaluate(parameters, client_context)

@dataclass
class RegexCondition(BaseCondition):
    """Condition that uses regex pattern matching."""
    field: str
    pattern: Union[str, Pattern]
    source: str = "parameters"

    def __post_init__(self):
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern)

    def evaluate(self, parameters: Dict[str, Any], client_context: Dict[str, Any]) -> bool:
        source_data = parameters if self.source == "parameters" else client_context
        if self.field not in source_data:
            return False
        return bool(self.pattern.match(str(source_data[self.field])))

@dataclass
class AnyOfCondition(BaseCondition):
    """Matches if the field matches any value in a list."""
    field: str
    values: List[Any]
    source: str = "parameters"

    def evaluate(self, parameters: Dict[str, Any], client_context: Dict[str, Any]) -> bool:
        source_data = parameters if self.source == "parameters" else client_context
        if self.field not in source_data:
            return False
        return source_data[self.field] in self.values

@dataclass
class AllCondition(BaseCondition):
    """Matches if all subconditions match."""
    conditions: List[BaseCondition]

    def evaluate(self, parameters: Dict[str, Any], client_context: Dict[str, Any]) -> bool:
        return all(c.evaluate(parameters, client_context) for c in self.conditions)

@dataclass
class AnyCondition(BaseCondition):
    """Matches if any subcondition matches."""
    conditions: List[BaseCondition]

    def evaluate(self, parameters: Dict[str, Any], client_context: Dict[str, Any]) -> bool:
        return any(c.evaluate(parameters, client_context) for c in self.conditions)

@dataclass
class NestedCondition(BaseCondition):
    """Evaluates nested fields using dot notation."""
    path: List[str]
    operator: ConditionOperator
    value: Any
    source: str = "parameters"

    def evaluate(self, parameters: Dict[str, Any], client_context: Dict[str, Any]) -> bool:
        source_data = parameters if self.source == "parameters" else client_context
        current = source_data
        
        # Navigate through nested structure
        for key in self.path[:-1]:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]
            
        last_key = self.path[-1]
        if not isinstance(current, dict) or last_key not in current:
            return False
            
        actual_value = current[last_key]
        
        # Reuse SimpleCondition's evaluation logic
        temp_condition = SimpleCondition(
            field="temp",
            operator=self.operator,
            value=self.value
        )
        return temp_condition.evaluate({"temp": actual_value}, {})

@dataclass
class TemplateGroup:
    """Groups related templates together with advanced ordering."""
    id: str
    name: str
    description: str
    templates: List[str]  # Template IDs
    order_rules: Optional[BaseCondition] = None  # Condition for group ordering
    subgroups: List['TemplateGroup'] = field(default_factory=list)  # Nested groups
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional group metadata
    priority_calculator: Optional[Callable[[Dict[str, Any]], int]] = None  # Dynamic priority function

    def calculate_priority(self, client_context: Dict[str, Any]) -> int:
        """Calculate group priority based on rules and calculator."""
        base_priority = 0
        if self.order_rules and self.order_rules.evaluate({}, client_context):
            base_priority = 100
        
        if self.priority_calculator:
            base_priority += self.priority_calculator(client_context)
            
        return base_priority

@dataclass
class DynamicOrdering:
    """Configures dynamic ordering behavior."""
    priority_function: Callable[[Dict[str, Any]], int]
    condition: Optional[BaseCondition] = None
    metadata_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SectionRule:
    """Enhanced rules for template section inclusion and ordering."""
    conditions: BaseCondition  # Can be simple or composite
    dependencies: List[str]
    priority: int
    required: bool
    group: Optional[str] = None  # Group ID
    dynamic_priority: Optional[Callable[[Dict[str, Any]], int]] = None  # Function to compute priority
    ordering_rules: List[DynamicOrdering] = field(default_factory=list)  # Multiple ordering rules
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata for ordering

@dataclass
class TemplateSection:
    """Enhanced template section with grouping support."""
    id: str
    category: TemplateCategory
    title: str
    content: str
    parameters: List[str]
    tags: List[str]
    version: str
    last_updated: datetime
    compliance_approved: bool = False
    requires_review: bool = True
    rules: Optional[SectionRule] = None
    group: Optional[str] = None  # Group ID

class TemplateLibrary:
    """Enhanced template library with grouping support."""
    
    def __init__(self):
        self._templates: Dict[str, TemplateSection] = {}
        self._version_history: Dict[str, List[TemplateSection]] = {}
        self._groups: Dict[str, TemplateGroup] = {}
    
    def add_template(self, template: TemplateSection) -> None:
        """Add a new template section to the library."""
        if template.id in self._templates:
            # Store the current version in history
            if template.id not in self._version_history:
                self._version_history[template.id] = []
            self._version_history[template.id].append(self._templates[template.id])
        
        self._templates[template.id] = template

    def get_template(self, template_id: str) -> Optional[TemplateSection]:
        """Retrieve a template by its ID."""
        return self._templates.get(template_id)

    def get_templates_by_category(self, category: TemplateCategory) -> List[TemplateSection]:
        """Get all templates in a specific category."""
        return [
            template for template in self._templates.values()
            if template.category == category
        ]

    def get_templates_by_tags(self, tags: List[str]) -> List[TemplateSection]:
        """Get templates that match all provided tags."""
        return [
            template for template in self._templates.values()
            if all(tag in template.tags for tag in tags)
        ]

    def get_version_history(self, template_id: str) -> List[TemplateSection]:
        """Get the version history of a template."""
        return self._version_history.get(template_id, [])

    def add_group(self, group: TemplateGroup) -> None:
        """Add a template group."""
        self._groups[group.id] = group
    
    def get_group(self, group_id: str) -> Optional[TemplateGroup]:
        """Get a template group by ID."""
        return self._groups.get(group_id)
    
    def get_group_templates(self, group_id: str) -> List[TemplateSection]:
        """Get all templates in a group."""
        group = self._groups.get(group_id)
        if not group:
            return []
        return [
            self._templates[tid] for tid in group.templates
            if tid in self._templates
        ]

class AdviceBuilder:
    """Enhanced advice document builder with advanced ordering."""
    
    def __init__(self, template_library: TemplateLibrary):
        self.template_library = template_library
    
    def build_section(
        self,
        template_id: str,
        parameters: Dict[str, Any],
        client_context: Dict[str, Any]
    ) -> Optional[str]:
        """Build a single section of advice."""
        template = self.template_library.get_template(template_id)
        if not template:
            return None
            
        # Check if section should be included based on rules
        if template.rules and not self._evaluate_rules(
            template.rules, parameters, client_context
        ):
            return None
            
        content = template.content
        
        # Replace template parameters
        for param_name, param_value in parameters.items():
            placeholder = f"{{{param_name}}}"
            content = content.replace(placeholder, str(param_value))
            
        content = self._apply_client_context(content, client_context)
        return content
    
    def build_advice_document(
        self,
        template_ids: List[str],
        parameters: Dict[str, Dict[str, Any]],
        client_context: Dict[str, Any]
    ) -> str:
        """Build a complete advice document with advanced sectioning logic."""
        # Get all templates
        templates = [
            self.template_library.get_template(tid) 
            for tid in template_ids
        ]
        templates = [t for t in templates if t is not None]
        
        # Sort templates by priority and resolve dependencies
        ordered_templates = self._order_templates(templates, client_context)
        
        # Build each section that meets its conditions
        sections = []
        included_ids = set()
        
        for template in ordered_templates:
            if template.id not in parameters:
                continue
                
            section_params = parameters[template.id]
            
            # Check dependencies
            if template.rules and template.rules.dependencies:
                if not all(dep in included_ids for dep in template.rules.dependencies):
                    if template.rules.required:
                        raise ValueError(
                            f"Required section {template.id} is missing dependencies: "
                            f"{[dep for dep in template.rules.dependencies if dep not in included_ids]}"
                        )
                    continue
            
            content = self.build_section(
                template.id,
                section_params,
                client_context
            )
            
            if content:
                sections.append(content)
                included_ids.add(template.id)
        
        # Check if all required sections are included
        self._validate_required_sections(templates, included_ids)
        
        return "\n\n".join(sections)
    
    def _evaluate_rules(
        self,
        rules: SectionRule,
        parameters: Dict[str, Any],
        client_context: Dict[str, Any]
    ) -> bool:
        """Evaluate if a section's rules are satisfied."""
        return all(
            condition.evaluate(parameters, client_context)
            for condition in rules.conditions
        )
    
    def _compute_dynamic_priority(
        self,
        template: TemplateSection,
        client_context: Dict[str, Any]
    ) -> int:
        """Compute dynamic priority based on client data."""
        if not template.rules or not template.rules.dynamic_priority:
            return template.rules.priority if template.rules else 0
        return template.rules.dynamic_priority(client_context)
    
    def _order_templates(
        self,
        templates: List[TemplateSection],
        client_context: Dict[str, Any]
    ) -> List[TemplateSection]:
        """Enhanced template ordering with groups and dynamic priorities."""
        # Group templates
        grouped_templates: Dict[Optional[str], List[TemplateSection]] = {}
        for template in templates:
            group_id = template.group
            if group_id not in grouped_templates:
                grouped_templates[group_id] = []
            grouped_templates[group_id].append(template)
        
        # Order groups based on their conditions and dynamic priorities
        ordered_groups = sorted(
            grouped_templates.keys(),
            key=lambda g: self._evaluate_group_priority(g, client_context)
        )
        
        # Order templates within each group
        ordered_templates = []
        for group_id in ordered_groups:
            group_templates = grouped_templates[group_id]
            
            # Apply all ordering rules
            ordered_group_templates = self._apply_ordering_rules(
                group_templates,
                client_context
            )
            
            ordered_templates.extend(ordered_group_templates)
        
        return ordered_templates
    
    def _evaluate_group_priority(
        self,
        group_id: Optional[str],
        client_context: Dict[str, Any]
    ) -> int:
        """Evaluate group priority including dynamic calculations."""
        if not group_id:
            return -1  # Ungrouped templates come last
            
        group = self.template_library.get_group(group_id)
        if not group:
            return 0
            
        return group.calculate_priority(client_context)
    
    def _apply_ordering_rules(
        self,
        templates: List[TemplateSection],
        client_context: Dict[str, Any]
    ) -> List[TemplateSection]:
        """Apply all ordering rules to a group of templates."""
        # First sort by dynamic priority
        templates.sort(
            key=lambda t: self._compute_template_priority(t, client_context),
            reverse=True
        )
        
        # Then apply dependency ordering
        ordered = self._order_group_templates(templates)
        
        # Finally, apply any template-specific ordering rules
        return self._apply_template_ordering_rules(ordered, client_context)
    
    def _compute_template_priority(
        self,
        template: TemplateSection,
        client_context: Dict[str, Any]
    ) -> int:
        """Compute final template priority including all rules."""
        if not template.rules:
            return 0
            
        base_priority = template.rules.priority
        
        # Apply dynamic priority if available
        if template.rules.dynamic_priority:
            base_priority = template.rules.dynamic_priority(client_context)
        
        # Apply additional ordering rules
        for rule in template.rules.ordering_rules:
            if (not rule.condition or 
                rule.condition.evaluate({}, client_context)):
                if all(
                    template.rules.metadata.get(k) == v 
                    for k, v in rule.metadata_requirements.items()
                ):
                    base_priority += rule.priority_function(client_context)
        
        return base_priority
    
    def _apply_template_ordering_rules(
        self,
        templates: List[TemplateSection],
        client_context: Dict[str, Any]
    ) -> List[TemplateSection]:
        """Apply template-specific ordering rules."""
        # Create groups based on metadata requirements
        metadata_groups: Dict[str, List[TemplateSection]] = {}
        
        for template in templates:
            if not template.rules or not template.rules.metadata:
                continue
                
            key = json.dumps(
                template.rules.metadata,
                sort_keys=True
            )
            if key not in metadata_groups:
                metadata_groups[key] = []
            metadata_groups[key].append(template)
        
        # Order within metadata groups
        ordered_templates = []
        for group in metadata_groups.values():
            group.sort(
                key=lambda t: self._compute_template_priority(t, client_context),
                reverse=True
            )
            ordered_templates.extend(group)
        
        # Add remaining templates
        remaining = [
            t for t in templates 
            if not t.rules or not t.rules.metadata
        ]
        ordered_templates.extend(remaining)
        
        return ordered_templates
    
    def _validate_required_sections(
        self,
        templates: List[TemplateSection],
        included_ids: Set[str]
    ) -> None:
        """Validate that all required sections are included."""
        missing = []
        for template in templates:
            if (
                template.rules 
                and template.rules.required 
                and template.id not in included_ids
            ):
                missing.append(template.id)
        
        if missing:
            raise ValueError(f"Required sections missing: {missing}")
    
    def _apply_client_context(self, content: str, context: Dict[str, Any]) -> str:
        """Apply client-specific customizations to the content."""
        # Replace client-specific placeholders
        for key, value in context.items():
            placeholder = f"{{client.{key}}}"
            content = content.replace(placeholder, str(value))
        
        return content

def create_sample_templates() -> TemplateLibrary:
    """Create sample templates with advanced features."""
    library = TemplateLibrary()
    
    # Create sophisticated template groups
    risk_group = TemplateGroup(
        id="risk_assessment",
        name="Risk Assessment",
        description="Risk profile and investment strategy templates",
        templates=[
            "risk_profile_assessment",
            "investment_strategy_balanced",
            "investment_strategy_growth",
            "investment_strategy_conservative"
        ],
        order_rules=CompositeCondition([
            SimpleCondition(
                "needs_risk_assessment",
                ConditionOperator.EQUALS,
                True,
                "client_context"
            ),
            AnyOfCondition(
                "risk_profile",
                ["Balanced", "Growth", "Conservative"],
                "client_context"
            )
        ], "AND"),
        priority_calculator=lambda ctx: 50 if ctx.get("age", 0) > 50 else 0,
        metadata={"category": "risk", "importance": "high"}
    )
    
    retirement_group = TemplateGroup(
        id="retirement_planning",
        name="Retirement Planning",
        description="Retirement strategy and superannuation templates",
        templates=[
            "retirement_goals",
            "super_contribution_strategy",
            "pension_strategy"
        ],
        order_rules=AllCondition([
            SimpleCondition(
                "age",
                ConditionOperator.GREATER,
                50,
                "client_context"
            ),
            NestedCondition(
                ["retirement", "planning_started"],
                ConditionOperator.EQUALS,
                True,
                "client_context"
            )
        ]),
        subgroups=[
            TemplateGroup(
                id="pension_planning",
                name="Pension Planning",
                description="Specific pension advice templates",
                templates=["pension_strategy", "centrelink_assessment"],
                order_rules=SimpleCondition(
                    "age",
                    ConditionOperator.GREATER,
                    60,
                    "client_context"
                )
            )
        ]
    )
    
    library.add_group(risk_group)
    library.add_group(retirement_group)
    
    # Add sophisticated templates with complex conditions
    library.add_template(TemplateSection(
        id="investment_strategy_balanced",
        category=TemplateCategory.INVESTMENT_STRATEGY,
        title="Balanced Investment Strategy",
        content="""
Based on our comprehensive assessment of your risk profile, investment objectives, and current market conditions,
we recommend a balanced investment strategy with the following characteristics:

Asset Allocation:
- Growth Assets: {equity_allocation}%
- Defensive Assets: {defensive_allocation}%

This strategy aims to provide a balance between capital growth and income generation over your
investment timeframe of {investment_timeframe} years.

Portfolio Composition:
- Australian Shares: {aus_shares}%
- International Shares: {intl_shares}%
- Property: {property}%
- Fixed Interest: {fixed_interest}%
- Cash: {cash}%

Key Considerations:
1. Your risk tolerance assessment indicates a {client.risk_profile} profile
2. Your investment timeframe of {investment_timeframe} years aligns with this strategy
3. Current market conditions and economic outlook have been considered
4. The strategy accounts for your current life stage and retirement planning needs

Regular Review:
This strategy should be reviewed {review_frequency} or when your circumstances change significantly.
        """.strip(),
        parameters=[
            "equity_allocation",
            "defensive_allocation",
            "investment_timeframe",
            "aus_shares",
            "intl_shares",
            "property",
            "fixed_interest",
            "cash",
            "review_frequency"
        ],
        tags=["investment", "asset_allocation", "balanced"],
        version="2.0",
        last_updated=datetime.now(),
        compliance_approved=True,
        group="risk_assessment",
        rules=SectionRule(
            conditions=CompositeCondition([
                SimpleCondition(
                    "risk_profile",
                    ConditionOperator.EQUALS,
                    "Balanced",
                    "client_context"
                ),
                AllCondition([
                    SimpleCondition(
                        "investment_timeframe",
                        ConditionOperator.GREATER,
                        5,
                        "parameters"
                    ),
                    SimpleCondition(
                        "investment_amount",
                        ConditionOperator.BETWEEN,
                        (100000, 1000000),
                        "parameters"
                    ),
                    NotCondition(
                        SimpleCondition(
                            "high_risk_flags",
                            ConditionOperator.EXISTS,
                            None,
                            "client_context"
                        )
                    )
                ])
            ], "AND"),
            dependencies=[],
            priority=100,
            required=True,
            dynamic_priority=lambda ctx: 110 if ctx.get("age", 0) > 50 else 100,
            ordering_rules=[
                DynamicOrdering(
                    priority_function=lambda ctx: 20 if ctx.get("urgent_review", False) else 0,
                    condition=SimpleCondition(
                        "review_required",
                        ConditionOperator.EQUALS,
                        True,
                        "client_context"
                    ),
                    metadata_requirements={"importance": "high"}
                )
            ],
            metadata={"category": "investment", "importance": "high"}
        )
    ))
    
    library.add_template(TemplateSection(
        id="retirement_goals",
        category=TemplateCategory.RETIREMENT_PLANNING,
        title="Retirement Goals and Strategy",
        content="""
Retirement Planning Analysis

Based on your stated retirement goals and current financial position:

Target Retirement Income: ${target_income:,} per year
Required Capital: ${required_capital:,}
Planned Retirement Age: {retirement_age}

Current Position:
- Superannuation Balance: ${current_super:,}
- Monthly Contributions: ${monthly_contributions:,}
- Projected Balance at Retirement: ${projected_balance:,}

Status: Your projected retirement balance is {projection_status} your target.

Recommendations:
{strategy_recommendations}

Additional Considerations:
1. Age Pension Eligibility: {client.age_pension_eligible}
2. Debt Position: {client.debt_status}
3. Health Factors: {client.health_considerations}

Implementation Timeline:
{timeline_recommendations}
        """.strip(),
        parameters=[
            "target_income",
            "required_capital",
            "retirement_age",
            "current_super",
            "monthly_contributions",
            "projected_balance",
            "projection_status",
            "strategy_recommendations",
            "timeline_recommendations"
        ],
        tags=["retirement", "superannuation", "planning"],
        version="2.0",
        last_updated=datetime.now(),
        compliance_approved=True,
        group="retirement_planning",
        rules=SectionRule(
            conditions=CompositeCondition([
                AnyCondition([
                    SimpleCondition(
                        "age",
                        ConditionOperator.GREATER,
                        50,
                        "client_context"
                    ),
                    SimpleCondition(
                        "retirement_planning_requested",
                        ConditionOperator.EQUALS,
                        True,
                        "client_context"
                    )
                ]),
                RegexCondition(
                    "employment_status",
                    r"^(Employed|Self-Employed|Semi-Retired)$",
                    "client_context"
                )
            ], "AND"),
            dependencies=[],
            priority=90,
            required=False,
            dynamic_priority=lambda ctx: 150 if ctx.get("retirement_priority", "low") == "high" else 90,
            ordering_rules=[
                DynamicOrdering(
                    priority_function=lambda ctx: 30 if ctx.get("super_review_required", False) else 0,
                    metadata_requirements={"category": "retirement"}
                )
            ],
            metadata={"category": "retirement", "importance": "high"}
        )
    ))
    
    return library 