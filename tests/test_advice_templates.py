"""
Tests for the financial advice template system.
"""
import pytest
from datetime import datetime
from app.templates.advice_templates import (
    TemplateCategory,
    TemplateSection,
    TemplateLibrary,
    AdviceBuilder,
    create_sample_templates,
    Condition,
    SectionRule,
    SimpleCondition,
    CompositeCondition,
    NotCondition,
    ConditionOperator,
    TemplateGroup,
    DynamicOrdering,
    RegexCondition,
    AnyOfCondition,
    AllCondition,
    AnyCondition,
    NestedCondition
)

def test_template_creation():
    """Test creating a template section."""
    template = TemplateSection(
        id="test_template",
        category=TemplateCategory.INVESTMENT_STRATEGY,
        title="Test Template",
        content="Test content with {param1} and {param2}",
        parameters=["param1", "param2"],
        tags=["test", "example"],
        version="1.0",
        last_updated=datetime.now()
    )
    
    assert template.id == "test_template"
    assert template.category == TemplateCategory.INVESTMENT_STRATEGY
    assert len(template.parameters) == 2
    assert not template.compliance_approved
    assert template.requires_review

def test_template_library_operations():
    """Test basic template library operations."""
    library = TemplateLibrary()
    
    # Create and add a template
    template = TemplateSection(
        id="test_template",
        category=TemplateCategory.INVESTMENT_STRATEGY,
        title="Test Template",
        content="Test content",
        parameters=[],
        tags=["test"],
        version="1.0",
        last_updated=datetime.now()
    )
    
    library.add_template(template)
    
    # Test retrieval
    retrieved = library.get_template("test_template")
    assert retrieved == template
    
    # Test category filtering
    category_templates = library.get_templates_by_category(TemplateCategory.INVESTMENT_STRATEGY)
    assert len(category_templates) == 1
    assert category_templates[0] == template
    
    # Test tag filtering
    tagged_templates = library.get_templates_by_tags(["test"])
    assert len(tagged_templates) == 1
    assert tagged_templates[0] == template

def test_template_versioning():
    """Test template version history."""
    library = TemplateLibrary()
    
    # Add initial version
    template_v1 = TemplateSection(
        id="test_template",
        category=TemplateCategory.INVESTMENT_STRATEGY,
        title="Test Template",
        content="Version 1",
        parameters=[],
        tags=["test"],
        version="1.0",
        last_updated=datetime.now()
    )
    library.add_template(template_v1)
    
    # Add updated version
    template_v2 = TemplateSection(
        id="test_template",
        category=TemplateCategory.INVESTMENT_STRATEGY,
        title="Test Template",
        content="Version 2",
        parameters=[],
        tags=["test"],
        version="2.0",
        last_updated=datetime.now()
    )
    library.add_template(template_v2)
    
    # Check version history
    history = library.get_version_history("test_template")
    assert len(history) == 1
    assert history[0].content == "Version 1"
    
    # Check current version
    current = library.get_template("test_template")
    assert current.content == "Version 2"

def test_advice_builder():
    """Test building advice documents from templates."""
    library = create_sample_templates()
    builder = AdviceBuilder(library)
    
    # Test building investment strategy section
    params = {
        "equity_allocation": 60,
        "defensive_allocation": 40,
        "investment_timeframe": 10,
        "aus_shares": 30,
        "intl_shares": 20,
        "property": 10,
        "fixed_interest": 30,
        "cash": 10
    }
    
    client_context = {
        "name": "John Smith",
        "risk_profile": "Balanced"
    }
    
    content = builder.build_section(
        "investment_strategy_balanced",
        params,
        client_context
    )
    
    assert content is not None
    assert "60% in growth assets" in content
    assert "40% in defensive assets" in content
    assert "Australian Shares: 30%" in content

def test_full_advice_document():
    """Test building a complete advice document."""
    library = create_sample_templates()
    builder = AdviceBuilder(library)
    
    template_ids = [
        "investment_strategy_balanced",
        "retirement_planning_basic"
    ]
    
    parameters = {
        "investment_strategy_balanced": {
            "equity_allocation": 60,
            "defensive_allocation": 40,
            "investment_timeframe": 10,
            "aus_shares": 30,
            "intl_shares": 20,
            "property": 10,
            "fixed_interest": 30,
            "cash": 10
        },
        "retirement_planning_basic": {
            "target_income": 80000,
            "required_capital": 1600000,
            "retirement_age": 65,
            "current_super": 500000,
            "monthly_contributions": 1000,
            "projected_balance": 1200000,
            "projection_status": "below",
            "strategy_recommendations": "- Increase monthly contributions\n- Review investment strategy"
        }
    }
    
    client_context = {
        "name": "John Smith",
        "risk_profile": "Balanced"
    }
    
    document = builder.build_advice_document(
        template_ids,
        parameters,
        client_context
    )
    
    assert document is not None
    assert "growth assets" in document
    assert "retirement income" in document
    assert "monthly contributions" in document

def test_invalid_template():
    """Test handling of invalid template IDs."""
    library = TemplateLibrary()
    builder = AdviceBuilder(library)
    
    content = builder.build_section(
        "non_existent_template",
        {},
        {}
    )
    
    assert content is None

def test_client_context_replacement():
    """Test client context placeholder replacement."""
    library = TemplateLibrary()
    
    template = TemplateSection(
        id="test_template",
        category=TemplateCategory.GENERAL_ADVICE,
        title="Test Template",
        content="Dear {client.name}, based on your {client.risk_profile} risk profile...",
        parameters=[],
        tags=["test"],
        version="1.0",
        last_updated=datetime.now()
    )
    
    library.add_template(template)
    builder = AdviceBuilder(library)
    
    client_context = {
        "name": "John Smith",
        "risk_profile": "Balanced"
    }
    
    content = builder.build_section(
        "test_template",
        {},
        client_context
    )
    
    assert content is not None
    assert "Dear John Smith" in content
    assert "your Balanced risk profile" in content

def test_condition_evaluation():
    """Test condition evaluation logic."""
    # Test parameter conditions
    condition = Condition("amount", ">", 1000, "parameters")
    assert condition.evaluate({"amount": 2000}, {}) is True
    assert condition.evaluate({"amount": 500}, {}) is False
    
    # Test client context conditions
    condition = Condition("risk_profile", "==", "Balanced", "client_context")
    assert condition.evaluate({}, {"risk_profile": "Balanced"}) is True
    assert condition.evaluate({}, {"risk_profile": "Conservative"}) is False
    
    # Test list operations
    condition = Condition("interests", "contains", "retirement", "client_context")
    assert condition.evaluate({}, {"interests": ["investment", "retirement"]}) is True
    assert condition.evaluate({}, {"interests": ["investment"]}) is False

def test_section_rules():
    """Test section rules evaluation."""
    rules = SectionRule(
        conditions=[
            Condition("age", "<", 65, "client_context"),
            Condition("balance", ">", 100000, "parameters")
        ],
        dependencies=["risk_profile"],
        priority=100,
        required=True
    )
    
    template = TemplateSection(
        id="test_template",
        category=TemplateCategory.INVESTMENT_STRATEGY,
        title="Test Template",
        content="Test content",
        parameters=[],
        tags=["test"],
        version="1.0",
        last_updated=datetime.now(),
        rules=rules
    )
    
    library = TemplateLibrary()
    library.add_template(template)
    builder = AdviceBuilder(library)
    
    # Test conditions met
    content = builder.build_section(
        "test_template",
        {"balance": 150000},
        {"age": 45}
    )
    assert content is not None
    
    # Test conditions not met
    content = builder.build_section(
        "test_template",
        {"balance": 50000},
        {"age": 45}
    )
    assert content is None

def test_template_dependencies():
    """Test template dependency resolution."""
    library = TemplateLibrary()
    
    # Create templates with dependencies
    base_template = TemplateSection(
        id="base",
        category=TemplateCategory.GENERAL_ADVICE,
        title="Base Template",
        content="Base content",
        parameters=[],
        tags=[],
        version="1.0",
        last_updated=datetime.now(),
        rules=SectionRule(
            conditions=[],
            dependencies=[],
            priority=100,
            required=True
        )
    )
    
    dependent_template = TemplateSection(
        id="dependent",
        category=TemplateCategory.GENERAL_ADVICE,
        title="Dependent Template",
        content="Dependent content",
        parameters=[],
        tags=[],
        version="1.0",
        last_updated=datetime.now(),
        rules=SectionRule(
            conditions=[],
            dependencies=["base"],
            priority=90,
            required=True
        )
    )
    
    library.add_template(base_template)
    library.add_template(dependent_template)
    builder = AdviceBuilder(library)
    
    # Test correct ordering
    document = builder.build_advice_document(
        ["dependent", "base"],  # Intentionally wrong order
        {
            "base": {},
            "dependent": {}
        },
        {}
    )
    
    assert document is not None
    assert document.index("Base content") < document.index("Dependent content")

def test_required_sections():
    """Test handling of required sections."""
    library = TemplateLibrary()
    
    required_template = TemplateSection(
        id="required",
        category=TemplateCategory.GENERAL_ADVICE,
        title="Required Template",
        content="Required content",
        parameters=[],
        tags=[],
        version="1.0",
        last_updated=datetime.now(),
        rules=SectionRule(
            conditions=[
                Condition("include_required", "==", True, "parameters")
            ],
            dependencies=[],
            priority=100,
            required=True
        )
    )
    
    library.add_template(required_template)
    builder = AdviceBuilder(library)
    
    # Test missing required section
    with pytest.raises(ValueError) as exc_info:
        builder.build_advice_document(
            ["required"],
            {"required": {"include_required": False}},
            {}
        )
    assert "Required sections missing" in str(exc_info.value)
    
    # Test included required section
    document = builder.build_advice_document(
        ["required"],
        {"required": {"include_required": True}},
        {}
    )
    assert document is not None
    assert "Required content" in document

def test_section_priority_ordering():
    """Test ordering of sections by priority."""
    library = TemplateLibrary()
    
    # Create templates with different priorities
    templates = []
    for i in range(3):
        template = TemplateSection(
            id=f"template_{i}",
            category=TemplateCategory.GENERAL_ADVICE,
            title=f"Template {i}",
            content=f"Content {i}",
            parameters=[],
            tags=[],
            version="1.0",
            last_updated=datetime.now(),
            rules=SectionRule(
                conditions=[],
                dependencies=[],
                priority=100 - i * 10,  # 100, 90, 80
                required=True
            )
        )
        library.add_template(template)
        templates.append(template)
    
    builder = AdviceBuilder(library)
    
    # Test priority ordering
    document = builder.build_advice_document(
        ["template_2", "template_0", "template_1"],  # Random order
        {
            "template_0": {},
            "template_1": {},
            "template_2": {}
        },
        {}
    )
    
    assert document is not None
    lines = document.split("\n\n")
    assert lines[0] == "Content 0"  # Highest priority
    assert lines[1] == "Content 1"
    assert lines[2] == "Content 2"  # Lowest priority

def test_conditional_section_inclusion():
    """Test conditional inclusion of sections based on client context and parameters."""
    library = create_sample_templates()
    builder = AdviceBuilder(library)
    
    # Test inclusion based on client risk profile and investment timeframe
    document = builder.build_advice_document(
        ["investment_strategy_balanced"],
        {
            "investment_strategy_balanced": {
                "equity_allocation": 60,
                "defensive_allocation": 40,
                "investment_timeframe": 10,
                "aus_shares": 30,
                "intl_shares": 20,
                "property": 10,
                "fixed_interest": 30,
                "cash": 10
            }
        },
        {"risk_profile": "Balanced"}
    )
    assert document is not None
    assert "balanced investment strategy" in document.lower()
    
    # Test exclusion based on risk profile
    document = builder.build_advice_document(
        ["investment_strategy_balanced"],
        {
            "investment_strategy_balanced": {
                "equity_allocation": 60,
                "defensive_allocation": 40,
                "investment_timeframe": 10,
                "aus_shares": 30,
                "intl_shares": 20,
                "property": 10,
                "fixed_interest": 30,
                "cash": 10
            }
        },
        {"risk_profile": "Conservative"}
    )
    assert document == ""  # Should be empty as conditions not met

def test_complex_conditions():
    """Test complex condition combinations."""
    # Test nested conditions with AND/OR logic
    condition = CompositeCondition([
        SimpleCondition(
            "age",
            ConditionOperator.BETWEEN,
            (50, 65),
            "client_context"
        ),
        CompositeCondition([
            SimpleCondition(
                "income",
                ConditionOperator.GREATER,
                100000,
                "parameters"
            ),
            NotCondition(
                SimpleCondition(
                    "has_pension",
                    ConditionOperator.EQUALS,
                    True,
                    "client_context"
                )
            )
        ], "AND")
    ], "AND")
    
    # Test condition that should pass
    assert condition.evaluate(
        {"income": 150000},
        {"age": 55, "has_pension": False}
    ) is True
    
    # Test condition that should fail due to age
    assert condition.evaluate(
        {"income": 150000},
        {"age": 45, "has_pension": False}
    ) is False
    
    # Test condition that should fail due to pension
    assert condition.evaluate(
        {"income": 150000},
        {"age": 55, "has_pension": True}
    ) is False

def test_regex_condition():
    """Test regex pattern matching condition."""
    condition = RegexCondition(
        "email",
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "client_context"
    )
    
    assert condition.evaluate({}, {"email": "test@example.com"}) is True
    assert condition.evaluate({}, {"email": "invalid-email"}) is False

def test_any_of_condition():
    """Test matching against a list of values."""
    condition = AnyOfCondition(
        "risk_profile",
        ["Conservative", "Balanced", "Growth"],
        "client_context"
    )
    
    assert condition.evaluate({}, {"risk_profile": "Balanced"}) is True
    assert condition.evaluate({}, {"risk_profile": "Aggressive"}) is False

def test_nested_condition():
    """Test evaluation of nested fields."""
    condition = NestedCondition(
        ["retirement", "planning", "started"],
        ConditionOperator.EQUALS,
        True,
        "client_context"
    )
    
    assert condition.evaluate({}, {
        "retirement": {
            "planning": {
                "started": True
            }
        }
    }) is True
    
    assert condition.evaluate({}, {
        "retirement": {
            "planning": {
                "started": False
            }
        }
    }) is False

def test_template_groups():
    """Test template grouping functionality."""
    library = TemplateLibrary()
    
    # Create a group with subgroups
    main_group = TemplateGroup(
        id="main_group",
        name="Main Group",
        description="Main group description",
        templates=["template_1", "template_2"],
        subgroups=[
            TemplateGroup(
                id="sub_group",
                name="Sub Group",
                description="Sub group description",
                templates=["template_3"],
                order_rules=SimpleCondition(
                    "include_sub",
                    ConditionOperator.EQUALS,
                    True,
                    "client_context"
                )
            )
        ],
        priority_calculator=lambda ctx: 50 if ctx.get("high_priority", False) else 0
    )
    
    # Add templates
    for i in range(1, 4):
        template = TemplateSection(
            id=f"template_{i}",
            category=TemplateCategory.GENERAL_ADVICE,
            title=f"Template {i}",
            content=f"Content {i}",
            parameters=[],
            tags=[],
            version="1.0",
            last_updated=datetime.now(),
            group="main_group" if i <= 2 else "sub_group"
        )
        library.add_template(template)
    
    library.add_group(main_group)
    
    # Test group priority calculation
    builder = AdviceBuilder(library)
    priority = builder._evaluate_group_priority(
        "main_group",
        {"high_priority": True}
    )
    assert priority == 50
    
    # Test subgroup template retrieval
    sub_templates = library.get_group_templates("sub_group")
    assert len(sub_templates) == 1
    assert sub_templates[0].id == "template_3"

def test_dynamic_ordering():
    """Test dynamic ordering of templates."""
    library = TemplateLibrary()
    
    # Create templates with dynamic ordering rules
    template1 = TemplateSection(
        id="template_1",
        category=TemplateCategory.GENERAL_ADVICE,
        title="Template 1",
        content="Content 1",
        parameters=[],
        tags=[],
        version="1.0",
        last_updated=datetime.now(),
        rules=SectionRule(
            conditions=SimpleCondition(
                "include",
                ConditionOperator.EQUALS,
                True,
                "parameters"
            ),
            dependencies=[],
            priority=100,
            required=True,
            ordering_rules=[
                DynamicOrdering(
                    priority_function=lambda ctx: 50 if ctx.get("urgent", False) else 0,
                    condition=SimpleCondition(
                        "review_needed",
                        ConditionOperator.EQUALS,
                        True,
                        "client_context"
                    ),
                    metadata_requirements={"importance": "high"}
                )
            ],
            metadata={"importance": "high"}
        )
    )
    
    template2 = TemplateSection(
        id="template_2",
        category=TemplateCategory.GENERAL_ADVICE,
        title="Template 2",
        content="Content 2",
        parameters=[],
        tags=[],
        version="1.0",
        last_updated=datetime.now(),
        rules=SectionRule(
            conditions=SimpleCondition(
                "include",
                ConditionOperator.EQUALS,
                True,
                "parameters"
            ),
            dependencies=[],
            priority=90,
            required=True,
            metadata={"importance": "medium"}
        )
    )
    
    library.add_template(template1)
    library.add_template(template2)
    
    builder = AdviceBuilder(library)
    
    # Test ordering with dynamic rules
    ordered = builder._apply_ordering_rules(
        [template1, template2],
        {
            "urgent": True,
            "review_needed": True
        }
    )
    
    assert ordered[0].id == "template_1"  # Should be first due to higher priority
    assert ordered[1].id == "template_2"

def test_metadata_based_ordering():
    """Test ordering based on template metadata."""
    library = TemplateLibrary()
    builder = AdviceBuilder(library)
    
    # Create templates with different metadata
    templates = []
    for i, importance in enumerate(["high", "medium", "low"]):
        template = TemplateSection(
            id=f"template_{i}",
            category=TemplateCategory.GENERAL_ADVICE,
            title=f"Template {i}",
            content=f"Content {i}",
            parameters=[],
            tags=[],
            version="1.0",
            last_updated=datetime.now(),
            rules=SectionRule(
                conditions=SimpleCondition(
                    "include",
                    ConditionOperator.EQUALS,
                    True,
                    "parameters"
                ),
                dependencies=[],
                priority=100 - i * 10,
                required=True,
                metadata={"importance": importance}
            )
        )
        templates.append(template)
        library.add_template(template)
    
    # Test metadata-based grouping and ordering
    ordered = builder._apply_template_ordering_rules(
        templates,
        {}
    )
    
    # Templates should be grouped by metadata and ordered by priority
    assert ordered[0].rules.metadata["importance"] == "high"
    assert ordered[1].rules.metadata["importance"] == "medium"
    assert ordered[2].rules.metadata["importance"] == "low"

def test_advanced_template_example():
    """Test the sophisticated template examples."""
    library = create_sample_templates()
    builder = AdviceBuilder(library)
    
    # Test investment strategy template
    params = {
        "investment_strategy_balanced": {
            "equity_allocation": 60,
            "defensive_allocation": 40,
            "investment_timeframe": 10,
            "investment_amount": 500000,
            "aus_shares": 30,
            "intl_shares": 20,
            "property": 10,
            "fixed_interest": 30,
            "cash": 10,
            "review_frequency": "annually"
        }
    }
    
    client_context = {
        "risk_profile": "Balanced",
        "age": 55,
        "needs_risk_assessment": True,
        "review_required": True,
        "urgent_review": True
    }
    
    document = builder.build_advice_document(
        ["investment_strategy_balanced"],
        params,
        client_context
    )
    
    assert document is not None
    assert "Based on our comprehensive assessment" in document
    assert "60%" in document
    assert "annually" in document
    
    # Test retirement goals template
    retirement_params = {
        "retirement_goals": {
            "target_income": 80000,
            "required_capital": 1600000,
            "retirement_age": 65,
            "current_super": 500000,
            "monthly_contributions": 1000,
            "projected_balance": 1200000,
            "projection_status": "below",
            "strategy_recommendations": "- Increase monthly contributions\n- Review investment strategy",
            "timeline_recommendations": "Implement changes within 3 months"
        }
    }
    
    client_context.update({
        "employment_status": "Employed",
        "retirement_priority": "high",
        "age_pension_eligible": "Yes",
        "debt_status": "Low",
        "health_considerations": "Good"
    })
    
    document = builder.build_advice_document(
        ["retirement_goals"],
        retirement_params,
        client_context
    )
    
    assert document is not None
    assert "Retirement Planning Analysis" in document
    assert "$80,000 per year" in document
    assert "Increase monthly contributions" in document 