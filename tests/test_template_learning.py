"""
Tests for the template analytics and learning system.
"""
import pytest
from datetime import datetime, timedelta
from app.templates.analytics import (
    TemplateAnalytics,
    TemplateUsage,
    TemplateInsight
)
from app.templates.learning import TemplateLearningService
from app.templates.advice_templates import (
    TemplateLibrary,
    TemplateSection,
    TemplateCategory,
    SectionRule,
    SimpleCondition,
    ConditionOperator
)

@pytest.fixture
def template_library():
    """Create a test template library."""
    library = TemplateLibrary()
    
    # Add a test template
    template = TemplateSection(
        id="test_template",
        category=TemplateCategory.GENERAL_ADVICE,
        title="Test Template",
        content="Test content with {param1} and {param2}",
        parameters=["param1", "param2", "rarely_used"],
        tags=["test"],
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
            required=True
        )
    )
    library.add_template(template)
    
    return library

@pytest.fixture
def analytics(template_library):
    """Create a test analytics instance."""
    return TemplateAnalytics(template_library)

@pytest.fixture
def learning_service(template_library, analytics):
    """Create a test learning service."""
    return TemplateLearningService(
        template_library,
        analytics,
        auto_apply_threshold=0.9,
        suggestion_threshold=0.7,
        review_interval=timedelta(days=1)
    )

def test_record_template_use(analytics):
    """Test recording template usage."""
    analytics.record_template_use(
        "test_template",
        {
            "param1": "value1",
            "param2": "value2"
        },
        {"context_key": "context_value"},
        modifications={"edit_type": "content_change"},
        feedback_score=4
    )
    
    usage = analytics.usage_data["test_template"]
    assert usage.use_count == 1
    assert usage.parameter_frequencies["param1"]["value1"] == 1
    assert "context_key" in usage.client_contexts[0]
    assert usage.modifications[0]["edit_type"] == "content_change"
    assert usage.feedback_scores[0] == 4

def test_parameter_analysis(analytics):
    """Test parameter usage analysis."""
    # Record multiple uses with same parameter value
    for _ in range(10):
        analytics.record_template_use(
            "test_template",
            {
                "param1": "common_value",
                "param2": "value2"
            },
            {}
        )
    
    # Record one use of rarely used parameter
    analytics.record_template_use(
        "test_template",
        {
            "param1": "common_value",
            "param2": "value2",
            "rarely_used": "value"
        },
        {}
    )
    
    # Analyze template
    analytics._analyze_template_usage("test_template")
    
    # Check insights
    insights = analytics.get_insights(
        insight_types={"parameter"}
    )
    
    # Should suggest removing rarely used parameter
    assert any(
        i.suggested_changes["action"] == "remove_parameter" and
        i.suggested_changes["parameter"] == "rarely_used"
        for i in insights
    )
    
    # Should suggest default value for commonly used parameter
    assert any(
        i.suggested_changes["action"] == "default_value" and
        i.suggested_changes["parameter"] == "param1" and
        i.suggested_changes["value"] == "common_value"
        for i in insights
    )

def test_modification_analysis(analytics):
    """Test analysis of template modifications."""
    # Record multiple similar modifications
    for _ in range(3):
        analytics.record_template_use(
            "test_template",
            {"param1": "value1"},
            {},
            modifications={
                "content_change": {
                    "type": "addition",
                    "section": "disclaimer"
                }
            }
        )
    
    # Analyze template
    analytics._analyze_template_usage("test_template")
    
    # Check insights
    insights = analytics.get_insights(
        insight_types={"content"}
    )
    
    # Should suggest content review
    assert any(
        i.suggested_changes["action"] == "review_content" and
        i.suggested_changes["edit_type"] == "content_change"
        for i in insights
    )

def test_client_context_analysis(analytics):
    """Test analysis of client contexts."""
    # Record uses with similar contexts
    contexts = [
        {
            "age": 55,
            "risk_profile": "Conservative",
            "investment_amount": 500000
        }
    ] * 3
    
    for ctx in contexts:
        analytics.record_template_use(
            "test_template",
            {"param1": "value1"},
            ctx
        )
    
    # Analyze template
    analytics._analyze_template_usage("test_template")
    
    # Check insights
    insights = analytics.get_insights(
        insight_types={"condition"}
    )
    
    # Should suggest new condition based on common context
    assert any(
        i.suggested_changes["action"] == "add_condition" and
        "age" in i.suggested_changes["pattern"]
        for i in insights
    )

@pytest.mark.asyncio
async def test_learning_service(learning_service):
    """Test the learning service functionality."""
    # Record template uses to generate insights
    for _ in range(10):
        learning_service.analytics.record_template_use(
            "test_template",
            {"param1": "common_value"},
            {"age": 55}
        )
    
    # Run review
    await learning_service.review_and_improve_templates()
    
    # Check auto-applied changes
    assert len(learning_service.applied_changes) > 0
    
    # Check pending suggestions
    suggestions = learning_service.get_pending_suggestions()
    assert len(suggestions) > 0
    
    # Test applying a suggestion
    if suggestions:
        suggestion_id = suggestions[0].id
        learning_service.apply_suggestion(suggestion_id)
        
        # Suggestion should be removed from pending
        assert not any(
            s.id == suggestion_id 
            for s in learning_service.get_pending_suggestions()
        )

@pytest.mark.asyncio
async def test_template_variant_creation(learning_service):
    """Test creation of template variants."""
    # Record multiple similar modifications
    for _ in range(5):
        learning_service.analytics.record_template_use(
            "test_template",
            {"param1": "value1"},
            {"age": 65},
            modifications={
                "content_addition": {
                    "section": "retirement",
                    "text": "Additional retirement planning considerations..."
                }
            }
        )
    
    # Run review
    await learning_service.review_and_improve_templates()
    
    # Check if variant was created
    variants = [
        t for t in learning_service.template_library.templates.values()
        if t.id.startswith("test_template_variant_")
    ]
    
    assert len(variants) > 0
    variant = variants[0]
    assert "auto_generated" in variant.tags
    
def test_insight_prioritization(analytics):
    """Test insight prioritization logic."""
    # Record high-usage template with low feedback
    for _ in range(51):  # More than 50 uses
        analytics.record_template_use(
            "test_template",
            {"param1": "value1"},
            {},
            feedback_score=2  # Low score
        )
    
    # Analyze template
    analytics._analyze_template_usage("test_template")
    
    # Get insights
    insights = analytics.get_insights()
    
    # First insight should have high priority
    assert insights[0].priority > 100  # Base priority + usage bonus + low feedback bonus

def test_suggestion_deduplication(learning_service):
    """Test deduplication of similar suggestions."""
    # Create two similar insights
    insight1 = TemplateInsight(
        template_id="test_template",
        insight_type="parameter",
        confidence=0.7,
        description="Test insight 1",
        suggested_changes={"action": "remove_parameter", "parameter": "param1"},
        supporting_data={}
    )
    
    insight2 = TemplateInsight(
        template_id="test_template",
        insight_type="parameter",
        confidence=0.8,  # Higher confidence
        description="Test insight 2",
        suggested_changes={"action": "remove_parameter", "parameter": "param1"},
        supporting_data={"additional": "data"}
    )
    
    # Add both insights
    learning_service._add_suggestion(insight1)
    learning_service._add_suggestion(insight2)
    
    # Should only have one suggestion with the higher confidence
    suggestions = learning_service.get_pending_suggestions()
    assert len(suggestions) == 1
    assert suggestions[0].confidence == 0.8 