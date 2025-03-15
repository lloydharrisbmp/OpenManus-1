"""
Admin interface for managing template suggestions and insights.
"""
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from .learning import TemplateLearningService
from .analytics import TemplateAnalytics
from .feedback import FeedbackService
from .advice_templates import TemplateLibrary

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Template Admin API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models for API
class SuggestionResponse(BaseModel):
    """Response model for template suggestions."""
    suggestion_id: str
    template_id: str
    suggestion_type: str
    confidence: float
    description: str
    suggested_changes: Dict[str, Any]
    supporting_data: Dict[str, Any]
    created_at: datetime
    status: str = "pending"
    priority: int

class FeedbackSummaryResponse(BaseModel):
    """Response model for template feedback summary."""
    template_id: str
    total_feedback: int
    average_rating: float
    rating_distribution: Dict[int, int]
    common_issues: List[Dict[str, Any]]
    common_suggestions: List[Dict[str, Any]]
    modification_patterns: List[Dict[str, Any]]
    conversion_rate: float
    avg_completion_time: float
    last_updated: datetime

class ABTestResponse(BaseModel):
    """Response model for A/B test results."""
    test_id: str
    status: str
    duration: int
    metrics: Dict[str, Any]
    segments: Dict[str, Any]
    recommendations: List[Dict[str, Any]]

class AdminService:
    """Service for managing template admin functionality."""
    
    def __init__(
        self,
        template_library: TemplateLibrary,
        learning_service: TemplateLearningService,
        analytics: TemplateAnalytics,
        feedback_service: FeedbackService
    ):
        self.template_library = template_library
        self.learning_service = learning_service
        self.analytics = analytics
        self.feedback_service = feedback_service
        
        # Register API routes
        self._register_routes()
    
    def _register_routes(self):
        """Register FastAPI routes."""
        
        @app.get(
            "/suggestions",
            response_model=List[SuggestionResponse],
            tags=["suggestions"]
        )
        async def get_suggestions(
            min_confidence: float = Query(0.7, ge=0.0, le=1.0),
            suggestion_type: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = Query(10, ge=1, le=100)
        ):
            """Get template suggestions."""
            suggestions = self.learning_service.get_pending_suggestions(
                min_confidence=min_confidence,
                insight_types={suggestion_type} if suggestion_type else None,
                limit=limit
            )
            
            return [
                SuggestionResponse(
                    suggestion_id=s.id,
                    template_id=s.template_id,
                    suggestion_type=s.insight_type,
                    confidence=s.confidence,
                    description=s.description,
                    suggested_changes=s.suggested_changes,
                    supporting_data=s.supporting_data,
                    created_at=s.created_at,
                    priority=s.priority
                )
                for s in suggestions
                if not status or status == "pending"
            ]
        
        @app.post(
            "/suggestions/{suggestion_id}/apply",
            response_model=Dict[str, str],
            tags=["suggestions"]
        )
        async def apply_suggestion(
            suggestion_id: str = Path(..., title="The ID of the suggestion to apply")
        ):
            """Apply a template suggestion."""
            try:
                self.learning_service.apply_suggestion(suggestion_id)
                return {"status": "success", "message": "Suggestion applied successfully"}
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=str(e)
                )
        
        @app.post(
            "/suggestions/{suggestion_id}/reject",
            response_model=Dict[str, str],
            tags=["suggestions"]
        )
        async def reject_suggestion(
            suggestion_id: str = Path(..., title="The ID of the suggestion to reject")
        ):
            """Reject a template suggestion."""
            try:
                self.learning_service.reject_suggestion(suggestion_id)
                return {"status": "success", "message": "Suggestion rejected successfully"}
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=str(e)
                )
        
        @app.get(
            "/feedback/summary/{template_id}",
            response_model=FeedbackSummaryResponse,
            tags=["feedback"]
        )
        async def get_feedback_summary(
            template_id: str = Path(..., title="The ID of the template"),
            recalculate: bool = Query(False)
        ):
            """Get feedback summary for a template."""
            summary = self.feedback_service.get_summary(
                template_id,
                recalculate=recalculate
            )
            
            if not summary:
                raise HTTPException(
                    status_code=404,
                    detail=f"No feedback found for template {template_id}"
                )
            
            return FeedbackSummaryResponse(**summary.__dict__)
        
        @app.get(
            "/feedback/trending-issues",
            response_model=List[Dict[str, Any]],
            tags=["feedback"]
        )
        async def get_trending_issues(
            min_occurrences: int = Query(3, ge=1),
            time_window_days: int = Query(30, ge=1)
        ):
            """Get trending issues across templates."""
            return self.feedback_service.get_trending_issues(
                min_occurrences=min_occurrences,
                time_window_days=time_window_days
            )
        
        @app.get(
            "/feedback/modification-patterns",
            response_model=List[Dict[str, Any]],
            tags=["feedback"]
        )
        async def get_modification_patterns(
            template_id: Optional[str] = None
        ):
            """Get modification patterns for templates."""
            return self.feedback_service.analyze_modification_patterns(
                template_id=template_id
            )
        
        @app.get(
            "/tests",
            response_model=List[ABTestResponse],
            tags=["tests"]
        )
        async def get_ab_tests(
            status: Optional[str] = None
        ):
            """Get A/B test results."""
            tests = []
            for test_id in self.learning_service.active_tests:
                test_results = self.learning_service.analyze_test(test_id)
                if not status or test_results['status'] == status:
                    tests.append(ABTestResponse(**test_results))
            return tests
        
        @app.post(
            "/tests/{test_id}/end",
            response_model=ABTestResponse,
            tags=["tests"]
        )
        async def end_ab_test(
            test_id: str = Path(..., title="The ID of the test to end")
        ):
            """End an A/B test."""
            try:
                results = self.learning_service.end_test(test_id)
                return ABTestResponse(**results)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=str(e)
                )
        
        @app.get(
            "/analytics/patterns",
            response_model=List[Dict[str, Any]],
            tags=["analytics"]
        )
        async def get_usage_patterns(
            min_confidence: float = Query(0.7, ge=0.0, le=1.0),
            pattern_type: Optional[str] = None
        ):
            """Get template usage patterns."""
            patterns = self.analytics.analyze_usage_patterns([])  # Pass actual usage data
            
            filtered_patterns = [
                p for p in patterns
                if (
                    p.confidence >= min_confidence and
                    (not pattern_type or p.pattern_type == pattern_type)
                )
            ]
            
            return [
                {
                    'pattern_type': p.pattern_type,
                    'confidence': p.confidence,
                    'description': p.description,
                    'evidence': p.evidence,
                    'affected_templates': list(p.affected_templates),
                    'detected_at': p.detected_at
                }
                for p in filtered_patterns
            ]
        
        @app.get(
            "/analytics/insights",
            response_model=List[Dict[str, Any]],
            tags=["analytics"]
        )
        async def get_insights(
            min_confidence: float = Query(0.7, ge=0.0, le=1.0),
            insight_type: Optional[str] = None,
            limit: int = Query(10, ge=1, le=100)
        ):
            """Get template insights."""
            insights = self.analytics.get_insights(
                min_confidence=min_confidence,
                insight_types={insight_type} if insight_type else None,
                limit=limit
            )
            
            return [
                {
                    'template_id': i.template_id,
                    'insight_type': i.insight_type,
                    'confidence': i.confidence,
                    'description': i.description,
                    'suggested_changes': i.suggested_changes,
                    'supporting_data': i.supporting_data,
                    'created_at': i.created_at,
                    'priority': i.priority
                }
                for i in insights
            ]
        
        @app.get("/health", tags=["system"])
        async def health_check():
            """Check system health."""
            return {
                "status": "healthy",
                "components": {
                    "template_library": len(self.template_library.templates),
                    "active_tests": len(self.learning_service.active_tests),
                    "feedback_items": len(self.feedback_service.feedback_items),
                    "pending_suggestions": len(
                        self.learning_service.get_pending_suggestions()
                    )
                }
            }

# Dependency for authentication
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validate user token."""
    # Implement your authentication logic here
    return {"username": "admin"}  # Placeholder 