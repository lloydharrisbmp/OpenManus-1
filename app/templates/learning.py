"""
Template learning system for automated improvements and suggestions.
"""
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import logging
from .analytics import TemplateAnalytics, TemplateInsight
from .advice_templates import (
    TemplateLibrary,
    TemplateSection,
    TemplateCategory,
    SectionRule,
    SimpleCondition,
    CompositeCondition,
    ConditionOperator
)

logger = logging.getLogger(__name__)

class TemplateLearningService:
    """Service for managing template learning and improvements."""
    
    def __init__(
        self,
        template_library: TemplateLibrary,
        analytics: TemplateAnalytics,
        auto_apply_threshold: float = 0.9,
        suggestion_threshold: float = 0.7,
        review_interval: timedelta = timedelta(days=7)
    ):
        self.template_library = template_library
        self.analytics = analytics
        self.auto_apply_threshold = auto_apply_threshold
        self.suggestion_threshold = suggestion_threshold
        self.review_interval = review_interval
        self.last_review = datetime.now()
        self.pending_suggestions: List[TemplateInsight] = []
        self.applied_changes: List[Dict[str, Any]] = []
        
    async def start_learning_loop(self):
        """Start the background learning loop."""
        while True:
            try:
                await self.review_and_improve_templates()
                await asyncio.sleep(self.review_interval.total_seconds())
            except Exception as e:
                logger.error(f"Error in learning loop: {e}", exc_info=True)
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def review_and_improve_templates(self):
        """Review analytics and generate improvements."""
        current_time = datetime.now()
        if (current_time - self.last_review) < self.review_interval:
            return
            
        self.last_review = current_time
        
        # Get insights sorted by priority
        insights = self.analytics.get_insights(
            min_confidence=self.suggestion_threshold
        )
        
        for insight in insights:
            if insight.confidence >= self.auto_apply_threshold:
                await self._auto_apply_improvement(insight)
            else:
                self._add_suggestion(insight)
    
    async def _auto_apply_improvement(self, insight: TemplateInsight) -> None:
        """Automatically apply a high-confidence improvement."""
        try:
            if insight.insight_type == "parameter":
                await self._apply_parameter_change(insight)
            elif insight.insight_type == "content":
                await self._apply_content_change(insight)
            elif insight.insight_type == "condition":
                await self._apply_condition_change(insight)
            elif insight.insight_type == "new_template":
                await self._create_new_template(insight)
            
            self.applied_changes.append({
                "insight": insight,
                "applied_at": datetime.now(),
                "auto_applied": True
            })
            
            logger.info(
                f"Auto-applied improvement: {insight.description} "
                f"to template {insight.template_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to auto-apply improvement: {e}",
                exc_info=True
            )
    
    async def _apply_parameter_change(self, insight: TemplateInsight) -> None:
        """Apply parameter-related changes to a template."""
        template = self.template_library.get_template(insight.template_id)
        if not template:
            return
            
        if insight.suggested_changes["action"] == "remove_parameter":
            param = insight.suggested_changes["parameter"]
            if param in template.parameters:
                template.parameters.remove(param)
                
        elif insight.suggested_changes["action"] == "default_value":
            param = insight.suggested_changes["parameter"]
            value = insight.suggested_changes["value"]
            # Add to template metadata for default values
            if not hasattr(template, "default_values"):
                template.default_values = {}
            template.default_values[param] = value
    
    async def _apply_content_change(self, insight: TemplateInsight) -> None:
        """Apply content-related changes to a template."""
        template = self.template_library.get_template(insight.template_id)
        if not template:
            return
            
        if insight.suggested_changes["action"] == "review_content":
            # Log for manual review if content changes are needed
            logger.info(
                f"Content review needed for template {template.id}: "
                f"{insight.description}"
            )
            self._add_suggestion(insight)
    
    async def _apply_condition_change(self, insight: TemplateInsight) -> None:
        """Apply condition-related changes to a template."""
        template = self.template_library.get_template(insight.template_id)
        if not template:
            return
            
        if insight.suggested_changes["action"] == "add_condition":
            pattern = insight.suggested_changes["pattern"]
            
            # Create new condition from pattern
            conditions = []
            for field, value in pattern.items():
                conditions.append(
                    SimpleCondition(
                        field,
                        ConditionOperator.EQUALS,
                        value,
                        "client_context"
                    )
                )
            
            new_condition = CompositeCondition(conditions, "AND")
            
            # Add to existing rules or create new ones
            if template.rules:
                if isinstance(template.rules.conditions, CompositeCondition):
                    template.rules.conditions.conditions.append(new_condition)
                else:
                    template.rules.conditions = CompositeCondition(
                        [template.rules.conditions, new_condition],
                        "AND"
                    )
            else:
                template.rules = SectionRule(
                    conditions=new_condition,
                    dependencies=[],
                    priority=50,
                    required=False
                )
    
    async def _create_new_template(self, insight: TemplateInsight) -> None:
        """Create a new template variant."""
        base_template = self.template_library.get_template(
            insight.suggested_changes["base_template"]
        )
        if not base_template:
            return
            
        # Create new template as a variant
        new_template = TemplateSection(
            id=f"{base_template.id}_variant_{datetime.now().strftime('%Y%m%d')}",
            category=base_template.category,
            title=f"{base_template.title} (Variant)",
            content=base_template.content,
            parameters=base_template.parameters.copy(),
            tags=base_template.tags + ["auto_generated"],
            version="1.0",
            last_updated=datetime.now(),
            group=base_template.group
        )
        
        # Apply the changes from the insight
        changes = insight.suggested_changes["changes"]
        for key, value in changes.items():
            if hasattr(new_template, key):
                setattr(new_template, key, value)
        
        # Add to library
        self.template_library.add_template(new_template)
        
        logger.info(
            f"Created new template variant: {new_template.id} "
            f"based on {base_template.id}"
        )
    
    def _add_suggestion(self, insight: TemplateInsight) -> None:
        """Add a suggestion for manual review."""
        # Check if similar suggestion already exists
        for existing in self.pending_suggestions:
            if (
                existing.template_id == insight.template_id and
                existing.insight_type == insight.insight_type and
                existing.suggested_changes == insight.suggested_changes
            ):
                # Update existing suggestion if new one has higher confidence
                if insight.confidence > existing.confidence:
                    existing.confidence = insight.confidence
                    existing.priority = insight.priority
                    existing.supporting_data.update(insight.supporting_data)
                return
        
        self.pending_suggestions.append(insight)
    
    def get_pending_suggestions(
        self,
        min_confidence: float = 0.0,
        insight_types: Optional[Set[str]] = None,
        limit: Optional[int] = None
    ) -> List[TemplateInsight]:
        """Get pending suggestions filtered by criteria."""
        filtered = [
            s for s in self.pending_suggestions
            if s.confidence >= min_confidence and
            (not insight_types or s.insight_type in insight_types)
        ]
        
        # Sort by priority
        sorted_suggestions = sorted(
            filtered,
            key=lambda x: x.priority,
            reverse=True
        )
        
        if limit:
            return sorted_suggestions[:limit]
        return sorted_suggestions
    
    def apply_suggestion(self, suggestion_id: str) -> None:
        """Manually apply a suggested improvement."""
        suggestion = next(
            (s for s in self.pending_suggestions if s.id == suggestion_id),
            None
        )
        if not suggestion:
            return
            
        asyncio.create_task(self._auto_apply_improvement(suggestion))
        self.pending_suggestions.remove(suggestion)
    
    def reject_suggestion(self, suggestion_id: str) -> None:
        """Reject a suggested improvement."""
        self.pending_suggestions = [
            s for s in self.pending_suggestions 
            if s.id != suggestion_id
        ]
    
    def get_applied_changes(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get history of applied changes."""
        filtered = self.applied_changes
        if since:
            filtered = [
                c for c in filtered 
                if c["applied_at"] >= since
            ]
            
        if limit:
            return filtered[-limit:]
        return filtered 