"""
Feedback collection system for templates.
"""
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from enum import Enum
from .advice_templates import TemplateSection, TemplateLibrary

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    RATING = "rating"  # Numerical rating
    COMMENT = "comment"  # Text comment
    MODIFICATION = "modification"  # Template modifications
    COMPLETION_TIME = "completion_time"  # Time to complete
    CONVERSION = "conversion"  # Conversion event
    ISSUE = "issue"  # Issue report
    SUGGESTION = "suggestion"  # Improvement suggestion

@dataclass
class FeedbackItem:
    """Represents a single piece of feedback."""
    feedback_id: str
    template_id: str
    feedback_type: FeedbackType
    value: Union[float, str, Dict[str, Any]]
    user_id: Optional[str] = None
    client_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class FeedbackSummary:
    """Summary of feedback for a template."""
    template_id: str
    total_feedback: int = 0
    average_rating: float = 0.0
    rating_distribution: Dict[int, int] = field(default_factory=dict)
    common_issues: List[Dict[str, Any]] = field(default_factory=list)
    common_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    modification_patterns: List[Dict[str, Any]] = field(default_factory=list)
    conversion_rate: float = 0.0
    avg_completion_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class FeedbackService:
    """Service for collecting and analyzing template feedback."""
    
    def __init__(
        self,
        template_library: TemplateLibrary,
        min_feedback_threshold: int = 5
    ):
        self.template_library = template_library
        self.min_feedback_threshold = min_feedback_threshold
        self.feedback_items: List[FeedbackItem] = []
        self.feedback_summaries: Dict[str, FeedbackSummary] = {}
        
    def add_feedback(
        self,
        template_id: str,
        feedback_type: Union[FeedbackType, str],
        value: Union[float, str, Dict[str, Any]],
        user_id: Optional[str] = None,
        client_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Add a new feedback item."""
        # Validate template exists
        if not self.template_library.get_template(template_id):
            raise ValueError(f"Template {template_id} not found")
        
        # Convert string to FeedbackType if necessary
        if isinstance(feedback_type, str):
            feedback_type = FeedbackType(feedback_type)
        
        # Validate feedback value based on type
        self._validate_feedback_value(feedback_type, value)
        
        # Create feedback item
        feedback_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedback_items)}"
        feedback_item = FeedbackItem(
            feedback_id=feedback_id,
            template_id=template_id,
            feedback_type=feedback_type,
            value=value,
            user_id=user_id,
            client_context=client_context or {},
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Add to collection
        self.feedback_items.append(feedback_item)
        
        # Update summary
        self._update_summary(feedback_item)
        
        logger.info(f"Added {feedback_type.value} feedback for template {template_id}")
        return feedback_id
    
    def get_feedback(
        self,
        template_id: Optional[str] = None,
        feedback_type: Optional[Union[FeedbackType, str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> List[FeedbackItem]:
        """Get feedback items matching criteria."""
        filtered_feedback = self.feedback_items
        
        if template_id:
            filtered_feedback = [
                f for f in filtered_feedback
                if f.template_id == template_id
            ]
        
        if feedback_type:
            if isinstance(feedback_type, str):
                feedback_type = FeedbackType(feedback_type)
            filtered_feedback = [
                f for f in filtered_feedback
                if f.feedback_type == feedback_type
            ]
        
        if start_date:
            filtered_feedback = [
                f for f in filtered_feedback
                if f.created_at >= start_date
            ]
        
        if end_date:
            filtered_feedback = [
                f for f in filtered_feedback
                if f.created_at <= end_date
            ]
        
        if tags:
            filtered_feedback = [
                f for f in filtered_feedback
                if any(tag in f.tags for tag in tags)
            ]
        
        if user_id:
            filtered_feedback = [
                f for f in filtered_feedback
                if f.user_id == user_id
            ]
        
        return filtered_feedback
    
    def get_summary(
        self,
        template_id: str,
        recalculate: bool = False
    ) -> Optional[FeedbackSummary]:
        """Get feedback summary for a template."""
        if template_id not in self.feedback_summaries or recalculate:
            self._calculate_summary(template_id)
        
        return self.feedback_summaries.get(template_id)
    
    def get_trending_issues(
        self,
        min_occurrences: int = 3,
        time_window_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get trending issues across templates."""
        recent_feedback = [
            f for f in self.feedback_items
            if (
                f.feedback_type == FeedbackType.ISSUE and
                (datetime.now() - f.created_at).days <= time_window_days
            )
        ]
        
        # Group issues by type/category
        issue_groups = {}
        for feedback in recent_feedback:
            issue_type = feedback.value.get('type', 'unknown')
            if issue_type not in issue_groups:
                issue_groups[issue_type] = {
                    'type': issue_type,
                    'count': 0,
                    'templates': set(),
                    'examples': []
                }
            
            group = issue_groups[issue_type]
            group['count'] += 1
            group['templates'].add(feedback.template_id)
            if len(group['examples']) < 3:  # Keep up to 3 examples
                group['examples'].append(feedback.value)
        
        # Filter and sort trending issues
        trending = [
            {
                'type': issue_type,
                'count': data['count'],
                'affected_templates': list(data['templates']),
                'examples': data['examples']
            }
            for issue_type, data in issue_groups.items()
            if data['count'] >= min_occurrences
        ]
        
        return sorted(trending, key=lambda x: x['count'], reverse=True)
    
    def analyze_modification_patterns(
        self,
        template_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in template modifications."""
        modifications = self.get_feedback(
            template_id=template_id,
            feedback_type=FeedbackType.MODIFICATION
        )
        
        if not modifications:
            return []
        
        # Group modifications by type
        patterns = {}
        for mod in modifications:
            mod_type = mod.value.get('type', 'unknown')
            if mod_type not in patterns:
                patterns[mod_type] = {
                    'type': mod_type,
                    'count': 0,
                    'templates': set(),
                    'common_changes': {},
                    'examples': []
                }
            
            pattern = patterns[mod_type]
            pattern['count'] += 1
            pattern['templates'].add(mod.template_id)
            
            # Track specific changes
            changes = mod.value.get('changes', {})
            for change_type, details in changes.items():
                if change_type not in pattern['common_changes']:
                    pattern['common_changes'][change_type] = {
                        'count': 0,
                        'examples': []
                    }
                
                change_data = pattern['common_changes'][change_type]
                change_data['count'] += 1
                if len(change_data['examples']) < 3:
                    change_data['examples'].append(details)
        
        # Convert to list and sort by frequency
        return sorted(
            [
                {
                    'type': mod_type,
                    'count': data['count'],
                    'affected_templates': list(data['templates']),
                    'common_changes': data['common_changes']
                }
                for mod_type, data in patterns.items()
            ],
            key=lambda x: x['count'],
            reverse=True
        )
    
    def _validate_feedback_value(
        self,
        feedback_type: FeedbackType,
        value: Any
    ) -> None:
        """Validate feedback value based on type."""
        if feedback_type == FeedbackType.RATING:
            if not isinstance(value, (int, float)) or value < 1 or value > 5:
                raise ValueError("Rating must be a number between 1 and 5")
        
        elif feedback_type == FeedbackType.COMMENT:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Comment must be a non-empty string")
        
        elif feedback_type == FeedbackType.MODIFICATION:
            if not isinstance(value, dict) or 'type' not in value:
                raise ValueError("Modification must be a dict with 'type' field")
        
        elif feedback_type == FeedbackType.COMPLETION_TIME:
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError("Completion time must be a positive number")
        
        elif feedback_type == FeedbackType.CONVERSION:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Conversion event must be a non-empty string")
        
        elif feedback_type == FeedbackType.ISSUE:
            if not isinstance(value, dict) or 'type' not in value:
                raise ValueError("Issue must be a dict with 'type' field")
        
        elif feedback_type == FeedbackType.SUGGESTION:
            if not isinstance(value, dict) or 'description' not in value:
                raise ValueError("Suggestion must be a dict with 'description' field")
    
    def _update_summary(self, feedback_item: FeedbackItem) -> None:
        """Update feedback summary with new item."""
        template_id = feedback_item.template_id
        if template_id not in self.feedback_summaries:
            self.feedback_summaries[template_id] = FeedbackSummary(template_id)
        
        summary = self.feedback_summaries[template_id]
        summary.total_feedback += 1
        summary.last_updated = datetime.now()
        
        if feedback_item.feedback_type == FeedbackType.RATING:
            rating = int(feedback_item.value)
            summary.rating_distribution[rating] = summary.rating_distribution.get(rating, 0) + 1
            total_ratings = sum(summary.rating_distribution.values())
            summary.average_rating = sum(
                rating * count
                for rating, count in summary.rating_distribution.items()
            ) / total_ratings
        
        elif feedback_item.feedback_type == FeedbackType.ISSUE:
            summary.common_issues = self._update_common_items(
                summary.common_issues,
                feedback_item.value,
                max_items=5
            )
        
        elif feedback_item.feedback_type == FeedbackType.SUGGESTION:
            summary.common_suggestions = self._update_common_items(
                summary.common_suggestions,
                feedback_item.value,
                max_items=5
            )
        
        elif feedback_item.feedback_type == FeedbackType.MODIFICATION:
            summary.modification_patterns = self._update_common_items(
                summary.modification_patterns,
                feedback_item.value,
                max_items=5
            )
    
    def _calculate_summary(self, template_id: str) -> None:
        """Calculate full feedback summary for a template."""
        feedback = self.get_feedback(template_id=template_id)
        if not feedback:
            return
        
        summary = FeedbackSummary(template_id)
        summary.total_feedback = len(feedback)
        
        # Calculate ratings
        ratings = [
            f.value for f in feedback
            if f.feedback_type == FeedbackType.RATING
        ]
        if ratings:
            for rating in ratings:
                rating = int(rating)
                summary.rating_distribution[rating] = summary.rating_distribution.get(rating, 0) + 1
            summary.average_rating = sum(ratings) / len(ratings)
        
        # Calculate conversion rate
        conversions = [
            f for f in feedback
            if f.feedback_type == FeedbackType.CONVERSION
        ]
        if conversions:
            summary.conversion_rate = len(conversions) / summary.total_feedback
        
        # Calculate average completion time
        completion_times = [
            f.value for f in feedback
            if f.feedback_type == FeedbackType.COMPLETION_TIME
        ]
        if completion_times:
            summary.avg_completion_time = sum(completion_times) / len(completion_times)
        
        # Collect common issues
        issues = [
            f.value for f in feedback
            if f.feedback_type == FeedbackType.ISSUE
        ]
        if issues:
            summary.common_issues = self._find_common_items(issues, max_items=5)
        
        # Collect common suggestions
        suggestions = [
            f.value for f in feedback
            if f.feedback_type == FeedbackType.SUGGESTION
        ]
        if suggestions:
            summary.common_suggestions = self._find_common_items(suggestions, max_items=5)
        
        # Analyze modification patterns
        modifications = [
            f.value for f in feedback
            if f.feedback_type == FeedbackType.MODIFICATION
        ]
        if modifications:
            summary.modification_patterns = self._find_common_items(modifications, max_items=5)
        
        self.feedback_summaries[template_id] = summary
    
    @staticmethod
    def _update_common_items(
        existing_items: List[Dict[str, Any]],
        new_item: Dict[str, Any],
        max_items: int = 5
    ) -> List[Dict[str, Any]]:
        """Update list of common items with new item."""
        # Try to find matching item
        for item in existing_items:
            if item['type'] == new_item['type']:
                item['count'] = item.get('count', 0) + 1
                return sorted(
                    existing_items,
                    key=lambda x: x.get('count', 0),
                    reverse=True
                )[:max_items]
        
        # Add new item
        new_entry = {
            'type': new_item['type'],
            'count': 1,
            'details': new_item
        }
        
        updated_items = existing_items + [new_entry]
        return sorted(
            updated_items,
            key=lambda x: x.get('count', 0),
            reverse=True
        )[:max_items]
    
    @staticmethod
    def _find_common_items(
        items: List[Dict[str, Any]],
        max_items: int = 5
    ) -> List[Dict[str, Any]]:
        """Find common items in a list of dictionaries."""
        item_counts = {}
        for item in items:
            item_type = item['type']
            if item_type not in item_counts:
                item_counts[item_type] = {
                    'type': item_type,
                    'count': 0,
                    'details': item
                }
            item_counts[item_type]['count'] += 1
        
        return sorted(
            item_counts.values(),
            key=lambda x: x['count'],
            reverse=True
        )[:max_items] 