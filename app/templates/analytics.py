"""
Analytics system for template usage and enhancement suggestions.
"""
from typing import Dict, List, Optional, Any, Set, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from .advice_templates import TemplateSection, TemplateLibrary

@dataclass
class TemplateUsage:
    """Tracks usage statistics for a template."""
    template_id: str
    use_count: int = 0
    last_used: Optional[datetime] = None
    parameter_frequencies: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    client_contexts: List[Dict[str, Any]] = field(default_factory=list)
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    feedback_scores: List[int] = field(default_factory=list)
    common_edits: Counter = field(default_factory=Counter)

@dataclass
class TemplateInsight:
    """Represents an insight or suggestion for template improvement."""
    template_id: str
    insight_type: str  # 'parameter', 'content', 'condition', 'new_template'
    confidence: float
    description: str
    suggested_changes: Dict[str, Any]
    supporting_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0

class TemplateAnalytics:
    """Analyzes template usage and generates insights."""
    
    def __init__(self, template_library: TemplateLibrary):
        self.template_library = template_library
        self.usage_data: Dict[str, TemplateUsage] = {}
        self.insights: List[TemplateInsight] = []
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.min_data_points = 10
        
    def record_template_use(
        self,
        template_id: str,
        parameters: Dict[str, Any],
        client_context: Dict[str, Any],
        modifications: Optional[Dict[str, Any]] = None,
        feedback_score: Optional[int] = None
    ) -> None:
        """Record a template usage instance."""
        if template_id not in self.usage_data:
            self.usage_data[template_id] = TemplateUsage(template_id)
            
        usage = self.usage_data[template_id]
        usage.use_count += 1
        usage.last_used = datetime.now()
        
        # Record parameter values
        for param, value in parameters.items():
            usage.parameter_frequencies[param][str(value)] += 1
        
        # Store client context for pattern analysis
        usage.client_contexts.append(client_context)
        
        # Record any modifications made to the template
        if modifications:
            usage.modifications.append(modifications)
            for edit_type, details in modifications.items():
                usage.common_edits[edit_type] += 1
        
        # Record feedback if provided
        if feedback_score is not None:
            usage.feedback_scores.append(feedback_score)
        
        # Generate insights if we have enough data
        if self._should_generate_insights(template_id):
            self._analyze_template_usage(template_id)
    
    def _should_generate_insights(self, template_id: str) -> bool:
        """Determine if we have enough data to generate meaningful insights."""
        usage = self.usage_data.get(template_id)
        if not usage:
            return False
            
        return (
            usage.use_count >= self.min_data_points or
            len(usage.modifications) >= 5 or
            (usage.feedback_scores and 
             sum(usage.feedback_scores) / len(usage.feedback_scores) < 4.0)
        )
    
    def _analyze_template_usage(self, template_id: str) -> None:
        """Analyze template usage and generate insights."""
        usage = self.usage_data[template_id]
        template = self.template_library.get_template(template_id)
        
        if not template:
            return
            
        # Analyze parameter patterns
        self._analyze_parameters(template, usage)
        
        # Analyze modifications
        self._analyze_modifications(template, usage)
        
        # Analyze client contexts
        self._analyze_client_contexts(template, usage)
        
        # Look for potential new templates
        self._analyze_potential_new_templates(template, usage)
    
    def _analyze_parameters(
        self,
        template: TemplateSection,
        usage: TemplateUsage
    ) -> None:
        """Analyze parameter usage patterns."""
        for param, frequencies in usage.parameter_frequencies.items():
            total_uses = sum(frequencies.values())
            
            # Check for unused parameters
            if total_uses < usage.use_count * 0.1:
                self._add_insight(
                    template.id,
                    "parameter",
                    0.8,
                    f"Parameter '{param}' is rarely used",
                    {
                        "action": "remove_parameter",
                        "parameter": param
                    },
                    {"usage_rate": total_uses / usage.use_count}
                )
            
            # Check for common values
            most_common = frequencies.most_common(1)
            if most_common and most_common[0][1] / total_uses > 0.9:
                self._add_insight(
                    template.id,
                    "parameter",
                    0.7,
                    f"Parameter '{param}' almost always uses value '{most_common[0][0]}'",
                    {
                        "action": "default_value",
                        "parameter": param,
                        "value": most_common[0][0]
                    },
                    {"frequency": most_common[0][1] / total_uses}
                )
    
    def _analyze_modifications(
        self,
        template: TemplateSection,
        usage: TemplateUsage
    ) -> None:
        """Analyze common modifications to suggest template improvements."""
        if not usage.modifications:
            return
            
        # Analyze common edits
        for edit_type, count in usage.common_edits.most_common():
            if count >= 3:  # If an edit type appears frequently
                self._add_insight(
                    template.id,
                    "content",
                    0.75,
                    f"Common {edit_type} modifications detected",
                    {
                        "action": "review_content",
                        "edit_type": edit_type,
                        "frequency": count
                    },
                    {"modifications": [
                        m for m in usage.modifications 
                        if edit_type in m
                    ]}
                )
    
    def _analyze_client_contexts(
        self,
        template: TemplateSection,
        usage: TemplateUsage
    ) -> None:
        """Analyze client contexts to identify potential condition improvements."""
        if not usage.client_contexts:
            return
            
        # Convert contexts to feature vectors
        context_data = []
        for ctx in usage.client_contexts:
            flat_ctx = self._flatten_dict(ctx)
            context_data.append(json.dumps(flat_ctx))
        
        # Vectorize and cluster contexts
        vectors = self.vectorizer.fit_transform(context_data)
        clusters = DBSCAN(eps=0.3, min_samples=3).fit(vectors.toarray())
        
        # Analyze clusters for patterns
        cluster_groups = defaultdict(list)
        for i, label in enumerate(clusters.labels_):
            if label != -1:  # Ignore noise points
                cluster_groups[label].append(usage.client_contexts[i])
        
        for cluster_contexts in cluster_groups.values():
            common_patterns = self._find_common_patterns(cluster_contexts)
            if common_patterns:
                self._add_insight(
                    template.id,
                    "condition",
                    0.7,
                    "Identified potential new condition pattern",
                    {
                        "action": "add_condition",
                        "pattern": common_patterns
                    },
                    {"contexts": cluster_contexts}
                )
    
    def _analyze_potential_new_templates(
        self,
        template: TemplateSection,
        usage: TemplateUsage
    ) -> None:
        """Analyze if variations warrant new templates."""
        if len(usage.modifications) < 5:
            return
            
        # Group similar modifications
        mod_texts = [json.dumps(m) for m in usage.modifications]
        vectors = self.vectorizer.fit_transform(mod_texts)
        clusters = DBSCAN(eps=0.3, min_samples=3).fit(vectors.toarray())
        
        # Analyze each significant cluster
        cluster_groups = defaultdict(list)
        for i, label in enumerate(clusters.labels_):
            if label != -1:
                cluster_groups[label].append(usage.modifications[i])
        
        for cluster_mods in cluster_groups.values():
            if len(cluster_mods) >= 3:  # If we see a pattern of similar modifications
                common_changes = self._extract_common_changes(cluster_mods)
                if common_changes:
                    self._add_insight(
                        template.id,
                        "new_template",
                        0.8,
                        "Identified potential new template variant",
                        {
                            "action": "create_template_variant",
                            "base_template": template.id,
                            "changes": common_changes
                        },
                        {"modifications": cluster_mods}
                    )
    
    def _add_insight(
        self,
        template_id: str,
        insight_type: str,
        confidence: float,
        description: str,
        suggested_changes: Dict[str, Any],
        supporting_data: Dict[str, Any]
    ) -> None:
        """Add a new insight with priority based on confidence and impact."""
        priority = int(confidence * 100)
        
        # Adjust priority based on insight type
        if insight_type == "new_template":
            priority += 20
        elif insight_type == "condition":
            priority += 10
        
        # Adjust priority based on usage frequency
        usage = self.usage_data.get(template_id)
        if usage and usage.use_count > 50:
            priority += 15
        
        # Adjust priority based on feedback scores
        if usage and usage.feedback_scores:
            avg_score = sum(usage.feedback_scores) / len(usage.feedback_scores)
            if avg_score < 3.0:
                priority += 25
        
        insight = TemplateInsight(
            template_id=template_id,
            insight_type=insight_type,
            confidence=confidence,
            description=description,
            suggested_changes=suggested_changes,
            supporting_data=supporting_data,
            priority=priority
        )
        
        self.insights.append(insight)
    
    def get_insights(
        self,
        min_confidence: float = 0.0,
        insight_types: Optional[Set[str]] = None,
        limit: Optional[int] = None
    ) -> List[TemplateInsight]:
        """Get filtered and sorted insights."""
        filtered_insights = [
            i for i in self.insights
            if i.confidence >= min_confidence and
            (not insight_types or i.insight_type in insight_types)
        ]
        
        # Sort by priority (highest first)
        sorted_insights = sorted(
            filtered_insights,
            key=lambda x: x.priority,
            reverse=True
        )
        
        if limit:
            return sorted_insights[:limit]
        return sorted_insights
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    TemplateAnalytics._flatten_dict(v, new_key).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def _find_common_patterns(contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common patterns in a list of contexts."""
        if not contexts:
            return {}
            
        patterns = {}
        reference = contexts[0]
        
        for key, value in reference.items():
            if all(
                key in ctx and ctx[key] == value 
                for ctx in contexts[1:]
            ):
                patterns[key] = value
        
        return patterns
    
    @staticmethod
    def _extract_common_changes(
        modifications: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract common changes from a list of modifications."""
        if not modifications:
            return {}
            
        common_changes = {}
        reference = modifications[0]
        
        for key, value in reference.items():
            if all(
                key in mod and mod[key] == value 
                for mod in modifications[1:]
            ):
                common_changes[key] = value
        
        return common_changes 