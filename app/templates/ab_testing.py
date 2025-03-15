"""
A/B testing system for template variants.
"""
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import numpy as np
from scipy import stats
import json
import logging
from .advice_templates import TemplateSection, TemplateLibrary

logger = logging.getLogger(__name__)

@dataclass
class ABTest:
    """Represents an A/B test for template variants."""
    test_id: str
    control_template_id: str
    variant_template_ids: List[str]
    start_date: datetime
    end_date: datetime
    metrics: List[str]  # List of metrics to track
    segment_by: Optional[List[str]] = None  # Client context fields to segment by
    min_sample_size: int = 100
    confidence_level: float = 0.95
    is_active: bool = True
    results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemplateVariant:
    """Tracks performance metrics for a template variant."""
    template_id: str
    impressions: int = 0
    feedback_scores: List[float] = field(default_factory=list)
    completion_time: List[float] = field(default_factory=list)
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    conversion_events: List[str] = field(default_factory=list)
    segment_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class ABTestingService:
    """Service for managing A/B tests of template variants."""
    
    def __init__(
        self,
        template_library: TemplateLibrary,
        min_test_duration: timedelta = timedelta(days=14),
        max_concurrent_tests: int = 5
    ):
        self.template_library = template_library
        self.min_test_duration = min_test_duration
        self.max_concurrent_tests = max_concurrent_tests
        self.active_tests: Dict[str, ABTest] = {}
        self.test_data: Dict[str, Dict[str, TemplateVariant]] = {}
        
    def create_test(
        self,
        control_template_id: str,
        variant_template_ids: List[str],
        metrics: List[str],
        segment_by: Optional[List[str]] = None,
        duration: Optional[timedelta] = None,
        min_sample_size: int = 100,
        confidence_level: float = 0.95
    ) -> str:
        """Create a new A/B test."""
        if len(self.active_tests) >= self.max_concurrent_tests:
            raise ValueError("Maximum number of concurrent tests reached")
        
        # Validate templates exist
        all_templates = [control_template_id] + variant_template_ids
        for template_id in all_templates:
            if not self.template_library.get_template(template_id):
                raise ValueError(f"Template {template_id} not found")
        
        # Create test
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        duration = duration or self.min_test_duration
        
        test = ABTest(
            test_id=test_id,
            control_template_id=control_template_id,
            variant_template_ids=variant_template_ids,
            start_date=datetime.now(),
            end_date=datetime.now() + duration,
            metrics=metrics,
            segment_by=segment_by,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level
        )
        
        self.active_tests[test_id] = test
        self.test_data[test_id] = {
            template_id: TemplateVariant(template_id)
            for template_id in all_templates
        }
        
        logger.info(f"Created A/B test {test_id} for templates {all_templates}")
        return test_id
    
    def select_template(
        self,
        test_id: str,
        client_context: Dict[str, Any]
    ) -> str:
        """Select a template variant for a given client context."""
        test = self.active_tests.get(test_id)
        if not test or not test.is_active:
            raise ValueError(f"Test {test_id} not found or inactive")
        
        # Get all template options
        templates = [test.control_template_id] + test.variant_template_ids
        
        # Select template based on Thompson Sampling
        scores = []
        for template_id in templates:
            variant = self.test_data[test_id][template_id]
            if variant.feedback_scores:
                alpha = sum(1 for s in variant.feedback_scores if s >= 4.0)
                beta = sum(1 for s in variant.feedback_scores if s < 4.0)
            else:
                alpha = 1
                beta = 1
            
            score = random.betavariate(alpha, beta)
            scores.append(score)
        
        selected_template = templates[np.argmax(scores)]
        
        # Record impression
        self.test_data[test_id][selected_template].impressions += 1
        
        return selected_template
    
    def record_metrics(
        self,
        test_id: str,
        template_id: str,
        metrics: Dict[str, Any],
        client_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record metrics for a template variant."""
        if test_id not in self.active_tests:
            return
            
        variant = self.test_data[test_id].get(template_id)
        if not variant:
            return
            
        # Record metrics
        if 'feedback_score' in metrics:
            variant.feedback_scores.append(metrics['feedback_score'])
        
        if 'completion_time' in metrics:
            variant.completion_time.append(metrics['completion_time'])
        
        if 'modifications' in metrics:
            variant.modifications.append(metrics['modifications'])
        
        if 'conversion_event' in metrics:
            variant.conversion_events.append(metrics['conversion_event'])
        
        # Record segment data if applicable
        if client_context and self.active_tests[test_id].segment_by:
            segment_key = self._get_segment_key(
                client_context,
                self.active_tests[test_id].segment_by
            )
            if segment_key not in variant.segment_data:
                variant.segment_data[segment_key] = TemplateVariant(template_id)
            
            segment_variant = variant.segment_data[segment_key]
            for metric, value in metrics.items():
                if hasattr(segment_variant, metric):
                    getattr(segment_variant, metric).append(value)
    
    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze test results."""
        test = self.active_tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")
            
        results = {
            'test_id': test_id,
            'status': 'active' if test.is_active else 'completed',
            'duration': (datetime.now() - test.start_date).days,
            'metrics': {},
            'segments': {},
            'recommendations': []
        }
        
        control_data = self.test_data[test_id][test.control_template_id]
        
        # Analyze each metric
        for metric in test.metrics:
            metric_results = self._analyze_metric(
                test,
                metric,
                control_data,
                self.test_data[test_id]
            )
            results['metrics'][metric] = metric_results
        
        # Analyze segments if applicable
        if test.segment_by:
            for segment in self._get_unique_segments(test_id):
                segment_results = self._analyze_segment(test, segment)
                if segment_results:
                    results['segments'][segment] = segment_results
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(
            test,
            results['metrics'],
            results['segments']
        )
        
        # Update test results
        test.results = results
        
        return results
    
    def end_test(self, test_id: str) -> Dict[str, Any]:
        """End an A/B test and return final results."""
        test = self.active_tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")
            
        test.is_active = False
        test.end_date = datetime.now()
        
        # Perform final analysis
        final_results = self.analyze_test(test_id)
        
        # Clean up
        self.active_tests.pop(test_id)
        
        return final_results
    
    def _analyze_metric(
        self,
        test: ABTest,
        metric: str,
        control_data: TemplateVariant,
        test_data: Dict[str, TemplateVariant]
    ) -> Dict[str, Any]:
        """Analyze a specific metric across variants."""
        results = {
            'control': self._calculate_metric_stats(control_data, metric),
            'variants': {}
        }
        
        for variant_id in test.variant_template_ids:
            variant_data = test_data[variant_id]
            variant_stats = self._calculate_metric_stats(variant_data, metric)
            
            # Perform statistical test
            if variant_stats['sample_size'] >= test.min_sample_size:
                p_value = self._calculate_p_value(
                    control_data,
                    variant_data,
                    metric
                )
                
                variant_stats['p_value'] = p_value
                variant_stats['significant'] = p_value < (1 - test.confidence_level)
            
            results['variants'][variant_id] = variant_stats
        
        return results
    
    def _analyze_segment(
        self,
        test: ABTest,
        segment: str
    ) -> Dict[str, Any]:
        """Analyze test results for a specific segment."""
        results = {}
        
        control_data = self.test_data[test_id][test.control_template_id]
        if segment not in control_data.segment_data:
            return None
            
        for metric in test.metrics:
            results[metric] = {
                'control': self._calculate_metric_stats(
                    control_data.segment_data[segment],
                    metric
                ),
                'variants': {}
            }
            
            for variant_id in test.variant_template_ids:
                variant_data = self.test_data[test_id][variant_id]
                if segment in variant_data.segment_data:
                    variant_stats = self._calculate_metric_stats(
                        variant_data.segment_data[segment],
                        metric
                    )
                    
                    if variant_stats['sample_size'] >= test.min_sample_size:
                        p_value = self._calculate_p_value(
                            control_data.segment_data[segment],
                            variant_data.segment_data[segment],
                            metric
                        )
                        
                        variant_stats['p_value'] = p_value
                        variant_stats['significant'] = p_value < (1 - test.confidence_level)
                    
                    results[metric]['variants'][variant_id] = variant_stats
        
        return results
    
    @staticmethod
    def _calculate_metric_stats(
        variant: TemplateVariant,
        metric: str
    ) -> Dict[str, Any]:
        """Calculate statistics for a metric."""
        data = getattr(variant, metric, [])
        if not data:
            return {'sample_size': 0}
            
        return {
            'sample_size': len(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data),
            'min': np.min(data),
            'max': np.max(data)
        }
    
    @staticmethod
    def _calculate_p_value(
        control: TemplateVariant,
        variant: TemplateVariant,
        metric: str
    ) -> float:
        """Calculate p-value for statistical significance."""
        control_data = getattr(control, metric, [])
        variant_data = getattr(variant, metric, [])
        
        if not control_data or not variant_data:
            return 1.0
            
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(control_data, variant_data)
        return p_value
    
    @staticmethod
    def _get_segment_key(
        context: Dict[str, Any],
        segment_by: List[str]
    ) -> str:
        """Create a key for segmentation."""
        segment_values = []
        for field in segment_by:
            value = context.get(field, 'unknown')
            segment_values.append(f"{field}={value}")
        return "|".join(segment_values)
    
    def _get_unique_segments(self, test_id: str) -> Set[str]:
        """Get unique segments across all variants."""
        segments = set()
        for variant in self.test_data[test_id].values():
            segments.update(variant.segment_data.keys())
        return segments
    
    def _generate_recommendations(
        self,
        test: ABTest,
        metric_results: Dict[str, Any],
        segment_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check if test has enough data
        total_impressions = sum(
            self.test_data[test.test_id][template_id].impressions
            for template_id in [test.control_template_id] + test.variant_template_ids
        )
        
        if total_impressions < test.min_sample_size:
            recommendations.append({
                'type': 'warning',
                'message': f"Insufficient data (need {test.min_sample_size} samples, have {total_impressions})"
            })
            return recommendations
        
        # Analyze each metric
        for metric, results in metric_results.items():
            control_stats = results['control']
            
            for variant_id, variant_stats in results['variants'].items():
                if variant_stats.get('significant'):
                    relative_change = (
                        (variant_stats['mean'] - control_stats['mean']) /
                        control_stats['mean']
                    )
                    
                    recommendations.append({
                        'type': 'variant_performance',
                        'metric': metric,
                        'variant_id': variant_id,
                        'relative_change': relative_change,
                        'p_value': variant_stats['p_value'],
                        'message': (
                            f"Variant {variant_id} shows significant "
                            f"{'improvement' if relative_change > 0 else 'decrease'} "
                            f"in {metric} ({relative_change:.1%} change)"
                        )
                    })
        
        # Analyze segments
        for segment, segment_data in segment_results.items():
            for metric, results in segment_data.items():
                control_stats = results['control']
                
                for variant_id, variant_stats in results['variants'].items():
                    if variant_stats.get('significant'):
                        relative_change = (
                            (variant_stats['mean'] - control_stats['mean']) /
                            control_stats['mean']
                        )
                        
                        recommendations.append({
                            'type': 'segment_performance',
                            'segment': segment,
                            'metric': metric,
                            'variant_id': variant_id,
                            'relative_change': relative_change,
                            'p_value': variant_stats['p_value'],
                            'message': (
                                f"For segment {segment}, variant {variant_id} shows "
                                f"significant {'improvement' if relative_change > 0 else 'decrease'} "
                                f"in {metric} ({relative_change:.1%} change)"
                            )
                        })
        
        # Add test duration recommendation
        test_duration = (datetime.now() - test.start_date).days
        if test_duration < self.min_test_duration.days:
            recommendations.append({
                'type': 'duration',
                'message': (
                    f"Test has been running for {test_duration} days, "
                    f"recommend running for at least {self.min_test_duration.days} days"
                )
            })
        
        return recommendations 