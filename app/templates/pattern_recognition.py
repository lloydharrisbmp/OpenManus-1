"""
Advanced pattern recognition system for template analytics.
"""
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
import json
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class PatternMatch:
    """Represents a detected pattern in template usage."""
    pattern_type: str  # 'sequence', 'cluster', 'anomaly', 'correlation'
    confidence: float
    description: str
    evidence: Dict[str, Any]
    affected_templates: Set[str]
    detected_at: datetime = field(default_factory=datetime.now)

class AdvancedPatternRecognition:
    """Advanced pattern recognition for template usage analysis."""
    
    def __init__(
        self,
        min_sequence_length: int = 3,
        min_cluster_size: int = 5,
        anomaly_threshold: float = 0.1
    ):
        self.min_sequence_length = min_sequence_length
        self.min_cluster_size = min_cluster_size
        self.anomaly_threshold = anomaly_threshold
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        
    def analyze_usage_patterns(
        self,
        usage_data: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Analyze usage data for patterns."""
        patterns = []
        
        # Convert usage data to features
        features = self._extract_features(usage_data)
        
        # Find sequence patterns
        sequence_patterns = self._detect_sequence_patterns(usage_data)
        patterns.extend(sequence_patterns)
        
        # Find clusters
        cluster_patterns = self._detect_clusters(features, usage_data)
        patterns.extend(cluster_patterns)
        
        # Detect anomalies
        anomaly_patterns = self._detect_anomalies(features, usage_data)
        patterns.extend(anomaly_patterns)
        
        # Find correlations
        correlation_patterns = self._detect_correlations(usage_data)
        patterns.extend(correlation_patterns)
        
        return patterns
    
    def _extract_features(
        self,
        usage_data: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract numerical features from usage data."""
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(usage_data)
        
        # Extract numerical features
        numerical_features = []
        for record in usage_data:
            features = {
                'parameter_count': len(record.get('parameters', {})),
                'context_complexity': len(json.dumps(record.get('client_context', {}))),
                'modification_count': len(record.get('modifications', [])),
                'feedback_score': record.get('feedback_score', 0),
                'template_length': len(record.get('content', '')),
            }
            numerical_features.append(features)
        
        # Convert to numpy array and normalize
        feature_array = pd.DataFrame(numerical_features).to_numpy()
        return self.scaler.fit_transform(feature_array)
    
    def _detect_sequence_patterns(
        self,
        usage_data: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect sequential patterns in template usage."""
        patterns = []
        
        # Create sequences of template usage
        template_sequences = defaultdict(list)
        for record in usage_data:
            user_id = record.get('user_id')
            if user_id:
                template_sequences[user_id].append(record['template_id'])
        
        # Find common subsequences
        common_sequences = self._find_common_subsequences(template_sequences.values())
        
        for seq, count in common_sequences.items():
            if len(seq) >= self.min_sequence_length:
                confidence = count / len(template_sequences)
                if confidence >= 0.3:  # At least 30% of users follow this pattern
                    patterns.append(PatternMatch(
                        pattern_type='sequence',
                        confidence=confidence,
                        description=f"Common template sequence: {' -> '.join(seq)}",
                        evidence={
                            'sequence': seq,
                            'occurrence_count': count,
                            'total_users': len(template_sequences)
                        },
                        affected_templates=set(seq)
                    ))
        
        return patterns
    
    def _detect_clusters(
        self,
        features: np.ndarray,
        usage_data: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect clusters in template usage patterns."""
        patterns = []
        
        # Try different clustering algorithms
        # DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.3, min_samples=self.min_cluster_size)
        dbscan_labels = dbscan.fit_predict(features)
        
        # KMeans for centroid-based clustering
        kmeans = KMeans(n_clusters=min(5, len(usage_data)))
        kmeans_labels = kmeans.fit_predict(features)
        
        # Analyze DBSCAN clusters
        cluster_groups = defaultdict(list)
        for i, label in enumerate(dbscan_labels):
            if label != -1:  # Not noise
                cluster_groups[label].append(usage_data[i])
        
        for label, cluster_data in cluster_groups.items():
            if len(cluster_data) >= self.min_cluster_size:
                common_traits = self._find_cluster_traits(cluster_data)
                patterns.append(PatternMatch(
                    pattern_type='cluster',
                    confidence=len(cluster_data) / len(usage_data),
                    description=f"Usage pattern cluster identified: {common_traits}",
                    evidence={
                        'cluster_size': len(cluster_data),
                        'common_traits': common_traits,
                        'cluster_label': label
                    },
                    affected_templates=set(
                        record['template_id'] for record in cluster_data
                    )
                ))
        
        return patterns
    
    def _detect_anomalies(
        self,
        features: np.ndarray,
        usage_data: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect anomalous template usage patterns."""
        patterns = []
        
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=self.anomaly_threshold)
        anomaly_labels = iso_forest.fit_predict(features)
        
        # Collect anomalies
        anomalies = [
            usage_data[i] for i, label in enumerate(anomaly_labels)
            if label == -1
        ]
        
        if anomalies:
            # Group similar anomalies
            anomaly_groups = self._group_similar_anomalies(anomalies)
            
            for group in anomaly_groups:
                patterns.append(PatternMatch(
                    pattern_type='anomaly',
                    confidence=1.0 - self.anomaly_threshold,
                    description="Unusual template usage pattern detected",
                    evidence={
                        'anomaly_count': len(group),
                        'anomaly_characteristics': self._analyze_anomalies(group)
                    },
                    affected_templates=set(
                        record['template_id'] for record in group
                    )
                ))
        
        return patterns
    
    def _detect_correlations(
        self,
        usage_data: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect correlations between template usage and outcomes."""
        patterns = []
        
        # Convert to DataFrame for correlation analysis
        df = pd.DataFrame([
            {
                'template_id': record['template_id'],
                'feedback_score': record.get('feedback_score', 0),
                'modification_count': len(record.get('modifications', [])),
                'parameter_count': len(record.get('parameters', {})),
                'context_complexity': len(json.dumps(record.get('client_context', {})))
            }
            for record in usage_data
        ])
        
        # Calculate correlations
        correlations = df.corr()
        
        # Find strong correlations
        for col1 in correlations.columns:
            for col2 in correlations.index:
                if col1 < col2:  # Avoid duplicate correlations
                    correlation = correlations.loc[col1, col2]
                    if abs(correlation) >= 0.7:  # Strong correlation threshold
                        patterns.append(PatternMatch(
                            pattern_type='correlation',
                            confidence=abs(correlation),
                            description=f"Strong correlation between {col1} and {col2}",
                            evidence={
                                'correlation_coefficient': correlation,
                                'feature1': col1,
                                'feature2': col2
                            },
                            affected_templates=set(df['template_id'].unique())
                        ))
        
        return patterns
    
    @staticmethod
    def _find_common_subsequences(
        sequences: List[List[str]]
    ) -> Dict[Tuple[str, ...], int]:
        """Find common subsequences in template usage."""
        subsequences = defaultdict(int)
        
        for sequence in sequences:
            n = len(sequence)
            # Look for subsequences of length 2 or more
            for length in range(2, n + 1):
                for i in range(n - length + 1):
                    subsequence = tuple(sequence[i:i + length])
                    subsequences[subsequence] += 1
        
        return subsequences
    
    @staticmethod
    def _find_cluster_traits(cluster_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze common traits in a cluster."""
        traits = defaultdict(int)
        
        for record in cluster_data:
            # Analyze parameters
            for param, value in record.get('parameters', {}).items():
                traits[f"param_{param}_{value}"] += 1
            
            # Analyze context
            for key, value in record.get('client_context', {}).items():
                traits[f"context_{key}_{value}"] += 1
            
            # Analyze modifications
            for mod in record.get('modifications', []):
                traits[f"modification_{mod['type']}"] += 1
        
        # Keep only significant traits (occurring in >50% of records)
        threshold = len(cluster_data) * 0.5
        return {
            k: v for k, v in traits.items()
            if v >= threshold
        }
    
    @staticmethod
    def _group_similar_anomalies(
        anomalies: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group similar anomalies together."""
        groups = []
        processed = set()
        
        for i, anomaly1 in enumerate(anomalies):
            if i in processed:
                continue
                
            group = [anomaly1]
            processed.add(i)
            
            # Find similar anomalies
            for j, anomaly2 in enumerate(anomalies[i + 1:], i + 1):
                if j not in processed and _are_anomalies_similar(anomaly1, anomaly2):
                    group.append(anomaly2)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    @staticmethod
    def _analyze_anomalies(anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of anomalous records."""
        characteristics = {
            'parameter_patterns': defaultdict(int),
            'context_patterns': defaultdict(int),
            'modification_patterns': defaultdict(int),
            'feedback_scores': []
        }
        
        for record in anomalies:
            # Analyze parameters
            for param, value in record.get('parameters', {}).items():
                characteristics['parameter_patterns'][f"{param}={value}"] += 1
            
            # Analyze context
            for key, value in record.get('client_context', {}).items():
                characteristics['context_patterns'][f"{key}={value}"] += 1
            
            # Analyze modifications
            for mod in record.get('modifications', []):
                characteristics['modification_patterns'][mod['type']] += 1
            
            # Record feedback scores
            if 'feedback_score' in record:
                characteristics['feedback_scores'].append(record['feedback_score'])
        
        # Calculate statistics for feedback scores
        if characteristics['feedback_scores']:
            characteristics['avg_feedback'] = sum(characteristics['feedback_scores']) / len(characteristics['feedback_scores'])
            characteristics['feedback_variance'] = np.var(characteristics['feedback_scores'])
        
        return characteristics

def _are_anomalies_similar(a1: Dict[str, Any], a2: Dict[str, Any]) -> bool:
    """Determine if two anomalous records are similar."""
    # Check template similarity
    if a1['template_id'] != a2['template_id']:
        return False
    
    # Check parameter similarity
    params1 = set(a1.get('parameters', {}).items())
    params2 = set(a2.get('parameters', {}).items())
    param_similarity = len(params1 & params2) / len(params1 | params2) if params1 or params2 else 1.0
    
    # Check context similarity
    ctx1 = set(a1.get('client_context', {}).items())
    ctx2 = set(a2.get('client_context', {}).items())
    ctx_similarity = len(ctx1 & ctx2) / len(ctx1 | ctx2) if ctx1 or ctx2 else 1.0
    
    # Check modification similarity
    mods1 = {m['type'] for m in a1.get('modifications', [])}
    mods2 = {m['type'] for m in a2.get('modifications', [])}
    mod_similarity = len(mods1 & mods2) / len(mods1 | mods2) if mods1 or mods2 else 1.0
    
    # Calculate overall similarity
    similarity = (param_similarity + ctx_similarity + mod_similarity) / 3
    return similarity >= 0.7  # 70% similarity threshold 