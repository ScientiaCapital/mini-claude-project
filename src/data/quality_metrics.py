"""
Data Quality Metrics Module
Implementation following TDD approach - making tests pass with minimal code.
"""

import pandas as pd
import numpy as np
import statistics
from typing import Dict, List, Any, Union, Optional
from scipy import stats
import json
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_completeness(df: pd.DataFrame) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate completeness metrics for a dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing overall and per-column completeness scores
    """
    if df.empty:
        return {'overall': 0.0, 'by_column': {}}
    
    # Calculate completeness for each column
    by_column = {}
    for col in df.columns:
        non_null_count = df[col].count()
        total_count = len(df)
        by_column[col] = non_null_count / total_count if total_count > 0 else 0.0
    
    # Calculate overall completeness as median of per-column completeness
    if by_column:
        overall = statistics.median(by_column.values())
    else:
        overall = 0.0
    
    return {
        'overall': overall,
        'by_column': by_column
    }


def detect_outliers(data: List[Union[int, float]], 
                   method: str = 'iqr', 
                   threshold: float = 1.5) -> List[Union[int, float]]:
    """
    Detect outliers in numerical data using specified method.
    
    Args:
        data: List of numerical values
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        List of detected outliers
    """
    if not data or len(data) < 3:
        return []
    
    data_array = np.array([x for x in data if x is not None and not np.isnan(x)])
    
    if len(data_array) < 3:
        return []
    
    outliers = []
    
    if method == 'iqr':
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = [x for x in data_array if x < lower_bound or x > upper_bound]
        
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data_array))
        outliers = [data_array[i] for i, z in enumerate(z_scores) if z > threshold]
    
    return list(outliers)


def assess_label_distribution(labels: List[str]) -> Dict[str, Union[float, bool, str]]:
    """
    Assess the distribution of labels in the dataset.
    
    Args:
        labels: List of label strings
        
    Returns:
        Dictionary containing distribution metrics
    """
    if not labels:
        return {
            'balance_ratio': 0.0,
            'entropy': 0.0,
            'is_imbalanced': True,
            'majority_class': None,
            'class_counts': {}
        }
    
    # Count occurrences of each label
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate balance ratio (min_count / max_count)
    counts = list(label_counts.values())
    balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0.0
    
    # Calculate entropy
    probabilities = [count / total_samples for count in counts]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    # Determine if dataset is imbalanced (threshold: balance_ratio < 0.5)
    is_imbalanced = balance_ratio < 0.5
    
    # Find majority class
    majority_class = max(label_counts, key=label_counts.get)
    
    return {
        'balance_ratio': balance_ratio,
        'entropy': entropy,
        'is_imbalanced': is_imbalanced,
        'majority_class': majority_class,
        'class_counts': dict(label_counts)
    }


def check_data_leakage(train_data: List[str], 
                      test_data: List[str]) -> Dict[str, Union[bool, int, List[str]]]:
    """
    Check for data leakage between training and test sets.
    
    Args:
        train_data: List of training samples
        test_data: List of test samples
        
    Returns:
        Dictionary containing leakage detection results
    """
    train_set = set(train_data)
    test_set = set(test_data)
    
    # Find intersection (leaked samples)
    leaked_samples = list(train_set.intersection(test_set))
    
    return {
        'has_leakage': len(leaked_samples) > 0,
        'leak_count': len(leaked_samples),
        'leaked_samples': leaked_samples,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'leakage_percentage': len(leaked_samples) / len(test_data) * 100 if test_data else 0
    }


def validate_alpaca_format(data: List[Dict[str, str]]) -> Dict[str, Union[bool, int, List[str]]]:
    """
    Validate data against Alpaca format requirements.
    
    Args:
        data: List of dictionaries representing Alpaca format data
        
    Returns:
        Dictionary containing validation results
    """
    required_fields = {'instruction', 'input', 'output'}
    errors = []
    
    for i, item in enumerate(data):
        # Check if all required fields are present
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            errors.append(f"Sample {i}: Missing required fields: {missing_fields}")
        
        # Check for empty required fields
        for field in required_fields:
            if field in item and (not item[field] or item[field].strip() == ""):
                errors.append(f"Sample {i}: Empty {field} field")
        
        # Check data types
        for field in required_fields:
            if field in item and not isinstance(item[field], str):
                errors.append(f"Sample {i}: Field '{field}' must be a string")
    
    return {
        'is_valid': len(errors) == 0,
        'error_count': len(errors),
        'errors': errors,
        'total_samples': len(data)
    }


class DataQualityAssessment:
    """
    Main class for comprehensive data quality assessment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data quality assessor.
        
        Args:
            config: Configuration dictionary for quality thresholds
        """
        self.config = config or {
            'completeness_threshold': 0.95,
            'outlier_threshold': 0.05,  # Max 5% outliers acceptable
            'balance_threshold': 0.3,   # Min balance ratio for balanced data
            'leakage_threshold': 0.0    # Zero tolerance for data leakage
        }
    
    def assess_dataset(self, df: pd.DataFrame, 
                      format_type: str = 'alpaca') -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment on a dataset.
        
        Args:
            df: Input DataFrame
            format_type: Expected format ('alpaca', 'sharegpt', etc.)
            
        Returns:
            Comprehensive quality report
        """
        logger.info(f"Starting quality assessment for dataset with {len(df)} samples")
        
        report = {}
        
        # 1. Completeness Assessment
        report['completeness'] = calculate_completeness(df)
        
        # 2. Outlier Detection (for numerical columns)
        outlier_info = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            col_data = df[col].dropna().tolist()
            if col_data:
                outliers = detect_outliers(col_data)
                outlier_info[col] = {
                    'outliers': outliers,
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(col_data) * 100
                }
        report['outliers'] = outlier_info
        
        # 3. Format Validation
        if format_type == 'alpaca':
            # Convert DataFrame to list of dicts for validation
            data_list = df.to_dict('records')
            report['format_validation'] = validate_alpaca_format(data_list)
        
        # 4. Label Distribution (if applicable)
        if 'output' in df.columns:
            labels = df['output'].tolist()
            report['label_distribution'] = assess_label_distribution(labels)
        
        # 5. Calculate Overall Quality Score
        report['overall_score'] = self._calculate_overall_score(report)
        
        # 6. Generate Recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        logger.info(f"Quality assessment completed. Overall score: {report['overall_score']:.3f}")
        
        return report
    
    def _calculate_overall_score(self, report: Dict[str, Any]) -> float:
        """
        Calculate an overall quality score based on various metrics.
        
        Args:
            report: Quality assessment report
            
        Returns:
            Overall quality score (0-1)
        """
        scores = []
        weights = []
        
        # Completeness score
        if 'completeness' in report:
            scores.append(report['completeness']['overall'])
            weights.append(0.3)
        
        # Format validation score
        if 'format_validation' in report:
            format_score = 1.0 if report['format_validation']['is_valid'] else 0.0
            scores.append(format_score)
            weights.append(0.3)
        
        # Label distribution score
        if 'label_distribution' in report:
            balance_score = min(1.0, report['label_distribution']['balance_ratio'] / 0.5)
            scores.append(balance_score)
            weights.append(0.2)
        
        # Outlier score (fewer outliers = higher score)
        if 'outliers' in report:
            outlier_scores = []
            for col_info in report['outliers'].values():
                outlier_pct = col_info['outlier_percentage']
                outlier_score = max(0.0, 1.0 - outlier_pct / 10)  # 10% outliers = 0 score
                outlier_scores.append(outlier_score)
            
            if outlier_scores:
                avg_outlier_score = sum(outlier_scores) / len(outlier_scores)
                scores.append(avg_outlier_score)
                weights.append(0.2)
        
        # Calculate weighted average
        if scores and weights:
            return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            return 0.0
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on quality assessment.
        
        Args:
            report: Quality assessment report
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Completeness recommendations
        if 'completeness' in report:
            completeness = report['completeness']
            if completeness['overall'] < self.config['completeness_threshold']:
                recommendations.append(
                    f"Dataset completeness ({completeness['overall']:.2f}) is below threshold "
                    f"({self.config['completeness_threshold']}). Consider data imputation or removal of incomplete samples."
                )
        
        # Format validation recommendations
        if 'format_validation' in report and not report['format_validation']['is_valid']:
            recommendations.append(
                f"Found {report['format_validation']['error_count']} format validation errors. "
                "Review and fix data format issues before training."
            )
        
        # Label distribution recommendations
        if 'label_distribution' in report:
            dist = report['label_distribution']
            if dist['is_imbalanced']:
                recommendations.append(
                    f"Dataset is imbalanced (balance ratio: {dist['balance_ratio']:.2f}). "
                    "Consider data augmentation, resampling, or class weighting techniques."
                )
        
        # Outlier recommendations
        if 'outliers' in report:
            high_outlier_cols = [
                col for col, info in report['outliers'].items()
                if info['outlier_percentage'] > self.config['outlier_threshold'] * 100
            ]
            if high_outlier_cols:
                recommendations.append(
                    f"High outlier percentage in columns: {high_outlier_cols}. "
                    "Review and consider outlier removal or capping."
                )
        
        if not recommendations:
            recommendations.append("Dataset quality looks good! No major issues detected.")
        
        return recommendations