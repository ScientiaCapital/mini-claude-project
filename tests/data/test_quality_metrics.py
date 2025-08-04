"""
Test suite for data quality metrics module.
Following TDD approach - tests written first to define expected behavior.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Import will fail initially - this is expected in TDD RED phase
try:
    from src.data.quality_metrics import (
        DataQualityAssessment,
        calculate_completeness,
        detect_outliers,
        assess_label_distribution,
        check_data_leakage,
        validate_alpaca_format
    )
except ImportError:
    # Expected during RED phase - tests define the interface
    pass


class TestDataQualityAssessment:
    """Test suite for comprehensive data quality assessment."""
    
    def setup_method(self):
        """Setup test data for each test method."""
        self.sample_alpaca_data = [
            {
                "instruction": "Translate the following English text to French:",
                "input": "Hello, how are you?",
                "output": "Bonjour, comment allez-vous?"
            },
            {
                "instruction": "Solve this math problem:",
                "input": "What is 15 + 27?",
                "output": "42"
            },
            {
                "instruction": "Explain the concept:",
                "input": "What is machine learning?",
                "output": "Machine learning is a subset of artificial intelligence..."
            }
        ]
        
        self.invalid_alpaca_data = [
            {
                "instruction": "Missing output field",
                "input": "Some input"
                # Missing output field
            },
            {
                "instruction": "",  # Empty instruction
                "input": "Valid input",
                "output": "Valid output"
            }
        ]

    def test_calculate_completeness_perfect_data(self):
        """Test completeness calculation with perfect data."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        completeness = calculate_completeness(df)
        
        assert completeness['overall'] == 1.0
        assert all(score == 1.0 for score in completeness['by_column'].values())

    def test_calculate_completeness_with_missing_values(self):
        """Test completeness calculation with missing values."""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['a', None, 'c', None, 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, None]
        })
        
        completeness = calculate_completeness(df)
        
        assert completeness['overall'] == pytest.approx(0.8, rel=1e-2)
        assert completeness['by_column']['col1'] == 0.8
        assert completeness['by_column']['col2'] == 0.6
        assert completeness['by_column']['col3'] == 0.8

    def test_detect_outliers_iqr_method(self):
        """Test outlier detection using IQR method."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is clear outlier
        
        outliers = detect_outliers(data, method='iqr')
        
        assert 100 in outliers
        assert len(outliers) == 1

    def test_detect_outliers_zscore_method(self):
        """Test outlier detection using Z-score method."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        
        outliers = detect_outliers(data, method='zscore', threshold=2)
        
        assert 100 in outliers

    def test_assess_label_distribution_balanced(self):
        """Test label distribution assessment with balanced data."""
        labels = ['A', 'B', 'A', 'B', 'A', 'B']
        
        distribution = assess_label_distribution(labels)
        
        assert distribution['balance_ratio'] == pytest.approx(1.0, rel=1e-2)
        assert distribution['entropy'] > 0.9  # High entropy for balanced data
        assert not distribution['is_imbalanced']

    def test_assess_label_distribution_imbalanced(self):
        """Test label distribution assessment with imbalanced data."""
        labels = ['A'] * 90 + ['B'] * 10
        
        distribution = assess_label_distribution(labels)
        
        assert distribution['balance_ratio'] < 0.5
        assert distribution['is_imbalanced']
        assert distribution['majority_class'] == 'A'

    def test_check_data_leakage_no_leakage(self):
        """Test data leakage detection with clean data."""
        train_data = ["sample 1", "sample 2", "sample 3"]
        test_data = ["sample 4", "sample 5", "sample 6"]
        
        leakage_report = check_data_leakage(train_data, test_data)
        
        assert leakage_report['has_leakage'] is False
        assert leakage_report['leak_count'] == 0

    def test_check_data_leakage_with_leakage(self):
        """Test data leakage detection with leaked data."""
        train_data = ["sample 1", "sample 2", "sample 3"]
        test_data = ["sample 2", "sample 4", "sample 5"]  # sample 2 is leaked
        
        leakage_report = check_data_leakage(train_data, test_data)
        
        assert leakage_report['has_leakage'] is True
        assert leakage_report['leak_count'] == 1
        assert "sample 2" in leakage_report['leaked_samples']

    def test_validate_alpaca_format_valid_data(self):
        """Test Alpaca format validation with valid data."""
        validation_result = validate_alpaca_format(self.sample_alpaca_data)
        
        assert validation_result['is_valid'] is True
        assert validation_result['error_count'] == 0
        assert len(validation_result['errors']) == 0

    def test_validate_alpaca_format_invalid_data(self):
        """Test Alpaca format validation with invalid data."""
        validation_result = validate_alpaca_format(self.invalid_alpaca_data)
        
        assert validation_result['is_valid'] is False
        assert validation_result['error_count'] > 0
        assert len(validation_result['errors']) > 0

    def test_validate_alpaca_format_empty_fields(self):
        """Test Alpaca format validation catches empty required fields."""
        data_with_empty_fields = [
            {
                "instruction": "",
                "input": "Valid input",
                "output": "Valid output"
            }
        ]
        
        validation_result = validate_alpaca_format(data_with_empty_fields)
        
        assert validation_result['is_valid'] is False
        assert any('empty instruction' in error.lower() for error in validation_result['errors'])


class TestDataQualityAssessmentIntegration:
    """Integration tests for the complete data quality assessment workflow."""
    
    def test_full_quality_assessment_pipeline(self):
        """Test the complete data quality assessment pipeline."""
        # This test will define the expected interface for the main assessment class
        sample_dataset = pd.DataFrame({
            'instruction': ['Task 1', 'Task 2', 'Task 3'],
            'input': ['Input 1', 'Input 2', 'Input 3'],
            'output': ['Output 1', 'Output 2', 'Output 3']
        })
        
        assessor = DataQualityAssessment()
        quality_report = assessor.assess_dataset(sample_dataset)
        
        # Expected report structure
        assert 'completeness' in quality_report
        assert 'outliers' in quality_report
        assert 'label_distribution' in quality_report
        assert 'format_validation' in quality_report
        assert 'overall_score' in quality_report
        
        # Quality score should be between 0 and 1
        assert 0 <= quality_report['overall_score'] <= 1


class TestPropertyBasedTests:
    """Property-based tests for data quality metrics."""
    
    def test_completeness_property_always_between_0_and_1(self):
        """Property: Completeness score should always be between 0 and 1."""
        # Generate random dataframes with varying missing data
        for _ in range(10):
            size = np.random.randint(10, 100)
            missing_ratio = np.random.random()
            
            data = np.random.randn(size)
            # Randomly introduce missing values
            mask = np.random.random(size) < missing_ratio
            data[mask] = np.nan
            
            df = pd.DataFrame({'col': data})
            completeness = calculate_completeness(df)
            
            assert 0 <= completeness['overall'] <= 1
            assert 0 <= completeness['by_column']['col'] <= 1

    def test_outlier_detection_empty_list_property(self):
        """Property: Outlier detection should return empty list for uniform data."""
        uniform_data = [5] * 20  # All same values
        
        outliers_iqr = detect_outliers(uniform_data, method='iqr')
        outliers_zscore = detect_outliers(uniform_data, method='zscore')
        
        assert len(outliers_iqr) == 0
        assert len(outliers_zscore) == 0