"""
Tests for the data pipeline components.

This module tests the core data processing functionality including:
- Data validation
- Data preprocessing
- Pipeline orchestration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.pipeline import DataPipeline
from data.validator import DataValidator
from data.preprocessor import DataPreprocessor

class TestDataValidator:
    """Test cases for DataValidator class."""
    
    def test_validate_empty_dataset(self):
        """Test validation of empty dataset."""
        validator = DataValidator()
        df = pd.DataFrame()
        result = validator.validate_dataset(df, "target")
        
        assert not result["is_valid"]
        assert "Dataset is empty" in result["issues"]
    
    def test_validate_missing_target_column(self):
        """Test validation when target column is missing."""
        validator = DataValidator()
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        result = validator.validate_dataset(df, "target")
        
        assert not result["is_valid"]
        assert "Target column 'target' not found" in result["issues"]
    
    def test_validate_good_dataset(self):
        """Test validation of a good dataset."""
        validator = DataValidator()
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [6, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0]
        })
        result = validator.validate_dataset(df, "target")
        
        assert result["is_valid"]
        assert len(result["issues"]) == 0
    
    def test_validate_constant_features(self):
        """Test detection of constant features."""
        validator = DataValidator()
        df = pd.DataFrame({
            "constant_feature": [1, 1, 1, 1, 1],
            "variable_feature": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0]
        })
        result = validator.validate_dataset(df, "target")
        
        assert "Constant features found" in str(result["issues"])

class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_handle_missing_values(self):
        """Test missing value imputation."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            "numeric_feature": [1, 2, np.nan, 4, 5],
            "categorical_feature": ["A", "B", np.nan, "A", "B"],
            "target": [0, 1, 0, 1, 0]
        })
        
        result = preprocessor._handle_missing_values(df, "target")
        
        # Check that missing values are filled
        assert not result["numeric_feature"].isnull().any()
        assert not result["categorical_feature"].isnull().any()
    
    def test_remove_constant_features(self):
        """Test removal of constant features."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            "constant_feature": [1, 1, 1, 1, 1],
            "variable_feature": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0]
        })
        
        result = preprocessor._remove_constant_features(df, "target")
        
        # Check that constant feature is removed
        assert "constant_feature" not in result.columns
        assert "variable_feature" in result.columns
    
    def test_is_classification(self):
        """Test classification vs regression detection."""
        preprocessor = DataPreprocessor()
        
        # Classification case (few unique values)
        y_class = pd.Series([0, 1, 0, 1, 0, 1])
        assert preprocessor._is_classification(y_class)
        
        # Regression case (many unique values)
        y_reg = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 12.2, 13.3, 14.4, 15.5, 16.6, 17.7, 18.8, 19.9, 20.0, 21.1])
        assert not preprocessor._is_classification(y_reg)

class TestDataPipeline:
    """Test cases for DataPipeline class."""
    
    def test_process_dataset_success(self):
        """Test successful dataset processing."""
        pipeline = DataPipeline()
        
        # Create a good test dataset
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "categorical": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Save to temporary file
        temp_file = "temp_test_data.csv"
        df.to_csv(temp_file, index=False)
        
        try:
            result = pipeline.process_dataset(temp_file, "target")
            
            # Check that processing was successful
            assert "X_train" in result
            assert "X_test" in result
            assert "y_train" in result
            assert "y_test" in result
            assert "feature_names" in result
            
            # Check data shapes
            assert len(result["X_train"]) > 0
            assert len(result["X_test"]) > 0
            assert len(result["feature_names"]) > 0
            
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
    
    def test_process_dataset_validation_failure(self):
        """Test dataset processing with validation failure."""
        pipeline = DataPipeline()
        
        # Create a bad dataset (empty)
        df = pd.DataFrame()
        temp_file = "temp_bad_data.csv"
        df.to_csv(temp_file, index=False)
        
        try:
            with pytest.raises(ValueError, match="Data validation failed"):
                pipeline.process_dataset(temp_file, "target")
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        pipeline = DataPipeline()
        
        # Process a dataset first
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [6, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0]
        })
        
        temp_file = "temp_summary_data.csv"
        df.to_csv(temp_file, index=False)
        
        try:
            pipeline.process_dataset(temp_file, "target")
            summary = pipeline.get_data_summary()
            
            assert "dataset_info" in summary
            assert "feature_info" in summary
            assert "preprocessing_steps" in summary
            assert "data_splits" in summary
            assert "target_info" in summary
            
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)

# Fixtures for common test data
@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [6, 7, 8, 9, 10],
        "categorical": ["A", "B", "A", "B", "A"],
        "target": [0, 1, 0, 1, 0]
    })

@pytest.fixture
def sample_dataframe_with_missing():
    """Create a sample DataFrame with missing values."""
    return pd.DataFrame({
        "feature1": [1, 2, np.nan, 4, 5],
        "feature2": [6, 7, 8, np.nan, 10],
        "categorical": ["A", "B", np.nan, "A", "B"],
        "target": [0, 1, 0, 1, 0]
    })

# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self, sample_dataframe):
        """Test the complete end-to-end pipeline."""
        # Save sample data
        temp_file = "temp_integration_test.csv"
        sample_dataframe.to_csv(temp_file, index=False)
        
        try:
            # Initialize pipeline
            pipeline = DataPipeline()
            
            # Process dataset
            result = pipeline.process_dataset(temp_file, "target")
            
            # Verify results
            assert result["X_train"].shape[0] > 0
            assert result["X_test"].shape[0] > 0
            assert len(result["feature_names"]) > 0
            
            # Check that target is properly separated
            assert "target" not in result["X_train"].columns
            assert "target" not in result["X_test"].columns
            
            # Check that target values are preserved
            assert len(result["y_train"]) > 0
            assert len(result["y_test"]) > 0
            
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
    
    def test_pipeline_with_missing_values(self, sample_dataframe_with_missing):
        """Test pipeline with missing values."""
        temp_file = "temp_missing_test.csv"
        sample_dataframe_with_missing.to_csv(temp_file, index=False)
        
        try:
            pipeline = DataPipeline()
            result = pipeline.process_dataset(temp_file, "target")
            
            # Should handle missing values gracefully
            assert not result["X_train"].isnull().any().any()
            assert not result["X_test"].isnull().any().any()
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
