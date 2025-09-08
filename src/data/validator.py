"""
Data validation module for the Autonomous ML Agent.

This module validates datasets to ensure they're suitable for machine learning:
- Checks data types and formats
- Validates target column exists and is appropriate
- Identifies potential issues (missing values, data leakage, etc.)
- Provides recommendations for data preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates datasets for machine learning suitability.
    
    This class performs comprehensive validation of tabular datasets to ensure
    they meet the requirements for automated ML processing.
    """
    
    def __init__(self):
        """Initialize the data validator with default settings."""
        self.validation_results = {}
        self.issues = []
        self.recommendations = []
    
    def validate_dataset(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Perform comprehensive validation of a dataset.
        
        Args:
            df: The dataset to validate
            target_column: Name of the target column
            
        Returns:
            Dict containing validation results, issues, and recommendations
        """
        logger.info(f"Starting validation of dataset with shape {df.shape}")
        
        # Reset validation state
        self.issues = []
        self.recommendations = []
        
        # Basic dataset validation
        self._validate_basic_structure(df, target_column)
        
        # Data type validation
        self._validate_data_types(df, target_column)
        
        # Missing value analysis
        self._validate_missing_values(df, target_column)
        
        # Target column validation
        self._validate_target_column(df, target_column)
        
        # Feature validation
        self._validate_features(df, target_column)
        
        # Data quality checks
        self._validate_data_quality(df, target_column)
        
        # Compile results
        validation_results = {
            "is_valid": len(self.issues) == 0,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "dataset_info": self._get_dataset_info(df, target_column),
            "feature_info": self._get_feature_info(df, target_column)
        }
        
        logger.info(f"Validation completed. Found {len(self.issues)} issues, {len(self.recommendations)} recommendations")
        return validation_results
    
    def _validate_basic_structure(self, df: pd.DataFrame, target_column: str):
        """Validate basic dataset structure."""
        # Check if dataset is empty
        if df.empty:
            self.issues.append("Dataset is empty")
            return
        
        # Check minimum size requirements
        if len(df) < 10:
            self.issues.append(f"Dataset too small: {len(df)} rows (minimum: 10)")
        
        if len(df.columns) < 2:
            self.issues.append(f"Dataset has too few columns: {len(df.columns)} (minimum: 2)")
        
        # Check if target column exists
        if target_column not in df.columns:
            self.issues.append(f"Target column '{target_column}' not found in dataset")
            return
        
        # Check for duplicate columns
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            self.issues.append(f"Duplicate columns found: {duplicate_columns}")
    
    def _validate_data_types(self, df: pd.DataFrame, target_column: str):
        """Validate data types and identify potential issues."""
        # Check for mixed data types in columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if object column might be numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    self.recommendations.append(f"Column '{col}' appears to be numeric but stored as text")
                except (ValueError, TypeError):
                    pass
        
        # Check for datetime columns that might need special handling
        datetime_columns = []
        for col in df.columns:
            if col != target_column and df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col], errors='raise')
                    datetime_columns.append(col)
                except (ValueError, TypeError):
                    pass
        
        if datetime_columns:
            self.recommendations.append(f"Potential datetime columns found: {datetime_columns}")
    
    def _validate_missing_values(self, df: pd.DataFrame, target_column: str):
        """Analyze missing values in the dataset."""
        missing_info = df.isnull().sum()
        total_rows = len(df)
        
        # Check target column missing values
        target_missing = missing_info[target_column]
        if target_missing > 0:
            self.issues.append(f"Target column has {target_missing} missing values ({target_missing/total_rows*100:.1f}%)")
        
        # Check feature columns with high missing values
        high_missing_features = []
        for col in df.columns:
            if col != target_column:
                missing_pct = missing_info[col] / total_rows
                if missing_pct > 0.5:  # More than 50% missing
                    high_missing_features.append((col, missing_pct))
        
        if high_missing_features:
            self.recommendations.append(f"Features with >50% missing values: {[f'{col} ({pct:.1%})' for col, pct in high_missing_features]}")
    
    def _validate_target_column(self, df: pd.DataFrame, target_column: str):
        """Validate the target column for ML suitability."""
        target_series = df[target_column]
        
        # Check for constant target (no variation)
        if target_series.nunique() <= 1:
            self.issues.append("Target column has no variation (all values are the same)")
        
        # Check for too many unique values (might be regression instead of classification)
        unique_count = target_series.nunique()
        total_count = len(target_series)
        
        if unique_count > total_count * 0.9:
            self.recommendations.append(f"Target column has {unique_count} unique values out of {total_count} total. Consider if this is a regression problem.")
        
        # Check for class imbalance in classification
        if unique_count <= 20:  # Likely classification
            value_counts = target_series.value_counts()
            min_class_size = value_counts.min()
            max_class_size = value_counts.max()
            
            if min_class_size < 5:
                self.issues.append(f"Some classes have very few samples (minimum: {min_class_size})")
            
            imbalance_ratio = max_class_size / min_class_size
            if imbalance_ratio > 10:
                self.recommendations.append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
    
    def _validate_features(self, df: pd.DataFrame, target_column: str):
        """Validate feature columns."""
        feature_columns = [col for col in df.columns if col != target_column]
        
        # Check for constant features
        constant_features = []
        for col in feature_columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            self.issues.append(f"Constant features found: {constant_features}")
        
        # Check for features with too many unique values (potential IDs)
        high_cardinality_features = []
        for col in feature_columns:
            if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.8:
                high_cardinality_features.append(col)
        
        if high_cardinality_features:
            self.recommendations.append(f"High cardinality features (potential IDs): {high_cardinality_features}")
    
    def _validate_data_quality(self, df: pd.DataFrame, target_column: str):
        """Perform data quality checks."""
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            self.recommendations.append(f"Found {duplicate_rows} duplicate rows")
        
        # Check for potential data leakage (features that are too predictive)
        # This is a simplified check - in practice, domain knowledge is needed
        feature_columns = [col for col in df.columns if col != target_column]
        
        for col in feature_columns:
            if df[col].dtype == 'object':
                # Check if feature has perfect correlation with target
                if df.groupby(col)[target_column].nunique().max() == 1:
                    self.recommendations.append(f"Feature '{col}' might have perfect correlation with target (potential data leakage)")
    
    def _get_dataset_info(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "target_column": target_column,
            "feature_count": len(df.columns) - 1,
            "row_count": len(df)
        }
    
    def _get_feature_info(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Get detailed feature information."""
        feature_columns = [col for col in df.columns if col != target_column]
        
        feature_info = {}
        for col in feature_columns:
            feature_info[col] = {
                "dtype": str(df[col].dtype),
                "missing_count": df[col].isnull().sum(),
                "missing_pct": df[col].isnull().sum() / len(df) * 100,
                "unique_count": df[col].nunique(),
                "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                "is_categorical": df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.1
            }
        
        return feature_info
