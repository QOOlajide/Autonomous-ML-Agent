"""
Data preprocessing module for the Autonomous ML Agent.

This module handles all data preprocessing tasks including:
- Missing value imputation
- Categorical encoding
- Feature scaling and normalization
- DateTime feature extraction
- Feature selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles comprehensive data preprocessing for machine learning.
    
    This class provides methods for cleaning, transforming, and preparing
    tabular data for machine learning model training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing settings
        """
        self.config = config or {}
        
        # Initialize transformers (will be fitted during preprocessing)
        self.scaler = StandardScaler()
        self.target_encoder = TargetEncoder()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_selector = None
        
        # Store preprocessing metadata
        self.preprocessing_info = {}
        self.feature_names = []
        
    def preprocess_data(self, 
                       df: pd.DataFrame, 
                       target_column: str,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
        """
        Perform comprehensive data preprocessing.
        
        Args:
            df: Raw dataset
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info(f"Starting preprocessing of dataset with shape {df.shape}")
        
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Step 1: Handle missing values
        df_processed = self._handle_missing_values(df_processed, target_column)
        
        # Step 2: Process datetime columns
        df_processed = self._process_datetime_columns(df_processed, target_column)
        
        # Step 3: Encode categorical variables
        df_processed = self._encode_categorical_variables(df_processed, target_column)
        
        # Step 4: Remove constant features
        df_processed = self._remove_constant_features(df_processed, target_column)
        
        # Step 5: Split data into features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Step 6: Feature selection
        X = self._select_features(X, y)
        
        # Step 7: Scale features
        X = self._scale_features(X)
        
        # Step 8: Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self._is_classification(y) else None
        )
        
        # Store preprocessing information
        self.preprocessing_info = {
            "original_shape": df.shape,
            "processed_shape": X.shape,
            "feature_names": list(X.columns),
            "target_info": {
                "name": target_column,
                "type": "classification" if self._is_classification(y) else "regression",
                "unique_values": y.nunique(),
                "class_distribution": y.value_counts().to_dict() if self._is_classification(y) else None
            },
            "preprocessing_steps": [
                "missing_value_imputation",
                "datetime_processing", 
                "categorical_encoding",
                "constant_feature_removal",
                "feature_selection",
                "feature_scaling"
            ]
        }
        
        logger.info(f"Preprocessing completed. Final shape: {X.shape}")
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": list(X.columns),
            "preprocessing_info": self.preprocessing_info
        }
    
    def _handle_missing_values(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Strategy:
        - Drop rows with missing target values
        - For features: use different imputation strategies based on data type
        """
        logger.info("Handling missing values...")
        
        # Remove rows with missing target values
        initial_rows = len(df)
        df = df.dropna(subset=[target_column])
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with missing target values")
        
        # Handle missing values in feature columns
        feature_columns = [col for col in df.columns if col != target_column]
        
        for col in feature_columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Numeric columns: use median imputation
                    imputer = SimpleImputer(strategy='median')
                    df[col] = imputer.fit_transform(df[[col]]).flatten()
                    self.imputers[col] = imputer
                else:
                    # Categorical columns: use mode imputation
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[col] = imputer.fit_transform(df[[col]]).flatten()
                    self.imputers[col] = imputer
        
        return df
    
    def _process_datetime_columns(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Process datetime columns by extracting useful features.
        
        Extracts: year, month, day, dayofweek, hour (if time present)
        """
        logger.info("Processing datetime columns...")
        
        datetime_columns = []
        for col in df.columns:
            if col != target_column and df[col].dtype == 'object':
                try:
                    # Try to convert to datetime
                    pd.to_datetime(df[col], errors='raise')
                    datetime_columns.append(col)
                except (ValueError, TypeError):
                    pass
        
        for col in datetime_columns:
            logger.info(f"Processing datetime column: {col}")
            
            # Convert to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Extract datetime features
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            
            # Extract time features if time component exists
            if df[col].dt.time.nunique() > 1:  # Has time component
                df[f"{col}_hour"] = df[col].dt.hour
                df[f"{col}_minute"] = df[col].dt.minute
            
            # Drop original datetime column
            df = df.drop(columns=[col])
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Encode categorical variables using appropriate encoding strategy.
        
        Strategy:
        - High cardinality (>10 unique values): Target encoding
        - Low cardinality (<=10 unique values): Label encoding
        """
        logger.info("Encoding categorical variables...")
        
        categorical_columns = []
        for col in df.columns:
            if col != target_column and df[col].dtype == 'object':
                categorical_columns.append(col)
        
        for col in categorical_columns:
            unique_count = df[col].nunique()
            
            if unique_count > 10:
                # High cardinality: use target encoding
                logger.info(f"Using target encoding for high cardinality column: {col}")
                df[col] = self.target_encoder.fit_transform(df[[col]], df[target_column]).flatten()
            else:
                # Low cardinality: use label encoding
                logger.info(f"Using label encoding for low cardinality column: {col}")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def _remove_constant_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Remove features with no variation (constant values)."""
        logger.info("Removing constant features...")
        
        feature_columns = [col for col in df.columns if col != target_column]
        constant_features = []
        
        for col in feature_columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            logger.info(f"Removing constant features: {constant_features}")
            df = df.drop(columns=constant_features)
        
        return df
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Select the most informative features.
        
        Uses mutual information for feature selection.
        """
        logger.info("Performing feature selection...")
        
        # Determine number of features to select (max 50 or 80% of features)
        max_features = min(50, int(X.shape[1] * 0.8))
        
        if X.shape[1] > max_features:
            # Use mutual information for feature selection
            if self._is_classification(y):
                selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
            else:
                selector = SelectKBest(score_func=mutual_info_regression, k=max_features)
            
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            self.feature_selector = selector
            logger.info(f"Selected {len(selected_features)} features out of {X.shape[1]}")
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features using standardization (mean=0, std=1).
        """
        logger.info("Scaling features...")
        
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def _is_classification(self, y: pd.Series) -> bool:
        """
        Determine if the problem is classification or regression.
        
        Heuristic: if target has <=20 unique values, treat as classification
        """
        return y.nunique() <= 20
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same preprocessing transformations to new data.
        
        This is used for inference on new data points.
        """
        logger.info("Transforming new data...")
        
        df_transformed = df.copy()
        
        # Apply missing value imputation
        for col, imputer in self.imputers.items():
            if col in df_transformed.columns:
                df_transformed[col] = imputer.transform(df_transformed[[col]]).flatten()
        
        # Apply categorical encoding
        for col, encoder in self.label_encoders.items():
            if col in df_transformed.columns:
                df_transformed[col] = encoder.transform(df_transformed[col].astype(str))
        
        # Apply feature selection
        if self.feature_selector is not None:
            df_transformed = df_transformed[self.feature_names]
            df_transformed = pd.DataFrame(
                self.feature_selector.transform(df_transformed),
                columns=self.feature_names,
                index=df_transformed.index
            )
        
        # Apply scaling
        df_transformed = pd.DataFrame(
            self.scaler.transform(df_transformed),
            columns=self.feature_names,
            index=df_transformed.index
        )
        
        return df_transformed
