"""
Main data pipeline for the Autonomous ML Agent.

This module orchestrates the entire data processing workflow:
1. Data loading and validation
2. Data preprocessing
3. Data splitting and preparation for ML
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

from .validator import DataValidator
from .preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Main data pipeline that orchestrates data processing workflow.
    
    This class coordinates data loading, validation, preprocessing, and preparation
    for machine learning model training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data pipeline.
        
        Args:
            config: Configuration dictionary for pipeline settings
        """
        self.config = config or {}
        
        # Initialize components
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor(config)
        
        # Store pipeline metadata
        self.pipeline_info = {}
        self.validation_results = {}
        
    def process_dataset(self, 
                       dataset_path: str, 
                       target_column: str,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
        """
        Process a complete dataset from loading to ML-ready format.
        
        Args:
            dataset_path: Path to the dataset file
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info(f"Starting data pipeline for dataset: {dataset_path}")
        
        # Step 1: Load data
        df = self._load_data(dataset_path)
        
        # Step 2: Validate data
        validation_results = self.validator.validate_dataset(df, target_column)
        self.validation_results = validation_results
        
        # Check if validation passed
        if not validation_results["is_valid"]:
            logger.error("Data validation failed. Issues found:")
            for issue in validation_results["issues"]:
                logger.error(f"  - {issue}")
            raise ValueError("Data validation failed. Please fix the issues and try again.")
        
        # Log recommendations
        if validation_results["recommendations"]:
            logger.info("Data validation recommendations:")
            for rec in validation_results["recommendations"]:
                logger.info(f"  - {rec}")
        
        # Step 3: Preprocess data
        processed_data = self.preprocessor.preprocess_data(
            df=df,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state
        )
        
        # Step 4: Compile pipeline information
        self.pipeline_info = {
            "dataset_path": dataset_path,
            "target_column": target_column,
            "validation_results": validation_results,
            "preprocessing_info": processed_data["preprocessing_info"],
            "data_splits": {
                "train_size": len(processed_data["X_train"]),
                "test_size": len(processed_data["X_test"]),
                "feature_count": len(processed_data["feature_names"])
            }
        }
        
        logger.info("Data pipeline completed successfully")
        logger.info(f"Final dataset: {processed_data['X_train'].shape[0]} train samples, "
                   f"{processed_data['X_test'].shape[0]} test samples, "
                   f"{len(processed_data['feature_names'])} features")
        
        return processed_data
    
    def _load_data(self, dataset_path: str) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Supported formats: CSV, Excel, JSON, Parquet
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from: {dataset_path}")
        
        file_path = Path(dataset_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Determine file format and load accordingly
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(dataset_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(dataset_path)
            elif file_extension == '.json':
                df = pd.read_json(dataset_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(dataset_path)
            else:
                # Try CSV as default
                logger.warning(f"Unknown file extension {file_extension}, trying CSV format")
                df = pd.read_csv(dataset_path)
            
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the processed data.
        
        Returns:
            Dictionary containing data summary information
        """
        if not self.pipeline_info:
            return {"error": "No data has been processed yet"}
        
        return {
            "dataset_info": self.pipeline_info["validation_results"]["dataset_info"],
            "feature_info": self.pipeline_info["validation_results"]["feature_info"],
            "preprocessing_steps": self.pipeline_info["preprocessing_info"]["preprocessing_steps"],
            "data_splits": self.pipeline_info["data_splits"],
            "target_info": self.pipeline_info["preprocessing_info"]["target_info"]
        }
    
    def get_feature_importance_info(self) -> Dict[str, Any]:
        """
        Get information about feature importance and selection.
        
        Returns:
            Dictionary containing feature importance information
        """
        if not hasattr(self.preprocessor, 'feature_selector') or self.preprocessor.feature_selector is None:
            return {"message": "No feature selection was performed"}
        
        selector = self.preprocessor.feature_selector
        feature_names = self.pipeline_info["preprocessing_info"]["feature_names"]
        
        # Get feature scores
        feature_scores = selector.scores_
        selected_features = feature_names[selector.get_support()]
        selected_scores = feature_scores[selector.get_support()]
        
        # Create feature importance ranking
        feature_importance = list(zip(selected_features, selected_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "selected_features": selected_features.tolist(),
            "feature_importance": feature_importance,
            "total_features_considered": len(feature_names),
            "features_selected": len(selected_features)
        }
    
    def export_preprocessing_pipeline(self, output_path: str):
        """
        Export the preprocessing pipeline for reuse.
        
        Args:
            output_path: Path where to save the pipeline
        """
        import pickle
        
        pipeline_data = {
            "preprocessor": self.preprocessor,
            "pipeline_info": self.pipeline_info,
            "validation_results": self.validation_results
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Preprocessing pipeline exported to: {output_path}")
    
    def load_preprocessing_pipeline(self, pipeline_path: str):
        """
        Load a previously exported preprocessing pipeline.
        
        Args:
            pipeline_path: Path to the saved pipeline
        """
        import pickle
        
        with open(pipeline_path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.preprocessor = pipeline_data["preprocessor"]
        self.pipeline_info = pipeline_data["pipeline_info"]
        self.validation_results = pipeline_data["validation_results"]
        
        logger.info(f"Preprocessing pipeline loaded from: {pipeline_path}")
