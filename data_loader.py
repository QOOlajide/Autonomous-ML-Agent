"""
Simple dataset loader for ML pipeline demonstration.

This module provides basic functionality to load and validate datasets
for machine learning pipelines.
"""

import pandas as pd
from typing import Optional, Dict, Any
import os


def load_dataset(file_path: str, file_type: str = "csv") -> Optional[pd.DataFrame]:
    """
    Load a dataset from a file and return as pandas DataFrame.
    
    Args:
        file_path (str): Path to the dataset file
        file_type (str): Type of file ('csv', 'json', 'excel'). Defaults to 'csv'
    
    Returns:
        Optional[pd.DataFrame]: Loaded dataset or None if loading fails
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If file_type is not supported
    """
    # Check if file exists before attempting to load
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load data based on file type
    if file_type.lower() == "csv":
        data = pd.read_csv(file_path)
    elif file_type.lower() == "json":
        data = pd.read_json(file_path)
    elif file_type.lower() == "excel":
        data = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Basic validation - check if we got any data
    if data.empty:
        print("Warning: Loaded dataset is empty")
    
    return data


if __name__ == "__main__":
    # Micro-test: Try to load a sample dataset
    print("Testing data loader...")
    
    # Create a simple test CSV file
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })
    test_data.to_csv("test_dataset.csv", index=False)
    
    # Test the loader
    loaded_data = load_dataset("test_dataset.csv")
    print(f"Loaded dataset shape: {loaded_data.shape}")
    print(f"Columns: {list(loaded_data.columns)}")
    print("✅ Data loader test passed!")
