"""
Data processing module for the Autonomous ML Agent.

This module handles all aspects of data processing including:
- Data loading and validation
- Missing value handling
- Categorical encoding
- Feature engineering
- Data splitting for training/validation
"""

from .pipeline import DataPipeline
from .preprocessor import DataPreprocessor
from .validator import DataValidator

__all__ = ["DataPipeline", "DataPreprocessor", "DataValidator"]
