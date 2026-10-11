"""
Configuration management for the Autonomous ML Agent.

This module handles all configuration settings including:
- LLM API keys and settings (OpenRouter by default)
- Model training parameters
- Compute budget constraints
- Logging configuration
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class LLMConfig:
    """Configuration for LLM providers (OpenRouter primary; OpenAI-compatible)."""
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "google/gemini-3.1-flash-lite"
    site_url: Optional[str] = None  # Optional OpenRouter ranking header
    site_name: Optional[str] = None  # Optional OpenRouter ranking header
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_provider: str = "openrouter"
    max_tokens: int = 4000
    temperature: float = 0.1  # Lower temperature for more consistent code generation

@dataclass
class ModelConfig:
    """Configuration for model training and optimization"""
    max_models_to_train: int = 10
    max_compute_budget: int = 3600  # seconds
    default_target_metric: str = "accuracy"
    supported_models: list = None
    hyperparameter_search_trials: int = 50
    
    def __post_init__(self):
        if self.supported_models is None:
            # Default set of models we'll support
            self.supported_models = [
                "logistic_regression",
                "random_forest", 
                "gradient_boosting",
                "knn",
                "mlp"
            ]

@dataclass
class DataConfig:
    """Configuration for data processing"""
    max_missing_threshold: float = 0.5  # Drop columns with >50% missing values
    categorical_encoding_method: str = "target_encoding"  # or "one_hot", "label"
    datetime_expansion: bool = True
    feature_selection_method: str = "mutual_info"  # or "chi2", "f_score"

@dataclass
class LoggingConfig:
    """Configuration for logging"""
    log_level: str = "INFO"
    log_file: str = "logs/ml_agent.log"
    console_output: bool = True

class Config:
    """
    Main configuration class that combines all configuration sections.
    
    This class loads configuration from environment variables and provides
    sensible defaults for all settings.
    """
    
    def __init__(self):
        # Load LLM configuration (OpenRouter-first)
        self.llm = LLMConfig(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            model_name=os.getenv("OPENROUTER_MODEL", "google/gemini-3.1-flash-lite"),
            site_url=os.getenv("OPENROUTER_SITE_URL"),
            site_name=os.getenv("OPENROUTER_SITE_NAME", "Autonomous ML Agent"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            default_provider=os.getenv("DEFAULT_LLM_PROVIDER", "openrouter"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1"))
        )
        
        # Load model configuration
        self.model = ModelConfig(
            max_models_to_train=int(os.getenv("MAX_MODELS_TO_TRAIN", "10")),
            max_compute_budget=int(os.getenv("MAX_COMPUTE_BUDGET", "3600")),
            default_target_metric=os.getenv("DEFAULT_TARGET_METRIC", "accuracy"),
            hyperparameter_search_trials=int(os.getenv("HP_SEARCH_TRIALS", "50"))
        )
        
        # Load data configuration
        self.data = DataConfig(
            max_missing_threshold=float(os.getenv("MAX_MISSING_THRESHOLD", "0.5")),
            categorical_encoding_method=os.getenv("CATEGORICAL_ENCODING", "target_encoding"),
            datetime_expansion=os.getenv("DATETIME_EXPANSION", "true").lower() == "true",
            feature_selection_method=os.getenv("FEATURE_SELECTION", "mutual_info")
        )
        
        # Load logging configuration
        self.logging = LoggingConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "logs/ml_agent.log"),
            console_output=os.getenv("CONSOLE_OUTPUT", "true").lower() == "true"
        )
    
    def validate(self) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        provider = (self.llm.default_provider or "openrouter").lower()

        if provider == "openrouter":
            if not self.llm.openrouter_api_key:
                print("ERROR: No OpenRouter API key found. Set OPENROUTER_API_KEY in .env")
                return False
        elif provider == "openai":
            if not self.llm.openai_api_key:
                print("ERROR: No OpenAI API key found. Set OPENAI_API_KEY in .env")
                return False
        elif provider == "anthropic":
            if not self.llm.anthropic_api_key:
                print("ERROR: No Anthropic API key found. Set ANTHROPIC_API_KEY in .env")
                return False
        else:
            # Accept any configured key if provider string is custom
            if not (
                self.llm.openrouter_api_key
                or self.llm.openai_api_key
                or self.llm.anthropic_api_key
            ):
                print(
                    "ERROR: No LLM API keys found. "
                    "Set OPENROUTER_API_KEY (recommended) in .env"
                )
                return False
        
        # Check that log directory exists
        log_dir = os.path.dirname(self.logging.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            "llm": {
                "default_provider": self.llm.default_provider,
                "model_name": self.llm.model_name,
                "openrouter_base_url": self.llm.openrouter_base_url,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature
            },
            "model": {
                "max_models_to_train": self.model.max_models_to_train,
                "max_compute_budget": self.model.max_compute_budget,
                "default_target_metric": self.model.default_target_metric,
                "supported_models": self.model.supported_models,
                "hyperparameter_search_trials": self.model.hyperparameter_search_trials
            },
            "data": {
                "max_missing_threshold": self.data.max_missing_threshold,
                "categorical_encoding_method": self.data.categorical_encoding_method,
                "datetime_expansion": self.data.datetime_expansion,
                "feature_selection_method": self.data.feature_selection_method
            },
            "logging": {
                "log_level": self.logging.log_level,
                "log_file": self.logging.log_file,
                "console_output": self.logging.console_output
            }
        }

# Global configuration instance
config = Config()
