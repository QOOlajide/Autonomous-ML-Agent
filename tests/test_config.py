"""
Tests for configuration management.

This module tests the configuration system including:
- Environment variable loading
- Configuration validation
- Default value handling
"""

import pytest
import os
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config, LLMConfig, ModelConfig, DataConfig, LoggingConfig

class TestLLMConfig:
    """Test cases for LLMConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        
        assert config.default_provider == "openai"
        assert config.max_tokens == 4000
        assert config.temperature == 0.1
        assert config.openai_api_key is None
        assert config.anthropic_api_key is None
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMConfig(
            openai_api_key="test_key",
            default_provider="anthropic",
            max_tokens=2000,
            temperature=0.5
        )
        
        assert config.openai_api_key == "test_key"
        assert config.default_provider == "anthropic"
        assert config.max_tokens == 2000
        assert config.temperature == 0.5

class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        assert config.max_models_to_train == 10
        assert config.max_compute_budget == 3600
        assert config.default_target_metric == "accuracy"
        assert config.hyperparameter_search_trials == 50
        assert "logistic_regression" in config.supported_models
        assert "random_forest" in config.supported_models
    
    def test_custom_values(self):
        """Test custom model configuration."""
        config = ModelConfig(
            max_models_to_train=5,
            max_compute_budget=1800,
            default_target_metric="precision"
        )
        
        assert config.max_models_to_train == 5
        assert config.max_compute_budget == 1800
        assert config.default_target_metric == "precision"

class TestDataConfig:
    """Test cases for DataConfig class."""
    
    def test_default_values(self):
        """Test default data configuration."""
        config = DataConfig()
        
        assert config.max_missing_threshold == 0.5
        assert config.categorical_encoding_method == "target_encoding"
        assert config.datetime_expansion is True
        assert config.feature_selection_method == "mutual_info"
    
    def test_custom_values(self):
        """Test custom data configuration."""
        config = DataConfig(
            max_missing_threshold=0.3,
            categorical_encoding_method="one_hot",
            datetime_expansion=False
        )
        
        assert config.max_missing_threshold == 0.3
        assert config.categorical_encoding_method == "one_hot"
        assert config.datetime_expansion is False

class TestLoggingConfig:
    """Test cases for LoggingConfig class."""
    
    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        
        assert config.log_level == "INFO"
        assert config.log_file == "logs/ml_agent.log"
        assert config.console_output is True
    
    def test_custom_values(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            log_level="DEBUG",
            log_file="custom.log",
            console_output=False
        )
        
        assert config.log_level == "DEBUG"
        assert config.log_file == "custom.log"
        assert config.console_output is False

class TestConfig:
    """Test cases for main Config class."""
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        # Set test environment variables
        test_env = {
            "OPENAI_API_KEY": "test_openai_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key",
            "DEFAULT_LLM_PROVIDER": "anthropic",
            "LLM_MAX_TOKENS": "2000",
            "LLM_TEMPERATURE": "0.3",
            "MAX_MODELS_TO_TRAIN": "5",
            "MAX_COMPUTE_BUDGET": "1800",
            "DEFAULT_TARGET_METRIC": "precision",
            "HP_SEARCH_TRIALS": "30",
            "MAX_MISSING_THRESHOLD": "0.3",
            "CATEGORICAL_ENCODING": "one_hot",
            "DATETIME_EXPANSION": "false",
            "FEATURE_SELECTION": "chi2",
            "LOG_LEVEL": "DEBUG",
            "LOG_FILE": "test.log",
            "CONSOLE_OUTPUT": "false"
        }
        
        # Backup original environment
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # Create config instance
            config = Config()
            
            # Test LLM config
            assert config.llm.openai_api_key == "test_openai_key"
            assert config.llm.anthropic_api_key == "test_anthropic_key"
            assert config.llm.default_provider == "anthropic"
            assert config.llm.max_tokens == 2000
            assert config.llm.temperature == 0.3
            
            # Test model config
            assert config.model.max_models_to_train == 5
            assert config.model.max_compute_budget == 1800
            assert config.model.default_target_metric == "precision"
            assert config.model.hyperparameter_search_trials == 30
            
            # Test data config
            assert config.data.max_missing_threshold == 0.3
            assert config.data.categorical_encoding_method == "one_hot"
            assert config.data.datetime_expansion is False
            assert config.data.feature_selection_method == "chi2"
            
            # Test logging config
            assert config.logging.log_level == "DEBUG"
            assert config.logging.log_file == "test.log"
            assert config.logging.console_output is False
            
        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    def test_validation_without_api_keys(self):
        """Test validation when no API keys are provided."""
        # Remove API keys from environment
        original_openai = os.environ.pop("OPENAI_API_KEY", None)
        original_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        
        try:
            config = Config()
            assert not config.validate()
        finally:
            # Restore API keys
            if original_openai:
                os.environ["OPENAI_API_KEY"] = original_openai
            if original_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = original_anthropic
    
    def test_validation_with_api_keys(self):
        """Test validation when API keys are provided."""
        # Set test API keys
        os.environ["OPENAI_API_KEY"] = "test_key"
        
        try:
            config = Config()
            assert config.validate()
        finally:
            # Clean up
            os.environ.pop("OPENAI_API_KEY", None)
    
    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        # Check that all sections are present
        assert "llm" in config_dict
        assert "model" in config_dict
        assert "data" in config_dict
        assert "logging" in config_dict
        
        # Check that sensitive data is not included
        assert "openai_api_key" not in config_dict["llm"]
        assert "anthropic_api_key" not in config_dict["llm"]
        
        # Check that other values are included
        assert config_dict["llm"]["default_provider"] == config.llm.default_provider
        assert config_dict["model"]["max_models_to_train"] == config.model.max_models_to_train
    
    def test_log_directory_creation(self):
        """Test that log directory is created during validation."""
        # Set a custom log file path
        custom_log_file = "custom_logs/test.log"
        os.environ["LOG_FILE"] = custom_log_file
        
        try:
            config = Config()
            config.validate()
            
            # Check that directory was created
            log_dir = Path(custom_log_file).parent
            assert log_dir.exists()
            
        finally:
            # Clean up
            os.environ.pop("LOG_FILE", None)
            # Remove created directory
            Path("custom_logs").rmdir()

# Fixtures
@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("OPENAI_API_KEY=test_key\n")
        f.write("MAX_MODELS_TO_TRAIN=5\n")
        f.write("LOG_LEVEL=DEBUG\n")
        return f.name

@pytest.fixture
def clean_env():
    """Provide a clean environment for testing."""
    # Store original environment
    original_env = dict(os.environ)
    
    # Clear test-related environment variables
    test_vars = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEFAULT_LLM_PROVIDER",
        "LLM_MAX_TOKENS", "LLM_TEMPERATURE", "MAX_MODELS_TO_TRAIN",
        "MAX_COMPUTE_BUDGET", "DEFAULT_TARGET_METRIC", "HP_SEARCH_TRIALS",
        "MAX_MISSING_THRESHOLD", "CATEGORICAL_ENCODING", "DATETIME_EXPANSION",
        "FEATURE_SELECTION", "LOG_LEVEL", "LOG_FILE", "CONSOLE_OUTPUT"
    ]
    
    for var in test_vars:
        os.environ.pop(var, None)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
