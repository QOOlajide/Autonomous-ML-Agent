"""
Main entry point for the Autonomous ML Agent.

This script orchestrates the entire ML pipeline:
1. Data ingestion and preprocessing
2. Model training with hyperparameter optimization
3. Model evaluation and selection
4. Pipeline export and deployment

Usage:
    python main.py --dataset path/to/data.csv --target_column target
    python main.py --dataset data.csv --target_column target --metric precision --max_models 5
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from config import config
from data.pipeline import DataPipeline
from models.trainer import ModelTrainer
from orchestration.llm_orchestrator import LLMOrchestrator
from ui.leaderboard import LeaderboardUI

def setup_logging():
    """
    Set up logging configuration.
    
    This function configures logging to both file and console output
    with appropriate formatting and log levels.
    """
    # Create log directory if it doesn't exist
    log_dir = Path(config.logging.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up logging configuration
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level.upper()),
        format=log_format,
        handlers=[
            # File handler for persistent logging
            logging.FileHandler(config.logging.log_file),
            # Console handler for real-time output (if enabled)
            logging.StreamHandler(sys.stdout) if config.logging.console_output else logging.NullHandler()
        ]
    )
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized successfully")
    return logger

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Autonomous ML Agent - Automatically train and optimize ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset data.csv --target_column target
  python main.py --dataset data.csv --target_column target --metric precision --max_models 5
  python main.py --dataset data.csv --target_column target --compute_budget 1800
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Path to the dataset file (CSV format)"
    )
    
    parser.add_argument(
        "--target_column", 
        type=str, 
        required=True,
        help="Name of the target column to predict"
    )
    
    # Optional arguments
    parser.add_argument(
        "--metric", 
        type=str, 
        default=config.model.default_target_metric,
        choices=["accuracy", "precision", "recall", "f1", "auc"],
        help="Target metric to optimize (default: accuracy)"
    )
    
    parser.add_argument(
        "--max_models", 
        type=int, 
        default=config.model.max_models_to_train,
        help=f"Maximum number of models to train (default: {config.model.max_models_to_train})"
    )
    
    parser.add_argument(
        "--compute_budget", 
        type=int, 
        default=config.model.max_compute_budget,
        help=f"Maximum compute time in seconds (default: {config.model.max_compute_budget})"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="artifacts",
        help="Directory to save trained models and artifacts (default: artifacts)"
    )
    
    parser.add_argument(
        "--ui", 
        type=str, 
        choices=["cli", "web"],
        default="cli",
        help="User interface type (default: cli)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def main():
    """
    Main function that orchestrates the entire ML pipeline.
    
    This function:
    1. Validates configuration and arguments
    2. Sets up logging
    3. Initializes the data pipeline
    4. Trains models with LLM orchestration
    5. Displays results and exports artifacts
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging()
    
    # Update config with command line arguments
    config.model.max_models_to_train = args.max_models
    config.model.max_compute_budget = args.compute_budget
    config.model.default_target_metric = args.metric
    
    if args.verbose:
        config.logging.log_level = "DEBUG"
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Autonomous ML Agent")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Target column: {args.target_column}")
    logger.info(f"Target metric: {args.metric}")
    logger.info(f"Max models: {args.max_models}")
    logger.info(f"Compute budget: {args.compute_budget}s")
    
    try:
        # Validate configuration
        if not config.validate():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # Initialize components
        logger.info("Initializing ML pipeline components...")
        
        # 1. Data Pipeline - handles data loading, cleaning, and preprocessing
        data_pipeline = DataPipeline()
        
        # 2. LLM Orchestrator - generates and modifies code dynamically
        llm_orchestrator = LLMOrchestrator()
        
        # 3. Model Trainer - trains models with hyperparameter optimization
        model_trainer = ModelTrainer()
        
        # 4. UI - displays results and leaderboard
        ui = LeaderboardUI(ui_type=args.ui)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        processed_data = data_pipeline.process_dataset(
            dataset_path=args.dataset,
            target_column=args.target_column
        )
        
        # Train models with LLM orchestration
        logger.info("Starting model training with LLM orchestration...")
        trained_models = model_trainer.train_models(
            data=processed_data,
            orchestrator=llm_orchestrator,
            target_metric=args.metric
        )
        
        # Display results
        logger.info("Displaying results...")
        ui.display_leaderboard(trained_models)
        
        # Export best model and pipeline
        logger.info("Exporting best model and pipeline...")
        best_model = max(trained_models, key=lambda m: m.metrics[args.metric])
        model_trainer.export_pipeline(
            model=best_model,
            data_pipeline=data_pipeline,
            output_dir=args.output_dir
        )
        
        logger.info("ML pipeline completed successfully!")
        logger.info(f"Best model: {best_model.name} with {args.metric}: {best_model.metrics[args.metric]:.4f}")
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
