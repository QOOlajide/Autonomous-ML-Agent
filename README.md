# Autonomous ML Agent

An intelligent machine learning agent that automatically processes tabular datasets, trains models, and optimizes them using LLM orchestration.

## Features

- **Automated Data Pipeline**: Ingests, cleans, and preprocesses tabular data
- **LLM Orchestration**: Uses language models to generate and modify code dynamically
- **Model Training**: Supports multiple algorithms with hyperparameter optimization
- **Meta-Learning**: Learns from previous runs to improve future performance
- **Ensembling**: Combines top models for better performance
- **Deployment Ready**: Exports pipelines as reusable artifacts

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the agent:
```bash
python main.py --dataset path/to/your/data.csv --target_column target
```

## Project Structure

```
├── src/
│   ├── data/           # Data processing modules
│   ├── models/         # Model implementations
│   ├── orchestration/  # LLM orchestration
│   ├── ui/            # User interface components
│   └── utils/         # Utility functions
├── tests/             # Test files
├── examples/          # Example datasets and configs
└── artifacts/         # Generated models and pipelines
```

## Development

This project follows agile methodology with iterative development sprints.
