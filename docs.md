For this project, you are tasked with building an autonomous machine learning agent that ingests tabular datasets, automatically cleans and preprocesses the data, trains models, and optimizes them for a target metric such as accuracy, precision, or recall. The entire pipeline should be orchestrated by LLMs, where they generate and modify code, select appropriate algorithms, and iteratively refine the pipeline until the best-performing model is achieved.

Project Requirements:

Implement an end-to-end pipeline orchestrated by an LLM that handles data ingestion, preprocessing (missing values, categorical encoding, datetime expansion), model training, evaluation, and final model selection
Support a curated set of models (Logistic/Linear Regression, Random Forest, Gradient Boosting, kNN, MLP), with hyperparameter optimization strategies (random or Bayesian search) that are dynamically generated and adapted by the LLM under a fixed compute/time budget
Provide a leaderboard UI (or CLI) showing candidate models, validation metrics, and selected best model. Include feature importance and basic interpretability, with the LLM generating natural language summaries of results for transparency
Export the final pipeline as a reusable artifact (e.g., Python package or FastAPI service) that preserves preprocessing steps and allows batch or single-instance inference, with auto-generated deployment scripts written by the LLM
Challenges:

Add meta-learning warm starts: use prior runs’ metadata (dataset size, feature types, class balance) to initialize hyperparameter search, with the LLM recommending promising starting configurations
Implement simple ensembling of top models (stacking or blending) and allow the LLM to propose ensemble strategies that balance performance, speed, and reproducibility
For your demo:

Make sure to clearly explain the problem that you're trying to solve and your solution.
Learn about the structure of a demo here
Check out this example demo
When uploading to YouTube, make sure captions are enabled
Where you can ask for help:

