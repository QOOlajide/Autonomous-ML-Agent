"""
Sample dataset generator for testing the Autonomous ML Agent.

This script creates sample datasets with various characteristics to test
the data processing pipeline and model training capabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_classification_dataset(n_samples: int = 1000, save_path: str = "examples/sample_classification.csv"):
    """
    Create a sample classification dataset.
    
    Features:
    - age: numeric feature with some missing values
    - income: numeric feature with outliers
    - education: categorical feature
    - city: high cardinality categorical feature
    - signup_date: datetime feature
    - target: binary classification target
    """
    np.random.seed(42)
    
    # Generate base features
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'city': np.random.choice([f'City_{i}' for i in range(50)], n_samples),
        'signup_date': pd.date_range('2020-01-01', '2023-12-31', periods=n_samples),
        'is_premium': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'age'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    # Create target based on features (with some noise)
    target_prob = (
        0.3 * (df['age'] > 30).astype(int) +
        0.2 * (df['income'] > df['income'].median()).astype(int) +
        0.2 * (df['education'].isin(['Master', 'PhD'])).astype(int) +
        0.1 * (df['is_premium'] == 1).astype(int) +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['target'] = (target_prob > 0.5).astype(int)
    
    # Add some noise to make it more realistic
    noise_indices = np.random.choice(df.index, size=int(n_samples * 0.1), replace=False)
    df.loc[noise_indices, 'target'] = 1 - df.loc[noise_indices, 'target']
    
    # Save dataset
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Created classification dataset with {n_samples} samples")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"Missing values: {df.isnull().sum().to_dict()}")
    print(f"Saved to: {save_path}")
    
    return df

def create_regression_dataset(n_samples: int = 1000, save_path: str = "examples/sample_regression.csv"):
    """
    Create a sample regression dataset.
    
    Features:
    - house_size: numeric feature
    - bedrooms: integer feature
    - location: categorical feature
    - year_built: datetime feature
    - target: continuous target (house price)
    """
    np.random.seed(42)
    
    # Generate base features
    data = {
        'house_size': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.poisson(3, n_samples),
        'location': np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples, p=[0.4, 0.5, 0.1]),
        'year_built': np.random.randint(1950, 2023, n_samples),
        'has_garage': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.02), replace=False)
    df.loc[missing_indices, 'house_size'] = np.nan
    
    # Create target based on features (house price)
    base_price = (
        100 * df['house_size'] +
        20000 * df['bedrooms'] +
        50000 * (df['location'] == 'Downtown').astype(int) +
        30000 * (df['location'] == 'Suburbs').astype(int) +
        10000 * (df['has_garage'] == 1).astype(int) +
        1000 * (df['year_built'] - 1950)
    )
    
    # Add noise
    df['target'] = base_price + np.random.normal(0, 20000, n_samples)
    
    # Ensure positive prices
    df['target'] = np.maximum(df['target'], 50000)
    
    # Save dataset
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Created regression dataset with {n_samples} samples")
    print(f"Target statistics: mean={df['target'].mean():.0f}, std={df['target'].std():.0f}")
    print(f"Missing values: {df.isnull().sum().to_dict()}")
    print(f"Saved to: {save_path}")
    
    return df

if __name__ == "__main__":
    # Create sample datasets
    print("Creating sample datasets...")
    
    # Classification dataset
    create_classification_dataset(1000, "examples/sample_classification.csv")
    
    # Regression dataset  
    create_regression_dataset(1000, "examples/sample_regression.csv")
    
    print("\nSample datasets created successfully!")
    print("You can now test the ML agent with these datasets.")
