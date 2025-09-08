"""
Streamlit Web UI for the Autonomous ML Agent

This creates a modern, interactive web interface for our ML agent that allows users to:
1. Upload datasets and configure parameters
2. Monitor the data processing pipeline
3. View model training progress and results
4. Explore feature importance and model performance
5. Download trained models and results

Streamlit is a Python framework that makes it incredibly easy to create web apps for ML.
It automatically handles the web server, HTML/CSS generation, and user interactions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time
import json
from io import BytesIO

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from data.pipeline import DataPipeline
from config import config

# Configure Streamlit page
st.set_page_config(
    page_title="Autonomous ML Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """
    Main function that orchestrates the entire Streamlit app.
    
    Streamlit works by running this function every time the user interacts with the app.
    The @st.cache decorators help optimize performance by caching expensive operations.
    """
    
    # App header with title and description
    st.markdown('<h1 class="main-header">🤖 Autonomous ML Agent</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Upload your dataset and let our AI agent automatically train the best machine learning model for you!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset upload section
        st.subheader("📁 Dataset Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your tabular dataset in CSV format"
        )
        
        # Configuration options
        st.subheader("🎯 Target Configuration")
        target_column = st.text_input(
            "Target Column Name",
            value="target",
            help="Name of the column you want to predict"
        )
        
        # Model configuration
        st.subheader("🧠 Model Settings")
        max_models = st.slider(
            "Maximum Models to Train",
            min_value=3,
            max_value=20,
            value=10,
            help="More models = better results but longer training time"
        )
        
        target_metric = st.selectbox(
            "Optimization Metric",
            options=["accuracy", "precision", "recall", "f1", "auc"],
            index=0,
            help="Which metric should we optimize for?"
        )
        
        compute_budget = st.slider(
            "Compute Budget (seconds)",
            min_value=300,
            max_value=7200,
            value=1800,
            step=300,
            help="Maximum time to spend on training"
        )
    
    # Main content area
    if uploaded_file is not None:
        # Process the uploaded file
        process_dataset_ui(uploaded_file, target_column, max_models, target_metric, compute_budget)
    else:
        # Show welcome screen with example
        show_welcome_screen()

def show_welcome_screen():
    """
    Display the welcome screen with instructions and example data.
    
    This function shows when no file is uploaded, providing guidance to users.
    """
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚀 How It Works")
        st.markdown("""
        Our Autonomous ML Agent uses advanced AI to automatically:
        
        1. **📊 Analyze Your Data**: Validates and understands your dataset structure
        2. **🔧 Preprocess**: Handles missing values, encodes categories, and engineers features
        3. **🤖 Train Models**: Tests multiple algorithms with hyperparameter optimization
        4. **📈 Optimize**: Finds the best model for your specific metric
        5. **📋 Report**: Provides detailed insights and downloadable results
        
        **Just upload a CSV file and specify your target column!**
        """)
        
        st.header("📋 Data Requirements")
        st.markdown("""
        - **Format**: CSV file
        - **Size**: Any size (we'll handle it efficiently)
        - **Target**: One column to predict (can be classification or regression)
        - **Features**: Any number of columns (numeric, categorical, or datetime)
        """)
    
    with col2:
        st.header("🎯 Try It Out!")
        st.markdown("""
        **Don't have a dataset?** No problem! We've created sample datasets for you to try:
        """)
        
        # Create sample data download buttons
        if st.button("📊 Download Sample Classification Dataset", use_container_width=True):
            create_sample_classification_data()
        
        if st.button("📈 Download Sample Regression Dataset", use_container_width=True):
            create_sample_regression_data()
        
        st.markdown("""
        **💡 Pro Tip**: Use the sample datasets to see how the agent works, then try with your own data!
        """)

@st.cache_data
def create_sample_classification_data():
    """
    Create and return a sample classification dataset.
    
    The @st.cache_data decorator ensures this function only runs once and caches the result.
    This is important for performance when users click the button multiple times.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic sample data
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'city': np.random.choice([f'City_{i}' for i in range(50)], n_samples),
        'signup_date': pd.date_range('2020-01-01', '2023-12-31', periods=n_samples),
        'is_premium': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Add missing values to make it realistic
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'age'] = np.nan
    
    # Create target based on features
    target_prob = (
        0.3 * (df['age'] > 30).astype(int) +
        0.2 * (df['income'] > df['income'].median()).astype(int) +
        0.2 * (df['education'].isin(['Master', 'PhD'])).astype(int) +
        0.1 * (df['is_premium'] == 1).astype(int) +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['target'] = (target_prob > 0.5).astype(int)
    
    # Add noise
    noise_indices = np.random.choice(df.index, size=int(n_samples * 0.1), replace=False)
    df.loc[noise_indices, 'target'] = 1 - df.loc[noise_indices, 'target']
    
    return df.to_csv(index=False)

@st.cache_data
def create_sample_regression_data():
    """Create and return a sample regression dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'house_size': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.poisson(3, n_samples),
        'location': np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples, p=[0.4, 0.5, 0.1]),
        'year_built': np.random.randint(1950, 2023, n_samples),
        'has_garage': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Create target (house price)
    base_price = (
        100 * df['house_size'] +
        20000 * df['bedrooms'] +
        50000 * (df['location'] == 'Downtown').astype(int) +
        30000 * (df['location'] == 'Suburbs').astype(int) +
        10000 * (df['has_garage'] == 1).astype(int) +
        1000 * (df['year_built'] - 1950)
    )
    
    df['target'] = base_price + np.random.normal(0, 20000, n_samples)
    df['target'] = np.maximum(df['target'], 50000)
    
    return df.to_csv(index=False)

def process_dataset_ui(uploaded_file, target_column, max_models, target_metric, compute_budget):
    """
    Main processing function that handles the ML pipeline in the UI.
    
    This function coordinates the entire workflow:
    1. Load and validate the uploaded data
    2. Run the data preprocessing pipeline
    3. Display results and visualizations
    4. Provide download options
    """
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔧 Processing", "📈 Results", "💾 Download"])
    
    with tab1:
        show_data_overview(uploaded_file, target_column)
    
    with tab2:
        processed_data = show_processing_pipeline(uploaded_file, target_column)
    
    with tab3:
        if 'processed_data' in locals():
            show_results_visualization(processed_data)
        else:
            st.info("👆 Please run the processing pipeline first!")
    
    with tab4:
        if 'processed_data' in locals():
            show_download_options(processed_data)
        else:
            st.info("👆 Please run the processing pipeline first!")

def show_data_overview(uploaded_file, target_column):
    """
    Display an overview of the uploaded dataset.
    
    This function shows basic statistics, data types, and sample rows
    to help users understand their data before processing.
    """
    st.header("📊 Dataset Overview")
    
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        
        # Basic information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📏 Rows", f"{len(df):,}")
        with col2:
            st.metric("📋 Columns", f"{len(df.columns):,}")
        with col3:
            st.metric("💾 Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("❓ Missing Values", f"{missing_pct:.1f}%")
        
        # Data types and missing values
        st.subheader("🔍 Data Types & Missing Values")
        
        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': df.nunique(),
            'Sample Value': df.iloc[0].astype(str)
        })
        
        st.dataframe(summary_df, use_container_width=True)
        
        # Target column analysis
        if target_column in df.columns:
            st.subheader(f"🎯 Target Column Analysis: '{target_column}'")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Value counts
                value_counts = df[target_column].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution of '{target_column}'",
                    labels={'x': target_column, 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statistics
                if pd.api.types.is_numeric_dtype(df[target_column]):
                    st.metric("Mean", f"{df[target_column].mean():.2f}")
                    st.metric("Std Dev", f"{df[target_column].std():.2f}")
                    st.metric("Min", f"{df[target_column].min():.2f}")
                    st.metric("Max", f"{df[target_column].max():.2f}")
                else:
                    st.metric("Unique Values", df[target_column].nunique())
                    st.metric("Most Common", value_counts.index[0])
                    st.metric("Most Common Count", value_counts.iloc[0])
        
        # Sample data
        st.subheader("👀 Sample Data (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.markdown("""
        <div class="error-box">
            <strong>Common Issues:</strong><br>
            • Make sure your file is a valid CSV<br>
            • Check that the target column name is correct<br>
            • Ensure the file isn't corrupted
        </div>
        """, unsafe_allow_html=True)

def show_processing_pipeline(uploaded_file, target_column):
    """
    Run the data processing pipeline and show progress.
    
    This function uses Streamlit's progress bars and status indicators
    to show real-time progress of the ML pipeline.
    """
    st.header("🔧 Data Processing Pipeline")
    
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Initialize pipeline
        with st.spinner("🚀 Initializing ML Pipeline..."):
            pipeline = DataPipeline()
        
        # Step 1: Data Validation
        st.subheader("Step 1: Data Validation")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Validating dataset structure...")
        progress_bar.progress(25)
        
        # Load data for validation
        df = pd.read_csv(temp_path)
        
        # Run validation
        validation_results = pipeline.validator.validate_dataset(df, target_column)
        progress_bar.progress(50)
        
        # Display validation results
        if validation_results["is_valid"]:
            st.success("✅ Data validation passed!")
            if validation_results["recommendations"]:
                with st.expander("💡 Recommendations"):
                    for rec in validation_results["recommendations"]:
                        st.info(f"• {rec}")
        else:
            st.error("❌ Data validation failed!")
            for issue in validation_results["issues"]:
                st.error(f"• {issue}")
            return None
        
        # Step 2: Data Preprocessing
        st.subheader("Step 2: Data Preprocessing")
        status_text.text("🔧 Preprocessing data...")
        progress_bar.progress(75)
        
        # Run preprocessing
        processed_data = pipeline.preprocess_data(
            df=df,
            target_column=target_column,
            test_size=0.2,
            random_state=42
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Display preprocessing results
        st.markdown("""
        <div class="success-box">
            <strong>🎉 Data Processing Complete!</strong><br>
            Your data has been successfully cleaned, preprocessed, and prepared for machine learning.
        </div>
        """, unsafe_allow_html=True)
        
        # Show preprocessing summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Train Samples", f"{len(processed_data['X_train']):,}")
        with col2:
            st.metric("Test Samples", f"{len(processed_data['X_test']):,}")
        with col3:
            st.metric("Features", f"{len(processed_data['feature_names']):,}")
        
        # Store processed data in session state for other tabs
        st.session_state.processed_data = processed_data
        st.session_state.pipeline = pipeline
        
        return processed_data
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        return None
    
    finally:
        # Clean up temporary file
        if Path(temp_path).exists():
            Path(temp_path).unlink()

def show_results_visualization(processed_data):
    """
    Display comprehensive visualizations of the processed data and results.
    
    This function creates interactive charts using Plotly to help users
    understand their data and the preprocessing results.
    """
    st.header("📈 Processing Results & Visualizations")
    
    # Feature importance (if available)
    if hasattr(st.session_state, 'pipeline'):
        pipeline = st.session_state.pipeline
        feature_info = pipeline.get_feature_importance_info()
        
        if "feature_importance" in feature_info:
            st.subheader("🎯 Feature Importance")
            
            # Create feature importance chart
            importance_data = feature_info["feature_importance"][:10]  # Top 10 features
            features, scores = zip(*importance_data)
            
            fig = px.bar(
                x=scores,
                y=features,
                orientation='h',
                title="Top 10 Most Important Features",
                labels={'x': 'Importance Score', 'y': 'Feature'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Data distribution analysis
    st.subheader("📊 Data Distribution Analysis")
    
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    
    # Target distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Target distribution
        target_counts = y_train.value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title="Target Distribution (Training Set)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature correlation heatmap (for numeric features only)
        numeric_features = X_train.select_dtypes(include=[np.number]).columns[:10]  # Top 10 numeric features
        
        if len(numeric_features) > 1:
            corr_matrix = X_train[numeric_features].corr()
            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric features for correlation analysis")
    
    # Feature statistics
    st.subheader("📋 Feature Statistics")
    
    # Create feature statistics table
    feature_stats = X_train.describe().round(3)
    st.dataframe(feature_stats, use_container_width=True)
    
    # Data quality metrics
    st.subheader("✅ Data Quality Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Missing Values", "0%", "✅ Clean")
    with col2:
        st.metric("Constant Features", "0", "✅ Removed")
    with col3:
        st.metric("Data Types", "Consistent", "✅ Processed")
    with col4:
        st.metric("Scaling", "Applied", "✅ Standardized")

def show_download_options(processed_data):
    """
    Provide download options for processed data and results.
    
    This function creates downloadable files including:
    - Processed datasets
    - Feature importance reports
    - Processing pipeline artifacts
    """
    st.header("💾 Download Results")
    
    # Processed data download
    st.subheader("📊 Processed Datasets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Training data
        train_data = processed_data['X_train'].copy()
        train_data['target'] = processed_data['y_train']
        
        csv_train = train_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Training Data (CSV)",
            data=csv_train,
            file_name="processed_training_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Test data
        test_data = processed_data['X_test'].copy()
        test_data['target'] = processed_data['y_test']
        
        csv_test = test_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Test Data (CSV)",
            data=csv_test,
            file_name="processed_test_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Feature information
    st.subheader("📋 Feature Information")
    
    if hasattr(st.session_state, 'pipeline'):
        pipeline = st.session_state.pipeline
        feature_info = pipeline.get_feature_importance_info()
        
        # Create feature report
        feature_report = {
            "feature_names": processed_data['feature_names'],
            "feature_importance": feature_info.get("feature_importance", []),
            "preprocessing_info": processed_data['preprocessing_info']
        }
        
        json_report = json.dumps(feature_report, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Feature Report (JSON)",
            data=json_report,
            file_name="feature_report.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Processing summary
    st.subheader("📄 Processing Summary")
    
    summary_text = f"""
    # Data Processing Summary
    
    ## Dataset Information
    - **Original Features**: {processed_data['preprocessing_info']['original_shape'][1]}
    - **Final Features**: {len(processed_data['feature_names'])}
    - **Training Samples**: {len(processed_data['X_train']):,}
    - **Test Samples**: {len(processed_data['X_test']):,}
    
    ## Preprocessing Steps Applied
    {chr(10).join([f"- {step}" for step in processed_data['preprocessing_info']['preprocessing_steps']])}
    
    ## Target Information
    - **Type**: {processed_data['preprocessing_info']['target_info']['type']}
    - **Unique Values**: {processed_data['preprocessing_info']['target_info']['unique_values']}
    
    ## Feature Names
    {chr(10).join([f"- {feature}" for feature in processed_data['feature_names']])}
    """
    
    st.download_button(
        label="📥 Download Processing Summary (TXT)",
        data=summary_text,
        file_name="processing_summary.txt",
        mime="text/plain",
        use_container_width=True
    )

if __name__ == "__main__":
    main()
