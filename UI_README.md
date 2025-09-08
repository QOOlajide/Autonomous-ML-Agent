# 🤖 Autonomous ML Agent - Web Interface

## Quick Start

### Option 1: Simple Launcher (Recommended)
```bash
python launch_ui.py
```

### Option 2: Direct Streamlit Command
```bash
streamlit run streamlit_app.py
```

## What You'll See

The web interface provides a modern, intuitive way to use our ML agent:

### 🏠 **Home Page**
- **Welcome screen** with instructions
- **Sample datasets** you can download to try the system
- **Configuration sidebar** for setting up your ML pipeline

### 📊 **Data Overview Tab**
- **Dataset statistics** (rows, columns, memory usage, missing values)
- **Data type analysis** with missing value information
- **Target column analysis** with distribution charts
- **Sample data preview** (first 10 rows)

### 🔧 **Processing Tab**
- **Real-time progress bars** showing pipeline steps
- **Data validation results** with recommendations
- **Preprocessing progress** with status updates
- **Success/error messages** with helpful guidance

### 📈 **Results Tab**
- **Feature importance charts** (top 10 most important features)
- **Target distribution pie charts**
- **Feature correlation heatmaps**
- **Data quality metrics** and statistics

### 💾 **Download Tab**
- **Processed training data** (CSV format)
- **Processed test data** (CSV format)
- **Feature importance report** (JSON format)
- **Processing summary** (TXT format)

## Key Features Explained

### 🎨 **Modern UI Framework: Streamlit**
- **Streamlit** is a Python framework that makes creating web apps incredibly easy
- It automatically handles HTML/CSS generation, user interactions, and web server setup
- Perfect for ML applications because it integrates seamlessly with pandas, plotly, and scikit-learn

### 📊 **Interactive Visualizations: Plotly**
- **Plotly** creates beautiful, interactive charts that users can zoom, pan, and hover over
- Much more engaging than static matplotlib charts
- Automatically responsive and mobile-friendly

### ⚡ **Performance Optimizations**
- **@st.cache_data** decorators cache expensive operations (like creating sample data)
- **Session state** stores processed data between tabs to avoid re-computation
- **Progress bars** and **spinners** provide real-time feedback

### 🎯 **User Experience Features**
- **File upload** with drag-and-drop support
- **Configuration sidebar** for easy parameter adjustment
- **Tabbed interface** for organized information display
- **Error handling** with helpful error messages and suggestions
- **Download buttons** for all processed data and reports

## Technical Architecture

### Frontend (Streamlit)
```python
# Main app structure
def main():
    # Page configuration and styling
    st.set_page_config(...)
    
    # Sidebar for configuration
    with st.sidebar:
        # File upload, target column, model settings
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs([...])
    
    # Process uploaded data
    if uploaded_file:
        process_dataset_ui(...)
```

### Data Flow
1. **Upload** → User uploads CSV file
2. **Validate** → Check data structure and quality
3. **Process** → Run preprocessing pipeline
4. **Visualize** → Show results and charts
5. **Download** → Export processed data

### State Management
- **Session state** (`st.session_state`) stores data between tabs
- **Caching** (`@st.cache_data`) optimizes performance
- **Temporary files** for handling uploaded data

## Customization

### Adding New Visualizations
```python
# In show_results_visualization()
fig = px.bar(data, x='feature', y='importance')
st.plotly_chart(fig, use_container_width=True)
```

### Adding New Download Options
```python
# In show_download_options()
st.download_button(
    label="Download Custom Report",
    data=custom_data,
    file_name="custom_report.csv",
    mime="text/csv"
)
```

### Styling
The app uses custom CSS for better appearance:
```python
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #1f77b4; }
    .metric-card { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)
```

## Troubleshooting

### Common Issues
1. **"Module not found"** → Run `pip install streamlit plotly`
2. **"File upload not working"** → Make sure file is CSV format
3. **"Processing fails"** → Check target column name is correct
4. **"Charts not showing"** → Refresh the page

### Performance Tips
- Use smaller datasets for testing (< 10MB)
- Close other browser tabs to free up memory
- Use the sample datasets first to test functionality

## Next Steps

This UI provides the foundation for:
- **Model training interface** (coming in next sprint)
- **Real-time training progress** with live metrics
- **Model comparison dashboard** with leaderboards
- **Hyperparameter tuning visualization**
- **Model deployment interface**

The architecture is designed to be easily extensible as we add more ML capabilities!
