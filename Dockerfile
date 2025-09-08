# Dockerfile for Autonomous ML Agent
# This creates a containerized version of your app that can run anywhere

# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
# These are needed for some Python packages like scikit-learn
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for logs and artifacts
RUN mkdir -p logs artifacts

# Expose port 8501 (Streamlit's default port)
EXPOSE 8501

# Health check to ensure app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Command to run the app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
