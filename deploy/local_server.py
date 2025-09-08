"""
Local Development Server

This creates a simple local server that you can run on your own machine
or any server you have access to. Great for testing and internal use.
"""

import subprocess
import sys
import os
from pathlib import Path

def setup_local_server():
    """Set up and run the local development server."""
    
    print("🚀 Setting up Local Development Server")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found. Make sure you're in the project root.")
        return False
    
    # Install dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    # Create necessary directories
    print("📁 Creating directories...")
    Path("logs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    print("✅ Directories created!")
    
    # Set environment variables
    print("⚙️ Setting up environment...")
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    print("✅ Environment configured!")
    
    return True

def run_server():
    """Run the Streamlit server."""
    
    if not setup_local_server():
        return
    
    print("🌐 Starting server...")
    print("📱 Your app will be available at: http://localhost:8501")
    print("🌍 For network access: http://YOUR_IP:8501")
    print("⏹️ Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error running server: {e}")

if __name__ == "__main__":
    run_server()
