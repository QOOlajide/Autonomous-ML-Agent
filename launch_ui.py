"""
Simple launcher script for the Streamlit UI.

This script provides an easy way to launch the web interface
without needing to remember the streamlit command.
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def launch_streamlit():
    """
    Launch the Streamlit app and automatically open it in the browser.
    
    This function:
    1. Checks if Streamlit is installed
    2. Launches the Streamlit server
    3. Opens the app in the default browser
    """
    
    print("🚀 Launching Autonomous ML Agent Web Interface...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Streamlit installed successfully!")
    
    # Check if our app file exists
    app_file = Path("streamlit_app.py")
    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        print("Make sure you're running this from the project root directory.")
        return
    
    print("🌐 Starting Streamlit server...")
    print("📱 The web interface will open automatically in your browser.")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down the web interface. Goodbye!")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

if __name__ == "__main__":
    launch_streamlit()
