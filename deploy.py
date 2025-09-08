"""
Deployment Script for Autonomous ML Agent

This script helps you choose and execute the right deployment option for your needs.
It provides an interactive menu to select from different deployment platforms.
"""

import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Print a nice banner for the deployment script."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🚀 ML Agent Deployer 🚀                   ║
    ║                                                              ║
    ║  Choose your deployment option and we'll handle the rest!    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_requirements():
    """Check if all required files exist for deployment."""
    required_files = [
        "streamlit_app.py",
        "requirements.txt",
        "src/config.py",
        "src/data/pipeline.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease make sure you're in the project root directory.")
        return False
    
    print("✅ All required files found!")
    return True

def deploy_streamlit_cloud():
    """Deploy to Streamlit Cloud."""
    print("\n🌐 Deploying to Streamlit Cloud...")
    print("=" * 50)
    
    print("""
    Steps to deploy to Streamlit Cloud:
    
    1. 📁 Push your code to GitHub:
       git add .
       git commit -m "Deploy ML Agent"
       git push origin main
    
    2. 🌐 Go to https://share.streamlit.io
    
    3. 🔗 Connect your GitHub repository
    
    4. ⚙️ Configure your app:
       - App URL: Choose a unique name
       - Main file: streamlit_app.py
       - Python version: 3.11
    
    5. 🚀 Deploy!
    
    Your app will be live at: https://your-app-name.streamlit.app
    """)
    
    # Check if git is initialized
    if not Path(".git").exists():
        print("⚠️ Git not initialized. Initializing now...")
        subprocess.run(["git", "init"])
        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", "Initial commit"])
        print("✅ Git initialized! Now push to GitHub and follow the steps above.")

def deploy_docker():
    """Deploy using Docker."""
    print("\n🐳 Deploying with Docker...")
    print("=" * 50)
    
    # Check if Docker is installed
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker is installed!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found. Please install Docker first:")
        print("   https://docs.docker.com/get-docker/")
        return
    
    # Build Docker image
    print("🔨 Building Docker image...")
    try:
        subprocess.run(["docker", "build", "-t", "ml-agent", "."], check=True)
        print("✅ Docker image built successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to build Docker image")
        return
    
    # Run Docker container
    print("🚀 Starting Docker container...")
    print("Your app will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the container")
    
    try:
        subprocess.run([
            "docker", "run", "-p", "8501:8501", 
            "-v", f"{Path.cwd()}/artifacts:/app/artifacts",
            "-v", f"{Path.cwd()}/logs:/app/logs",
            "ml-agent"
        ])
    except KeyboardInterrupt:
        print("\n👋 Container stopped.")

def deploy_heroku():
    """Deploy to Heroku."""
    print("\n🟣 Deploying to Heroku...")
    print("=" * 50)
    
    # Check if Heroku CLI is installed
    try:
        subprocess.run(["heroku", "--version"], check=True, capture_output=True)
        print("✅ Heroku CLI is installed!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Heroku CLI not found. Please install it first:")
        print("   https://devcenter.heroku.com/articles/heroku-cli")
        return
    
    # Get app name from user
    app_name = input("Enter your Heroku app name (or press Enter for auto-generated): ").strip()
    
    if not app_name:
        app_name = "ml-agent-" + str(hash(Path.cwd()) % 10000)
    
    print(f"📱 Creating Heroku app: {app_name}")
    
    try:
        # Create Heroku app
        subprocess.run(["heroku", "create", app_name], check=True)
        print(f"✅ Heroku app created: https://{app_name}.herokuapp.com")
        
        # Set environment variables (if needed)
        print("⚙️ Setting up environment variables...")
        # Add your API keys here if needed
        # subprocess.run(["heroku", "config:set", "OPENAI_API_KEY=your_key"], check=True)
        
        # Deploy
        print("🚀 Deploying to Heroku...")
        subprocess.run(["git", "push", "heroku", "main"], check=True)
        
        # Open app
        subprocess.run(["heroku", "open"])
        
        print(f"🎉 Your app is live at: https://{app_name}.herokuapp.com")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Heroku deployment failed: {e}")
        print("Make sure you're logged in: heroku login")

def deploy_local():
    """Deploy locally."""
    print("\n💻 Deploying locally...")
    print("=" * 50)
    
    # Install dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    
    # Run the app
    print("🚀 Starting local server...")
    print("Your app will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")

def main():
    """Main deployment script."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        return
    
    # Show deployment options
    print("\n🎯 Choose your deployment option:")
    print("1. 🌐 Streamlit Cloud (Free, Easiest)")
    print("2. 🐳 Docker (Portable)")
    print("3. 🟣 Heroku (Simple Cloud)")
    print("4. 💻 Local Server (Development)")
    print("5. 📖 Show deployment guide")
    print("0. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                deploy_streamlit_cloud()
                break
            elif choice == "2":
                deploy_docker()
                break
            elif choice == "3":
                deploy_heroku()
                break
            elif choice == "4":
                deploy_local()
                break
            elif choice == "5":
                print("\n📖 Opening deployment guide...")
                if Path("deploy/README.md").exists():
                    print("Check deploy/README.md for detailed instructions!")
                else:
                    print("Deployment guide not found.")
                break
            else:
                print("❌ Invalid choice. Please enter 0-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
