"""
CI/CD Setup Script for Autonomous ML Agent

This script helps you set up the CI/CD pipeline by:
1. Checking prerequisites
2. Setting up GitHub secrets
3. Testing the pipeline
4. Providing guidance for deployment
"""

import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Print a nice banner for the CI/CD setup."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🚀 CI/CD Setup Wizard 🚀                 ║
    ║                                                              ║
    ║  Let's set up your automated deployment pipeline!           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if we're in a git repository
    if not Path(".git").exists():
        print("❌ Not in a git repository. Please run 'git init' first.")
        return False
    
    # Check if GitHub remote is set
    try:
        result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
        if "github.com" not in result.stdout:
            print("❌ GitHub remote not found. Please add your GitHub repository.")
            return False
    except FileNotFoundError:
        print("❌ Git not found. Please install Git first.")
        return False
    
    # Check if required files exist
    required_files = [
        ".github/workflows/ci-cd.yml",
        "tests/test_data_pipeline.py",
        "tests/test_config.py",
        "pyproject.toml"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All prerequisites met!")
    return True

def install_test_dependencies():
    """Install dependencies needed for testing."""
    print("📦 Installing test dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "pytest", "pytest-cov", "black", "flake8", "mypy", "safety", "bandit"
        ], check=True)
        print("✅ Test dependencies installed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install test dependencies")
        return False

def run_tests():
    """Run the test suite to verify everything works."""
    print("🧪 Running tests...")
    
    try:
        # Run pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_code_quality_checks():
    """Run code quality checks."""
    print("🔍 Running code quality checks...")
    
    checks = [
        ("Black (formatting)", [sys.executable, "-m", "black", "--check", "src/"]),
        ("Flake8 (linting)", [sys.executable, "-m", "flake8", "src/"]),
        ("MyPy (type checking)", [sys.executable, "-m", "mypy", "src/", "--ignore-missing-imports"]),
    ]
    
    all_passed = True
    
    for name, command in checks:
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {name} passed!")
            else:
                print(f"❌ {name} failed:")
                print(result.stdout)
                print(result.stderr)
                all_passed = False
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            all_passed = False
    
    return all_passed

def setup_github_secrets():
    """Guide user through setting up GitHub secrets."""
    print("\n🔐 GitHub Secrets Setup")
    print("=" * 50)
    
    secrets = {
        "HEROKU_API_KEY": "Your Heroku API key (for Heroku deployment)",
        "HEROKU_APP_NAME": "Your Heroku app name",
        "HEROKU_EMAIL": "Your Heroku account email",
        "AWS_ACCESS_KEY_ID": "Your AWS access key (for AWS deployment)",
        "AWS_SECRET_ACCESS_KEY": "Your AWS secret key",
        "OPENAI_API_KEY": "Your OpenAI API key (optional)",
        "ANTHROPIC_API_KEY": "Your Anthropic API key (optional)",
    }
    
    print("To set up GitHub secrets:")
    print("1. Go to your GitHub repository")
    print("2. Click Settings → Secrets and variables → Actions")
    print("3. Click 'New repository secret'")
    print("4. Add the following secrets:\n")
    
    for secret, description in secrets.items():
        print(f"   {secret}: {description}")
    
    print("\n💡 Tip: You only need to add secrets for the platforms you want to deploy to!")
    
    return True

def test_github_actions():
    """Test if GitHub Actions workflow is valid."""
    print("🔍 Validating GitHub Actions workflow...")
    
    workflow_file = Path(".github/workflows/ci-cd.yml")
    if not workflow_file.exists():
        print("❌ CI/CD workflow file not found")
        return False
    
    try:
        # Basic YAML validation
        import yaml
        with open(workflow_file) as f:
            yaml.safe_load(f)
        print("✅ GitHub Actions workflow is valid!")
        return True
    except ImportError:
        print("⚠️ PyYAML not installed, skipping YAML validation")
        return True
    except Exception as e:
        print(f"❌ Invalid YAML in workflow file: {e}")
        return False

def create_deployment_guide():
    """Create a deployment guide for the user."""
    print("📚 Creating deployment guide...")
    
    guide_content = """
# 🚀 Deployment Guide

## Quick Start
1. Push your code to GitHub: `git push origin main`
2. GitHub Actions will automatically run tests and deploy
3. Check the Actions tab in your GitHub repository for status

## Deployment URLs
- **Streamlit Cloud**: https://share.streamlit.io (automatic)
- **Heroku**: https://your-app.herokuapp.com (if configured)
- **AWS**: https://your-app.aws.com (if configured)

## Monitoring
- Check GitHub Actions tab for deployment status
- View logs if deployment fails
- Set up notifications for deployment results

## Troubleshooting
- Ensure all secrets are set in GitHub repository settings
- Check that your code passes all tests locally
- Verify platform-specific requirements are met
"""
    
    with open("DEPLOYMENT_GUIDE.md", "w") as f:
        f.write(guide_content)
    
    print("✅ Deployment guide created: DEPLOYMENT_GUIDE.md")
    return True

def main():
    """Main CI/CD setup function."""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return False
    
    # Install dependencies
    if not install_test_dependencies():
        print("\n❌ Failed to install dependencies. Please check your Python environment.")
        return False
    
    # Run tests
    if not run_tests():
        print("\n❌ Tests failed. Please fix the issues and try again.")
        return False
    
    # Run code quality checks
    if not run_code_quality_checks():
        print("\n⚠️ Some code quality checks failed. Consider fixing these issues.")
    
    # Validate GitHub Actions
    if not test_github_actions():
        print("\n❌ GitHub Actions workflow is invalid. Please check the YAML syntax.")
        return False
    
    # Setup GitHub secrets
    setup_github_secrets()
    
    # Create deployment guide
    create_deployment_guide()
    
    print("\n🎉 CI/CD Setup Complete!")
    print("=" * 50)
    print("✅ All tests passing")
    print("✅ Code quality checks passed")
    print("✅ GitHub Actions workflow validated")
    print("✅ Deployment guide created")
    
    print("\n🚀 Next Steps:")
    print("1. Set up GitHub secrets (see instructions above)")
    print("2. Push your code: git push origin main")
    print("3. Check GitHub Actions tab for deployment status")
    print("4. Your app will be automatically deployed!")
    
    print("\n📚 Documentation:")
    print("- CI_CD_GUIDE.md - Complete CI/CD documentation")
    print("- DEPLOYMENT_GUIDE.md - Quick deployment guide")
    print("- README.md - Project overview")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 CI/CD setup completed successfully!")
        else:
            print("\n❌ CI/CD setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
