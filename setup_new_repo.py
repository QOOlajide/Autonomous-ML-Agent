"""
Git repository setup helper for creating a new GitHub repository.

This script helps disconnect from the current repo and prepare for a new one.
"""

import subprocess
import sys
from typing import List


def run_git_command(command: List[str]) -> str:
    """
    Run a git command and return the output.
    
    Args:
        command (List[str]): Git command as a list of strings
    
    Returns:
        str: Command output or error message
    """
    try:
        # Run the git command and capture output
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"
    except FileNotFoundError:
        return "Error: Git is not installed or not in PATH"


def setup_new_repository(new_repo_name: str) -> None:
    """
    Set up a new git repository by disconnecting from current remote.
    
    Args:
        new_repo_name (str): Name for the new repository
    """
    print(f"Setting up new repository: {new_repo_name}")
    print("=" * 50)
    
    # Step 1: Remove current remote connection
    print("1. Removing current remote connection...")
    result = run_git_command(["git", "remote", "remove", "origin"])
    if "Error" not in result:
        print("✅ Disconnected from current repository")
    else:
        print(f"⚠️  {result}")
    
    # Step 2: Add and commit current changes
    print("\n2. Adding current files...")
    result = run_git_command(["git", "add", "."])
    if not result:
        print("✅ Files added to staging")
    
    # Step 3: Commit changes
    print("\n3. Committing changes...")
    result = run_git_command(["git", "commit", "-m", f"Initial commit for {new_repo_name}"])
    if "Error" not in result:
        print("✅ Changes committed")
    else:
        print(f"⚠️  {result}")
    
    print(f"\n🎉 Ready to create new repository: {new_repo_name}")
    print("\nNext steps:")
    print(f"1. Go to GitHub.com and create a new repository named '{new_repo_name}'")
    print("2. Copy the repository URL")
    print("3. Run: git remote add origin <your-new-repo-url>")
    print("4. Run: git push -u origin main")


if __name__ == "__main__":
    # Get new repository name from user
    if len(sys.argv) > 1:
        repo_name = sys.argv[1]
    else:
        repo_name = input("Enter new repository name: ")
    
    setup_new_repository(repo_name)
