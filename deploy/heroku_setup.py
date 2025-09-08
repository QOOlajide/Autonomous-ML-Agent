"""
Heroku Deployment Setup

Heroku is a popular platform-as-a-service that makes deployment easy.
It's great for MVPs and small to medium applications.

Steps to deploy:
1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
2. Create Heroku app: heroku create your-app-name
3. Set environment variables: heroku config:set OPENAI_API_KEY=your_key
4. Deploy: git push heroku main

Cost: Free tier available, then $7/month for basic dyno
"""

# This file documents the Heroku deployment process
# The actual deployment uses the Procfile and requirements.txt we'll create
