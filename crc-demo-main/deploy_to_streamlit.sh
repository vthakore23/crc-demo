#!/bin/bash

echo "🚀 CRC Analysis Platform - Streamlit Cloud Deployment Script"
echo "==========================================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: CRC Analysis Platform"
fi

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "📝 Committing current changes..."
    git add .
    git commit -m "Update: Prepare for Streamlit Cloud deployment"
fi

echo ""
echo "✅ Repository is ready for deployment!"
echo ""
echo "📋 Next Steps:"
echo "1. Create a GitHub repository at https://github.com/new"
echo "2. Add the remote: git remote add origin https://github.com/YOUR_USERNAME/crc-analysis-platform.git"
echo "3. Push code: git push -u origin main"
echo "4. Go to https://share.streamlit.io"
echo "5. Click 'New app' and select your repository"
echo "6. Set main file path to: app.py"
echo "7. Click 'Deploy'"
echo ""
echo "🎉 Your app will be live in a few minutes!" 