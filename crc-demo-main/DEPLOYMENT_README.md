# 🚀 Quick Deployment Guide - CRC Analysis Platform

## Local Testing
The app is currently running at: http://localhost:8502

If you see only the background, try:
1. Hard refresh (Ctrl+F5 or Cmd+Shift+R)
2. Clear browser cache
3. Open in incognito/private window

## Deploy to Streamlit Cloud (Recommended)

### Step 1: Push to GitHub
```bash
# If not already done
git init
git add .
git commit -m "CRC Analysis Platform - Ready for deployment"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/crc-analysis-platform.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Visit https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - Repository: `your-username/crc-analysis-platform`
   - Branch: `main`
   - Main file: `app.py`
5. Click "Deploy"

### Step 3: Share Your App
Your app will be available at:
`https://your-username-crc-analysis-platform.streamlit.app`

## Alternative: Deploy to Hugging Face Spaces

```bash
pip install huggingface-hub
huggingface-cli login
huggingface-cli repo create crc-analysis-platform --type space --space_sdk streamlit
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/crc-analysis-platform
git push hf main
```

## Features Deployed
✅ Landing page with animations  
✅ Upload & Analyze interface  
✅ Real-time demo  
✅ EPOC dashboard  
✅ Analysis history  
✅ SNF molecular subtypes (not CMS)  
✅ Professional biotech UI  

## Troubleshooting
- **White screen**: Check browser console for errors
- **Slow loading**: First load downloads models (~200MB)
- **Import errors**: Ensure all files in app/ directory are uploaded

## Support
- Issues: Create GitHub issue
- Streamlit help: https://discuss.streamlit.io 