# CRC Analysis Platform - Deployment Guide

## 🚀 Deployment to Streamlit Cloud

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at share.streamlit.io)
3. All code pushed to GitHub repository

### Step 1: Prepare Repository
Ensure your repository has the following structure:
```
CRC_Analysis_Project/
├── app.py                    # Main entry point
├── app/                      # Application modules
│   ├── crc_unified_platform.py
│   ├── molecular_subtype_mapper.py
│   ├── report_generator.py
│   └── ...
├── models/                   # Model files
│   └── balanced_tissue_classifier.pth
├── demo_assets/             # Demo images
├── requirements.txt         # Python dependencies
├── packages.txt            # System dependencies
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

### Step 2: Deploy to Streamlit Cloud

1. **Visit Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub

2. **Create New App**
   - Click "New app"
   - Select your repository: `your-username/CRC_Analysis_Project`
   - Branch: `main` (or your default branch)
   - Main file path: `app.py`

3. **Advanced Settings**
   - Python version: 3.10 (recommended)
   - Add secrets if needed (none required for basic deployment)

4. **Deploy**
   - Click "Deploy"
   - Wait for build to complete (5-10 minutes first time)

### Step 3: Monitor Deployment

Watch the deployment logs for any errors. Common issues:
- **Memory limits**: Free tier has 1GB limit
- **Missing dependencies**: Check requirements.txt
- **File paths**: Ensure all paths are relative

### 🌐 Alternative Deployment Options

#### Deploy to Hugging Face Spaces

1. **Create Space**
   ```bash
   # Install Hugging Face CLI
   pip install huggingface-hub
   
   # Login
   huggingface-cli login
   
   # Create space
   huggingface-cli repo create crc-analysis-platform --type space --space_sdk streamlit
   ```

2. **Push Code**
   ```bash
   git remote add hf https://huggingface.co/spaces/your-username/crc-analysis-platform
   git push hf main
   ```

#### Deploy to Google Cloud Run

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       libgl1-mesa-glx \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender-dev \
       libgomp1 \
       wget \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy app files
   COPY . .
   
   # Expose port
   EXPOSE 8501
   
   # Run app
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Deploy**
   ```bash
   # Build and push to Container Registry
   gcloud builds submit --tag gcr.io/PROJECT-ID/crc-platform
   
   # Deploy to Cloud Run
   gcloud run deploy --image gcr.io/PROJECT-ID/crc-platform --platform managed
   ```

#### Deploy to AWS EC2

1. **Launch EC2 Instance**
   - Amazon Linux 2 or Ubuntu 20.04
   - t2.medium or larger
   - Open port 8501 in security group

2. **Install Dependencies**
   ```bash
   # Update system
   sudo yum update -y  # or apt-get update
   
   # Install Python 3.10
   sudo yum install python3.10 -y
   
   # Clone repository
   git clone https://github.com/your-username/CRC_Analysis_Project.git
   cd CRC_Analysis_Project
   
   # Install requirements
   pip3.10 install -r requirements.txt
   ```

3. **Run with systemd**
   ```bash
   # Create service file
   sudo nano /etc/systemd/system/crc-platform.service
   ```
   
   Content:
   ```ini
   [Unit]
   Description=CRC Analysis Platform
   After=network.target
   
   [Service]
   Type=simple
   User=ec2-user
   WorkingDirectory=/home/ec2-user/CRC_Analysis_Project
   ExecStart=/usr/bin/python3.10 -m streamlit run app.py
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

4. **Start Service**
   ```bash
   sudo systemctl start crc-platform
   sudo systemctl enable crc-platform
   ```

### 📊 Performance Optimization

1. **Model Loading**
   - Use `@st.cache_resource` for model loading
   - Consider model quantization for faster loading

2. **Image Processing**
   - Implement lazy loading for demo images
   - Use image compression for web display

3. **Memory Management**
   - Clear unused variables
   - Use generators for large datasets

### 🔒 Security Considerations

1. **Authentication** (if needed)
   ```python
   # Add to app.py
   import hmac
   
   def check_password():
       """Returns `True` if the user had the correct password."""
       def password_entered():
           """Checks whether a password entered by the user is correct."""
           if hmac.compare_digest(st.session_state["password"], 
                                 st.secrets["password"]):
               st.session_state["password_correct"] = True
               del st.session_state["password"]
           else:
               st.session_state["password_correct"] = False
   
       if st.session_state.get("password_correct", False):
           return True
   
       st.text_input("Password", type="password", 
                    on_change=password_entered, key="password")
       if "password_correct" in st.session_state:
           st.error("😕 Password incorrect")
       return False
   
   if not check_password():
       st.stop()
   ```

2. **Data Privacy**
   - Don't store uploaded images permanently
   - Clear temporary files after analysis
   - Add GDPR compliance notices if needed

### 🐛 Troubleshooting

**Common Issues:**

1. **"Module not found" errors**
   - Check requirements.txt includes all imports
   - Verify file paths are correct

2. **Memory errors**
   - Reduce model size
   - Implement batch processing
   - Use smaller demo images

3. **Slow loading**
   - Optimize model loading with caching
   - Use CDN for static assets
   - Enable Streamlit's file watching only in dev

### 📧 Support

For deployment issues:
- Streamlit Cloud: https://discuss.streamlit.io
- GitHub Issues: Create issue in your repository
- Documentation: https://docs.streamlit.io/streamlit-cloud 