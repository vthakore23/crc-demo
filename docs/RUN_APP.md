# 🚀 How to Run the CRC Molecular Subtype Predictor

## Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Run the Application
```bash
python3 -m streamlit run app.py
```

**Alternative method:**
```bash
streamlit run app.py
```

### 3. Access the App
- The app will automatically open in your browser
- Or visit: http://localhost:8501

## Features Available

✅ **Molecular Subtype Analysis** - Upload histopathology images  
✅ **Live Demo** - Interactive examples  
✅ **EPOC Dashboard** - Ready for validated data  
✅ **Model Performance** - Architecture details  
✅ **Analysis History** - Track results  

## Architecture

- **State-of-the-art ensemble**: 247.3M parameters
- **Three backbone models**: Swin-V2, ConvNeXt-V2, EfficientNet-V2
- **Advanced features**: Uncertainty quantification, attention visualization
- **Ready for EPOC data**: Molecular ground truth integration

## Troubleshooting

**Issue**: `command not found: streamlit`  
**Solution**: Use `python3 -m streamlit run app.py`

**Issue**: Module import errors  
**Solution**: Install missing dependencies: `pip3 install [module_name]`

**Issue**: Port already in use  
**Solution**: Kill existing processes: `pkill -f streamlit`

---

**Note**: This is a research tool ready for clinical validation with molecular ground truth data. 