# 🧠 Memory Optimization Guide for CRC Analysis Platform

## 🚨 Common Issue: Server Disconnection with Large Images

If you're experiencing server disconnections when processing large images, this guide will help you resolve the issue.

## 🔍 Problem Identification

**Symptoms:**
- Server disconnects during image processing
- "Server disconnected" error messages
- Application freezes or becomes unresponsive
- Browser tab crashes during analysis

**Root Cause:**
The platform runs out of available memory when processing large histopathology images, causing the Streamlit server to crash.

## ⚡ Immediate Solutions

### 1. Reduce Image Size
**Before uploading:**
- Resize images to maximum 3000x3000 pixels
- Keep file size under 100MB
- Use JPEG format for better compression

### 2. Close Other Applications
**Free up system memory:**
- Close web browsers (except the one running the platform)
- Close image editing software (Photoshop, GIMP, ImageJ)
- Close video players and streaming applications
- Close other Python/Jupyter notebooks

### 3. Check Available Memory
**Minimum requirements:**
- At least 2GB RAM available
- 4GB+ recommended for large images
- 8GB+ recommended for molecular analysis

## 🔧 Advanced Configuration

### Memory Settings Configuration

The platform uses `app/memory_config.py` for memory management:

```python
# File size limits (in MB)
MAX_FILE_SIZE_MB = 100  # Reduce to 50 if still having issues

# Image processing limits  
MAX_PIXELS = 16_000_000  # Reduce to 9_000_000 for lower memory
MAX_DIMENSION = 3000     # Reduce to 2000 for lower memory

# Memory requirements
MIN_AVAILABLE_MEMORY_GB = 2.0  # Increase if you have more RAM
```

### Reducing Memory Usage

**Option 1: Edit memory_config.py**
```python
# Conservative settings for limited memory
MAX_FILE_SIZE_MB = 50
MAX_PIXELS = 9_000_000
MAX_DIMENSION = 2000
MIN_AVAILABLE_MEMORY_GB = 1.5
```

**Option 2: Use smaller images**
- Resize images to 2000x2000 pixels maximum
- Use 50MB file size limit
- Process one image at a time

## 🖥️ System Optimization

### For Windows Users:
1. Close unnecessary programs via Task Manager
2. Increase virtual memory (pagefile)
3. Disable Windows visual effects
4. Run platform with administrator privileges

### For Mac Users:
1. Close applications via Activity Monitor
2. Free up disk space (minimum 5GB free)
3. Restart your Mac to clear memory caches
4. Use `sudo` when running the platform

### For Linux Users:
1. Monitor memory with `htop` or `free -h`
2. Increase swap space if needed
3. Close X11 applications
4. Consider running in headless mode

## 🚀 Performance Tips

### Image Preprocessing
**Before uploading to platform:**
```python
from PIL import Image

# Resize large images
image = Image.open('large_image.tiff')
if image.width > 3000 or image.height > 3000:
    image.thumbnail((3000, 3000), Image.Resampling.LANCZOS)
    image.save('resized_image.jpg', quality=95)
```

### Batch Processing
- Process images one at a time
- Close platform between large analyses
- Restart browser periodically
- Clear browser cache regularly

### Memory Monitoring
Enable memory monitoring in the platform:
```python
# In app/memory_config.py
SHOW_MEMORY_USAGE = True
```

## 🔄 Recovery Steps

### If Server Disconnects:
1. **Don't panic** - your analysis data is preserved
2. Refresh the browser page
3. Restart the Streamlit application
4. Try with a smaller image
5. Check system memory availability

### If Platform Won't Start:
1. Restart your computer
2. Close all unnecessary applications
3. Use smaller images (under 50MB)
4. Try the tissue-only analysis mode

## 📊 Memory Usage Guidelines

| Image Size | RAM Required | Processing Time | Recommendation |
|------------|--------------|-----------------|----------------|
| < 1000x1000 | 1-2GB | < 30 seconds | ✅ Optimal |
| 1000x2000 | 2-3GB | 30-60 seconds | ✅ Good |
| 2000x3000 | 3-4GB | 1-2 minutes | ⚠️ Caution |
| > 3000x3000 | 4GB+ | 2+ minutes | ❌ May fail |

## 🛠️ Advanced Troubleshooting

### Enable Debug Mode
```bash
# Run with debug information
streamlit run app.py --logger.level=debug
```

### Check Memory Usage
```python
import psutil
memory = psutil.virtual_memory()
print(f"Available: {memory.available / (1024**3):.1f} GB")
print(f"Used: {memory.percent:.1f}%")
```

### Force Garbage Collection
```python
import gc
gc.collect()  # Force memory cleanup
```

## 🆘 When All Else Fails

1. **Use tissue-only analysis mode** (requires less memory)
2. **Split large images into smaller tiles**
3. **Use a computer with more RAM**
4. **Contact support with your system specifications**

## 📞 Getting Help

**Include this information when seeking help:**
- Operating system and version
- Available RAM
- Image dimensions and file size
- Error messages received
- Steps that led to the problem

**Memory diagnostic command:**
```bash
python -c "import psutil; m=psutil.virtual_memory(); print(f'Available: {m.available/(1024**3):.1f}GB, Used: {m.percent:.1f}%')"
```

## 🎯 Success Indicators

**Platform is working optimally when:**
- ✅ Images process without disconnection
- ✅ Memory usage stays below 80%
- ✅ Analysis completes in under 2 minutes
- ✅ Browser remains responsive throughout

Remember: **The platform automatically optimizes memory usage**, but very large images may still cause issues on systems with limited RAM. 