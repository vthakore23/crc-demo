# Memory Management Configuration for CRC Analysis Platform

# File size limits (in MB)
MAX_FILE_SIZE_MB = 100  # Reduced from 200MB for memory safety

# Image processing limits
MAX_PIXELS = 16_000_000  # 16 megapixels (e.g., 4000x4000)
MAX_DIMENSION = 3000     # Max width/height for large images

# Memory requirements
MIN_AVAILABLE_MEMORY_GB = 2.0  # Minimum available memory to start processing

# Warning thresholds (in megapixels)
LARGE_IMAGE_WARNING_MP = 16    # Warn user about memory resizing
LARGE_IMAGE_INFO_MP = 9        # Info about potential resizing

# Model loading settings
LAZY_MODEL_LOADING = True      # Load molecular models only when needed
AGGRESSIVE_CLEANUP = True      # Force cleanup after each analysis step

# Memory monitoring
SHOW_MEMORY_USAGE = True       # Display memory usage during analysis
MEMORY_CHECK_INTERVAL = 0.5    # Seconds between memory checks (if monitoring enabled) 