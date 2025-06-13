# CRC Analysis Platform - API Documentation

## 🔧 Core Functions

### Tissue Classification

#### `analyze_tissue_patch(image, model, demo_mode=False)`
Analyzes a tissue patch using the trained model or generates demo predictions.

**Parameters:**
- `image`: PIL Image or numpy array - Input tissue image
- `model`: torch.nn.Module - Trained tissue classifier model
- `demo_mode`: bool - Whether to use demo predictions instead of model

**Returns:**
```python
{
    'primary_class': str,           # Predicted tissue type
    'confidence': float,            # Confidence percentage (0-100)
    'all_predictions': list,        # Top 3 predictions with confidence
    'probabilities': np.array,      # Raw probability distribution
    'tissue_composition': dict      # Tissue type percentages
}
```

#### `load_models()`
Loads all required models for the platform.

**Returns:**
- `tissue_model`: CRCClassifier - Tissue classification model
- `model_loaded`: bool - Whether pre-trained weights were loaded
- `molecular_mapper`: MolecularSubtypeMapper - Molecular prediction model
- `report_generator`: PDFReportGenerator - Report generation module

### Molecular Subtyping

#### `MolecularSubtypeMapper.classify_molecular_subtype(image, transform)`
Predicts molecular subtype from tissue image.

**Parameters:**
- `image`: PIL Image - Input tissue image
- `transform`: torchvision.transforms - Image preprocessing pipeline

**Returns:**
```python
{
    'predicted_subtype': str,       # SNF1, SNF2, or SNF3
    'confidence': float,            # Confidence percentage
    'probabilities': dict,          # Subtype probabilities
    'features': dict               # Extracted spatial features
}
```

### WSI Handling

#### `load_wsi_region(slide_path, location, level, size)`
Loads a region from a whole slide image.

**Parameters:**
- `slide_path`: str - Path to WSI file
- `location`: tuple - (x, y) coordinates of top-left corner
- `level`: int - Pyramid level (0 = highest resolution)
- `size`: tuple - (width, height) of region to extract

**Returns:**
- `region`: np.array - Extracted image region

#### `is_wsi_file(filename)`
Checks if a file is a supported WSI format.

**Parameters:**
- `filename`: str - File name to check

**Returns:**
- `bool`: True if WSI format, False otherwise

### Report Generation

#### `CRCReportGenerator.generate_report(analysis_results, output_path)`
Generates a comprehensive PDF report from analysis results.

**Parameters:**
- `analysis_results`: dict - Combined tissue and molecular results
- `output_path`: str - Path to save PDF report

**Returns:**
- `success`: bool - Whether report was generated successfully

## 🎨 Demo Functions

### `generate_demo_predictions(image)`
Generates realistic demo predictions based on image characteristics.

**Features:**
- Analyzes color distribution and texture
- Uses filename hints for demo samples
- Produces realistic confidence scores
- No model bias issues

### `RealTimeAnalysisDemo.run_analysis(image, results)`
Displays step-by-step analysis visualization.

**Parameters:**
- `image`: np.array - Input image
- `results`: dict - Analysis results to visualize

**Features:**
- Progressive visualization
- Heatmap generation
- Feature extraction display
- Confidence breakdown

## 📊 Visualization Functions

### `create_tissue_distribution_chart(probabilities, classes)`
Creates a horizontal bar chart for tissue type distribution.

### `create_subtype_distribution_chart(probabilities, subtypes)`
Creates a bar chart for molecular subtype probabilities.

### `create_tissue_composition_radar(tissue_comp)`
Creates a radar chart showing tissue composition breakdown.

### `create_confidence_gauge(confidence, certainty)`
Creates a gauge chart for confidence visualization.

## 🔌 Integration Examples

### Basic Tissue Analysis
```python
from app.crc_unified_platform import load_models, analyze_tissue_patch
from PIL import Image

# Load models
tissue_model, _, _, _ = load_models()

# Load image
image = Image.open("path/to/tissue.jpg")

# Analyze
results = analyze_tissue_patch(image, tissue_model)
print(f"Tissue type: {results['primary_class']}")
print(f"Confidence: {results['confidence']:.1f}%")
```

### Molecular Subtype Prediction
```python
# Get molecular predictions
molecular_results = molecular_mapper.classify_molecular_subtype(
    image, 
    get_transform()
)
print(f"Predicted subtype: {molecular_results['predicted_subtype']}")
```

### WSI Processing
```python
from app.wsi_handler import load_wsi_region, is_wsi_file

if is_wsi_file("slide.svs"):
    # Load region at 20x magnification
    region = load_wsi_region(
        "slide.svs",
        location=(1000, 1000),
        level=0,
        size=(512, 512)
    )
```

## 🚀 Deployment

### Streamlit Configuration
The platform uses custom Streamlit configuration in `config/.streamlit/config.toml`:
- Dark theme enabled
- Custom color scheme
- Optimized for medical imaging

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Server port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)

### Production Deployment
```bash
# Using Docker
docker build -t crc-platform .
docker run -p 8501:8501 crc-platform

# Using Streamlit Cloud
# Fork repo and deploy via streamlit.io
```

## 📝 Notes

- All confidence scores are returned as percentages (0-100)
- Demo mode provides realistic predictions without model bias
- WSI support includes SVS, NDPI, and other OpenSlide formats
- Report generation requires wkhtmltopdf system package