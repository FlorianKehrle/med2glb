# Quickstart: med2glb Development

## Prerequisites

- Python 3.10+
- pip

## Setup

```bash
# Clone the repo
git clone https://github.com/FlorianKehrle/med2glb.git
cd med2glb

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install in development mode (core dependencies)
pip install -e ".[dev]"

# Install with AI methods (optional, large download)
pip install -e ".[ai,dev]"
```

### Optional: KTX2 Texture Compression

For best GLB compression, install the [Khronos KTX-Software](https://github.com/KhronosGroup/KTX-Software/releases) (`toktx`). Without it, `--compress` falls back to texture downscaling.

## Run

```bash
# Interactive wizard (auto-detects data type)
med2glb ./path/to/data/

# CARTO conversion
med2glb ./Export_Study/ --animate --vectors --subdivide 2

# DICOM conversion
med2glb ./echo_folder/ -o heart.glb --animate

# Batch CARTO
med2glb ../CARTO_exports/ --batch --animate --vectors

# Gallery mode
med2glb ./dicom_folder/ --gallery

# GLB compression
med2glb large.glb --compress --max-size 10

# List available methods
med2glb --list-methods
```

## Test

```bash
# Run all tests
pytest

# Run unit tests only (fast)
pytest tests/unit/ -x -q

# Run specific test module
pytest tests/unit/test_carto_reader.py

# Run with coverage
pytest --cov=med2glb

# Skip integration tests
pytest -m "not integration"
```

## Project Structure

```
src/med2glb/
├── cli.py              # CLI entry point
├── cli_wizard.py       # Interactive wizard (data-driven prompts)
├── _pipeline_carto.py  # CARTO conversion pipeline
├── _pipeline_dicom.py  # DICOM conversion pipeline
├── _pipeline_gallery.py # Gallery mode pipeline
├── core/               # Data structures and types
├── io/                 # DICOM reading, CARTO reader/mapper, colormaps, export
├── methods/            # Pluggable conversion methods (registry pattern)
├── mesh/               # Mesh processing (smoothing, decimation, LAT vectors)
├── glb/                # GLB construction, animation, textures, compression
└── gallery/            # Gallery mode (individual, lightbox, spatial)
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| pydicom | Read DICOM files |
| numpy, scipy | Volume processing, spatial indexing (KDTree), filtering |
| scikit-image | Marching cubes isosurface extraction |
| trimesh | Mesh smoothing, decimation, Loop subdivision |
| pygltflib | GLB export with morph targets, emissive animation, PBR materials |
| xatlas | UV unwrapping for CARTO texture baking |
| typer + rich | CLI framework with interactive wizard and progress bars |
| totalsegmentator (optional) | AI cardiac CT segmentation |

## Adding a New Conversion Method

1. Create `src/med2glb/methods/your_method.py`
2. Implement `ConversionMethod` ABC (define `convert()`, `supports_animation()`, `check_dependencies()`)
3. Register with `@register_method("your-method")`
4. The method auto-appears in `--list-methods` and `--method` choices

## Adding a New CARTO Coloring Scheme

1. Add colormap function in `src/med2glb/io/carto_colormaps.py`
2. Add field extraction in `src/med2glb/io/carto_mapper.py`
3. Register scheme name in the CARTO pipeline and wizard
