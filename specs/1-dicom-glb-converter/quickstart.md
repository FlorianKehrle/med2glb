# Quickstart: dicom2glb Development

## Prerequisites

- Python 3.10+
- pip

## Setup

```bash
# Clone the repo
git clone https://github.com/FlorianKehrle/dicom2glb.git
cd dicom2glb

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install in development mode (core dependencies)
pip install -e ".[dev]"

# Install with AI methods (optional, large download)
pip install -e ".[ai,dev]"
```

## Run

```bash
# Basic conversion
dicom2glb ./path/to/dicom/ -o output.glb

# With animation (3D echo)
dicom2glb ./path/to/echo/ -o heart.glb --animate

# List available methods
dicom2glb --list-methods
```

## Test

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/unit/test_mesh_processing.py

# Run with coverage
pytest --cov=dicom2glb
```

## Project Structure

```
src/dicom2glb/
├── cli.py           # CLI entry point
├── io/              # DICOM reading + file export (extensible for future input types)
├── methods/         # Pluggable conversion methods
├── mesh/            # Mesh processing (smoothing, decimation)
├── glb/             # GLB construction + animation
└── core/            # Data structures and types
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| pydicom | Read DICOM files |
| numpy, scipy | Volume processing, Gaussian filtering |
| scikit-image | Marching cubes isosurface extraction |
| trimesh | Mesh smoothing, decimation, repair |
| pyvista | Taubin smoothing, mesh operations |
| pygltflib | GLB export with morph target animation |
| typer + rich | CLI framework with progress bars |

## Adding a New Conversion Method

1. Create `src/dicom2glb/methods/your_method.py`
2. Implement `ConversionMethod` ABC
3. Register with `@register_method("your-method")`
4. The method auto-appears in `--list-methods` and `--method` choices
