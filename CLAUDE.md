# med2glb Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-02-09

## Active Technologies

- **Language**: Python 3.10+
- **CLI Framework**: typer + rich
- **DICOM**: pydicom
- **Volume Processing**: numpy, scipy, scikit-image
- **Mesh Processing**: trimesh, pyvista
- **GLB Export**: pygltflib (with morph target animation support)
- **AI Segmentation (optional)**: totalsegmentator, MedSAM2, torch, monai
- **Testing**: pytest
- **Build**: pyproject.toml with src layout

## Project Structure

```text
src/med2glb/
├── cli.py              # Typer CLI entry point
├── io/                 # DICOM reading, CARTO reader/mapper, file export
├── methods/            # Pluggable conversion methods (registry pattern)
├── mesh/               # Mesh processing (Taubin smoothing, decimation)
├── glb/                # GLB construction + morph target animation + CARTO wavefront animation
└── core/               # Data structures and types

tests/
├── unit/               # Unit tests per module
├── integration/        # End-to-end pipeline tests
└── conftest.py         # Shared fixtures
```

## Commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Install with AI
pip install -e ".[ai,dev]"

# Run
med2glb ./input/ -o output.glb

# Test
pytest
pytest --cov=med2glb
```

## Code Style

- Type hints on all public functions
- Docstrings on public API
- Use `from __future__ import annotations` for modern type syntax
- Prefer dataclasses for data structures
- ABC for method interface, decorator for registration

## Recent Changes

- **carto-support**: CARTO 3 EP mapping support — .mesh/.car parser, per-vertex COLOR_0 heatmaps (LAT/bipolar/unipolar), animated LAT wavefront sweep, auto-detection
- **1-dicom-glb-converter**: Initial feature — DICOM to GLB conversion with pluggable methods, morph target animation, cardiac segmentation

<!-- MANUAL ADDITIONS START -->

## Maintenance Rules

- **README.md**: When adding features, changing CLI options, or modifying user-facing workflow, update README.md to reflect the changes. Keep it in sync with actual capabilities.

<!-- MANUAL ADDITIONS END -->
