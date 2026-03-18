# Implementation Plan: med2glb — Medical Imaging to GLB Converter

**Branch**: `main` | **Date**: 2026-02-09 (initial), 2026-03-18 (updated) | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/med2glb/spec.md`
**Status**: Implemented — this plan documents the delivered architecture.

## Summary

Python CLI tool (`med2glb`) that converts medical imaging data — DICOM (cardiac CT, MRI, 3D echo) and CARTO 3 electro-anatomical mapping exports — to GLB 3D models optimized for AR viewing on HoloLens 2 and other AR/MR devices. Features an interactive wizard, pluggable DICOM conversion methods, CARTO clinical heatmaps with animated excitation wavefronts (glow + ring via vertex color morph targets) and conduction vectors, gallery mode for DICOM slices, and GLB compression.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- Core: pydicom, numpy, scipy, scikit-image, trimesh, pygltflib, typer, rich, xatlas
- Optional AI: totalsegmentator, torch
**Storage**: Filesystem (DICOM/CARTO files in → GLB files out)
**Testing**: pytest + synthetic fixtures
**Target Platform**: Cross-platform CLI (Windows, macOS, Linux); primary AR target: HoloLens 2
**Project Type**: Single Python package with CLI entry point
**Performance Goals**: Quality over speed; offline conversion tool
**Constraints**: Output GLB practical for AR devices; 50K-100K triangles default; pip-installable without system deps
**Scale/Scope**: Single-user CLI tool, processes one dataset at a time (batch mode for CARTO)

## Project Structure

### Documentation (this feature)

```text
specs/med2glb/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Technical decisions
├── data-model.md        # Data model definitions
├── quickstart.md        # Developer quickstart
├── contracts/
│   └── cli-contract.md  # CLI contract
└── checklists/
    └── requirements.md  # Specification quality checklist
```

### Source Code (repository root)

```text
src/med2glb/
├── __init__.py              # Package version
├── __main__.py              # python -m med2glb support
├── _console.py              # Rich console utilities
├── cli.py                   # Typer CLI entry point, option parsing, pipeline dispatch
├── cli_wizard.py            # Interactive wizard (data-driven prompts, quality assessment)
├── _pipeline_carto.py       # CARTO conversion pipeline orchestration
├── _pipeline_dicom.py       # DICOM conversion pipeline orchestration
├── _pipeline_gallery.py     # Gallery mode pipeline
├── core/
│   ├── types.py             # All dataclasses (MeshData, CartoStudy, configs, etc.)
│   └── volume.py            # DicomVolume, TemporalSequence
├── io/
│   ├── carto_reader.py      # Parse .mesh + _car.txt files (v4, v5, v6)
│   ├── carto_mapper.py      # Sparse-to-dense mapping (KDTree, IDW), Loop subdivision
│   ├── carto_colormaps.py   # Clinical colormaps (LAT, bipolar, unipolar)
│   ├── dicom_reader.py      # DICOM series analysis, volume assembly
│   ├── echo_reader.py       # Vendor-specific 3D echo (Philips, GE)
│   ├── exporters.py         # Multi-format export (OBJ, etc.)
│   └── conversion_log.py    # Conversion statistics logging
├── methods/
│   ├── base.py              # ConversionMethod ABC
│   ├── registry.py          # @register_method decorator, discovery
│   ├── marching_cubes.py    # Basic isosurface + multi-threshold
│   ├── classical.py         # Region-growing + Otsu segmentation
│   ├── totalseg.py          # TotalSegmentator AI wrapper
│   └── chamber_detect.py    # Cardiac chamber detection
├── mesh/
│   ├── processing.py        # Taubin smoothing, quadric decimation, normals
│   ├── temporal.py          # Morph target animation from frame sequences
│   └── lat_vectors.py       # LAT gradient, streamline tracing, dash animation
├── glb/
│   ├── builder.py           # Core GLB construction with PBR materials
│   ├── carto_builder.py     # CARTO GLB with textures (static) + vertex color morph targets (animated)
│   ├── vertex_color_bake.py # xatlas UV unwrap + barycentric rasterization
│   ├── animation.py         # Morph target animation builder
│   ├── arrow_builder.py     # LAT streamline arrow/dash geometry
│   ├── legend_builder.py    # Color legend + info panel nodes
│   ├── materials.py         # PBR material definitions (cardiac color map)
│   ├── texture.py           # Textured plane (DICOM image as quad)
│   └── compress.py          # GLB compression (Draco, KTX2, downscale, JPEG)
└── gallery/
    ├── _glb_utils.py        # Shared quad/texture utilities
    ├── individual.py        # One GLB per slice
    ├── lightbox.py          # Grid layout GLB
    ├── loader.py            # DICOM slice loader (no shape filtering)
    └── spatial.py           # Spatial fan GLB (real-world positions)

tests/
├── conftest.py              # Shared fixtures (synthetic DICOM, CARTO, meshes)
├── unit/
│   ├── test_carto_reader.py
│   ├── test_carto_mapper.py
│   ├── test_carto_glb.py
│   ├── test_arrow_builder.py
│   ├── test_legend_builder.py
│   ├── test_lat_vectors.py
│   ├── test_vertex_color_bake.py
│   ├── test_glb_builder.py
│   ├── test_gallery.py
│   ├── test_dicom_reader.py
│   ├── test_methods.py
│   ├── test_mesh_processing.py
│   ├── test_pipeline_dicom.py
│   ├── test_compress.py
│   └── test_cli_wizard.py
└── integration/
    ├── conftest.py
    ├── test_cli.py
    ├── test_pipeline.py
    ├── test_carto_pipeline.py
    ├── test_animated_cache.py
    └── test_dicom_real.py

pyproject.toml               # Build config, dependencies, entry points, [ai] extra
README.md                    # Comprehensive user documentation
CLAUDE.md                    # AI development guidelines
```

## Architecture Decisions

### AD-001: Pluggable Method Architecture
- **Pattern**: Registry + Abstract Base Class
- **How**: `ConversionMethod` ABC defines `convert(volume, params) -> ConversionResult`. Methods register via `@register_method("name")` decorator. `registry.py` discovers and instantiates.
- **Why**: Clean separation, easy to add new methods, graceful degradation when AI deps missing.

### AD-002: GLB Animation — Morph Targets for DICOM
- **Library**: `pygltflib` (trimesh doesn't support animation)
- **How**: Base mesh = first cardiac phase. Subsequent phases stored as morph targets (vertex displacements). Consistent topology via cKDTree nearest-surface-point correspondence.

### AD-003: GLB Animation — Vertex Color Morph Targets for CARTO
- **How**: Single mesh with `COLOR_0` morph targets. Per-frame vertex colors computed from LAT timing (ring = Gaussian at activation frontier, glow = exponential decay behind frontier). Weight-based animation with GPU interpolation.
- **Why**: Replaces previous emissive overlay (30 mesh copies, 30+ draw calls, thin ring only). Single draw call, matches real CARTO 3 visual behavior (glow + ring, ~25–30% surface coverage), glTFast supports `COLOR_0` morph targets.

### AD-004: Texture Baking via xatlas (CARTO Static Heatmaps)
- **How**: xatlas UV unwrap → barycentric rasterization → baseColorTexture with gutter bleeding
- **Why**: HoloLens 2 glTFast does NOT render static `COLOR_0` vertex attributes. UV unwrap computed once, shared across static heatmap variants (LAT, bipolar, unipolar). The animated excitation variant does not require xatlas — it uses `COLOR_0` morph targets instead.

### AD-005: Interactive Wizard
- **How**: `cli_wizard.py` detects data type, shows Rich tables, prompts for relevant options. Bypassed when any pipeline flag is provided or in non-interactive mode.
- **Why**: Most users don't know the optimal settings. Data-driven prompts with smart defaults.

### AD-006: CARTO Multi-Version Support
- **How**: Single parser handles v4 (~2015), v5 (v7.1), v6 (v7.2+). VERSION header detection in `_car.txt`. Vectorized parsing via numpy.
- **Why**: Clinical CARTO exports span multiple system versions.

### AD-007: Sparse-to-Dense CARTO Mapping
- **How**: KDTree + k-NN IDW (k=6, power=2.0) maps ~100-500 measurement points to dense per-vertex fields. Loop subdivision (16× at level 2) produces smooth gradients.
- **Why**: CARTO exports have sparse measurement points that need smooth interpolation for clinical visualization.

### AD-008: LAT Conduction Vectors
- **How**: Per-face Gram-matrix gradient → vertex averaging → streamline tracing (half-edge traversal) → animated dash geometry. Quality gate prevents vectors on unsuitable data.
- **Why**: Physically meaningful conduction direction visualization.

### AD-009: Gallery Mode
- **How**: Separate module (`gallery/`) with three output modes: individual GLBs, lightbox grid, spatial fan. Frame-switching animation via scale toggle.
- **Why**: 2D imaging data presented as AR-ready textured quads with spatial awareness.

### AD-010: GLB Compression
- **How**: Four strategies (KTX2, Draco, downscale, JPEG) with automatic fallback. Draco skips animated GLBs. KTX2 requires external `toktx`.
- **Why**: AR devices have file size constraints. Different content types benefit from different strategies.
