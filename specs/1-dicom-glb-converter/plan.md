# Implementation Plan: DICOM to GLB Converter

**Branch**: `1-dicom-glb-converter` | **Date**: 2026-02-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/1-dicom-glb-converter/spec.md`

## Summary

Build a Python CLI tool (`dicom2glb`) that converts DICOM medical imaging data — primarily 3D echocardiography cardiac loops — to GLB 3D models with animation, optimized for AR viewing. The tool supports pluggable conversion methods (marching-cubes, classical, TotalSegmentator, MedSAM2) and outputs animated GLB using morph targets via pygltflib. Core pipeline: DICOM → volume assembly → segmentation/thresholding → marching cubes → Taubin smoothing → decimation → GLB export with PBR materials and morph target animation.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- Core: pydicom, numpy, scipy, scikit-image, trimesh, pyvista, pygltflib, typer, rich
- Optional AI: totalsegmentator, segment-anything-2 (MedSAM2), torch, monai
**Storage**: Filesystem (DICOM files in → GLB/STL/OBJ files out)
**Testing**: pytest + synthetic DICOM fixtures
**Target Platform**: Cross-platform CLI (Windows, macOS, Linux)
**Project Type**: Single Python package with CLI entry point
**Performance Goals**: Quality over speed; processing time is not a constraint
**Constraints**: Output GLB < 50MB per frame; 50K-100K triangles default; pip-installable without system deps
**Scale/Scope**: Single-user CLI tool, processes one dataset at a time

## Constitution Check

*GATE: No project constitution defined (template only). Proceeding with standard best practices.*

No constitution violations — the constitution file contains only placeholder templates. Standard Python best practices apply:
- Test coverage for core pipeline
- Type hints throughout
- Clear module boundaries
- Optional dependencies via extras

## Project Structure

### Documentation (this feature)

```text
specs/1-dicom-glb-converter/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (CLI contract)
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/dicom2glb/
├── __init__.py              # Package version and public API
├── __main__.py              # python -m dicom2glb support
├── cli.py                   # Typer CLI: entry point, argument parsing, --list-methods
├── io/
│   ├── __init__.py
│   ├── dicom_reader.py      # DICOM loading, series detection, volume assembly
│   ├── echo_reader.py       # Vendor-specific 3D echo (Philips, GE) DICOM handling
│   └── exporters.py         # Multi-format export: GLB (pygltflib), STL, OBJ
├── methods/
│   ├── __init__.py
│   ├── base.py              # ConversionMethod ABC: convert(volume) -> list[Mesh]
│   ├── registry.py          # Method registry: discover, list, instantiate methods
│   ├── marching_cubes.py    # Basic isosurface extraction + Taubin + decimation
│   ├── classical.py         # Full pipeline: Gaussian → adaptive thresh → morphological → Taubin → decimate → fill
│   ├── totalseg.py          # TotalSegmentator wrapper (optional, CT-optimized)
│   └── medsam2.py           # MedSAM2 wrapper (optional, echo/general)
├── mesh/
│   ├── __init__.py
│   ├── processing.py        # Taubin smoothing, quadric decimation, hole filling, normals
│   └── temporal.py          # Temporal vertex smoothing across animation frames
├── glb/
│   ├── __init__.py
│   ├── builder.py           # pygltflib GLB construction: scenes, nodes, meshes, buffers
│   ├── materials.py         # PBR material definitions: cardiac structure color map, alpha
│   └── animation.py         # Morph target creation + glTF animation channels/samplers
└── core/
    ├── __init__.py
    ├── volume.py             # DicomVolume, TemporalSequence data classes
    └── types.py              # Shared types: MeshData, MaterialConfig, MethodParams

tests/
├── conftest.py               # Shared fixtures: synthetic DICOM data, small test volumes
├── unit/
│   ├── test_dicom_reader.py  # DICOM parsing, series grouping, metadata extraction
│   ├── test_mesh_processing.py # Smoothing, decimation, hole filling
│   ├── test_glb_builder.py   # GLB structure validation, material assignment
│   ├── test_animation.py     # Morph target creation, animation channel wiring
│   └── test_methods.py       # Each method produces valid mesh output
└── integration/
    ├── test_pipeline.py      # End-to-end: DICOM dir → GLB file validation
    └── test_cli.py           # CLI argument parsing, --list-methods, error messages

pyproject.toml                # Build config, dependencies, entry points, [ai] extra
README.md                     # Comprehensive documentation
```

**Structure Decision**: Single Python package (`src/dicom2glb/`) with clear module boundaries: `io` (DICOM reading + export), `methods` (pluggable conversion pipelines), `mesh` (shared mesh processing), `glb` (GLB-specific construction), `core` (data structures). This flat structure avoids unnecessary abstraction while keeping concerns separated. The `methods/registry.py` enables the pluggable architecture via a simple decorator pattern.

## Complexity Tracking

No constitution violations to justify.

## Architecture Decisions

### AD-001: Pluggable Method Architecture
- **Pattern**: Registry + Abstract Base Class
- **How**: `ConversionMethod` ABC defines `convert(volume, params) -> list[MeshData]`. Methods register via `@register_method("name")` decorator. `registry.py` discovers and instantiates methods, checks optional dependency availability.
- **Why**: Clean separation, easy to add new methods, graceful degradation when AI deps missing.

### AD-002: GLB Animation via Morph Targets
- **Library**: `pygltflib` (not trimesh — trimesh doesn't support animation)
- **How**: Base mesh = first cardiac phase. Each subsequent phase stored as morph target (vertex displacements). glTF AnimationChannel targets node weights, AnimationSampler provides keyframe times.
- **Constraint**: All frames must share identical mesh topology (same vertex/face count). Marching cubes at different thresholds may produce different topologies — need to use a consistent topology approach (e.g., march on first frame, then deform vertices to match subsequent frames, or use consistent mesh via remeshing).

### AD-003: Consistent Mesh Topology for Animation
- **Approach**: Extract mesh from first time frame. For subsequent frames, use the same mesh topology but update vertex positions by finding nearest-surface-point correspondence. Alternative: use volumetric mesh registration.
- **Why**: glTF morph targets require identical vertex count across all targets. Independent marching cubes per frame would produce different topology.
- **Trade-off**: Some anatomical accuracy loss at extreme deformations vs. guaranteed smooth animation.

### AD-004: Vendor Echo DICOM Reading
- **Approach**: Build on pydicom with vendor-specific decoders referencing SlicerHeart's open-source implementations for Philips and GE 3D echo Cartesian volume extraction.
- **Why**: Standard DICOM readers can't handle proprietary 3D echo encodings.

### AD-005: CLI Framework
- **Library**: `typer` with `rich` for progress bars and colored output
- **Why**: Type-hinted CLI arguments, auto-generated help, built-in shell completion. Rich provides beautiful progress indication for long processing.
