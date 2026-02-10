# Research: DICOM to GLB Converter

**Date**: 2026-02-09
**Feature**: 1-dicom-glb-converter

## Decision Log

### R-001: GLB Animation Library
- **Decision**: `pygltflib` for GLB export with morph target animations
- **Rationale**: Only Python library with full glTF 2.0 spec support including animations, morph targets, and PBR materials. Low-level but complete control over the output.
- **Alternatives considered**:
  - `trimesh` — Excellent for mesh processing (smoothing, decimation) but does NOT support GLB animation export. Used for mesh processing only.
  - `PyVista` — Has `export_gltf()` but no animation support and limited material control.
  - `gltflib` — Similar to pygltflib but less maintained.

### R-002: Mesh Processing Pipeline
- **Decision**: scikit-image `marching_cubes` → PyVista/trimesh for Taubin smoothing → `fast-simplification` or PyVista for quadric decimation
- **Rationale**: Taubin smoothing preserves volume (critical for cardiac chambers — Laplacian shrinks them). Quadric edge collapse decimation preserves shape better than uniform decimation. `fast-simplification` is 4-5x faster than VTK decimation.
- **Alternatives considered**:
  - VTK `vtkMarchingCubes` + `vtkWindowedSincPolyDataFilter` — Equivalent quality but heavier dependency.
  - PyMCubes — Fast marching cubes but less ecosystem integration.
  - Surface Nets — Smoother initial output but less control over topology.

### R-003: 3D Echo Segmentation Model
- **Decision**: MedSAM2 as the primary AI segmentation for echo data
- **Rationale**: Most versatile open-source model (2025). Handles both 3D volumes and video. Achieved 96.1% LV / 93.1% LV epi / 95.8% LA Dice on echo data. Reduced annotation time by 91.8%. Code, weights, and datasets all publicly available.
- **Alternatives considered**:
  - EchoNet-Dynamic — Mature but 2D echo only (A4C views), not true 3D volumes.
  - SimLVSeg — State-of-art for 2D+time LV segmentation but not 3D volumetric.
  - MONAI Auto3DSeg — Won MVSEG2023 mitral valve challenge on 3D TEE but narrower scope (valve-specific).
  - MemSAM — SAM adapted for echo video (CVPR 2024 Oral) but 2D only.

### R-004: CT Segmentation Model
- **Decision**: TotalSegmentator for cardiac CT segmentation
- **Rationale**: Most mature option. nnU-Net-based, 117+ structures including cardiac (LA, RA, aorta, pulmonary artery). Feb 2025 update added more cardiovascular structures. Pip-installable, well-maintained. Strong community.
- **Alternatives considered**:
  - MONAI VISTA3D — 127 automatic classes but CT only and more experimental.
  - Custom nnU-Net — Would need labeled training data and training pipeline.

### R-005: Vendor-Specific 3D Echo DICOM
- **Decision**: Custom readers referencing SlicerHeart implementations for Philips and GE
- **Rationale**: Standard pydicom can't decode vendor-specific 3D echo volume encodings. SlicerHeart (open-source, peer-reviewed, actively maintained) provides reference implementations for Philips iE33/EPIQ and GE Vivid Cartesian volume extraction.
- **Alternatives considered**:
  - Relying on DICOM standard tags only — Would miss most real-world 3D echo data which uses vendor-specific private tags.
  - Using 3D Slicer as a pre-processing step — Too heavy, requires GUI, breaks CLI workflow.

### R-006: Consistent Mesh Topology for Animation
- **Decision**: Extract mesh from first frame, then deform to match subsequent frames via nearest-surface-point correspondence
- **Rationale**: glTF morph targets require identical vertex count across all targets. Running marching cubes independently per frame produces different topologies (different vertex/face counts). By fixing topology from frame 1 and deforming vertices, we guarantee compatible morph targets.
- **Alternatives considered**:
  - Independent marching cubes per frame — Simplest but incompatible with morph targets.
  - Volumetric mesh registration — Higher quality deformation but significantly more complex.
  - Re-meshing each frame to match target topology — Possible via trimesh but introduces artifacts.

### R-007: AR Platform Compatibility
- **Decision**: Target GLB as primary format. Document AR viewer compatibility in README.
- **Rationale**:
  - Android Scene Viewer: Natively supports animated GLB with morph targets.
  - iOS: Apple Quick Look does NOT support GLB and does NOT support morph target animations in USDZ. Recommend `<model-viewer>` web component for cross-platform.
  - Web: Google's `<model-viewer>` plays animated GLB on both platforms.
- **Alternatives considered**:
  - USDZ export for iOS — Would lose morph target animation (Apple doesn't support it).
  - Dual export (GLB + USDZ) — Added complexity for limited gain since USDZ can't animate.

### R-008: CLI Framework
- **Decision**: `typer` + `rich`
- **Rationale**: Typer provides type-hinted CLI with auto-generated help, shell completion, and clean argument parsing. Rich provides beautiful progress bars and colored output for long-running medical image processing. Both are lightweight and pip-installable.
- **Alternatives considered**:
  - `click` — More established but requires more boilerplate. Typer is built on click.
  - `argparse` — Standard library but verbose, no type hints, no progress bars.

### R-009: AR-Optimized Mesh Parameters
- **Decision**: Default 50K-100K triangles, Taubin smoothing (10-20 iterations), PBR materials with `metallicFactor: 0.0`, `roughnessFactor: 0.6-0.8`, `alphaMode: BLEND`
- **Rationale**: Industry research shows 50K-100K triangles is optimal for single close-up AR models on mobile devices. Apple Quick Look caps at ~500K vertices. PBR metallic=0 with moderate roughness gives realistic tissue appearance. BLEND alpha mode has broadest AR viewer support for transparency.
- **Alternatives considered**:
  - Higher poly counts (200K+) — Risk AR performance issues on mobile.
  - `KHR_materials_transmission` for glass-like transparency — Very limited AR viewer support.
