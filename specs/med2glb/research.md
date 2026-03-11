# Research: med2glb — Medical Imaging to GLB Converter

**Date**: 2026-02-09 (initial), 2026-03-11 (updated)
**Feature**: med2glb

## Decision Log

### R-001: GLB Animation Library
- **Decision**: `pygltflib` for GLB export with morph target animations and emissive overlay animations
- **Rationale**: Only Python library with full glTF 2.0 spec support including animations, morph targets, PBR materials, and `KHR_materials_unlit`. Low-level but complete control over the output.
- **Alternatives considered**:
  - `trimesh` — Excellent for mesh processing (smoothing, decimation) but does NOT support GLB animation export. Used for mesh processing only.
  - `PyVista` — Has `export_gltf()` but no animation support and limited material control.
  - `gltflib` — Similar to pygltflib but less maintained.

### R-002: Mesh Processing Pipeline
- **Decision**: scikit-image `marching_cubes` → trimesh for Taubin smoothing → quadric edge collapse decimation
- **Rationale**: Taubin smoothing preserves volume (critical for cardiac chambers — Laplacian shrinks them). Quadric edge collapse decimation preserves shape better than uniform decimation.
- **Alternatives considered**:
  - VTK `vtkMarchingCubes` + `vtkWindowedSincPolyDataFilter` — Equivalent quality but heavier dependency.
  - Surface Nets — Smoother initial output but less control over topology.

### R-003: 3D Echo Segmentation Model
- **Decision**: MedSAM2 as the primary AI segmentation for echo data
- **Rationale**: Most versatile open-source model (2025). Handles both 3D volumes and video, achieving 96.1% LV / 93.1% LV epi / 95.8% LA Dice on echo data. Code, weights, and datasets all publicly available.
- **Alternatives considered**:
  - EchoNet-Dynamic — 2D echo only, not true 3D volumes.
  - SimLVSeg — 2D+time LV only.
  - MONAI Auto3DSeg — Narrower scope (valve-specific).

### R-004: CT Segmentation Model
- **Decision**: TotalSegmentator for cardiac CT segmentation
- **Rationale**: Most mature option. nnU-Net-based, 117+ structures including cardiac. Feb 2025 update added more cardiovascular structures. Pip-installable.
- **Alternatives considered**:
  - MONAI VISTA3D — 127 automatic classes but more experimental.
  - Custom nnU-Net — Would need labeled training data.

### R-005: Vendor-Specific 3D Echo DICOM
- **Decision**: Custom readers referencing SlicerHeart implementations for Philips and GE
- **Rationale**: Standard pydicom can't decode vendor-specific 3D echo volume encodings. SlicerHeart provides reference implementations.
- **Alternatives considered**:
  - Standard DICOM tags only — Misses most real-world 3D echo data.
  - 3D Slicer as pre-processing — Too heavy, breaks CLI workflow.

### R-006: Consistent Mesh Topology for DICOM Animation
- **Decision**: Extract mesh from first frame, deform to match subsequent frames via nearest-surface-point correspondence (cKDTree)
- **Rationale**: glTF morph targets require identical vertex count across all targets. Independent marching cubes per frame produces incompatible topologies.
- **Alternatives considered**:
  - Independent marching cubes per frame — Incompatible with morph targets.
  - Volumetric mesh registration — Higher quality but significantly more complex.

### R-007: AR Platform Compatibility
- **Decision**: Target GLB as primary format. HoloLens 2 as primary AR device.
- **Rationale**:
  - HoloLens 2: Full PBR + animation support via glTFast/MRTK.
  - Android Scene Viewer: Natively supports animated GLB with morph targets.
  - Web: Google's `<model-viewer>` plays animated GLB on both platforms.
  - iOS: Apple Quick Look does NOT support GLB or morph target animations in USDZ.
- **Alternatives considered**:
  - USDZ for iOS — Would lose morph target animation.
  - Dual export — Added complexity for limited gain.

### R-008: CLI Framework
- **Decision**: `typer` + `rich` with interactive wizard
- **Rationale**: Typer provides type-hinted CLI with auto-generated help. Rich provides progress bars, tables, and colored output. The interactive wizard guides users through data-specific options without requiring CLI expertise.
- **Alternatives considered**:
  - `click` — More boilerplate, no type hints.
  - `argparse` — No progress bars, verbose.

### R-009: AR-Optimized Mesh Parameters
- **Decision**: Default 50K-100K triangles, Taubin smoothing (15 iterations), PBR materials with `metallicFactor: 0.0`, `roughnessFactor: 0.7`, `alphaMode: BLEND`
- **Rationale**: Optimal for single close-up AR models on mobile/HoloLens. BLEND alpha mode has broadest viewer support.

### R-010: CARTO Vertex Color Baking via xatlas
- **Decision**: UV unwrap via xatlas + barycentric rasterization into baseColorTexture
- **Rationale**: HoloLens 2 (glTFast/MRTK) does NOT render glTF `COLOR_0` vertex attributes. Baking colors into textures is the only reliable cross-platform approach. xatlas provides high-quality UV parameterization.
- **Implementation Details**:
  - xatlas called once per mesh; result shared across all coloring variants
  - Time estimation: `t = 6.50e-08 * n_faces^1.79` (fitted from real benchmarks)
  - Texture resolution: 512px (≤5K faces), 1024px (≤20K), 2048px (≤80K), 4096px (>80K)
  - Gutter bleeding (10 iterations) eliminates seam artifacts in mipmaps
- **Alternatives considered**:
  - Direct vertex colors (`COLOR_0`) — Not rendered on HoloLens 2.
  - Manual UV layout — xatlas is automatic and high quality.

### R-011: CARTO Animated Excitation via Emissive Overlay
- **Decision**: Per-frame emissive textures with Gaussian ring, frame visibility via scale animation
- **Rationale**: The emissive overlay technique (base color texture + per-frame emissive textures) works universally on HoloLens 2 via glTFast/MRTK. Using node scale [1,1,1]/[0,0,0] for frame switching is simpler and more compatible than weight-based morph targets for texture-driven animation.
- **Implementation Details**:
  - Ring width (σ): 0.025 (narrow Gaussian band)
  - Highlight color: [0.55, 0.55, 0.55] additive white
  - Ring position: `exp(-((lat_norm - t)² / (2σ²)))` per frame
  - Default: 30 frames, 2s duration → 15fps loop
  - Full mesh geometry shared across all frames; only emissive texture differs
- **Alternatives considered**:
  - Morph targets for color animation — Not supported for texture coordinates.
  - Video texture — Not supported in glTF.
  - Per-frame separate meshes — Wastes memory, solved by shared geometry + emissive.

### R-012: LAT Streamline Vectors via Gradient Field
- **Decision**: Analytic per-face gradient (Gram matrix solve) → vertex averaging → streamline tracing → animated dash geometry
- **Rationale**: Provides physically meaningful conduction direction from LAT activation times. Gradient computation is exact within each triangle. Momentum coasting handles sparse data regions.
- **Implementation Details**:
  - Face gradients: 2×2 Gram matrix solve per triangle (e1·e1, e1·e2 system)
  - Vertex gradients: area-weighted accumulation with tangent-plane projection
  - Streamline tracing: half-edge traversal + barycentric intersection
  - Momentum coasting: coast up to 20 steps when gradient is zero (for sparse data)
  - Seeding: ~300 seeds via grid-based spatial partitioning (highest gradient per cell)
  - Dash animation: 8% of median streamline length, speed scaled by local gradient magnitude
  - Quality gate: min 10 streamlines with ≥3 points, median length ≥2% of mesh bbox
- **Alternatives considered**:
  - Finite differences — Less accurate on irregular triangulations.
  - Particle tracing — More complex, similar visual result.

### R-013: Loop Subdivision for CARTO Meshes
- **Decision**: `trimesh.remesh.subdivide_loop` with pre-subdivision singularity smoothing and KDTree-based metadata propagation
- **Rationale**: Loop subdivision (~4× faces per level) produces smooth gradients via k-NN IDW interpolation instead of blocky nearest-neighbor on the original mesh. Pre-subdivision smoothing snaps spike vertices to prevent subdivision artifacts.
- **Implementation Details**:
  - Strip inactive faces (group_id=-1000000) before subdivision
  - Snap spike vertices (Laplacian displacement > 10× median) to neighbor average
  - Post-subdivision: propagate group_ids via nearest-vertex KDTree lookup
  - Default level 2: ~16× face increase
- **Alternatives considered**:
  - Catmull-Clark — Not suitable for triangle meshes.
  - Simple midpoint subdivision — No smoothing, produces flat facets.

### R-014: CARTO File Format Support
- **Decision**: Support CARTO v4 (~2015), v5 (v7.1), v6 (v7.2+) with vectorized parsing
- **Rationale**: Clinical CARTO exports span multiple system versions. The `.mesh` INI-style format and tab-separated `_car.txt` are consistent enough to parse with a single reader, with version-specific LAT sentinel handling.
- **Implementation Details**:
  - `.mesh`: INI-style with [GeneralAttributes], [VerticesSection], [TrianglesSection]
  - `_car.txt`: Tab-separated with VERSION_X_Y header for version detection
  - Bulk loading via `numpy.genfromtxt` for performance
  - LAT sentinel -9999.0 converted to NaN
  - CARTO_INACTIVE_GROUP_ID = -1000000

### R-015: GLB Compression Strategies
- **Decision**: Four-strategy system: KTX2 (default), Draco, downscale, JPEG
- **Rationale**: Different strategies suit different GLB content. KTX2 gives best compression for texture-heavy CARTO files. Draco works for geometry-heavy DICOM meshes without animation. Downscale/JPEG are universal fallbacks.
- **Implementation Details**:
  - KTX2: UASTC + Zstandard-5 via external `toktx` tool
  - Draco: trimesh integration, skips if animations present
  - Downscale: progressive halving (75%, 50%, 37.5%, 25%)
  - JPEG: quality 90→30, scale 1.0→0.25
  - All strategies maintain 4-byte buffer alignment

### R-016: Legend and Info Panel Embedding
- **Decision**: Embedded geometry nodes with `KHR_materials_unlit` textures
- **Rationale**: Self-contained GLB with clinical context — no external overlay or separate legend file needed. Unlit materials ensure readability regardless of AR scene lighting.
- **Implementation Details**:
  - Legend: 32-segment cylinder, 512×256 texture with gradient + labels
  - Info panel: double-sided quad, 320×224 texture with study metadata
  - Positioned outside mesh bounding sphere to avoid occlusion
  - `gltf.extras` dict stores metadata for programmatic access
