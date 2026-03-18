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
- **Decision**: UV unwrap via xatlas + barycentric rasterization into baseColorTexture for static heatmap variants
- **Rationale**: HoloLens 2 (glTFast/MRTK) does NOT render static glTF `COLOR_0` vertex attributes. Baking colors into textures is the only reliable approach for static heatmaps. xatlas provides high-quality UV parameterization. The animated excitation variant no longer requires xatlas since it uses `COLOR_0` morph targets.
- **Implementation Details**:
  - xatlas called once per mesh; result shared across all coloring variants
  - Time estimation: `t = 6.50e-08 * n_faces^1.79` (fitted from real benchmarks)
  - Texture resolution: 512px (≤5K faces), 1024px (≤20K), 2048px (≤80K), 4096px (>80K)
  - Gutter bleeding (10 iterations) eliminates seam artifacts in mipmaps
- **Alternatives considered**:
  - Direct vertex colors (`COLOR_0`) — Not rendered on HoloLens 2.
  - Manual UV layout — xatlas is automatic and high quality.

### R-011: CARTO Animated Excitation via Vertex Color Morph Targets
- **Decision**: Single mesh with `COLOR_0` morph targets animating a glow+ring excitation wavefront, replacing the previous emissive overlay technique
- **Rationale**: The previous approach (30 mesh copies with per-frame emissive textures, scale-toggle visibility) required 30+ draw calls and ~3.5MB of emissive textures. `COLOR_0` morph targets enable a single mesh with GPU-side vertex color interpolation (1 draw call). glTFast on HoloLens 2 supports `COLOR_0` morph targets. The visual model now matches real CARTO 3 behavior: bright leading-edge ring + broad diffuse trailing glow (~25–30% surface coverage).
- **Implementation Details**:
  - Wavefront = Gaussian ring (narrow σ) + exponential decay glow (broad trailing region)
  - Ring tracks LAT activation frontier; glow represents recently-activated tissue fading over time
  - Combined coverage: ~25–30% of mesh surface per frame (matching real CARTO 3)
  - Per-frame vertex colors stored as `COLOR_0` morph targets (VEC4 RGBA)
  - Single mesh, single draw call, weight-based animation (GPU interpolation)
  - No xatlas UV unwrap needed for animation (only for static heatmaps)
  - Loop duration: ~4–5 seconds (matching real CARTO 3 excitation cycle)
- **Alternatives considered**:
  - Emissive overlay (previous approach) — 30+ draw calls, thin ring only, doesn't match real CARTO 3 visual behavior (missing glow). Replaced.
  - Video texture — Not supported in glTF.
  - Per-frame separate meshes — Wastes memory, too many draw calls for AR.

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
