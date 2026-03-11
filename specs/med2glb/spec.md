# Feature Specification: med2glb — Medical Imaging to GLB Converter

**Feature Branch**: `main`
**Created**: 2026-02-09
**Updated**: 2026-03-11
**Status**: Implemented
**Input**: User description: "Python CLI tool for converting medical imaging data — DICOM (cardiac CT, MRI, 3D echocardiography) and CARTO 3 electro-anatomical mapping exports — to GLB 3D models optimized for augmented reality viewing, with pluggable conversion methods, animated output, clinical heatmaps, and an interactive wizard."

## Research Findings

The following findings from a comprehensive survey of existing tools and the state of the art inform this specification:

1. **No existing end-to-end CLI tool exists** for medical imaging to animated GLB, especially for 3D echo or CARTO EP mapping. Existing tools (3D Slicer, DicomToMesh, InVesalius) produce static STL/OBJ via GUI. SlicerHeart handles 3D echo DICOM reading but exports to STL for 3D printing, not animated GLB for AR.

2. **Animation in GLB requires morph targets** built via `pygltflib`. The commonly-used `trimesh` library does NOT support GLB animations. Each cardiac phase becomes a morph target, and the glTF animation system interpolates between them. `trimesh` remains useful for mesh processing (smoothing, decimation) before final export.

3. **HoloLens 2 (glTFast/MRTK) does not render glTF `COLOR_0` vertex attributes**, requiring vertex colors to be baked into textures via xatlas UV unwrap + rasterization as a workaround. This is critical for CARTO heatmap visualization.

4. **CARTO 3 EP mapping exports** use proprietary `.mesh` (INI-style geometry) and `_car.txt` (measurement points) file formats. Multiple format versions exist (v4 ~2015, v5/v7.1, v6/v7.2+). No existing tool converts these to AR-ready GLB with animated clinical heatmaps.

5. **3D echo segmentation**: MedSAM2 (2025) is the most versatile open-source model — handles 3D volumes and video, achieving 96.1% LV / 95.8% LA Dice on echo data.

6. **CT segmentation**: TotalSegmentator (nnU-Net-based, 117+ structures including cardiac chambers, aorta, pulmonary artery) is the most mature option. Feb 2025 update added more cardiovascular structures.

7. **Vendor-specific 3D echo DICOM** (Philips, GE) requires special handling. SlicerHeart provides reference implementations.

8. **Optimal AR mesh**: 50K-100K triangles for a single close-up model. Apple Quick Look caps at ~500K vertices.

9. **Recommended mesh pipeline**: Marching cubes → Taubin smoothing (preserves volume, unlike Laplacian which shrinks chambers) → Quadric edge collapse decimation → Final light Taubin smooth.

10. **CARTO color mapping** requires sparse-to-dense interpolation: k-NN inverse-distance weighting (IDW) maps ~100-500 measurement points to dense per-vertex fields. Loop subdivision (~16× faces at level 2) produces smooth color gradients.

11. **Animated excitation visualization** uses emissive overlay technique — frame-by-frame emissive textures with a sweeping ring that tracks LAT activation timing. This works universally on HoloLens 2 (glTFast/MRTK).

12. **LAT conduction vectors** require per-face gradient computation via analytic Gram-matrix solve, streamline tracing via half-edge traversal + barycentric intersection, and Gaussian smoothing to prevent jitter at mesh boundaries.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Convert CARTO 3 EP Mapping to Clinical AR Heatmaps (Priority: P0)

An electrophysiologist has a CARTO 3 export directory containing `.mesh` geometry and `_car.txt` measurement files from a cardiac ablation procedure. They want to convert this into GLB files with clinical voltage/LAT heatmaps viewable in AR on HoloLens, with animated excitation rings showing activation wavefronts and optional streamline arrows showing conduction direction.

**Why this priority**: CARTO EP mapping with animated heatmaps is the primary clinical use case, targeting HoloLens AR workflows for procedural planning and review.

**Independent Test**: Provide a CARTO export directory, run `med2glb ./Export_Study/`, use the interactive wizard to select meshes and options, verify GLB files show correct heatmaps in HoloLens.

**Acceptance Scenarios**:

1. **Given** a CARTO 3 export directory with `.mesh` and `_car.txt` files, **When** the user runs med2glb and follows the wizard, **Then** GLB files are produced for each selected coloring scheme (LAT, bipolar, unipolar) with correct clinical colormaps baked as textures.
2. **Given** a CARTO export, **When** animation is enabled, **Then** an animated GLB is produced with a sweeping excitation ring overlay that tracks LAT activation timing across 30 frames.
3. **Given** a CARTO export with sufficient LAT data (≥30 points, ≥20ms range, ≥15% gradient coverage), **When** vectors are enabled, **Then** animated streamline arrows show conduction direction with dashes advancing in sync with the excitation ring.
4. **Given** a CARTO export, **When** subdivision level 2 is selected, **Then** the mesh is Loop-subdivided (~16× faces) with k-NN IDW interpolation producing smooth color gradients instead of blocky nearest-neighbor.
5. **Given** CARTO exports from different versions (v4/2015, v7.1, v7.2+), **When** processed, **Then** all versions are correctly parsed and converted.
6. **Given** a parent directory containing multiple CARTO exports, **When** batch mode is used, **Then** all exports are processed with shared settings and output goes to per-export `glb/` subfolders.
7. **Given** a CARTO GLB, **When** viewed on HoloLens 2, **Then** heatmaps render correctly via baked textures (not vertex colors) and a color legend with study metadata is embedded.

---

### User Story 2 — Convert 3D Echo Cardiac Loop to Animated GLB (Priority: P1)

A cardiologist or researcher has 3D echocardiography data consisting of multiple DICOM files representing a cardiac cycle (3D volumes over time). They want to convert this 4D dataset into a single animated GLB file showing the beating heart, viewable in augmented reality.

**Why this priority**: 3D echo with animation is the primary DICOM use case. This drives the DICOM pipeline architecture — if this works well, the simpler cases (static CT, single slices) are subsets.

**Independent Test**: Provide 3D echo DICOM files, run the CLI with animation enabled, open the resulting GLB in an AR viewer to see a smooth cardiac loop.

**Acceptance Scenarios**:

1. **Given** a directory of 3D echo DICOM files representing a full cardiac cycle, **When** the user runs the converter with animation enabled, **Then** a single animated GLB file is produced using morph targets with smooth mesh transitions between time frames.
2. **Given** a 3D echo dataset, **When** the user switches between different methods (e.g., `marching-cubes`, `classical`), **Then** each produces a valid output with visibly different quality characteristics.
3. **Given** a 3D echo animated GLB output, **When** viewed in an AR application, **Then** the animation loops smoothly without visible temporal flickering or mesh jumping between frames.
4. **Given** 3D echo data from Philips or GE ultrasound systems, **When** provided, **Then** the system correctly reads and assembles the 3D volumes.

---

### User Story 3 — Convert Cardiac CT/MRI Volume to Segmented GLB (Priority: P2)

A medical professional has a cardiac CT or MRI scan. They want to convert this into a GLB file where different cardiac structures are individually colored and some are semi-transparent.

**Why this priority**: Static 3D volume conversion with segmentation is the second most common DICOM use case.

**Independent Test**: Provide cardiac CT DICOM slices, run the converter with `--method totalseg`, verify the GLB contains separately colored structures.

**Acceptance Scenarios**:

1. **Given** a cardiac CT dataset, **When** the user uses `totalseg` method, **Then** individual cardiac structures are extracted as separate meshes with distinct colors and configurable transparency.
2. **Given** a cardiac CT dataset, **When** the user uses `chamber-detect` method, **Then** myocardium and up to four blood pool chambers are detected and exported with per-structure PBR materials.
3. **Given** any conversion method, **When** the user specifies a target face count, **Then** the output mesh is decimated to approximately that count while preserving important anatomical detail.

---

### User Story 4 — Interactive Wizard Experience (Priority: P1)

A user who is unfamiliar with the tool's options points med2glb at a data directory and is guided through relevant choices via an interactive wizard with smart defaults.

**Why this priority**: The wizard is the primary user experience — most users don't know which options to set. A good wizard eliminates the need to read documentation for basic usage.

**Independent Test**: Run `med2glb ./data/` on a TTY terminal, verify the wizard detects data type, presents relevant prompts, and converts with selected options.

**Acceptance Scenarios**:

1. **Given** a directory containing CARTO data, **When** the user runs med2glb interactively, **Then** the wizard shows a summary table with mesh stats (vertices, triangles, points, LAT range, dimensions, estimated time) and prompts for mesh selection, coloring, subdivision, animation, and vector options.
2. **Given** a directory containing DICOM data, **When** the user runs med2glb interactively, **Then** the wizard shows series classification (modality, data type, recommended method) and prompts for series selection, method, quality, and animation.
3. **Given** a non-interactive environment (piped input, CI), **When** the user runs med2glb, **Then** the wizard is skipped and sensible defaults are used.
4. **Given** any explicit pipeline flag (`--method`, `--animate`, `--coloring`, etc.), **When** present, **Then** the wizard is bypassed entirely.
5. **Given** a CARTO mesh with insufficient LAT data for vectors, **When** the wizard runs, **Then** it automatically detects poor vector quality and either skips the vector prompt or warns the user.

---

### User Story 5 — Gallery Mode for DICOM Slices (Priority: P3)

A user wants to convert every DICOM slice in a series to individual GLB textured quads, with overview layouts (lightbox grid, spatial fan) for browsing.

**Why this priority**: Useful for AR presentation of 2D imaging data, but not the primary 3D visualization goal.

**Independent Test**: Run `med2glb ./dicom/ --gallery`, verify individual GLBs, lightbox grid, and spatial fan are created.

**Acceptance Scenarios**:

1. **Given** a DICOM series, **When** gallery mode is used, **Then** individual GLBs (one per slice/position) are created, plus a lightbox grid GLB.
2. **Given** a DICOM series with spatial metadata (ImagePositionPatient), **When** gallery mode is used, **Then** a spatial fan GLB is also created with quads at real-world positions.
3. **Given** temporal DICOM data (cine), **When** gallery mode is used with animation, **Then** each position gets frame-switching animation.

---

### User Story 6 — GLB Compression (Priority: P2)

A user has large GLB files that exceed AR device limits and needs to compress them to a target size.

**Why this priority**: Compressed GLBs are necessary for practical AR deployment on HoloLens and mobile.

**Independent Test**: Run `med2glb model.glb --compress --max-size 10`, verify output is under 10MB.

**Acceptance Scenarios**:

1. **Given** a large GLB file, **When** `--compress` is used with a target size, **Then** the file is compressed to fit within the target using the selected strategy.
2. **Given** `--strategy ktx2`, **When** `toktx` is installed, **Then** textures are GPU-compressed via Basis Universal for smallest file size.
3. **Given** `--strategy draco`, **When** the GLB has no animations, **Then** mesh geometry is Draco-compressed.
4. **Given** a GLB with animations, **When** Draco is selected, **Then** the system falls back gracefully since Draco doesn't preserve glTF animations.

---

### User Story 7 — Switch Between Conversion Methods (Priority: P1)

A user wants to experiment with different conversion methods on the same DICOM dataset to find the best visual result.

**Why this priority**: Comparing methods is essential for finding optimal results across different data types.

**Independent Test**: Run the same input through different `--method` flags, verify each produces distinct, valid output.

**Acceptance Scenarios**:

1. **Given** any DICOM input, **When** the user runs `--list-methods`, **Then** all available methods are displayed with descriptions, recommended data types, and dependency status.
2. **Given** the same input, **When** `--method compare` is used, **Then** all methods run side-by-side with a comparison table of mesh stats and timing.
3. **Given** an AI method is not installed, **When** the user tries to use it, **Then** a clear error explains how to install the optional AI dependencies.

---

### User Story 8 — Install and First Use (Priority: P2)

A new user installs the tool via pip and runs their first conversion, guided by clear documentation and the interactive wizard.

**Independent Test**: Follow the README on a clean Python environment and successfully convert a dataset.

**Acceptance Scenarios**:

1. **Given** a Python 3.10+ environment, **When** the user runs `pip install med2glb`, **Then** the tool installs with all core dependencies and the `med2glb` command is available.
2. **Given** the README, **When** a new user reads it, **Then** they understand CARTO and DICOM workflows, method comparison, AR viewer compatibility, and can follow examples.

---

### Edge Cases

- What happens when the CARTO `.mesh` file has no matching `_car.txt`?
  - The system warns and produces geometry-only output without heatmaps.
- What happens when CARTO points have all-NaN LAT values?
  - Animation is skipped with a warning; static coloring uses available voltage data.
- What happens when a CARTO mesh is non-manifold and cannot be Loop-subdivided?
  - Falls back to original geometry with nearest-neighbor coloring.
- What happens when the DICOM directory contains mixed series?
  - Series are grouped by Series Instance UID; the wizard presents a selection table.
- What happens when DICOM files have inconsistent slice spacing?
  - The system detects irregular spacing and either interpolates or warns.
- What happens when 3D echo data has missing time frames?
  - The system detects gaps and warns, optionally interpolating.
- What happens when the AI segmentation method produces poor results?
  - The system still produces output with a warning; user can fall back.
- What happens when the input contains no valid data?
  - Clear error message indicating no DICOM or CARTO data was found.
- What happens when `toktx` is not installed and KTX2 compression is requested?
  - Falls back to texture downscaling with a warning about installing KTX-Software.
- What happens when a CARTO mesh has too few points for vectors?
  - The wizard auto-assesses quality (≥30 points, ≥20ms LAT range, ≥15% gradient coverage) and skips or warns.

## Requirements *(mandatory)*

### Functional Requirements

#### Core Platform

- **FR-001**: System MUST accept a file path or directory path as input and automatically detect whether it contains CARTO 3 export data, DICOM volumes, temporal DICOM data, or single DICOM images.
- **FR-002**: System MUST provide an interactive wizard (TTY-only) that detects data type, shows summary tables, and prompts for relevant options with smart defaults.
- **FR-003**: System MUST bypass the wizard when any explicit pipeline flag is provided or when running in non-interactive mode (piped input, CI).
- **FR-004**: System MUST display progress indication during processing for long-running operations via `rich`.
- **FR-005**: System MUST provide clear, actionable error messages for common failure cases.
- **FR-006**: System MUST be installable via `pip install` with a `med2glb` CLI entry point via `pyproject.toml`.
- **FR-007**: System MUST include comprehensive README with CARTO and DICOM workflows, method comparison, AR viewer compatibility, and examples.

#### CARTO 3 EP Mapping

- **FR-010**: System MUST auto-detect CARTO 3 export directories by presence of `.mesh` files.
- **FR-011**: System MUST parse CARTO `.mesh` files (INI-style geometry with vertices, faces, group IDs) and `_car.txt` measurement files (LAT, bipolar/unipolar voltage) across CARTO versions v4, v5, and v6.
- **FR-012**: System MUST strip inactive geometry (GroupID `-1000000`) and fill/cap vertices.
- **FR-013**: System MUST support three coloring schemes: LAT (activation time), bipolar voltage (scar mapping), and unipolar voltage, with clinical colormaps matching CARTO 3 conventions.
- **FR-014**: System MUST generate all available colorings automatically in a single run, with `--coloring` flag to restrict to a single scheme.
- **FR-015**: System MUST perform Loop subdivision (levels 0-3, default 2) with k-NN IDW interpolation (k=6) for smooth color gradients.
- **FR-016**: System MUST bake vertex colors into textures via xatlas UV unwrap + barycentric rasterization with gutter bleeding, since HoloLens 2 glTFast/MRTK does not render `COLOR_0`.
- **FR-017**: System MUST produce animated GLB with sweeping excitation ring via emissive overlay textures (30 frames), with the full mesh shared across frames.
- **FR-018**: System MUST support animated LAT streamline vectors (conduction arrows) with vertex-gradient interpolation, momentum coasting, and Gaussian smoothing.
- **FR-019**: System MUST automatically assess vector quality (minimum 30 points, ≥20ms LAT range, ≥15% gradient coverage) and skip vectors for unsuitable meshes.
- **FR-020**: System MUST embed a color legend cylinder and study info panel in each CARTO GLB for AR readability.
- **FR-021**: System MUST support batch processing of multiple CARTO exports from a parent directory via `--batch` flag.
- **FR-022**: System MUST compute xatlas UV unwrap once per mesh and share it across all variants (static, animated, vector) to minimize processing time.

#### DICOM Conversion

- **FR-030**: System MUST support at minimum five conversion methods switchable via `--method` CLI flag: `marching-cubes`, `classical`, `totalseg`, `chamber-detect`, and `compare`.
- **FR-031**: System MUST default to GLB output format.
- **FR-032**: System MUST produce animated GLB output for 4D echo data using morph targets via `pygltflib`, with consistent mesh topology across frames via nearest-surface-point correspondence.
- **FR-033**: System MUST apply temporal smoothing across animation frames (weighted moving average) to prevent flickering and mesh jumping.
- **FR-034**: System MUST support configurable mesh quality parameters: Taubin smoothing iterations (default 15), target face count (default 80K), and threshold values.
- **FR-035**: System MUST support per-structure PBR materials in GLB output including distinct colors and configurable transparency (`alphaMode: BLEND`, `metallicFactor: 0.0`, `roughnessFactor: 0.6-0.8`).
- **FR-036**: System MUST support multi-threshold extraction for layered visualizations.
- **FR-037**: AI segmentation methods (`totalseg`) MUST be optional dependencies installable via `pip install med2glb[ai]`.
- **FR-038**: System MUST provide `--list-methods` and `--list-series` commands.
- **FR-039**: System MUST group DICOM files by Series Instance UID with `--series` flag for selection.
- **FR-040**: System MUST correctly read DICOM metadata (pixel spacing, slice thickness, image orientation) for anatomically accurate models.
- **FR-041**: System MUST use Taubin smoothing (not Laplacian) as default to preserve volume.
- **FR-042**: System MUST support reading vendor-specific 3D echo DICOM formats (Philips and GE).

#### Gallery Mode

- **FR-050**: System MUST support `--gallery` mode producing individual GLBs per slice, a lightbox grid GLB, and (if spatial metadata exists) a spatial fan GLB.
- **FR-051**: Gallery lightbox MUST support configurable column count (default 6) with 5% gap between cells.
- **FR-052**: Gallery spatial fan MUST use `ImagePositionPatient` and `ImageOrientationPatient` to place quads at real-world positions.
- **FR-053**: Gallery MUST support frame-switching animation for temporal (cine) data.

#### GLB Compression

- **FR-060**: System MUST support `--compress` mode with four strategies: `ktx2` (default), `draco`, `downscale`, `jpeg`.
- **FR-061**: System MUST support configurable target size via `--max-size` (default 25MB).
- **FR-062**: KTX2 strategy MUST use external `toktx` tool for Basis Universal compression, with graceful fallback if not installed.
- **FR-063**: Draco strategy MUST skip compression for GLBs containing animations.

#### Pluggable Architecture

- **FR-070**: The method architecture MUST allow adding new conversion methods without modifying existing code via `@register_method` decorator and `ConversionMethod` ABC.

### Key Entities

- **CartoStudy**: Complete CARTO 3 export with meshes, measurement points, version, and study name.
- **CartoMesh**: Surface mesh with vertices, faces, normals, group IDs (active vs. inactive geometry), and face group IDs.
- **CartoPoint**: Measurement point with position, orientation, bipolar/unipolar voltage, and LAT value.
- **CartoConfig**: Pipeline configuration for CARTO processing (mesh selection, colorings, subdivision, animation, vectors).
- **DicomVolume**: A 3D array of voxel intensities assembled from DICOM files, with pixel spacing, slice thickness, orientation.
- **TemporalSequence**: Ordered collection of DicomVolumes representing time frames in a cardiac cycle.
- **DicomConfig**: Pipeline configuration for DICOM processing (method, animation, threshold, etc.).
- **SeriesInfo**: DICOM series classification with modality, data type, recommended method.
- **MeshData**: Triangulated surface with vertices, faces, normals, structure name, material, and optional vertex colors.
- **MaterialConfig**: PBR material properties: base color, alpha, metallic, roughness, unlit flag.
- **ConversionResult**: Output from a conversion method with meshes, warnings, timing.
- **AnimatedResult**: Extends conversion result with morph targets, frame times, loop duration.
- **GallerySlice**: Individual DICOM slice for gallery mode with pixel data and spatial metadata.
- **ConversionStats**: Metrics for conversion logging.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can convert a CARTO export to AR-ready GLBs with clinical heatmaps within 2 CLI commands (install + `med2glb ./export/`).
- **SC-002**: CARTO heatmaps render correctly on HoloLens 2 via baked textures with correct clinical colormaps.
- **SC-003**: Animated excitation ring GLBs play back as a smooth 30-frame loop with the ring tracking LAT activation timing.
- **SC-004**: All three CARTO coloring schemes (LAT, bipolar, unipolar) are generated automatically in a single run.
- **SC-005**: Users can convert a 3D echo dataset to an animated GLB and view it in AR within 3 CLI commands.
- **SC-006**: Animated GLB output plays back the cardiac cycle as a smooth morph-target loop without frame-to-frame mesh jumping.
- **SC-007**: Users can switch between all available conversion methods and receive valid, visually distinct outputs.
- **SC-008**: The tool installs on a clean Python 3.10+ environment via `pip install` without requiring external system dependencies.
- **SC-009**: Output GLB files are practical for AR device loading (compression available for large files).
- **SC-010**: The interactive wizard enables a new user to produce GLBs without reading documentation.
- **SC-011**: AI-segmented cardiac CT output (via `totalseg`) contains separately colored cardiac structures.
- **SC-012**: The README enables a new user with medical imaging data to produce their first GLB within 10 minutes.
- **SC-013**: CARTO versions v4 (~2015), v7.1, and v7.2+ are all correctly parsed.
- **SC-014**: Gallery mode produces individual, lightbox, and spatial GLBs from any DICOM series.

## Assumptions

- Users have Python 3.10 or newer installed.
- DICOM files follow standard DICOM format. Vendor-specific 3D echo formats (Philips, GE) are supported.
- CARTO exports contain standard `.mesh` and `_car.txt` file structures across supported versions.
- For 3D echo data, temporal frames are distinguishable via DICOM tags.
- TotalSegmentator provides reasonable out-of-the-box results on standard cardiac CT.
- HoloLens 2 is the primary AR target; GLB also works on Android Scene Viewer and web via `<model-viewer>`.
- Processing time is not a primary concern — users prefer quality over speed for this offline conversion tool.
- The `toktx` external tool is optional — KTX2 compression degrades gracefully without it.
- The pluggable method architecture allows adding new segmentation models as they mature.
