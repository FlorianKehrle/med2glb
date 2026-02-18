# Feature Specification: DICOM to GLB Converter

**Feature Branch**: `1-dicom-glb-converter`
**Created**: 2026-02-09
**Status**: Draft
**Input**: User description: "Python CLI tool for converting DICOM medical imaging data (cardiac CT, MRI, 3D echocardiography) to GLB 3D models optimized for augmented reality viewing, with pluggable conversion methods and animated output for 4D echo data."

## Research Findings

The following findings from a comprehensive survey of existing tools and the state of the art inform this specification:

1. **No existing end-to-end CLI tool exists** for DICOM-to-animated-GLB, especially for 3D echo. Existing tools (3D Slicer, DicomToMesh, InVesalius) produce static STL/OBJ via GUI. SlicerHeart handles 3D echo DICOM reading but exports to STL for 3D printing, not animated GLB for AR.

2. **Animation in GLB requires morph targets** built via `pygltflib`. The commonly-used `trimesh` library does NOT support GLB animations. Each cardiac phase becomes a morph target, and the glTF animation system interpolates between them. `trimesh` remains useful for mesh processing (smoothing, decimation) before final export.

3. **Apple Quick Look does NOT support GLB** and does NOT support morph target animations in USDZ. For cross-platform AR, Google's `<model-viewer>` web component is the best option (plays animated GLB on both iOS and Android). Android Scene Viewer natively supports animated GLB with morph targets.

4. **3D echo segmentation**: Most open-source models (EchoNet-Dynamic, SimLVSeg) work on 2D echo video, not true 3D volumes. MedSAM2 (2025) is the most versatile — handles 3D volumes and video, achieving 96.1% LV / 95.8% LA Dice on echo data. MONAI Auto3DSeg won the MVSEG2023 mitral valve segmentation challenge on 3D TEE data.

5. **CT segmentation**: TotalSegmentator (nnU-Net-based, 117+ structures including cardiac chambers, aorta, pulmonary artery) is the most mature option. Feb 2025 update added more cardiovascular structures.

6. **Vendor-specific 3D echo DICOM** (Philips, GE) requires special handling. SlicerHeart provides reference implementations.

7. **Optimal AR mesh**: 50K-100K triangles for a single close-up model. Apple Quick Look caps at ~500K vertices.

8. **Recommended mesh pipeline**: Marching cubes -> Taubin smoothing (preserves volume, unlike Laplacian which shrinks chambers) -> Quadric edge collapse decimation -> Final light Taubin smooth.

9. **Output format quality is identical** across GLB, STL, and OBJ — the mesh is the same, only the container differs. GLB adds materials (colors, transparency) and animation (morph targets). STL/OBJ are useful for post-processing in Blender.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Convert 3D Echo Cardiac Loop to Animated GLB (Priority: P1)

A cardiologist or researcher has 3D echocardiography data consisting of multiple DICOM files representing a cardiac cycle (3D volumes over time). They want to convert this 4D dataset into a single animated GLB file showing the beating heart, viewable in augmented reality. They need to choose between conversion methods to find the best visual result for their specific data.

**Why this priority**: 3D echo with animation is the primary use case. This drives the core architecture — if this works well, the simpler cases (static CT, single slices) are subsets of this pipeline.

**Independent Test**: Can be tested by providing a set of 3D echo DICOM files, running the CLI with `--animate`, and opening the resulting GLB in an AR viewer (Android Scene Viewer, or cross-platform via `<model-viewer>` web component) to see a smooth cardiac loop.

**Acceptance Scenarios**:

1. **Given** a directory of 3D echo DICOM files representing a full cardiac cycle, **When** the user runs the converter with animation enabled, **Then** a single animated GLB file is produced using morph targets with smooth mesh transitions between time frames.
2. **Given** a 3D echo dataset, **When** the user switches between different methods (e.g., `marching-cubes`, `classical`, `medsam2`), **Then** each produces a valid output with visibly different quality characteristics, allowing comparison.
3. **Given** a 3D echo animated GLB output, **When** viewed in an AR application, **Then** the animation loops smoothly without visible temporal flickering or mesh jumping between frames.
4. **Given** a 3D echo dataset with noisy data, **When** the user selects the `classical` method with high smoothing, **Then** speckle noise is visibly reduced and the cardiac surface appears smooth.
5. **Given** 3D echo data from Philips or GE ultrasound systems, **When** the user provides the vendor-specific DICOM files, **Then** the system correctly reads and assembles the 3D volumes.

---

### User Story 2 - Convert Cardiac CT/MRI Volume to Segmented GLB (Priority: P2)

A medical professional has a cardiac CT or MRI scan as a series of DICOM slices forming a 3D volume. They want to convert this into a GLB file where different cardiac structures (chambers, valves, vessels) are individually colored and some are semi-transparent, enabling exploration of the heart anatomy in AR.

**Why this priority**: Static 3D volume conversion is the second most common use case. Segmentation with per-structure materials is critical for AR viewing quality.

**Independent Test**: Can be tested by providing a directory of cardiac CT DICOM slices, running the converter, and verifying the GLB contains separately colored/transparent structures viewable in AR.

**Acceptance Scenarios**:

1. **Given** a directory of cardiac CT DICOM slices, **When** the user runs the converter with the default method, **Then** a GLB file is produced containing a 3D mesh of the cardiac anatomy.
2. **Given** a cardiac CT dataset, **When** the user uses `totalseg` method, **Then** individual cardiac structures are extracted as separate meshes with distinct colors and configurable transparency.
3. **Given** any conversion method, **When** the user specifies a target face count, **Then** the output mesh is decimated to approximately that count while preserving important anatomical detail.
4. **Given** a cardiac CT dataset with contrast agent, **When** the user uses multi-threshold extraction, **Then** the blood pool and myocardium are exported as separate layers with distinct transparency levels.

---

### User Story 3 - Convert Single DICOM Slice to GLB (Priority: P3)

A user has a single DICOM file (one 2D image, e.g., a single echocardiogram frame or X-ray) and wants to convert it to a GLB for display in AR as a textured 3D plane.

**Why this priority**: Simplest use case, useful for quick previews but not the primary 3D visualization goal.

**Independent Test**: Can be tested by providing a single .dcm file, running the converter, and verifying a GLB with the image content is produced.

**Acceptance Scenarios**:

1. **Given** a single DICOM file, **When** the user runs the converter, **Then** a GLB file is produced containing the image as a textured plane in 3D space.
2. **Given** a single DICOM file, **When** the output GLB is viewed in AR, **Then** the image is clearly visible with correct aspect ratio and orientation.

---

### User Story 4 - Switch Between Conversion Methods (Priority: P1)

A user wants to easily experiment with different conversion methods on the same dataset to find the approach that produces the best visual result for their specific data type and AR viewing needs.

**Why this priority**: The ability to compare methods is essential for finding optimal results across different data types (echo vs. CT vs. MRI). This is a core usability requirement.

**Independent Test**: Can be tested by running the same input through different `--method` flags and verifying each produces distinct, valid output.

**Acceptance Scenarios**:

1. **Given** any DICOM input, **When** the user runs `--list-methods`, **Then** all available conversion methods are displayed with descriptions and recommended data types.
2. **Given** the same DICOM input, **When** the user runs the converter with different `--method` values, **Then** each method produces a valid output with different visual characteristics.
3. **Given** an AI segmentation method is not installed, **When** the user tries to use it, **Then** a clear error message explains how to install the optional AI dependencies.

---

### User Story 5 - Install and First Use (Priority: P2)

A new user installs the tool via pip and runs their first conversion, guided by clear documentation and helpful CLI output.

**Why this priority**: Good onboarding experience is critical for adoption. The README and CLI help must make the tool approachable.

**Independent Test**: Can be tested by following the README installation instructions on a clean Python environment and successfully converting a sample dataset.

**Acceptance Scenarios**:

1. **Given** a Python environment, **When** the user runs `pip install med2glb`, **Then** the tool installs with all core dependencies and the `med2glb` command is available.
2. **Given** a freshly installed tool, **When** the user runs `med2glb --help`, **Then** clear usage instructions with examples are displayed.
3. **Given** the README documentation, **When** a new user reads it, **Then** they can understand what each conversion method is best suited for, AR viewer compatibility, and follow step-by-step examples.

---

### Edge Cases

- What happens when the DICOM directory contains mixed series (different patients, different scans)?
  - The system detects series by Series Instance UID and lists available series, processing the largest by default or allowing the user to select via `--series` flag.
- What happens when DICOM files have inconsistent slice spacing?
  - The system detects irregular spacing and either interpolates or warns the user.
- What happens when 3D echo data has missing time frames?
  - The system detects gaps and warns, optionally interpolating the missing frames for animation continuity.
- What happens when the AI segmentation method produces poor results?
  - The system still produces output with a warning, and the user can fall back to a different method.
- What happens when the input directory contains no valid DICOM files?
  - The system provides a clear error message indicating no DICOM data was found.
- What happens when the target face count is too low for the anatomy's complexity?
  - The system warns if decimation causes significant quality loss.
- What happens when the system runs out of memory on very large volumes?
  - The system provides a clear error and suggests reducing resolution or processing a subregion.
- What happens with vendor-specific 3D echo DICOM files (Philips, GE)?
  - The system supports reading vendor-specific formats with reference to SlicerHeart's handling of these proprietary encodings.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a file path or directory path as input and automatically detect whether it contains single DICOM files, a multi-slice 3D volume, or 4D temporal echo data.
- **FR-002**: System MUST support at minimum four conversion methods switchable via a `--method` CLI flag: `marching-cubes`, `classical`, `totalseg` (CT-optimized), and `medsam2` (echo/general).
- **FR-003**: System MUST default to GLB output format, with optional `--format` flag supporting `glb` (default), `stl`, and `obj` for users who want to post-process in tools like Blender.
- **FR-004**: System MUST produce animated GLB output for 4D echo data using morph targets (via `pygltflib`), with the mesh smoothly transitioning between cardiac cycle frames. Animation is only available in GLB format.
- **FR-005**: System MUST apply temporal smoothing across animation frames to prevent flickering and mesh jumping between frames.
- **FR-006**: System MUST support configurable mesh quality parameters: smoothing level (Taubin iterations), target face count (default 50K-100K), and threshold values.
- **FR-007**: System MUST support per-structure PBR materials in GLB output including distinct colors and configurable transparency (`alphaMode: BLEND`, `metallicFactor: 0.0`, `roughnessFactor: 0.6-0.8`).
- **FR-008**: System MUST support multi-threshold extraction to produce layered visualizations (e.g., blood pool as semi-transparent, myocardium as opaque).
- **FR-009**: AI segmentation methods (`totalseg`, `medsam2`) MUST be optional dependencies installable via `pip install med2glb[ai]`, not required for core functionality.
- **FR-010**: System MUST display progress indication during processing for long-running operations.
- **FR-011**: System MUST provide a `--list-methods` command showing available methods with descriptions and recommended data types.
- **FR-012**: System MUST provide clear, actionable error messages for common failure cases (no DICOM found, missing dependencies, insufficient memory).
- **FR-013**: The `marching-cubes` method MUST extract an isosurface at a configurable intensity threshold with basic Taubin smoothing and decimation.
- **FR-014**: The `classical` method MUST apply: Gaussian volume smoothing, adaptive thresholding, morphological operations, Taubin mesh smoothing (volume-preserving), quadric edge collapse decimation, and hole filling.
- **FR-015**: The `totalseg` method MUST leverage TotalSegmentator to segment cardiac CT structures (at minimum: LV, RV, LA, RA, aorta, myocardium) into separate colored/transparent meshes.
- **FR-016**: The `medsam2` method MUST leverage MedSAM2 for segmentation of cardiac structures from echo and other modalities, with per-frame segmentation for 4D data producing segmented animated output.
- **FR-017**: System MUST be installable via `pip install` with a `med2glb` CLI entry point via `pyproject.toml`.
- **FR-018**: System MUST include comprehensive README with installation instructions, usage examples, method comparison guide, AR viewer compatibility matrix, and cross-platform viewing tips.
- **FR-019**: System MUST group DICOM files by Series Instance UID when a directory contains multiple series, with a `--series` flag to select a specific series.
- **FR-020**: System MUST correctly read DICOM metadata (pixel spacing, slice thickness, image orientation) to produce anatomically accurate 3D models.
- **FR-021**: System MUST use Taubin smoothing (not Laplacian) as the default mesh smoothing algorithm to preserve volume — critical for anatomically accurate cardiac chamber sizing.
- **FR-022**: System MUST support reading vendor-specific 3D echo DICOM formats (at minimum Philips and GE).
- **FR-023**: The pluggable method architecture MUST allow adding new conversion methods without modifying existing code (open/closed principle).

### Key Entities

- **DICOM Volume**: A 3D array of voxel intensities assembled from one or more DICOM slice files, characterized by pixel spacing, slice thickness, and patient orientation metadata. May originate from CT, MRI, or 3D echo (including vendor-specific Philips/GE formats).
- **Temporal Sequence**: An ordered collection of DICOM Volumes representing time frames in a cardiac cycle, characterized by frame count and temporal resolution. Identified via Temporal Position Index or Instance Number ordering.
- **Conversion Method**: A self-contained processing pipeline that transforms a DICOM Volume (or Temporal Sequence) into one or more 3D meshes. Each method has its own parameters, trade-offs, and recommended data types.
- **3D Mesh**: A triangulated surface representation of anatomical structures, with vertex positions, face indices, normals, and associated material properties. Target: 50K-100K triangles for AR.
- **GLB Output**: A binary glTF 2.0 container holding one or more meshes with PBR materials, and optionally morph target animations for temporal data. Exported via `pygltflib`.
- **Morph Target**: A set of vertex displacements representing one cardiac phase. The glTF animation system interpolates weights between targets to create the beating heart animation.
- **Material**: PBR material properties: base color (RGB), transparency (alpha), `metallicFactor: 0.0`, `roughnessFactor: 0.6-0.8`. Each cardiac structure gets a distinct material.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can convert a 3D echo dataset to an animated GLB and view it in AR within 5 CLI commands or fewer (install + convert + view).
- **SC-002**: Animated GLB output plays back the cardiac cycle as a smooth loop with no visible frame-to-frame mesh jumping when viewed in AR.
- **SC-003**: Users can switch between all available conversion methods on the same dataset and receive valid, visually distinct outputs for comparison.
- **SC-004**: The tool installs on a clean Python 3.10+ environment via `pip install` without requiring external system dependencies or compilation.
- **SC-005**: Output GLB files are under 50MB for typical cardiac datasets (single time frame), making them practical for AR device loading.
- **SC-006**: The README enables a new user with DICOM data to produce their first GLB within 10 minutes of reading the documentation.
- **SC-007**: AI-segmented cardiac CT output (via `totalseg`) contains at least 4 separately colored/transparent structures identifiable as cardiac anatomy.
- **SC-008**: 95% of valid DICOM datasets (CT, MRI, 3D echo) are automatically detected and processed without manual configuration of input type.
- **SC-009**: Output GLB meshes contain 50K-100K triangles by default, balancing anatomical detail with smooth AR rendering performance.

## Assumptions

- Users have Python 3.10 or newer installed.
- DICOM files follow standard DICOM format with required metadata tags. Vendor-specific 3D echo formats (Philips, GE) are supported as a priority.
- For 3D echo data, temporal frames are distinguishable via DICOM tags (Temporal Position Index, Instance Number ordering).
- TotalSegmentator provides reasonable out-of-the-box results on standard cardiac CT without fine-tuning.
- MedSAM2 provides usable segmentation on 3D echo data — quality may vary and users can fall back to non-AI methods.
- Processing time is not a primary concern — users prefer quality over speed for this offline conversion tool.
- The pluggable method architecture allows adding new segmentation models as they mature.
- Users may want STL/OBJ output for post-processing in Blender, but GLB is the default and preferred format.
