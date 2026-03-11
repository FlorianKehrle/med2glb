# Data Model: med2glb — Medical Imaging to GLB Converter

**Date**: 2026-02-09 (initial), 2026-03-11 (updated)
**Feature**: med2glb

## Core Entities

### DicomVolume
Represents a 3D array of voxel intensities assembled from DICOM files.

| Field | Type | Description |
|-------|------|-------------|
| voxels | ndarray (float32) | 3D array [Z, Y, X] of intensity values |
| pixel_spacing | tuple[float, float] | Row/column spacing in mm |
| slice_thickness | float | Distance between slices in mm |
| image_orientation | tuple[float, ...] | Patient orientation cosines (6 values) |
| image_position_first | tuple[float, float, float] | Position of first slice in mm |
| series_uid | str | Series Instance UID |
| modality | str | DICOM modality (CT, MR, US) |
| vendor | str or None | Detected vendor for echo (Philips, GE, None) |
| metadata | dict | Additional DICOM metadata (patient, study info) |
| rgb_data | ndarray or None | RGB pixel data if available |

**Validation**:
- voxels must be 3D with shape > (2, 2, 2)
- pixel_spacing values must be > 0
- slice_thickness must be > 0

**Properties**:
- `spacing` → (z, y, x) tuple combining slice_thickness and pixel_spacing
- `shape` → voxel array shape

### TemporalSequence
Ordered collection of DicomVolumes representing time frames.

| Field | Type | Description |
|-------|------|-------------|
| frames | list[DicomVolume] | Ordered volumes per cardiac phase |
| temporal_resolution | float or None | Time between frames in ms |
| is_loop | bool | Whether sequence represents a complete cardiac cycle |

**Validation**:
- frame_count >= 2
- All frames must have identical spatial dimensions (Z, Y, X)
- All frames must have identical pixel_spacing and slice_thickness

**Properties**:
- `frame_count` → len(frames)

### SeriesInfo
Classification of a DICOM series for wizard display and method recommendation.

| Field | Type | Description |
|-------|------|-------------|
| series_uid | str | Series Instance UID |
| modality | str | "US", "MR", "CT", "XA", "NM" |
| description | str | Series description from DICOM |
| file_count | int | Number of DICOM files |
| data_type | str | "2D cine", "3D volume", "3D+T volume", "still image" |
| detail | str | Human-readable detail ("132 frames", "64 slices") |
| dimensions | str | "600x800", "512x512x120" |
| recommended_method | str | Suggested conversion method |
| recommended_output | str | Suggested output type |
| is_multiframe | bool | Whether files contain multiple frames |
| number_of_frames | int | Total frame count |

---

## CARTO Entities

### CartoStudy
Complete CARTO 3 export containing all meshes and measurement points.

| Field | Type | Description |
|-------|------|-------------|
| meshes | list[CartoMesh] | All mesh surfaces in the export |
| points | dict[str, list[CartoPoint]] | Measurement points keyed by mesh/map name |
| version | str | "4.0", "5.0", or "6.0" |
| study_name | str | Patient/study identifier |

### CartoMesh
Surface mesh from a CARTO `.mesh` file with active/inactive geometry marking.

| Field | Type | Description |
|-------|------|-------------|
| mesh_id | int | Mesh identifier |
| vertices | ndarray (float64) | Vertex positions [N, 3] |
| faces | ndarray (int32) | Triangle indices [M, 3] |
| normals | ndarray (float64) | Per-vertex normals [N, 3] |
| group_ids | ndarray (int32) | Per-vertex group IDs [N] (-1000000 = inactive fill) |
| face_group_ids | ndarray (int32) | Per-face group IDs [M] |
| mesh_color | tuple[float, float, float, float] | Default RGBA color |
| color_names | list[str] | Per-vertex color channel names |
| transparent_group_ids | list[int] | Group IDs for inactive/fill geometry |
| structure_name | str | Map/structure label |

### CartoPoint
Single electro-anatomical measurement point.

| Field | Type | Description |
|-------|------|-------------|
| point_id | int | Point identifier |
| position | ndarray (float64) | [3] X, Y, Z coordinates |
| orientation | ndarray (float64) | [3] catheter orientation |
| bipolar_voltage | float | Bipolar voltage in mV |
| unipolar_voltage | float | Unipolar voltage in mV |
| lat | float | Local activation time in ms (NaN if sentinel -10000) |

### CartoConfig
Pipeline configuration for CARTO processing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| input_path | Path | — | Path to CARTO export directory |
| name | str | — | Descriptive label |
| output_dir | Path or None | None | Output directory (default: `<input>/glb/`) |
| selected_mesh_indices | list[int] or None | None | Mesh indices to process (None = all) |
| colorings | list[str] | ["lat", "bipolar", "unipolar"] | Coloring schemes to generate |
| subdivide | int | 2 | Loop subdivision level (0-3) |
| animate | bool | True | Generate animated GLB |
| static | bool | True | Generate static GLB |
| vectors | str | "no" | Vector mode: "no", "yes" (both), "only" |
| vector_mesh_indices | list[int] or None | None | Meshes to generate vectors for |
| target_faces | int | 80000 | Decimation target |

---

## Mesh & Material Entities

### MeshData
Triangulated surface mesh with material properties.

| Field | Type | Description |
|-------|------|-------------|
| vertices | ndarray (float32) | Vertex positions [N, 3] |
| faces | ndarray (int32) | Triangle indices [M, 3] |
| normals | ndarray (float32) or None | Vertex normals [N, 3] |
| structure_name | str | Anatomical label (e.g., "left_ventricle") |
| material | MaterialConfig | Color, transparency, PBR properties |
| vertex_colors | ndarray (float32) or None | Per-vertex RGBA [N, 4] |

**Validation**:
- vertices.shape[1] == 3
- faces.shape[1] == 3
- All face indices < len(vertices)

### MaterialConfig
PBR material properties for a structure.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| base_color | tuple[float, float, float] | (0.8, 0.2, 0.2) | RGB color [0-1] |
| alpha | float | 1.0 | Transparency [0-1], 0=transparent |
| metallic | float | 0.0 | PBR metallic factor |
| roughness | float | 0.7 | PBR roughness factor |
| name | str | "" | Material display name |
| unlit | bool | False | Skip lighting calculations (for legends) |

### MethodParams
Configuration parameters for a DICOM conversion method.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| threshold | float or None | None | Intensity threshold for isosurface |
| smoothing_iterations | int | 15 | Taubin smoothing iterations |
| target_faces | int | 80000 | Target triangle count after decimation |
| multi_threshold | list[ThresholdLayer] or None | None | Multi-threshold config |

### ThresholdLayer
Single layer in multi-threshold extraction.

| Field | Type | Description |
|-------|------|-------------|
| threshold | float | Intensity threshold for this layer |
| label | str | Layer name (e.g., "blood_pool") |
| material | MaterialConfig | Visual properties for this layer |

---

## Conversion Output Entities

### ConversionResult
Output from a DICOM conversion method.

| Field | Type | Description |
|-------|------|-------------|
| meshes | list[MeshData] | One or more meshes (multiple for segmented output) |
| method_name | str | Which method produced this result |
| processing_time | float | Duration in seconds |
| warnings | list[str] | Quality or data warnings |

### AnimatedResult
Extends ConversionResult for temporal DICOM data.

| Field | Type | Description |
|-------|------|-------------|
| base_meshes | list[MeshData] | Meshes for the first frame (base topology) |
| morph_targets | list[list[ndarray]] | Per-mesh vertex displacements per frame |
| frame_times | list[float] | Keyframe timestamps in seconds |
| loop_duration | float | Total animation cycle duration in seconds |
| method_name | str | Method identifier |
| processing_time | float | Duration in seconds |
| warnings | list[str] | Quality or data warnings |

**Property**: `meshes` → alias for `base_meshes`

### ConversionStats
Metrics for conversion logging.

| Field | Type | Description |
|-------|------|-------------|
| method_name | str | Conversion method used |
| output_path | str | Output file path |
| file_size_kb | float | Output file size |
| vertex_count | int | Total vertices |
| face_count | int | Total faces |
| mesh_count | int | Number of meshes |
| elapsed_seconds | float | Processing time |
| success | bool | Whether conversion succeeded |
| error | str or None | Error message if failed |

---

## Gallery Entities

### GallerySlice
Individual DICOM slice for gallery mode.

| Field | Type | Description |
|-------|------|-------------|
| pixel_data | ndarray | [Y, X] or [Y, X, 3] uint8 pixel data |
| pixel_spacing | tuple[float, float] or None | Physical spacing |
| image_position | tuple[float, float, float] or None | World position |
| image_orientation | tuple[float, ...] or None | Orientation cosines |
| instance_number | int | Slice ordering |
| filename | str | Source filename |
| rows | int | Image height |
| cols | int | Image width |
| temporal_index | int or None | Frame index for cine data |
| series_uid | str | Series identifier |
| modality | str | DICOM modality |

---

## Configuration Entities

### DicomConfig
Pipeline configuration for DICOM processing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| input_path | Path | — | Input file or directory |
| name | str | — | Descriptive label |
| output | Path or None | None | Output path |
| method | str | "classical" | Conversion method |
| format | str | "glb" | Output format |
| animate | bool | True | Enable animation |
| threshold | float or None | None | Isosurface threshold |
| smoothing | int | 15 | Taubin iterations |
| target_faces | int | 80000 | Decimation target |
| alpha | float | 1.0 | Global transparency |
| series_uid | str or None | None | Series selection |
| gallery | bool | False | Gallery mode |
| verbose | bool | False | Detailed output |

---

## Relationships

```
CartoStudy --contains--> CartoMesh (1:N)
CartoStudy --contains--> CartoPoint (1:N per mesh, keyed by name)
CartoConfig --configures--> CARTO pipeline
CartoMesh --subdivided--> CartoMesh (Loop subdivision)
CartoMesh + CartoPoint --mapped--> per-vertex values (IDW/NN)
per-vertex values --colormapped--> vertex RGBA (LAT/bipolar/unipolar)
vertex RGBA --baked--> baseColorTexture (xatlas UV + rasterization)

TemporalSequence --contains--> DicomVolume (1:N, ordered by time)
DicomVolume --processed-by--> ConversionMethod --> ConversionResult
ConversionResult --contains--> MeshData (1:N, one per structure)
MeshData --has--> MaterialConfig (1:1)
AnimatedResult --extends--> ConversionResult
AnimatedResult --contains--> morph_targets (per MeshData, per frame)

SeriesInfo --classifies--> DICOM series for wizard display
DicomConfig --configures--> DICOM pipeline
GallerySlice --extracted-from--> DicomVolume
```

## Cardiac Structure Color Map

Default PBR materials for segmented cardiac structures:

| Structure | Color (RGB) | Alpha | Label |
|-----------|------------|-------|-------|
| Left Ventricle | (0.85, 0.20, 0.20) | 0.9 | left_ventricle |
| Right Ventricle | (0.20, 0.40, 0.85) | 0.9 | right_ventricle |
| Left Atrium | (0.90, 0.45, 0.45) | 0.7 | left_atrium |
| Right Atrium | (0.45, 0.55, 0.90) | 0.7 | right_atrium |
| Myocardium | (0.80, 0.60, 0.50) | 1.0 | myocardium |
| Aorta | (0.90, 0.30, 0.30) | 0.8 | aorta |
| Pulmonary Artery | (0.30, 0.30, 0.90) | 0.8 | pulmonary_artery |
| Generic/Unknown | (0.70, 0.70, 0.70) | 0.8 | unknown |

## CARTO Clinical Colormaps

| Scheme | Clinical Use | Color Stops | Default Range |
|--------|-------------|-------------|---------------|
| LAT | Local activation time | Red → Yellow → Green → Cyan → Blue → Purple | min–max of data |
| Bipolar | Substrate/scar mapping | Red → Yellow → Green → Cyan → Purple | 0.05–1.5 mV |
| Unipolar | Voltage mapping | Red → Yellow → Green → Blue | 3.0–10.0 mV |

## Constants

| Name | Value | Description |
|------|-------|-------------|
| CARTO_LAT_SENTINEL | -10000 | LAT value meaning "no measurement" |
| CARTO_INACTIVE_GROUP_ID | -1000000 | Vertex group ID for inactive fill geometry |
