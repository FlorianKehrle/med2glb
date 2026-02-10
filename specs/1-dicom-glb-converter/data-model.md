# Data Model: DICOM to GLB Converter

**Date**: 2026-02-09
**Feature**: 1-dicom-glb-converter

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

**Validation**:
- voxels must be 3D with shape > (2, 2, 2)
- pixel_spacing values must be > 0
- slice_thickness must be > 0

### TemporalSequence
Ordered collection of DicomVolumes representing time frames.

| Field | Type | Description |
|-------|------|-------------|
| frames | list[DicomVolume] | Ordered volumes per cardiac phase |
| frame_count | int | Number of time frames |
| temporal_resolution | float or None | Time between frames in ms |
| is_loop | bool | Whether sequence represents a complete cardiac cycle |

**Validation**:
- frame_count >= 2
- All frames must have identical spatial dimensions (Z, Y, X)
- All frames must have identical pixel_spacing and slice_thickness

**State transitions**:
- Loading → Validated → Processed (after conversion method applied)

### MeshData
Triangulated surface mesh with material properties.

| Field | Type | Description |
|-------|------|-------------|
| vertices | ndarray (float32) | Vertex positions [N, 3] |
| faces | ndarray (int32) | Triangle indices [M, 3] |
| normals | ndarray (float32) or None | Vertex normals [N, 3] |
| structure_name | str | Anatomical label (e.g., "left_ventricle", "myocardium") |
| material | MaterialConfig | Color, transparency, PBR properties |

**Validation**:
- vertices.shape[1] == 3
- faces.shape[1] == 3
- All face indices < len(vertices)
- No degenerate triangles (area > 0)

### MaterialConfig
PBR material properties for a cardiac structure.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| base_color | tuple[float, float, float] | (0.8, 0.2, 0.2) | RGB color [0-1] |
| alpha | float | 1.0 | Transparency [0-1], 0=transparent |
| metallic | float | 0.0 | PBR metallic factor |
| roughness | float | 0.7 | PBR roughness factor |
| name | str | "" | Material display name |

### MethodParams
Configuration parameters for a conversion method.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| threshold | float or None | None | Intensity threshold for isosurface |
| smoothing_iterations | int | 15 | Taubin smoothing iterations |
| target_faces | int | 80000 | Target triangle count after decimation |
| multi_threshold | list[ThresholdLayer] or None | None | Multi-threshold config for layered output |

### ThresholdLayer
Single layer in multi-threshold extraction.

| Field | Type | Description |
|-------|------|-------------|
| threshold | float | Intensity threshold for this layer |
| label | str | Layer name (e.g., "blood_pool", "myocardium") |
| material | MaterialConfig | Visual properties for this layer |

### ConversionResult
Output from a conversion method.

| Field | Type | Description |
|-------|------|-------------|
| meshes | list[MeshData] | One or more meshes (multiple for segmented output) |
| method_name | str | Which method produced this result |
| processing_time | float | Duration in seconds |
| warnings | list[str] | Any quality or data warnings |

### AnimatedResult
Extends ConversionResult for temporal data.

| Field | Type | Description |
|-------|------|-------------|
| base_meshes | list[MeshData] | Meshes for the first frame (base topology) |
| morph_targets | list[list[ndarray]] | Per-mesh vertex displacements for each frame |
| frame_times | list[float] | Keyframe timestamps in seconds |
| loop_duration | float | Total animation cycle duration in seconds |

## Relationships

```
TemporalSequence --contains--> DicomVolume (1:N, ordered by time)
DicomVolume --processed-by--> ConversionMethod --> ConversionResult
ConversionResult --contains--> MeshData (1:N, one per structure)
MeshData --has--> MaterialConfig (1:1)
AnimatedResult --extends--> ConversionResult
AnimatedResult --contains--> morph_targets (per MeshData, per frame)
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
