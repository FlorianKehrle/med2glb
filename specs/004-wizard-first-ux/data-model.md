# Data Model: Wizard-First UX

**Branch**: `004-wizard-first-ux` | **Date**: 2026-03-11

## Entities

### CartoConfig (existing — no changes)

The CARTO wizard already returns a `CartoConfig` dataclass with all user choices. No schema changes needed.

| Field | Type | Description |
|-------|------|-------------|
| input_path | Path | CARTO export directory |
| name | str | Auto-generated output name prefix |
| selected_mesh_indices | list[int] \| None | User-selected meshes (None = all) |
| colorings | list[str] | Selected coloring schemes |
| subdivide | int | Subdivision level (0-3) |
| animate | bool | Enable animated output |
| static | bool | Enable static output |
| vectors | str | "yes" / "no" / "only" |
| vector_mesh_indices | list[int] \| None | Meshes suitable for vectors |

### DicomConfig (existing — no changes)

The DICOM wizard already returns a `DicomConfig` dataclass. No schema changes needed.

| Field | Type | Description |
|-------|------|-------------|
| input_path | Path | DICOM file or directory |
| name | str | Auto-generated output name prefix |
| method | str | Conversion method |
| animate | bool | Enable temporal animation |
| smoothing | int | Taubin smoothing iterations |
| target_faces | int | Decimation target |
| series_uid | str \| None | Selected series UID |

### SeriesInfo (existing — extended)

Add `spacing` and `est_time` fields.

| Field | Type | Status | Description |
|-------|------|--------|-------------|
| series_uid | str | existing | DICOM Series Instance UID |
| modality | str | existing | CT, US, MR, etc. |
| description | str | existing | Series description |
| file_count | int | existing | Number of DICOM files |
| data_type | str | existing | "2D cine", "3D volume", etc. |
| detail | str | existing | Summary string |
| dimensions | str | existing | Spatial dims (e.g., "512×512×120") |
| recommended_method | str | existing | Best method for this series |
| recommended_output | str | existing | Expected output type |
| is_multiframe | bool | existing | Whether multi-frame |
| number_of_frames | int | existing | Frame count |
| **spacing** | **str \| None** | **NEW** | Pixel spacing + slice thickness (e.g., "0.5 × 0.5 × 1.0 mm") |
| **est_time** | **str \| None** | **NEW** | Estimated processing time (e.g., "~30s", "~2 min") |

### EquivalentCommand (new concept — not a dataclass)

A plain string built from config + paths. Not persisted as structured data.

**CARTO template**:
```
med2glb "{input_path}" --coloring {colorings} --subdivide {level} [--animate] [--vectors] -o "{output}"
```

**DICOM template**:
```
med2glb "{input_path}" --method {method} --smoothing {n} --faces {n} [--animate] [--series {uid}] -o "{output}"
```

**Batch template**:
```
med2glb "{input_path}" --batch --coloring {colorings} --subdivide {level} [--animate] [--vectors]
```

## State Transitions

No state machines in this feature. The wizard flow is linear:
1. Load data → 2. Show summary → 3. Prompt choices → 4. Convert → 5. Print equivalent command → 6. Log

## Validation Rules

- Equivalent command must contain only hidden + visible flags (no wizard-only concepts).
- All paths in equivalent command must be quoted if they contain spaces.
- Duration format must be `{N}s`, `{M}m {S}s`, or `{H}h {M}m {S}s` — never raw float seconds.
