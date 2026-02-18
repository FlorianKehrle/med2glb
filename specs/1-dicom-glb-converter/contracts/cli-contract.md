# CLI Contract: med2glb

**Date**: 2026-02-09

## Command: `med2glb`

### Synopsis

```
med2glb [OPTIONS] INPUT_PATH
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| INPUT_PATH | path | Yes | Path to a DICOM file or directory containing DICOM files |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o`, `--output` | path | `output.glb` | Output file path |
| `-m`, `--method` | choice | `classical` | Conversion method: `marching-cubes`, `classical`, `totalseg`, `medsam2` |
| `-f`, `--format` | choice | `glb` | Output format: `glb`, `stl`, `obj` |
| `--animate` | flag | false | Enable animation for temporal (4D) data |
| `--threshold` | float | auto | Intensity threshold for isosurface extraction |
| `--smoothing` | int | 15 | Taubin smoothing iterations (0 to disable) |
| `--faces` | int | 80000 | Target triangle count after decimation |
| `--alpha` | float | 1.0 | Global transparency (0.0-1.0) for non-segmented output |
| `--multi-threshold` | str | None | Multi-threshold config: "val1:label1:alpha1,val2:label2:alpha2" |
| `--series` | str | None | Select specific DICOM series by UID (partial match supported) |
| `--list-methods` | flag | false | List available conversion methods and exit |
| `--list-series` | flag | false | List DICOM series found in input directory and exit |
| `-v`, `--verbose` | flag | false | Show detailed processing information |
| `--version` | flag | false | Show version and exit |
| `--help` | flag | false | Show help and exit |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (invalid input, processing failure) |
| 2 | Invalid arguments or usage error |
| 3 | Missing optional dependency (AI method not installed) |
| 4 | No valid DICOM data found in input |

### Usage Examples

```bash
# Basic conversion (auto-detect input type, classical method)
med2glb ./dicom_dir/ -o heart.glb

# 3D echo with animation
med2glb ./echo_data/ -o beating_heart.glb --animate

# Try different methods for comparison
med2glb ./echo_data/ -o heart_mc.glb --method marching-cubes --threshold 400
med2glb ./echo_data/ -o heart_cl.glb --method classical --smoothing 20
med2glb ./echo_data/ -o heart_ai.glb --method medsam2 --animate

# CT with TotalSegmentator segmentation
med2glb ./ct_scan/ -o heart_segmented.glb --method totalseg

# Multi-threshold layered output
med2glb ./ct_scan/ -o heart_layers.glb --multi-threshold "200:blood_pool:0.3,500:myocardium:1.0"

# Export as STL for Blender post-processing
med2glb ./dicom_dir/ -o heart.stl --format stl

# List available methods
med2glb --list-methods

# List series in a directory
med2glb ./dicom_dir/ --list-series

# Select a specific series
med2glb ./dicom_dir/ -o heart.glb --series "1.2.840.113619"
```

### Output: `--list-methods`

```
Available conversion methods:

  marching-cubes   Basic isosurface extraction at configurable threshold.
                   Best for: Quick preview of any modality.
                   Requires: No additional dependencies.

  classical        Full pipeline: volume smoothing, adaptive threshold,
                   morphological ops, Taubin smoothing, decimation.
                   Best for: 3D echo data, noisy datasets.
                   Requires: No additional dependencies.

  totalseg         AI cardiac segmentation via TotalSegmentator.
                   Segments: LV, RV, LA, RA, aorta, myocardium.
                   Best for: Cardiac CT with contrast.
                   Requires: pip install med2glb[ai]

  medsam2          AI segmentation via MedSAM2.
                   Best for: 3D echo, general cardiac imaging.
                   Requires: pip install med2glb[ai]
```

### Standard Output Behavior

- Progress bar shown during processing (via `rich`)
- Summary printed on completion: input type, method used, output path, mesh stats (vertices, faces, file size)
- Warnings printed to stderr (quality issues, data anomalies)
- `--verbose` adds detailed per-step timing and intermediate stats
