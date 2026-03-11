# CLI Contract: med2glb

**Date**: 2026-02-09 (initial), 2026-03-11 (updated)

## Command: `med2glb`

### Synopsis

```
med2glb [OPTIONS] INPUT_PATH
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| INPUT_PATH | path | Yes | Path to DICOM file/directory, CARTO export, or GLB file (for --compress) |

### Options

#### Core Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o`, `--output` | path | `<input>/glb/<name>.glb` | Output file path |
| `-v`, `--verbose` | flag | false | Show detailed processing information |
| `--version` | flag | false | Show version and exit |

#### DICOM Method & Processing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-m`, `--method` | choice | `classical` | Conversion method: `marching-cubes`, `classical`, `totalseg`, `chamber-detect`, `compare` |
| `--threshold` | float | auto | Intensity threshold for isosurface extraction |
| `--smoothing` | int | 15 | Taubin smoothing iterations (0 to disable) |
| `--faces` | int | 80000 | Target triangle count after decimation |
| `--alpha` | float | 1.0 | Global transparency (0.0-1.0) for non-segmented output |
| `--multi-threshold` | str | None | Multi-threshold config: "val1:label1:alpha1,val2:label2:alpha2" |
| `--animate` | flag | false | Enable animation for temporal (4D) data |
| `--no-animate` | flag | false | Force static output even if temporal data detected |
| `--series` | str | None | Select specific DICOM series by UID (partial match) |

#### CARTO-Specific Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--coloring` | choice | `all` | Coloring scheme: `lat`, `bipolar`, `unipolar`, `all` |
| `--subdivide` | int | 2 | CARTO mesh subdivision level (0-3) |
| `--vectors` | flag | false | Add animated LAT streamline arrows (CARTO LAT maps only) |

#### Gallery Mode

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--gallery` | flag | false | Gallery mode: individual GLBs, lightbox grid, spatial fan |
| `--columns` | int | 6 | Lightbox grid columns |

#### Batch Processing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--batch` | flag | false | Find all CARTO exports in subdirectories and convert with shared settings |

#### GLB Compression

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--compress` | flag | false | Compress a GLB file to fit a target size |
| `--max-size` | int | 25 | Target size in MB for --compress |
| `--strategy` | choice | `ktx2` | Compression strategy: `ktx2`, `draco`, `downscale`, `jpeg` |

#### Utility Commands

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--list-methods` | flag | false | List available conversion methods and exit |
| `--list-series` | flag | false | List DICOM series found in input directory and exit |

### Control Flow

1. If `--compress`: compress the input GLB file and exit
2. If interactive (TTY) + no pipeline flags: run the interactive wizard
3. If `--batch`: find CARTO subdirectories, convert each with shared settings
4. If directory auto-detects as CARTO: run CARTO pipeline
5. Otherwise: run DICOM pipeline
6. If `--gallery`: build gallery GLBs instead of 3D mesh

**Pipeline flags** (any of these bypasses the wizard): `--method`, `--coloring`, `--animate`, `--no-animate`, `--vectors`, `--subdivide`, `--threshold`, `--smoothing`, `--faces`, `--alpha`, `--multi-threshold`, `--series`, `--gallery`, `--batch`, `--compress`

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (invalid input, processing failure) |
| 2 | Invalid arguments or usage error |
| 3 | Missing optional dependency (AI method not installed) |
| 4 | No valid data found in input (no DICOM or CARTO) |

### Usage Examples

#### CARTO Conversion

```bash
# Interactive wizard (recommended)
med2glb ./Export_Study_AF/

# Non-interactive with defaults (all colorings, both static+animated)
med2glb ./Export_Study_AF/ --animate --vectors --subdivide 2

# Only bipolar voltage map
med2glb ./Export_Study_AF/ --coloring bipolar

# Quick preview without subdivision
med2glb ./Export_Study_AF/ --subdivide 0

# Batch all exports in a parent directory
med2glb ../CARTO_exports/ --batch --animate --vectors
```

#### DICOM Conversion

```bash
# Interactive wizard
med2glb ./echo_folder/

# Explicit method and output
med2glb ./ct_folder/ -o heart.glb --method totalseg

# Multi-threshold extraction
med2glb ./ct_folder/ -o layers.glb --method marching-cubes \
  --multi-threshold "200:bone:1.0,100:tissue:0.5,50:skin:0.3"

# Select specific series
med2glb ./multi_series/ --series 1.2.840...

# Compare all methods
med2glb ./data/ --method compare
```

#### Gallery Mode

```bash
med2glb ./dicom_folder/ --gallery
med2glb ./dicom_folder/ --gallery --columns 8
med2glb ./dicom_folder/ --gallery --no-animate
```

#### GLB Compression

```bash
med2glb model.glb --compress
med2glb model.glb --compress --max-size 10
med2glb model.glb --compress --strategy draco
med2glb model.glb --compress -o small.glb
```

### Output: `--list-methods`

```
Available conversion methods:

  marching-cubes   Basic isosurface extraction at configurable threshold.
                   Best for: Quick preview of any modality.
                   Requires: No additional dependencies.

  classical        Region-growing segmentation with Otsu thresholding.
                   Best for: Generic cardiac segmentation.
                   Requires: No additional dependencies.

  totalseg         AI segmentation of 104 anatomical structures.
                   Best for: Multi-organ cardiac/thoracic segmentation.
                   Requires: pip install med2glb[ai]

  chamber-detect   Cardiac chamber detection (LV, RV, LA, RA).
                   Best for: Focused cardiac chamber meshes.
                   Requires: pip install med2glb[ai]

  compare          Run all methods side-by-side with comparison table.
                   Best for: Finding the best method for your data.
                   Requires: No additional dependencies (AI methods included if installed).
```

### CARTO Output Files

For each mesh, the pipeline produces (depending on options):

| File | Description |
|------|-------------|
| `{Map}_lat.glb` | Static LAT heatmap |
| `{Map}_lat_animated.glb` | Animated excitation ring overlay |
| `{Map}_lat_animated_vectors.glb` | Animated ring + streamline arrows |
| `{Map}_bipolar.glb` | Static bipolar voltage map |
| `{Map}_unipolar.glb` | Static unipolar voltage map |
| `med2glb_log.txt` | Conversion metadata |

### Standard Output Behavior

- Interactive wizard on TTY with data summary tables
- Progress bar shown during processing (via `rich`)
- Summary printed on completion: input type, method used, output path, mesh stats
- Warnings printed to stderr (quality issues, data anomalies)
- `--verbose` adds detailed per-step timing and intermediate stats
- Non-interactive/piped input: wizard skipped, sensible defaults applied
