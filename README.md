# med2glb

Convert medical imaging data (DICOM, CARTO 3 EP mapping) to GLB 3D models optimized for augmented reality.

Built for clinical AR workflows — point it at a DICOM directory or CARTO export, and get AR-ready GLB files with PBR materials, animations, and clinical heatmaps. Designed for HoloLens and other AR/MR headsets.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Data Types](#supported-data-types)
  - [CARTO 3 EP Mapping](#carto-3-ep-mapping)
  - [DICOM Volumes & Cine](#dicom-volumes--cine)
  - [Gallery Mode](#gallery-mode)
- [Conversion Methods](#conversion-methods)
- [GLB Compression](#glb-compression)
- [CLI Reference](#cli-reference)
- [Architecture](#architecture)
- [Development](#development)

---

## Installation

```bash
pip install med2glb
```

For AI-powered cardiac segmentation (TotalSegmentator):

```bash
pip install med2glb[ai]
```

Development setup:

```bash
git clone https://github.com/FlorianKehrle/med2glb.git
cd med2glb
pip install -e ".[dev]"
```

### Optional: Compression tools

For best GLB compression (especially animated models for HoloLens/AR), install these external tools:

**KTX2 texture compression** — [Khronos KTX-Software](https://github.com/KhronosGroup/KTX-Software/releases) (`toktx`)

1. Download from [KTX-Software releases](https://github.com/KhronosGroup/KTX-Software/releases) (`KTX-Software-x.x.x-Windows-x64.exe`)
2. Run the installer — installs to `C:\Program Files\KTX-Software\bin`
3. Add to PATH if not done automatically: **System Environment Variables → Path → New →** `C:\Program Files\KTX-Software\bin`
4. Verify: `toktx --version`

**Meshopt compression** — [meshoptimizer / gltfpack](https://github.com/zeux/meshoptimizer/releases) (`gltfpack`)

1. Download from [meshoptimizer releases](https://github.com/zeux/meshoptimizer/releases) (`gltfpack-x.xx-windows.exe`)
2. Rename to `gltfpack.exe` and place in a directory on your PATH
3. Verify: `gltfpack`

> Both tools are optional. Without them, `--compress` falls back to texture downscaling.

---

## Quick Start

Point med2glb at any directory. It detects the data type, shows a summary, and guides you through relevant options:

```bash
med2glb ./patient_data/
```

The tool will:
1. **Detect** — CARTO export, DICOM volume, cine clip, or single image
2. **Summarize** — Rich table with maps/series, metadata, and recommendations
3. **Prompt** — only relevant options with smart defaults (press Enter to accept)
4. **Convert** — GLB files placed in a `glb/` subfolder

Non-interactive environments (CI/piped input) skip the wizard and use sensible defaults. Pass explicit flags to bypass the wizard entirely:

```bash
med2glb ./Export_Study/ --animate --vectors
med2glb ./echo_folder/ -o heart.glb --method totalseg
```

---

## Supported Data Types

### CARTO 3 EP Mapping

Auto-detects CARTO 3 export directories (containing `.mesh` files) and converts them to GLB with clinical voltage/LAT heatmaps. Supports CARTO versions: old (~2015, v4), v7.1 (v5), and v7.2+ (v6).

**All available coloring variants (LAT, bipolar, unipolar) are generated automatically** in a single run. The expensive subdivision and UV unwrap are done once per mesh; only the lightweight vertex-color mapping is repeated per coloring, adding ~5% overhead. Use `--coloring <scheme>` to restrict to a single coloring when needed.

#### Coloring Schemes

| Scheme | Clinical Use | Color Range |
|---|---|---|
| `lat` | Local activation time | Red (early) → yellow → green → cyan → blue → purple (late) |
| `bipolar` | Substrate/scar mapping | Red (scar, <0.5 mV) → yellow → green → cyan → purple (normal, >1.5 mV) |
| `unipolar` | Voltage mapping | Red (low) → yellow → green → blue (high) |

Animation and conduction vectors are LAT-only; bipolar and unipolar produce static GLBs only.

#### Output Files

For each mesh, the pipeline produces:

| File | Description |
|---|---|
| `{Map}_lat.glb` | Static LAT heatmap |
| `{Map}_lat_animated.glb` | Animated excitation ring overlay |
| `{Map}_lat_animated_vectors.glb` | Animated ring + streamline arrows |
| `{Map}_bipolar.glb` | Static bipolar voltage map |
| `{Map}_unipolar.glb` | Static unipolar voltage map |
| `med2glb_log.txt` | Conversion metadata |

Only colorings with valid data are produced. A legend cylinder and study info panel are embedded in each GLB for AR readability.

#### CARTO Example

```
$ med2glb ./Export_Study/

  CARTO Study: Patient_AF_Ablation (v7.2, 3 maps, 4218 points)

┌───┬────────────────┬──────────┬───────────┬────────┬─────────────────┐
│ # │ Map            │ Vertices │ Triangles │ Points │ LAT Range       │
├───┼────────────────┼──────────┼───────────┼────────┼─────────────────┤
│ 1 │ ReBS_V_SR_11   │   12840  │   25676   │  2105  │ -48 ms → 92 ms  │
│ 2 │ LA_PVI_Post    │    8432  │   16860   │  1650  │ -32 ms → 78 ms  │
│ 3 │ RA_Flutter     │    6210  │   12416   │   463  │ -15 ms → 45 ms  │
└───┴────────────────┴──────────┴───────────┴────────┴─────────────────┘

Mesh selection [all]: 1
Output mode (static / animated / both) [both]:
LAT vectors (yes / no / only) [yes]:
Subdivision level (0-3) [2]:

Done: ReBS_V_SR_11
  Output:
    ReBS_V_SR_11_lat.glb  (1245 KB, static)
    ReBS_V_SR_11_lat_animated.glb  (1823 KB, animated)
    ReBS_V_SR_11_lat_animated_vectors.glb  (2104 KB, animated + vectors)
    ReBS_V_SR_11_bipolar.glb  (1198 KB, static)
    ReBS_V_SR_11_unipolar.glb  (1210 KB, static)
  Time: 2m 15s
```

#### Mesh Overview Table

The wizard displays a summary table with geometry stats, point coverage, and estimated processing time:

| Column | Description |
|---|---|
| **Vertices** | Active vertices (excluding fill geometry with GroupID `-1000000`) |
| **Triangles** | Face count in the original mesh (each face is a triangle) |
| **After Subdiv** | Estimated face count after Loop subdivision at default level 2 (~16× increase) |
| **Active** | Percentage of real geometry vs fill/cap vertices |
| **Points** | Electro-anatomical measurement points from the `_car.txt` file |
| **LAT range** | Min–max local activation time (ms) |
| **Density** | Point-to-vertex ratio — higher means denser mapping coverage |
| **Volts** | Available voltage data: **B** = bipolar, **U** = unipolar |
| **Dimensions** | Bounding box of active vertices (W × H × D in mm) |
| **Est Time** | Rough processing time estimate (subdivide=2, both static+animated) |

#### Mesh Subdivision

Loop subdivision increases mesh resolution before mapping measurement points, producing smooth color gradients via k-NN IDW interpolation instead of blocky nearest-neighbor.

| Level | Faces | Mapping | Use Case |
|---|---|---|---|
| `0` | Original | Nearest-neighbor + linear | Fast preview |
| `1` | ~4x | k-NN IDW (k=6) | Moderate quality |
| `2` (default) | ~16x | k-NN IDW (k=6) | Good balance |
| `3` | ~64x | k-NN IDW (k=6) | Maximum smoothness |

Non-manifold meshes that cannot be subdivided fall back to the original geometry.

#### LAT Conduction Vectors

Animated streamline arrows showing electrical conduction direction, derived from the LAT gradient field. Arrows flow along curved paths with dashes that advance each frame in sync with the excitation ring.

The wizard automatically assesses vector quality (minimum 30 points, ≥20 ms LAT range, ≥15% gradient coverage) and skips vectors for meshes where streamlines would be meaningless. Vector choices:

| Option | Behavior |
|---|---|
| `yes` | Both with-vectors and without-vectors variants |
| `no` | No vector output |
| `only` | Only the animated+vectors variant |

#### Processing Pipeline

1. **Parse** — `.mesh` and `_car.txt` files; strip inactive vertices (GroupID `-1000000`) and fill/cap geometry
2. **Subdivide** — Loop subdivision (~4× faces per level) with k-NN IDW interpolation
3. **Color map** — per-vertex values mapped through clinical colormaps to RGBA (repeated per coloring)
4. **xatlas UV unwrap** (~60-90s) — computed once, shared across all variants
5. **Rasterize** — vertex colors baked into texture with 10-iteration gutter bleeding
6. **Encode** — lossless PNG textures
7. **Build GLB** — geometry, textures, PBR materials, animation keyframes
8. **Compress** (optional) — auto-selects best strategy: meshopt geometry compression + KTX2 textures for animated models, Draco + KTX2 for static

Texture resolution scales with face count: 512px (≤5k), 1024px (≤20k), 2048px (≤80k), 4096px (>80k). For animated output, the full mesh is shared across all 30 frames — only the emissive texture changes per frame.

> **Tip:** Install [`gltfpack`](https://github.com/zeux/meshoptimizer/releases) and [`toktx`](https://github.com/KhronosGroup/KTX-Software/releases) for optimal compression. Animated GLBs (morph targets) benefit most from gltfpack's meshopt compression — KTX2 alone only compresses textures (~15% of total size).

#### Batch Processing

Point med2glb at a parent folder containing multiple CARTO exports:

```
$ med2glb ../clinicalData/CARTO3/Version_7.1.80.33/

  Directory Scan: Version_7.1.80.33
    Type:     CARTO 3 electro-anatomical mapping
    Exports:  3 dataset(s) found

┌───┬─────────────────┬───────────────┬──────┬──────────────────┬────────┬─────────────────────┐
│ # │ Dataset         │ Version       │ Maps │ Mesh names       │ Points │ LAT data            │
├───┼─────────────────┼───────────────┼──────┼──────────────────┼────────┼─────────────────────┤
│ 1 │ Export_Study_AF │ CARTO 3 v7.1  │   2  │ 1-LA, 2-RA       │  3,412 │ 1-LA: -48..92 ms    │
│   │                 │               │      │                  │        │ 2-RA: -15..45 ms    │
│ 2 │ Export_Study_VT │ CARTO 3 v7.1  │   1  │ 1-LV             │  1,820 │ 1-LV: -30..110 ms   │
│ 3 │ Export_Study_FL │ CARTO 3 v7.1  │   1  │ 1-RA             │    463 │ 1-RA: -12..38 ms    │
└───┴─────────────────┴───────────────┴──────┴──────────────────┴────────┴─────────────────────┘

Output [both]:
LAT vectors [yes]:
Subdivision level [2]:

=== Dataset 1/3: Export_Study_AF ===
→ ./Export_Study_AF/glb/1-LA_lat.glb
→ ./Export_Study_AF/glb/1-LA_bipolar.glb
...
```

Each dataset's output goes into its own `glb/` subfolder. Use `--batch` to force batch mode in non-interactive environments:

```bash
med2glb ../CARTO_exports/ --batch --animate --vectors
```

#### CARTO CLI Examples

```bash
# Interactive wizard (recommended)
med2glb ./Export_Study_AF/

# Non-interactive with defaults (all colorings, both static+animated)
med2glb ./Export_Study_AF/ --animate --vectors --subdivide 2

# Only bipolar voltage
med2glb ./Export_Study_AF/ --coloring bipolar

# Quick preview without subdivision
med2glb ./Export_Study_AF/ --subdivide 0

# Maximum smoothness
med2glb ./Export_Study_AF/ --subdivide 3

# Batch all exports in a directory
med2glb ../CARTO_exports/ --batch --animate --vectors
```

---

### DICOM Volumes & Cine

Converts 3D DICOM volumes to isosurface meshes and 2D cine clips to animated textured planes. Multi-series folders are analyzed and classified automatically.

#### DICOM Example

```
$ med2glb ./echo_folder/

  DICOM Series in echo_folder

┌───┬──────────┬──────────────────────────┬───────────┬────────────┬────────────────────┐
│ # │ Modality │ Description              │ Data Type │ Detail     │ Recommended Method │
├───┼──────────┼──────────────────────────┼───────────┼────────────┼────────────────────┤
│ 1 │ US       │ Echokardiographie 4D     │ 2D cine   │ 132 frames │ classical          │
│ 2 │ CT       │ Thorax mit Kontrastmittel│ 3D volume │ 120 slices │ marching-cubes     │
└───┴──────────┴──────────────────────────┴───────────┴────────────┴────────────────────┘

Recommendation: Series 1 (2D cine, 132 frames) → classical

Series [1]:
Method [classical]:
Quality (draft / standard / high) [standard]:
Animate [yes]:

→ ./echo_folder/glb/echo_folder_Echo_3D_animated.glb
```

#### DICOM CLI Examples

```bash
# Interactive wizard
med2glb ./echo_folder/

# Explicit method
med2glb ./ct_folder/ -o heart.glb --method totalseg

# Multi-threshold extraction
med2glb ./ct_folder/ -o layers.glb --method marching-cubes \
  --multi-threshold "200:bone:1.0,100:tissue:0.5,50:skin:0.3"

# Select specific series by UID
med2glb ./multi_series/ --series 1.2.840...
```

---

### Gallery Mode

Converts every DICOM slice to a textured quad GLB. Each series gets individual GLBs, a lightbox grid, and (if spatial metadata exists) a spatial fan layout.

```
output/
  Echokardiographie_4D/
    slice_001.glb
    slice_002.glb
    ...
    lightbox.glb        # grid overview of all slices
    spatial.glb          # only if ImagePositionPatient is present
  Thorax_mit_Kontrastmittel/
    slice_001.glb
    ...
    lightbox.glb
    spatial.glb
```

Multi-frame DICOMs (e.g. ultrasound cine clips) are expanded into one slice per frame with animation. The spatial layout uses `ImagePositionPatient` and `ImageOrientationPatient` to place quads at real-world positions.

```bash
med2glb ./dicom_folder/ -o output/ --gallery
med2glb ./dicom_folder/ -o output/ --gallery --columns 8
med2glb ./dicom_folder/ -o output/ --gallery --no-animate
```

---

## Conversion Methods

| Method | Best For | AI Required | Animation |
|---|---|---|---|
| `classical` (default) | 3D echo, noisy data | No | Yes |
| `marching-cubes` | Quick preview, any modality | No | Yes |
| `totalseg` | Cardiac CT with contrast | Yes | No |
| `chamber-detect` | 3D echo, cardiac volumes | No | Yes |
| `compare` | Method comparison | No* | No |

\* `compare` includes AI methods if installed.

**Classical** — Full pipeline: Gaussian smoothing → adaptive Otsu threshold → morphological cleanup → largest-component extraction → marching cubes. Best for noisy 3D echo data.

**Marching Cubes** — Minimal: threshold → marching cubes. Fast but noisier. Supports multi-threshold mode.

**TotalSegmentator** — AI segmentation of 7 cardiac structures (myocardium, LV, RV, LA, RA, aorta, pulmonary artery) with per-structure PBR materials. Requires contrast-enhanced cardiac CT.

**Chamber Detect** — Multi-structure detection via intensity heuristics (Otsu + morphological ops + connected components). Myocardium plus up to four blood pool chambers.

**Compare** — Runs all methods side-by-side, producing a comparison table with mesh stats and timing.

```bash
med2glb --list-methods          # Show available methods
med2glb ./data/ --method compare  # Compare all methods
```

---

## GLB Compression

Shrink GLB files to a target size with automatic strategy selection:

```bash
med2glb model.glb --compress                          # Auto-selects best strategy
med2glb model.glb --compress --max-size 10            # Custom target (MB)
med2glb model.glb --compress -o small.glb             # Explicit output path
med2glb model.glb --compress --strategy gltfpack      # Force meshopt compression
med2glb model.glb --compress --strategy ktx2          # Force KTX2 textures only
```

| Strategy | Method | Best For | Requires |
|---|---|---|---|
| `auto` (default) | Picks best strategy based on content | Everything | — |
| `gltfpack` | Meshopt geometry + buffer compression | Animated GLBs (morph targets) | `gltfpack` |
| `ktx2` | KTX2 GPU textures via Basis Universal | Texture-heavy static models | `toktx` |
| `draco` | Draco mesh compression (static only) | Static models | — |
| `downscale` | Progressive texture resolution reduction | No tools available | — |
| `jpeg` | JPEG re-encoding with decreasing quality | Quick size reduction | — |

**Auto strategy** selects: animated GLB → gltfpack + KTX2, static GLB → Draco + KTX2, graceful fallback if tools aren't installed.

> **HoloLens/AR note:** Animated CARTO GLBs are typically 50+ MB with 80%+ morph target data. KTX2 alone achieves only ~15% reduction. Install `gltfpack` for meshopt compression to reach 70-85% reduction.

---

## CLI Reference

```
med2glb [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH              Path to DICOM file, directory, or CARTO export

Options:
  -o, --output PATH       Output file path (default: <input>/glb/<name>.glb)
  --batch                 Batch mode: convert all CARTO exports in subdirectories
  --compress              Compress a GLB file to fit a target size
  --max-size INTEGER      Target size in MB for --compress (default: 25)
  --strategy TEXT         Compression strategy: auto, gltfpack, ktx2, draco, downscale, jpeg (default: auto)
  --list-methods          List available methods and exit
  --list-series           List DICOM series and exit
  -v, --verbose           Show detailed processing information
  --version               Show version and exit
```

The wizard is the primary interface — just run `med2glb ./your-data/` and follow the prompts. All pipeline-specific flags (`--method`, `--coloring`, `--animate`, `--subdivide`, `--vectors`, `--threshold`, `--smoothing`, `--faces`, etc.) are accepted but hidden from `--help` to keep the CLI clean. Passing any of these flags bypasses the wizard and runs directly with a hint suggesting the interactive mode.

After each wizard-guided conversion, an **equivalent command** is printed that reproduces the same result non-interactively — useful for scripting, CI pipelines, or sharing with colleagues. The equivalent command is also logged to `med2glb_log.txt`.

---

## AR Viewer Compatibility

| Platform | Viewer | Notes |
|---|---|---|
| **HoloLens** | 3D Viewer / custom MRTK app | Primary target — PBR + animation fully supported |
| **Android** | Scene Viewer | Native GLB support |
| **Web** | [model-viewer](https://modelviewer.dev/) | Web component for browser viewing |
| **iOS** | USDZ conversion needed | Use external tools for Apple AR |

---

## Architecture

```
src/med2glb/
├── cli.py              # Typer CLI entry point
├── cli_wizard.py       # Interactive wizard (data-driven prompts)
├── core/               # Data types (MeshData, DicomVolume, CartoConfig, etc.)
├── io/                 # DICOM reading, echo reader, CARTO reader/mapper, exporters
├── methods/            # Pluggable conversion methods (registry pattern)
│   ├── classical.py    # Gaussian smoothing + adaptive threshold
│   ├── marching_cubes.py  # Basic isosurface extraction
│   ├── totalseg.py     # TotalSegmentator AI segmentation
│   └── chamber_detect.py  # Cardiac chamber detection
├── gallery/            # Gallery mode (individual, lightbox, spatial)
├── mesh/               # Taubin smoothing, decimation, temporal processing, LAT vectors
└── glb/                # GLB builder, morph targets, CARTO animation, arrows, textures, compression
```

---

## Development

```bash
# Run tests
pytest
pytest --cov=med2glb

# Run only unit tests (fast)
pytest tests/unit/ -x -q

# Install dev dependencies
pip install -e ".[dev]"
```
