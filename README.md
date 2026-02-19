# med2glb

Convert DICOM and CARTO 3 electro-anatomical mapping data to GLB 3D models optimized for AR viewing.

Supports 3D echocardiography, cardiac CT/MRI, 2D cine clips, single DICOM slices, and CARTO 3 EP mapping data. Outputs GLB files with PBR materials, animated cardiac cycles, per-vertex voltage/LAT heatmaps, and multi-structure segmentation with per-structure coloring.

## Why med2glb?

No existing end-to-end CLI tool converts DICOM directly to animated GLB for augmented reality. Existing tools (3D Slicer, DicomToMesh, InVesalius) produce static STL/OBJ via a GUI. med2glb fills this gap as a single command that takes DICOM data in and produces AR-ready GLB out.

**Key features:**

- **CARTO 3 EP mapping support** -- auto-detects CARTO export directories; renders LAT, bipolar voltage, and unipolar voltage heatmaps as per-vertex colored GLBs; animated LAT wavefront sweep; Loop subdivision with IDW interpolation for smooth color maps
- **Animated cardiac output** -- 2D cine clips become animated GLB with per-frame texture planes; 3D temporal volumes use morph targets; CARTO LAT wavefront animation
- **Gallery mode** -- convert every slice to textured quads with three layouts: individual GLBs, lightbox grid, and spatial fan positioned using DICOM metadata
- **Pluggable conversion methods** -- classical (Gaussian + adaptive threshold), marching cubes, TotalSegmentator (CT), and MedSAM2 (echo/general)
- **Automatic series detection** -- multi-series DICOM folders are analyzed and classified (3D volume, 2D cine, still image) with per-series conversion recommendations
- **Interactive series selection** -- choose which series to convert from a Rich table, or let the tool auto-select the best one
- **Multi-format export** -- GLB (with animation and PBR materials), STL, and OBJ
- **AR-optimized meshes** -- Taubin smoothing (volume-preserving), decimation to configurable triangle count, and configurable transparency
- **GLB size constraint** -- automatic compression to fit AR viewer limits (default 99 MB) with three strategies: Draco mesh compression, texture downscaling, or JPEG re-encoding
- **Multi-threshold layered output** -- extract multiple structures at different intensity thresholds with per-layer colors and transparency

## Installation

```bash
pip install med2glb
```

For AI-powered segmentation (TotalSegmentator, MedSAM2):

```bash
pip install med2glb[ai]
```

Development:

```bash
git clone https://github.com/FlorianKehrle/med2glb.git
cd med2glb
pip install -e ".[dev]"
```

## Quick Start

```bash
# Convert a DICOM directory — output placed next to input with auto-detected type
med2glb ./echo_folder/        # → ./echo_folder_Echo_3D_animated.glb
med2glb ./ct_scan/            # → ./ct_scan_CT_3D.glb
med2glb image.dcm             # → ./image_Echo_2D.glb

# Explicit output name
med2glb ./echo_folder/ -o heart.glb

# Output to a directory — filename derived from input
med2glb ./echo_folder/ -o output/   # → output/echo_folder_Echo_3D_animated.glb

# Animated 3D echo (cardiac cycle with morph targets)
med2glb ./echo_folder/ -o heart.glb --animate

# Gallery mode: individual GLBs + lightbox grid + spatial fan
med2glb ./dicom_folder/ --gallery

# Cardiac CT with AI segmentation (7 structures)
med2glb ./ct_folder/ -o heart.glb --method totalseg

# Multi-threshold layered output
med2glb ./data/ -o layers.glb --multi-threshold "200:bone:1.0,100:tissue:0.5"

# Export as STL or OBJ
med2glb ./data/ -o model.stl -f stl

# Limit output to 50 MB with JPEG compression
med2glb ./data/ --max-size 50 --compress jpeg
```

## CARTO 3 Electro-Anatomical Mapping

med2glb auto-detects CARTO 3 export directories (containing `.mesh` files) and converts them to GLB with per-vertex coloring from the mapping data. Supports old CARTO (~2015, v4), v7.1 (v5), and v7.2+ (v6) formats.

```bash
# Auto-detect and convert with LAT coloring (default)
med2glb ./Export_Study-1-01_09_2023-20-30-09/

# Bipolar voltage map (scar mapping)
med2glb ./Export_Study/ --coloring bipolar

# Unipolar voltage map
med2glb ./Export_Study/ --coloring unipolar

# Animated LAT wavefront sweep
med2glb ./Export_Study/ --animate

# Smoother color maps with more subdivision (default: 1)
med2glb ./Export_Study/ --subdivide 2

# Original nearest-neighbor mapping (no subdivision)
med2glb ./Export_Study/ --subdivide 0

# Explicit output path
med2glb ./Export_Study/ -o left_atrium_lat.glb --coloring lat
```

**Mesh subdivision (`--subdivide`):**

By default, CARTO meshes are Loop-subdivided once (`--subdivide 1`) before mapping measurement points. This increases mesh resolution (~4x faces per level) and uses k-NN inverse-distance weighting (IDW) interpolation instead of single nearest-neighbor, producing smooth color gradients between measurement points. Use `--subdivide 0` to get the original blocky nearest-neighbor behavior, or `--subdivide 2`/`3` for even smoother results at the cost of more vertices. Non-manifold meshes that cannot be subdivided automatically fall back to the original geometry.

| Level | Faces | Mapping | Use Case |
|---|---|---|---|
| `0` | Original | Nearest-neighbor + linear interpolation | Fast preview, original behavior |
| `1` (default) | ~4x | k-NN IDW (k=6) | Good balance of quality and speed |
| `2` | ~16x | k-NN IDW (k=6) | High quality, more vertices |
| `3` | ~64x | k-NN IDW (k=6) | Maximum smoothness, large meshes |

**Coloring schemes:**

| Scheme | Clinical Use | Color Range |
|---|---|---|
| `lat` (default) | Local activation time | Red (early) → yellow → green → cyan → blue → purple (late) |
| `bipolar` | Substrate/scar mapping | Red (scar, <0.5 mV) → yellow → green → cyan → purple (normal, >1.5 mV) |
| `unipolar` | Voltage mapping | Red (low) → yellow → green → blue (high) |

**What gets parsed:**
- `.mesh` files: 3D surface geometry with per-vertex group IDs (inactive vertices filtered out)
- `_car.txt` files: sparse measurement points mapped to mesh vertices via k-NN IDW interpolation (with subdivision) or nearest-neighbor + linear interpolation (without)
- LAT sentinel value `-10000` is treated as unmapped (rendered as transparent gray)

When a CARTO export contains multiple meshes (e.g. LA, RA), an interactive selection table is displayed.

## Gallery Mode

Gallery mode converts every DICOM slice to a textured quad GLB -- nothing is filtered or dropped, regardless of slice dimensions. Each series gets its own subfolder with individual GLBs, a lightbox overview, and (if spatial metadata is available) a spatial fan.

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

Multi-frame DICOMs (e.g. ultrasound cine clips) are automatically expanded into one slice per frame with animation enabled by default. Use `--no-animate` to force static output.

```bash
# All series, each in its own subfolder
med2glb ./dicom_folder/ -o output/ --gallery

# Custom grid columns (default: 6)
med2glb ./dicom_folder/ -o output/ --gallery --columns 8

# Force static output from temporal data
med2glb ./dicom_folder/ -o output/ --gallery --no-animate

# Select a specific series by UID
med2glb ./dicom_folder/ -o output/ --gallery --series 1.2.840...
```

The spatial layout uses `ImagePositionPatient` and `ImageOrientationPatient` from the DICOM metadata to place each quad at its real-world position. If no spatial metadata is available, the spatial output is skipped.

## GLB Size Constraint

Output GLB files are automatically constrained to 99 MB (configurable) to fit AR viewer limits. If a GLB exceeds the limit, the original file is kept and a compressed copy is saved alongside it with a `_compressed` suffix (e.g. `lightbox.glb` + `lightbox_compressed.glb`). Three compression strategies are available:

| Strategy | Method | Quality | Speed |
|---|---|---|---|
| `draco` (default) | Draco mesh compression + texture downscale fallback | Best | Moderate |
| `downscale` | Progressive texture resolution reduction (lossless PNG) | High | Fast |
| `jpeg` | JPEG re-encoding with decreasing quality | Lower | Fast |

Draco compresses mesh geometry losslessly and falls back to texture downscaling if still over the limit. The downscale strategy keeps PNG format but reduces resolution in steps (75%, 50%, 37.5%, 25%). The JPEG strategy re-encodes textures with progressively lower quality (90 down to 30) and scaling.

```bash
# Default: 99 MB limit with Draco strategy
med2glb ./data/ -o output.glb

# Custom size limit
med2glb ./data/ -o output.glb --max-size 50

# Use JPEG compression (smaller but lossy)
med2glb ./data/ -o output.glb --compress jpeg

# Disable size constraint
med2glb ./data/ -o output.glb --max-size 0
```

## Series Selection

When a DICOM folder contains multiple series, med2glb analyzes each series and presents an interactive selection table:

```
  DICOM Series in echo_folder
┌───┬──────────┬──────────────────────────┬─────────────┬─────────────┬──────────┬────────────────────┬────────────────────┐
│ # │ Modality │ Description              │ Data Type   │ Detail      │ Animated │ Recommended Output │ Recommended Method │
├───┼──────────┼──────────────────────────┼─────────────┼─────────────┼──────────┼────────────────────┼────────────────────┤
│ 1 │ US       │ Echokardiographie 4D     │ 2D cine     │ 132 frames  │   Yes    │ textured plane     │ classical          │
│ 2 │ US       │ (no desc)                │ 2D cine     │ 65 frames   │   Yes    │ textured plane     │ classical          │
│ 3 │ CT       │ Thorax mit Kontrastmittel│ 3D volume   │ 120 slices  │    No    │ 3D mesh            │ marching-cubes     │
└───┴──────────┴──────────────────────────┴─────────────┴─────────────┴──────────┴────────────────────┴────────────────────┘

Recommendation: Series 1 (2D cine, 132 frames) → classical

Enter number (1-3), comma-separated (1,3), or 'all' [1]: 1,3
```

- Enter a number (`1`), comma-separated list (`1,3`), or `all`
- The recommended conversion method is auto-selected per series unless `--method` is explicitly set
- Single-series folders proceed automatically without prompting
- Non-interactive (piped) input auto-selects the best series

Use `--list-series` to view the table without converting, or `--series <UID>` to skip selection entirely.

## Methods

| Method | Best For | AI Required | Animation |
|---|---|---|---|
| `classical` (default) | 3D echo, noisy data | No | Yes |
| `marching-cubes` | Quick preview, any modality | No | Yes |
| `totalseg` | Cardiac CT with contrast | Yes | No |
| `medsam2` | 3D echo, general cardiac | Yes | Yes |

List available methods and their install status:

```bash
med2glb --list-methods
```

### Classical (default)

Full pipeline: Gaussian smoothing, adaptive Otsu threshold, morphological cleanup, largest-component extraction, then marching cubes for surface extraction. Best for noisy 3D echo data.

```bash
# Basic -- auto-detects threshold via Otsu's method
med2glb ./echo_folder/ -o heart.glb

# Explicit intensity threshold
med2glb ./echo_folder/ -o heart.glb --threshold 400

# Animated cardiac cycle (morph targets from temporal 3D echo)
med2glb ./echo_folder/ -o heart.glb --animate

# More smoothing for noisy data (default: 15 iterations)
med2glb ./echo_folder/ -o heart.glb --smoothing 30

# Fewer triangles for a lighter file (default: 80000)
med2glb ./echo_folder/ -o heart.glb --faces 40000

# Semi-transparent output for layered AR viewing
med2glb ./echo_folder/ -o heart.glb --alpha 0.6
```

### Marching Cubes

Minimal pipeline: threshold then marching cubes. No morphological cleanup -- fast but noisier. Supports multi-threshold mode for extracting multiple structures at different intensity levels.

```bash
# Basic with auto threshold
med2glb ./data/ -o model.glb --method marching-cubes

# Multi-threshold: extract bone and soft tissue as separate layers
med2glb ./ct_folder/ -o layers.glb --method marching-cubes \
  --multi-threshold "200:bone:1.0,100:tissue:0.5,50:skin:0.3"
```

### TotalSegmentator

Uses TotalSegmentator's `heartchambers_highres` task to segment 7 cardiac structures into a single GLB with per-structure PBR materials:

| Structure | Color |
|---|---|
| Myocardium | Brown-red |
| Left ventricle | Red |
| Right ventricle | Blue |
| Left atrium | Orange |
| Right atrium | Teal |
| Aorta | Pink-red |
| Pulmonary artery | Purple |

Requires a contrast-enhanced cardiac CT for best results. The `heartchambers_highres` model may require a TotalSegmentator license for commercial use.

```bash
# Segment all 7 cardiac structures
med2glb ./ct_folder/ -o heart.glb --method totalseg

# Reduce mesh complexity (default: 80000 faces per structure)
med2glb ./ct_folder/ -o heart.glb --method totalseg --faces 50000

# Make all structures semi-transparent
med2glb ./ct_folder/ -o heart.glb --method totalseg --alpha 0.7
```

The `--threshold` and `--animate` options have no effect on `totalseg` (AI-driven segmentation, single time point).

### MedSAM2

AI segmentation for 3D echo and general cardiac imaging. Produces multi-structure output with per-structure colors. Currently uses a heuristic pseudo-segmentation; full MedSAM2 model integration is planned.

```bash
# Segment cardiac structures from 3D echo
med2glb ./echo_folder/ -o heart.glb --method medsam2

# Animated cardiac cycle
med2glb ./echo_folder/ -o heart.glb --method medsam2 --animate
```

Requires AI dependencies: `pip install med2glb[ai]`

## CLI Reference

```
med2glb [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH              Path to DICOM file or directory

Options:
  -o, --output PATH       Output file path (default: <input>_<modality>_<type>.glb next to input)
  -m, --method TEXT        Conversion method: classical, marching-cubes, totalseg, medsam2
  -f, --format TEXT        Output format: glb, stl, obj (default: glb)
  --coloring TEXT         CARTO coloring: lat, bipolar, unipolar (default: lat)
  --subdivide INTEGER     CARTO mesh subdivision level 0-3 (default: 1)
  --animate               Enable animation for temporal data
  --no-animate            Force static output even if temporal data is detected
  --threshold FLOAT       Intensity threshold for isosurface extraction
  --smoothing INTEGER     Taubin smoothing iterations, 0 to disable (default: 15)
  --faces INTEGER         Target triangle count after decimation (default: 80000)
  --alpha FLOAT           Global transparency 0.0-1.0 (default: 1.0)
  --multi-threshold TEXT   Multi-threshold config: "val:label:alpha,..."
  --max-size INTEGER      Maximum output GLB file size in MB, 0 to disable (default: 99)
  --compress TEXT         Compression strategy: draco, downscale, jpeg (default: draco)
  --gallery               Gallery mode: individual GLBs, lightbox grid, and spatial fan
  --columns INTEGER       Lightbox grid columns in gallery mode (default: 6)
  --series TEXT           Select DICOM series by UID (partial match)
  --list-methods          List available methods and exit
  --list-series           List DICOM series in input directory and exit
  -v, --verbose           Show detailed processing information
  --version               Show version and exit
```

## Output Formats

| Format | Animation | Materials | Use Case |
|---|---|---|---|
| GLB | Yes (per-frame planes / morph targets) | PBR with transparency | AR viewers, web (model-viewer) |
| STL | No | No | 3D printing, CAD |
| OBJ | No | Basic | General 3D software |

## AR Viewer Compatibility

- **Android**: Scene Viewer (native GLB support)
- **Web**: [model-viewer](https://modelviewer.dev/) web component
- **iOS**: USDZ conversion needed (use external tools)

## Architecture

```
src/med2glb/
├── cli.py              # Typer CLI entry point
├── core/               # Data types (MeshData, DicomVolume, GallerySlice, etc.)
├── io/                 # DICOM reading, echo reader, CARTO reader/mapper, exporters
├── methods/            # Pluggable conversion methods (registry pattern)
│   ├── classical.py    # Gaussian smoothing + adaptive threshold
│   ├── marching_cubes.py  # Basic isosurface extraction
│   ├── totalseg.py     # TotalSegmentator AI segmentation
│   └── medsam2.py      # MedSAM2 AI segmentation
├── gallery/            # Gallery mode (individual, lightbox, spatial)
├── mesh/               # Taubin smoothing, decimation, temporal processing
└── glb/                # GLB builder, morph target animation, CARTO animation, textures, compression
```

## Testing

```bash
pytest
pytest --cov=med2glb
```

## License

MIT
