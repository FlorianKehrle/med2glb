# med2glb

Convert DICOM and CARTO 3 electro-anatomical mapping data to GLB 3D models optimized for AR viewing.

Supports 3D echocardiography, cardiac CT/MRI, 2D cine clips, single DICOM slices, and CARTO 3 EP mapping data. Outputs GLB files with PBR materials, animated cardiac cycles, per-vertex voltage/LAT heatmaps, and multi-structure segmentation with per-structure coloring.

## Why med2glb?

No existing end-to-end CLI tool converts DICOM directly to animated GLB for augmented reality. Existing tools (3D Slicer, DicomToMesh, InVesalius) produce static STL/OBJ via a GUI. med2glb fills this gap as a single command that takes DICOM data in and produces AR-ready GLB out.

**Key features:**

- **Analyze-and-prompt workflow** -- point it at a directory, it analyzes the data, shows a summary, and guides you through relevant options with smart defaults
- **CARTO 3 EP mapping support** -- auto-detects CARTO export directories; renders LAT, bipolar voltage, and unipolar voltage heatmaps as per-vertex colored GLBs; animated excitation ring via emissive overlay (full-quality mesh shared across all variants); animated LAT streamline vectors showing conduction direction; Loop subdivision with IDW interpolation for smooth color maps; lossless PNG textures compressed to KTX2/Basis Universal for optimal AR quality; color scale legend cylinder and study info panel embedded alongside the mesh for AR readability
- **Animated cardiac output** -- 2D cine clips become animated GLB with per-frame texture planes; 3D temporal volumes use morph targets; CARTO excitation ring animation
- **Gallery mode** -- convert every slice to textured quads with three layouts: individual GLBs, lightbox grid, and spatial fan positioned using DICOM metadata
- **Pluggable conversion methods** -- classical (Gaussian + adaptive threshold), marching cubes, TotalSegmentator (CT), and chamber-detect (echo/general)
- **Automatic series detection** -- multi-series DICOM folders are analyzed and classified (3D volume, 2D cine, still image) with per-series conversion recommendations
- **GLB output** -- with animation and PBR materials
- **AR-optimized meshes** -- Taubin smoothing (volume-preserving), decimation to configurable triangle count, and configurable transparency
- **GLB compression** -- standalone `compress` subcommand to shrink GLB files with four strategies: KTX2 GPU textures, Draco mesh compression, texture downscaling, or JPEG re-encoding
- **Multi-threshold layered output** -- extract multiple structures at different intensity thresholds with per-layer colors and transparency

## Installation

```bash
pip install med2glb
```

For AI-powered segmentation (TotalSegmentator):

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

Just point med2glb at a directory. It analyzes the input, shows you what it found, and walks you through the relevant options:

```bash
med2glb ./patient_data/
```

That's it. The tool will:

1. **Detect the data type** -- CARTO export, DICOM volume, cine clip, or single image
2. **Display a summary** -- a Rich table with all maps or series, their metadata, and recommendations
3. **Prompt for options** -- only the choices relevant to your data, with smart defaults you can accept by pressing Enter
4. **Convert and export** -- output placed in a `glb/` subfolder with a descriptive filename

### CARTO example

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
Coloring (lat / bipolar / unipolar) [lat]:
Output mode (static / animated / both) [both]:
LAT vectors (yes / no / only) [yes]:
Subdivision level (0-3) [2]:

Done: ReBS_V_SR_11
  Output:
    ReBS_V_SR_11_lat.glb  (1245 KB, static)
    ReBS_V_SR_11_lat_animated.glb  (1823 KB, animated)
    ReBS_V_SR_11_lat_animated_vectors.glb  (2104 KB, animated + vectors)
  Time: 2m 15s
```

### DICOM example

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

### Multiple CARTO datasets (batch)

Point med2glb at a parent folder containing multiple CARTO exports. It auto-detects all datasets, shows an overview, and asks settings once:

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

Coloring (applied to all datasets) [lat]:
Output [both]:
LAT vectors [yes]:
Subdivision level [2]:

=== Dataset 1/3: Export_Study_AF ===
→ ./Export_Study_AF/glb/1-LA_lat.glb
...

=== Dataset 2/3: Export_Study_VT ===
...
```

Each dataset's output goes into its own `glb/` subfolder. The `--batch` flag can also be used explicitly to force batch mode in non-interactive environments:

```bash
med2glb ../CARTO_exports/ --batch --coloring lat --animate --vectors
```

### Non-interactive / CI usage

In non-TTY environments (piped input, CI), the wizard is skipped and sensible defaults are applied automatically. You can also bypass the wizard by passing explicit flags:

```bash
# Explicit flags skip the wizard entirely
med2glb ./Export_Study/ --coloring bipolar --animate
med2glb ./echo_folder/ -o heart.glb --method totalseg
med2glb ./dicom_folder/ --gallery
```

## CARTO 3 Electro-Anatomical Mapping

med2glb auto-detects CARTO 3 export directories (containing `.mesh` files) and converts them to GLB with per-vertex coloring from the mapping data. Output files are placed in a `glb/` subfolder inside the input directory by default. Supports old CARTO (~2015, v4), v7.1 (v5), and v7.2+ (v6) formats.

The interactive wizard handles all options, but you can also set them explicitly:

```bash
# Bipolar voltage map (scar mapping)
med2glb ./Export_Study/ --coloring bipolar

# Animated excitation ring with vectors
med2glb ./Export_Study/ --animate --vectors

# Smoother color maps with more subdivision (default: 2)
med2glb ./Export_Study/ --subdivide 3

# Explicit output path
med2glb ./Export_Study/ -o left_atrium_lat.glb --coloring lat
```

**Mesh subdivision (`--subdivide`):**

By default, CARTO meshes are Loop-subdivided twice (`--subdivide 2`) before mapping measurement points. This increases mesh resolution (~4x faces per level) and uses k-NN inverse-distance weighting (IDW) interpolation instead of single nearest-neighbor, producing smooth color gradients between measurement points. Use `--subdivide 0` to get the original blocky nearest-neighbor behavior, or `--subdivide 3` for even smoother results at the cost of more vertices. Non-manifold meshes that cannot be subdivided automatically fall back to the original geometry.

| Level | Faces | Mapping | Use Case |
|---|---|---|---|
| `0` | Original | Nearest-neighbor + linear interpolation | Fast preview, original behavior |
| `1` | ~4x | k-NN IDW (k=6) | Moderate quality |
| `2` (default) | ~16x | k-NN IDW (k=6) | Good balance of quality and speed |
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

**CARTO processing pipeline:**

When you run a CARTO conversion, the following steps happen under the hood:

1. **Parse** -- `.mesh` and `_car.txt` files are read; inactive vertices (GroupID `-1000000`) and fill/cap geometry (negative GroupIDs) are stripped
2. **Subdivide** (if `--subdivide > 0`) -- Loop subdivision increases mesh resolution (~4x faces per level); measurement points are mapped via k-NN IDW interpolation for smooth color gradients
3. **Color map** -- per-vertex LAT/bipolar/unipolar values are mapped through clinical colormaps to RGBA vertex colors
4. **xatlas UV unwrap** (~60-90s) -- the mesh is UV-parameterized for texture baking (computed once, shared across all output variants)
5. **Rasterize** -- vertex colors are baked into a texture via barycentric interpolation with 10-iteration gutter bleeding to prevent mipmap seam artifacts
6. **Encode textures** -- base color and per-frame emissive ring textures are encoded as lossless PNG
7. **Build GLB** -- geometry, textures, materials (metallic=0, roughness=0.7), and animation keyframes are assembled into a glTF binary
8. **KTX2 compress** (if `toktx` is installed) -- PNG textures are GPU-compressed to KTX2/Basis Universal (UASTC + Zstandard) as the single lossy step, producing smaller files with better quality than double-lossy JPEG

Texture resolution scales with face count: 512px (≤5k faces), 1024px (≤20k), 2048px (≤80k), 4096px (>80k). Emissive ring textures are capped at 1024px since the thin Gaussian highlight doesn't need higher resolution.

For animated output, the full-quality mesh is shared across all 30 frames -- only the emissive texture (mostly black with a thin ring) changes per frame, keeping file size small.

**LAT conduction vectors:**

Animated streamline arrows overlaid on the mesh showing the direction of electrical conduction derived from the LAT gradient field. The arrows flow along curved paths following the gradient, with dashes that advance each frame in sync with the excitation ring animation. In static mode, a single frame of arrows is rendered as an extra mesh node.

The wizard automatically assesses vector quality using three checks: minimum valid LAT points (≥30), minimum LAT range (≥20 ms), and a gradient coverage trial that computes the actual IDW-interpolated LAT field and face gradients to verify at least 15% of faces have non-zero gradient. This ensures streamlines will be meaningful rather than terminating immediately on locally constant LAT patches. The vectors prompt offers three choices:
- **yes** -- produce both with-vectors and without-vectors variants (for comparison)
- **no** -- no vector output
- **only** -- produce only the animated+vectors variant (no static, no non-vector files)

**Mesh overview table:**

The interactive wizard displays a summary table for each mesh in the CARTO study to help you decide which maps to convert and what to expect:

| Column | Description |
|---|---|
| **Vertices** | Number of active vertices (excluding fill geometry marked with GroupID `-1000000`) |
| **Triangles** | Number of faces in the original mesh (each face is a triangle) |
| **After Subdiv** | Estimated triangle count after Loop subdivision at the default level 2 (~16× increase) |
| **Active** | Percentage of vertices that are real geometry vs fill/cap vertices |
| **Points** | Number of electro-anatomical measurement points from the `_car.txt` file |
| **LAT range** | Min–max local activation time across all valid measurement points (ms) |
| **Density** | Point-to-vertex ratio (points ÷ active vertices) — higher means denser mapping coverage |
| **Volts** | Available voltage data: **B** = bipolar, **U** = unipolar |
| **Dimensions** | Bounding box of active vertices (width × height × depth in mm) |
| **Est Time** | Rough processing time estimate assuming default settings (subdivide=2, both static+animated output) |

### CARTO workflow example

A typical workflow for converting CARTO data for AR review on HoloLens:

```bash
# 1. Point med2glb at the CARTO export -- the wizard guides you through options
med2glb ./Export_Study_AF/

# 2. The wizard shows a mesh overview table with geometry stats,
#    point coverage, voltage data, and estimated processing time.
#    Pick a map, choose coloring (lat/bipolar/unipolar), and output mode.
#    Default settings (subdivide=2, lat coloring, both static+animated) work
#    well for most cases.

# 3. Output is placed in ./Export_Study_AF/glb/:
#      MapName_lat.glb                    -- static heatmap
#      MapName_lat_animated.glb           -- animated excitation ring
#      MapName_lat_animated_vectors.glb   -- animated ring + streamline arrows
#      med2glb_log.txt                    -- conversion metadata

# For non-interactive / scripted usage, pass flags directly:
med2glb ./Export_Study_AF/ --coloring lat --animate --vectors --subdivide 2

# Bipolar voltage for scar mapping (no animation needed):
med2glb ./Export_Study_AF/ --coloring bipolar

# Quick preview without subdivision (original mesh, fastest):
med2glb ./Export_Study_AF/ --subdivide 0

# Maximum smoothness for presentation (slower, more vertices):
med2glb ./Export_Study_AF/ --subdivide 3

# Batch-convert all CARTO exports in a parent directory:
med2glb ../CARTO_exports/ --batch --coloring lat --animate --vectors
```

For best AR quality, install the [Khronos `toktx` tool](https://github.com/KhronosGroup/KTX-Software) so textures are automatically GPU-compressed to KTX2. Without it, textures remain as lossless PNG (larger files but same visual quality).

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

## Compress

The `compress` subcommand shrinks a GLB file to fit a target size.

```bash
# Compress in-place to 25 MB (default)
med2glb compress model.glb

# Custom target size
med2glb compress model.glb --max-size 10

# Output to a different file
med2glb compress model.glb -o small.glb

# Choose a compression strategy
med2glb compress model.glb --strategy draco
```

Four compression strategies are available:

| Strategy | Method | Quality | Speed |
|---|---|---|---|
| `ktx2` (default) | KTX2 GPU-compressed textures via Basis Universal (requires `toktx`) | Best | Moderate |
| `draco` | Draco mesh compression + texture downscale fallback | Good | Moderate |
| `downscale` | Progressive texture resolution reduction (lossless PNG) | High | Fast |
| `jpeg` | JPEG re-encoding with decreasing quality | Lower | Fast |

## Methods

| Method | Best For | AI Required | Animation |
|---|---|---|---|
| `classical` (default) | 3D echo, noisy data | No | Yes |
| `marching-cubes` | Quick preview, any modality | No | Yes |
| `totalseg` | Cardiac CT with contrast | Yes | No |
| `chamber-detect` | 3D echo, cardiac volumes with contrast | No | Yes |
| `compare` | Method comparison | No* | No |

\* `compare` runs all non-AI methods by default; AI methods are included if installed.

The wizard auto-selects the recommended method based on the data type. You can override it in the wizard or with `--method`. List available methods and their install status:

```bash
med2glb --list-methods
```

### Classical (default)

Full pipeline: Gaussian smoothing, adaptive Otsu threshold, morphological cleanup, largest-component extraction, then marching cubes for surface extraction. Best for noisy 3D echo data.

### Marching Cubes

Minimal pipeline: threshold then marching cubes. No morphological cleanup -- fast but noisier. Supports multi-threshold mode for extracting multiple structures at different intensity levels.

```bash
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

### Chamber Detect

Multi-structure cardiac chamber detection using intensity heuristics (Otsu thresholding, morphological operations, connected component labeling). Produces multi-structure output with per-structure colors — myocardium plus up to four blood pool chambers assigned by size. Best for 3D echo and cardiac volumes with contrast.

### Compare

Runs all available conversion methods side-by-side on the same input and produces a comparison table showing mesh stats (vertices, faces), file size, and processing time for each method. Useful for choosing the best method for a new dataset. AI methods are included automatically if installed.

```bash
med2glb ./echo_folder/ --method compare
```

## CLI Reference

```
med2glb [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH              Path to DICOM file or directory

Options:
  -o, --output PATH       Output file path (default: <input>/glb/<name>_<type>.glb)
  -m, --method TEXT        Conversion method: classical, marching-cubes, totalseg, chamber-detect, compare
  --coloring TEXT         CARTO coloring: lat, bipolar, unipolar (default: lat)
  --subdivide INTEGER     CARTO mesh subdivision level 0-3 (default: 2)
  --vectors               Add animated LAT streamline arrows (CARTO LAT maps)
  --animate               Enable animation for temporal data
  --no-animate            Force static output even if temporal data is detected
  --threshold FLOAT       Intensity threshold for isosurface extraction
  --smoothing INTEGER     Taubin smoothing iterations, 0 to disable (default: 15)
  --faces INTEGER         Target triangle count after decimation (default: 80000)
  --alpha FLOAT           Global transparency 0.0-1.0 (default: 1.0)
  --multi-threshold TEXT   Multi-threshold config: "val:label:alpha,..."
  --batch                 Batch mode: find all CARTO exports in subdirectories and convert with shared settings
  --gallery               Gallery mode: individual GLBs, lightbox grid, and spatial fan
  --columns INTEGER       Lightbox grid columns in gallery mode (default: 6)
  --series TEXT           Select DICOM series by UID (partial match)
  --list-methods          List available methods and exit
  --list-series           List DICOM series in input directory and exit
  -v, --verbose           Show detailed processing information
  --version               Show version and exit
```

```
med2glb compress [OPTIONS] GLB_PATH

Arguments:
  GLB_PATH                GLB file to compress

Options:
  -s, --max-size INTEGER  Target size in MB (default: 25)
  --strategy TEXT         Compression strategy: ktx2, draco, downscale, jpeg (default: ktx2)
  -o, --output PATH       Output path (default: compress in-place)
```

Passing any pipeline flag (`--method`, `--coloring`, `--animate`, `--vectors`, etc.) bypasses the interactive wizard and runs with the specified settings directly. The `-o` flag can be combined with the wizard to control the output location.

## AR Viewer Compatibility

- **Android**: Scene Viewer (native GLB support)
- **Web**: [model-viewer](https://modelviewer.dev/) web component
- **iOS**: USDZ conversion needed (use external tools)

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
│   └── chamber_detect.py  # Cardiac chamber detection (intensity heuristic)
├── gallery/            # Gallery mode (individual, lightbox, spatial)
├── mesh/               # Taubin smoothing, decimation, temporal processing, LAT vectors
└── glb/                # GLB builder, morph target animation, CARTO animation, arrow builder, textures, compression
```

## Testing

```bash
pytest
pytest --cov=med2glb
```

## Code Review with Claude Code

To run a comprehensive code review using [Claude Code](https://claude.com/claude-code):

```bash
claude "run a code review"
```

This launches a read-only analysis covering architecture overview, latest features and changes, supported pipelines, code quality, AR-specific optimizations, and improvement suggestions. The review is aware that GLB outputs target augmented reality viewers and that inputs are clinical data formats (DICOM, CARTO).

## License

MIT
