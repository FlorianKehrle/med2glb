# med2glb

Convert DICOM and CARTO 3 electro-anatomical mapping data to GLB 3D models optimized for AR viewing.

Supports 3D echocardiography, cardiac CT/MRI, 2D cine clips, single DICOM slices, and CARTO 3 EP mapping data. Outputs GLB files with PBR materials, animated cardiac cycles, per-vertex voltage/LAT heatmaps, and multi-structure segmentation with per-structure coloring.

## Why med2glb?

No existing end-to-end CLI tool converts DICOM directly to animated GLB for augmented reality. Existing tools (3D Slicer, DicomToMesh, InVesalius) produce static STL/OBJ via a GUI. med2glb fills this gap as a single command that takes DICOM data in and produces AR-ready GLB out.

**Key features:**

- **Analyze-and-prompt workflow** -- point it at a directory, it analyzes the data, shows a summary, and guides you through relevant options with smart defaults
- **CARTO 3 EP mapping support** -- auto-detects CARTO export directories; renders LAT, bipolar voltage, and unipolar voltage heatmaps as per-vertex colored GLBs; animated excitation ring overlay; animated LAT streamline vectors showing conduction direction; Loop subdivision with IDW interpolation for smooth color maps
- **Animated cardiac output** -- 2D cine clips become animated GLB with per-frame texture planes; 3D temporal volumes use morph targets; CARTO excitation ring animation
- **Gallery mode** -- convert every slice to textured quads with three layouts: individual GLBs, lightbox grid, and spatial fan positioned using DICOM metadata
- **Pluggable conversion methods** -- classical (Gaussian + adaptive threshold), marching cubes, TotalSegmentator (CT), and MedSAM2 (echo/general)
- **Automatic series detection** -- multi-series DICOM folders are analyzed and classified (3D volume, 2D cine, still image) with per-series conversion recommendations
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

→ ./Export_Study/glb/ReBS_V_SR_11_lat.glb
→ ./Export_Study/glb/ReBS_V_SR_11_lat_animated.glb
→ ./Export_Study/glb/ReBS_V_SR_11_lat_animated_vectors.glb
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
→ ./Export_Study_AF/glb/1-LA_lat_AR.glb
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

**Dual output (standard + AR):**

Every CARTO conversion produces two variants of each GLB:
- **Standard** (`_lat.glb`) -- PBR metallic-roughness material with scene lighting, suitable for desktop/web viewers
- **AR-optimized** (`_lat_AR.glb`) -- uses `KHR_materials_unlit` extension for lighting-independent rendering on AR headsets (HoloLens 2, etc.)

Both variants are always generated automatically.

**What gets parsed:**
- `.mesh` files: 3D surface geometry with per-vertex group IDs (inactive vertices filtered out)
- `_car.txt` files: sparse measurement points mapped to mesh vertices via k-NN IDW interpolation (with subdivision) or nearest-neighbor + linear interpolation (without)
- LAT sentinel value `-10000` is treated as unmapped (rendered as transparent gray)

**LAT conduction vectors:**

Animated streamline arrows overlaid on the mesh showing the direction of electrical conduction derived from the LAT gradient field. The arrows flow along curved paths following the gradient, with dashes that advance each frame in sync with the excitation ring animation. In static mode, a single frame of arrows is rendered as an extra mesh node.

The wizard automatically assesses vector quality using three checks: minimum valid LAT points (≥30), minimum LAT range (≥20 ms), and a gradient coverage trial that computes the actual IDW-interpolated LAT field and face gradients to verify at least 15% of faces have non-zero gradient. This ensures streamlines will be meaningful rather than terminating immediately on locally constant LAT patches. The vectors prompt offers three choices:
- **yes** -- produce both with-vectors and without-vectors variants (for comparison)
- **no** -- no vector output
- **only** -- produce only the animated+vectors variant (no static, no non-vector files)

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
# Custom size limit
med2glb ./data/ --max-size 50

# Use JPEG compression (smaller but lossy)
med2glb ./data/ --compress jpeg

# Disable size constraint
med2glb ./data/ --max-size 0
```

## Methods

| Method | Best For | AI Required | Animation |
|---|---|---|---|
| `classical` (default) | 3D echo, noisy data | No | Yes |
| `marching-cubes` | Quick preview, any modality | No | Yes |
| `totalseg` | Cardiac CT with contrast | Yes | No |
| `medsam2` | 3D echo, general cardiac | Yes | Yes |

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

### MedSAM2

AI segmentation for 3D echo and general cardiac imaging. Produces multi-structure output with per-structure colors. Currently uses a heuristic pseudo-segmentation; full MedSAM2 model integration is planned.

Requires AI dependencies: `pip install med2glb[ai]`

## CLI Reference

```
med2glb [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH              Path to DICOM file or directory

Options:
  -o, --output PATH       Output file path (default: <input>/glb/<name>_<type>.glb)
  -m, --method TEXT        Conversion method: classical, marching-cubes, totalseg, medsam2
  -f, --format TEXT        Output format: glb, stl, obj (default: glb)
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
  --max-size INTEGER      Maximum output GLB file size in MB, 0 to disable (default: 99)
  --compress TEXT         Compression strategy: draco, downscale, jpeg (default: draco)
  --batch                 Batch mode: find all CARTO exports in subdirectories and convert with shared settings
  --gallery               Gallery mode: individual GLBs, lightbox grid, and spatial fan
  --columns INTEGER       Lightbox grid columns in gallery mode (default: 6)
  --series TEXT           Select DICOM series by UID (partial match)
  --list-methods          List available methods and exit
  --list-series           List DICOM series in input directory and exit
  -v, --verbose           Show detailed processing information
  --version               Show version and exit
```

Passing any pipeline flag (`--method`, `--coloring`, `--animate`, `--vectors`, etc.) bypasses the interactive wizard and runs with the specified settings directly. The `-o` flag can be combined with the wizard to control the output location.

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
├── cli_wizard.py       # Interactive wizard (data-driven prompts)
├── core/               # Data types (MeshData, DicomVolume, CartoConfig, etc.)
├── io/                 # DICOM reading, echo reader, CARTO reader/mapper, exporters
├── methods/            # Pluggable conversion methods (registry pattern)
│   ├── classical.py    # Gaussian smoothing + adaptive threshold
│   ├── marching_cubes.py  # Basic isosurface extraction
│   ├── totalseg.py     # TotalSegmentator AI segmentation
│   └── medsam2.py      # MedSAM2 AI segmentation
├── gallery/            # Gallery mode (individual, lightbox, spatial)
├── mesh/               # Taubin smoothing, decimation, temporal processing, LAT vectors
└── glb/                # GLB builder, morph target animation, CARTO animation, arrow builder, textures, compression
```

## Testing

```bash
pytest
pytest --cov=med2glb
```

## License

MIT
