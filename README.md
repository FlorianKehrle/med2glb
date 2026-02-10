# dicom2glb

Convert DICOM medical imaging data to GLB 3D models optimized for AR viewing.

Supports 3D echocardiography, cardiac CT/MRI, and single DICOM slices. Outputs GLB files with PBR materials, morph target animation for cardiac cycles, and multi-structure segmentation with per-structure coloring.

## Installation

```bash
pip install dicom2glb
```

For AI-powered segmentation (TotalSegmentator, MedSAM2):

```bash
pip install dicom2glb[ai]
```

Development:

```bash
git clone https://github.com/FlorianKehrle/dicom2glb.git
cd dicom2glb
pip install -e ".[dev]"
```

## Quick Start

```bash
# Convert a DICOM directory to GLB
dicom2glb ./dicom_folder/ -o output.glb

# Animated 3D echo (cardiac cycle with morph targets)
dicom2glb ./echo_folder/ -o heart.glb --animate

# Multi-series folder (interactive selection)
dicom2glb ./echo_folder/ -o output.glb

# Single DICOM image to textured plane
dicom2glb image.dcm -o plane.glb

# Cardiac CT with AI segmentation
dicom2glb ./ct_folder/ -o heart.glb --method totalseg

# Multi-threshold layered output
dicom2glb ./data/ -o layers.glb --multi-threshold "200:bone:1.0,100:tissue:0.5"

# Export as STL or OBJ
dicom2glb ./data/ -o model.stl -f stl
```

## Series Selection

When a DICOM folder contains multiple series, dicom2glb analyzes each series and presents an interactive selection table:

```
  DICOM Series in echo_folder
┏━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ # ┃ Modality ┃ Description     ┃ Data Type   ┃ Detail      ┃ Recommended      ┃
┡━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ 1 │ US       │ Echokardiogr... │ 2D cine     │ 132 frames  │ textured plane   │
│ 2 │ US       │ (no desc)       │ 2D cine     │ 65 frames   │ textured plane   │
│ 3 │ CT       │ Thorax          │ 3D volume   │ 120 slices  │ 3D mesh          │
└───┴──────────┴─────────────────┴─────────────┴─────────────┴──────────────────┘

Recommendation: Series 1 (2D cine, 132 frames)

Select series to convert [1]: 1,3
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
dicom2glb --list-methods
```

## CLI Reference

```
dicom2glb [OPTIONS] INPUT_PATH

Arguments:
  INPUT_PATH              Path to DICOM file or directory

Options:
  -o, --output PATH       Output file path (default: output.glb)
  -m, --method TEXT        Conversion method (default: classical)
  -f, --format TEXT        Output format: glb, stl, obj (default: glb)
  --animate               Enable morph target animation for temporal data
  --threshold FLOAT       Intensity threshold for isosurface extraction
  --smoothing INTEGER     Taubin smoothing iterations, 0 to disable (default: 15)
  --faces INTEGER         Target triangle count after decimation (default: 80000)
  --alpha FLOAT           Global transparency 0.0-1.0 (default: 1.0)
  --multi-threshold TEXT   Multi-threshold: "val:label:alpha,..."
  --series TEXT           Select DICOM series by UID (partial match)
  --list-methods          List available methods and exit
  --list-series           List DICOM series in input directory and exit
  -v, --verbose           Show detailed processing information
  --version               Show version and exit
```

## Output Formats

| Format | Animation | Materials | Use Case |
|---|---|---|---|
| GLB | Yes (morph targets) | PBR with transparency | AR viewers, web (model-viewer) |
| STL | No | No | 3D printing, CAD |
| OBJ | No | Basic | General 3D software |

## AR Viewer Compatibility

- **Android**: Scene Viewer (native GLB support)
- **Web**: [model-viewer](https://modelviewer.dev/) web component
- **iOS**: USDZ conversion needed (use external tools)

## Architecture

```
src/dicom2glb/
├── cli.py              # Typer CLI entry point
├── core/               # Data types (MeshData, DicomVolume, etc.)
├── io/                 # DICOM reading, echo reader, exporters
├── methods/            # Pluggable conversion methods (registry pattern)
│   ├── classical.py    # Gaussian smoothing + adaptive threshold
│   ├── marching_cubes.py  # Basic isosurface extraction
│   ├── totalseg.py     # TotalSegmentator AI segmentation
│   └── medsam2.py      # MedSAM2 AI segmentation
├── mesh/               # Taubin smoothing, decimation, temporal processing
└── glb/                # GLB builder, morph target animation, textures
```

## Testing

```bash
pytest
pytest --cov=dicom2glb
```

## License

MIT
