# MED2GLB ANALYSIS - EXECUTIVE SUMMARY

## ✅ Analysis Status: COMPLETE

Three comprehensive documents have been generated:

1. **CARTO_ANALYSIS.md** (11 KB)
   - Complete technical reference with all code locations
   - Detailed CARTO data parsing explanation
   - Voltage → Color mapping pipeline
   - Alpha channel opportunity analysis
   - Implementation roadmap

2. **ANALYSIS_SUMMARY.txt** (This file)
   - Quick reference of key findings
   - Architecture overview
   - Next steps for implementation

## 📊 ANSWERS TO YOUR QUESTIONS

### 1. Overall Architecture
- **Language**: Python 3.10+
- **Frameworks**: Typer (CLI), Rich (UI), pygltflib (GLB/glTF 2.0), trimesh, numpy, scipy
- **Design**: Pluggable conversion methods with registry pattern
- **Target**: HoloLens 2 AR visualization (doesn't support COLOR_0 vertex attributes)

### 2. Medical Format Support
**Primary**: CARTO 3 electro-anatomical mapping (.mesh + _car.txt)
**Secondary**: DICOM medical imaging
**Mentioned but not active**: NIfTI, inHEART, STL

### 3. CARTO Voltage/Activation Map Reading
- **Mesh file** [VerticesColorsSection]: Pre-computed per-vertex LAT/bipolar/unipolar
  - Type: [N] float64, stored in CartoMesh.vertex_color_values dict
  - File location: carto_reader.py:305-361
- **Car file** (_car.txt): Sparse per-point measurements (50-1000s points)
  - Parsed into CartoPoint objects with voltage + LAT values
  - File location: carto_reader.py:182-255
- **Mapping**: KDTree nearest-neighbor or IDW interpolation to mesh
  - File location: carto_mapper.py:46-130

### 4. Vertex Color Generation from Voltage
- **Flow**: Raw voltage → colormap function → RGBA [N,4] float32
- **Colormaps**:
  - LAT: Red (early) → Yellow → Green → Cyan → Blue → Purple (late)
  - Bipolar: Red (scar <0.5mV) → ... → Purple (healthy >1.5mV)
  - Unipolar: Red (low) → ... → Blue (high)
- **Implementation**: carto_colormaps.py:16-157
- **Raw voltage available until**: carto_mapper.py:559 (before colormap)

### 5. Color Gradient Mapping
- **Normalization**: t = (value - min) / (max - min), clipped to [0,1]
- **Interpolation**: Linear through defined color stops
- **Clamping**: Optional custom range or auto min/max
- **Default ranges**: LAT (auto), Bipolar (0.05-1.5mV), Unipolar (3.0-10.0mV)
- **Code**: carto_colormaps.py:114-147 (_apply_colormap function)

### 6. GLB File Generation
- **Library**: pygltflib (pure Python glTF 2.0)
- **Process**:
  1. Parse CARTO files
  2. Apply colormap to voltage values
  3. xatlas UV unwrap mesh
  4. Rasterize vertex colors to PNG texture (barycentric interpolation)
  5. Embed texture in GLB as baseColorTexture
  6. Write geometry (positions, normals, UVs, indices)
- **Key files**: vertex_color_bake.py (texture), builder.py (GLB construction)

### 7. Vertex Colors → GLB Alpha Handling
- **Current**: Always 1.0 (fully opaque) → alphaMode=OPAQUE
- **Texture output**: RGB-only PNG (alpha channel discarded)
- **No impact on rendering**: HoloLens 2 doesn't use vertex COLOR_0 attributes
- **Location of discard**: vertex_color_bake.py:439 (PNG mode="RGB")

### 8. Metadata Embedding
- **Yes**: Legend nodes with color gradients, info panels
- **Stored in**: gltf.extras dict
- **Includes**: coloring type, clamp_range, study name, method
- **Files**: legend_builder.py (legend), animation.py (metadata)

### 9. File Structure & Entry Points
- **Entry point**: src/med2glb/cli.py (Typer CLI)
- **CARTO pipeline**: src/med2glb/_pipeline_carto.py (wizard-driven)
- **Main conversion**: src/med2glb/io/carto_mapper.py:carto_mesh_to_mesh_data()
- **Directory structure**: See CARTO_ANALYSIS.md for full breakdown

## 🎯 CRITICAL FINDING: VOLTAGE ENCODING IN ALPHA

### Current State
`
Raw Voltage Values [N] float64
    ↓
Colormap Normalization: (v - lo) / (hi - lo)
    ↓
RGB Color Interpolation [N,3] float32
    ↓
Alpha Channel Set to 1.0 [N] float32 ← ALWAYS 1.0
    ↓
Texture Rasterization (barycentric)
    ↓
PNG Encoding: RGB-only (ALPHA DISCARDED) ← ❌ Lost here
    ↓
GLB with RGB Texture, alphaMode=OPAQUE
`

### Opportunity
`
At carto_colormaps.py:145-148, INSTEAD OF:
    colors[valid, 3] = 1.0

DO:
    alpha_normalized = (raw_values - lo) / (hi - lo)
    colors[valid, 3] = clip(alpha_normalized, 0.0, 1.0)
`

### Implementation Path
1. **Modify colormap functions** to accept encode_raw_to_alpha parameter
2. **Update texture rasterization** to preserve alpha (RGBA PNG)
3. **Update material detection** to recognize meaningful alpha
4. **Store metadata** in GLB extras for shader decoding
   - clamp_range
   - coloring type ("lat", "bipolar", "unipolar")
   - units ("ms" or "mV")

## 📍 KEY CODE LOCATIONS (Line Numbers)

| Task | File | Lines | Function |
|------|------|-------|----------|
| Parse .mesh | carto_reader.py | 116-179 | parse_mesh_file() |
| Parse _car.txt | carto_reader.py | 182-255 | parse_car_file() |
| Extract vertex voltages | carto_reader.py | 305-361 | _parse_vertices_colors_section() |
| Map to mesh vertices | carto_mapper.py | 46-130 | map_points_to_vertices_idw() |
| **Apply colormap** | carto_mapper.py | 504-559 | carto_mesh_to_mesh_data() |
| **Colormap core** | carto_colormaps.py | 94-157 | _apply_colormap() ← **MODIFY HERE** |
| **Texture baking** | vertex_color_bake.py | 310-441 | rasterize_vertex_colors() |
| **PNG encoding** | vertex_color_bake.py | 436-440 | PNG write ← **MODIFY HERE** |
| GLB construction | builder.py | 113-357 | _add_mesh_to_gltf() |
| **Alpha detection** | builder.py | 180-190 | Alpha mode logic ← **MODIFY HERE** |
| Metadata embedding | legend_builder.py | Various | Legend + extras |

## 🔧 Implementation Checklist

- [ ] Add ncode_raw_to_alpha parameter to colormap functions
- [ ] Modify _apply_colormap() to set alpha = normalized_value when enabled
- [ ] Update PNG encoding to output RGBA instead of RGB
- [ ] Update texture rasterization to preserve alpha
- [ ] Update alpha mode detection for meaningful alpha values
- [ ] Store clamp_range + coloring in GLB extras
- [ ] Add CLI flag or config option for encoding
- [ ] Write unit tests for alpha encoding
- [ ] Test shader decoding with alpha values

## 💾 Saved Documents

All analysis saved to: **C:\Users\flo\git\med2glb\**

- **CARTO_ANALYSIS.md** - Complete technical reference (11 KB)
  - Full code flow with all line numbers
  - Detailed explanations of each stage
  - Implementation roadmap
  - Timeline of raw voltage availability

- **ANALYSIS_SUMMARY.txt** - This quick reference

Generated: 2026-03-18 11:16:21
Analysis scope: Complete CARTO data pipeline, voltage processing, GLB export
