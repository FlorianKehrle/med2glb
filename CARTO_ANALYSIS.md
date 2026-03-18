# MED2GLB CARTO COMPREHENSIVE ANALYSIS

## EXECUTIVE SUMMARY

**med2glb** is a Python 3.10+ pipeline that converts medical imaging data (primarily CARTO 3 electro-anatomical mapping and DICOM) to GLB 3D models optimized for AR viewing on HoloLens 2.

### Architecture:
- **CLI**: Typer + Rich
- **Mesh Processing**: trimesh, numpy, scipy
- **3D Export**: pygltflib (glTF 2.0)
- **UV Unwrap**: xatlas
- **CARTO Support**: Custom .mesh/.car.txt parser

---

## CARTO DATA FLOW

### INPUT: CARTO 3 Export Files

Two files per anatomical structure:

1. **Structure.mesh** (INI-style)
   - [GeneralAttributes]: MeshColor (RGBA), ColorsNames ("lat", "bipolar", "unipolar")
   - [VerticesSection]: X Y Z NX NY NZ GroupID (geometry + normals + group membership)
   - [TrianglesSection]: V0 V1 V2 NX NY NZ GroupID (topology)
   - **[VerticesColorsSection]**: Pre-computed per-vertex color values from CARTO
     - One column per ColorsName (LAT, bipolar, unipolar in ms or mV)
     - Stored as: vertex_color_values["lat"/"bipolar"/"unipolar"][N] float64

2. **Structure_car.txt** (tab-separated)
   - Header: VERSION_X_Y MapName
   - Data rows: P(marker) idx pointID 0 X Y Z orientX orientY orientZ bipolarV unipolarV LAT
   - Creates sparse point measurements (50-1000s of points across mesh)
   - LAT: milliseconds (activation timing), -10000 = sentinel (no measurement)
   - Bipolar: mV, Unipolar: mV

### PARSING (carto_reader.py)

**parse_mesh_file()** [lines 116-179]:
`
CartoMesh {
  vertices[N,3]: float64
  faces[M,3]: int32
  normals[N,3]: float64
  group_ids[N]: int32
  face_group_ids[M]: int32
  vertex_color_values: dict[str, np.ndarray]  # Keys: "lat", "bipolar", "unipolar"
  mesh_color: (r,g,b,a) RGBA fallback
}
`

**parse_car_file()** [lines 182-255]:
`
list[CartoPoint] {
  point_id: int
  position[3]: float64
  orientation[3]: float64  
  bipolar_voltage: float (mV)
  unipolar_voltage: float (mV)
  lat: float (ms) or NaN
}
`

### VOLTAGE → COLOR MAPPING (carto_mapper.py & carto_colormaps.py)

**Step 1: Extract per-vertex voltage values** [carto_mapper.py:504-559]
`python
# Prefer pre-computed from mesh file:
mesh_lat = mesh.vertex_color_values.get("lat")  # [N] float64, raw values (ms)

# Fallback: map sparse car-file points via KDTree NN or IDW
lat_values = map_points_to_vertices_idw(mesh, points, field="lat")
`

**⚠️ KEY POINT: Raw voltage values are available here before colormap**

**Step 2: Apply colormap** [carto_colormaps.py:16-157]

**LAT colormap** (milliseconds):
- Input: [N] float64 values in ms
- Clamp range: data min/max (or custom)
- Output: [N,4] float32 RGBA [0,1]
- Gradient: Red (early) → Yellow → Green → Cyan → Blue → Purple (late)
- Color stops:
  `
  (0.0, 1.0, 0.0, 0.0),    # red [earliest]
  (0.2, 1.0, 1.0, 0.0),    # yellow
  (0.4, 0.0, 1.0, 0.0),    # green
  (0.6, 0.0, 1.0, 1.0),    # cyan
  (0.8, 0.0, 0.0, 1.0),    # blue
  (1.0, 0.8, 0.0, 1.0),    # purple [latest]
  `

**Bipolar colormap** (mV, scar assessment):
- Default clamp: (0.05, 1.5) mV
- Red: scar (<0.5), Purple: healthy (>1.5)

**Unipolar colormap** (mV):
- Default clamp: (3.0, 10.0) mV
- Red: low, Blue: high

**Normalization inside _apply_colormap()** [lines 114-147]:
`python
# Normalize to [0, 1] for color interpolation
lo = nanmin(v) if clamp_range is None else clamp_range[0]
hi = nanmax(v) if clamp_range is None else clamp_range[1]
t = (v - lo) / (hi - lo)          # ← Raw value normalization
t = clip(t, 0.0, 1.0)

# Interpolate RGB through color stops
R = interp(t, stop_positions, r_values)
G = interp(t, stop_positions, g_values)
B = interp(t, stop_positions, b_values)

# Alpha channel
colors[valid, :3] = [R, G, B]
colors[valid, 3] = 1.0            # ← ALWAYS 1.0 (fully opaque)
colors[~valid, :3] = [0.5, 0.5, 0.5]
colors[~valid, 3] = 1.0           # Unmapped: gray + opaque
`

**Output**: [N,4] float32 RGBA in [0,1]

---

## GLB GENERATION

### Texture Baking (vertex_color_bake.py)

**Process**:
1. **UV Unwrap**: xatlas on full mesh → vmapping, faces, uvs
2. **Rasterization**: For each pixel in texture, find which triangle it maps to
3. **Color interpolation**: Use barycentric weights to interpolate vertex colors
4. **PNG encode**: Write RGB only (alpha discarded!)

**Key code** [lines 408-440]:
`python
# Gather vertex colors via barycentric interpolation
colors = (bw0[:,np.newaxis] * vertex_colors[v0] +
          bw1[:,np.newaxis] * vertex_colors[v1] +
          bw2[:,np.newaxis] * vertex_colors[v2])
texture[pixel_y, pixel_x] = colors  # Includes [R,G,B, 1.0]

# PNG encoding - ALPHA DISCARDED
img = Image.fromarray(texture_u8[:, :, :3], mode="RGB")  # RGB only!
img.save(buf, format="PNG")
`

### GLB Construction (builder.py)

**Material Alpha Mode** [lines 180-190]:
`python
min_alpha = mesh_data.vertex_colors[:, 3].min()
if min_alpha > 0.99:
    alpha_mode = OPAQUE           # All alpha close to 1.0
elif min_alpha < 0.01:
    alpha_mode = MASK             # Some alpha near 0.0
    alpha_cutoff = 0.5
else:
    alpha_mode = BLEND            # Some alpha in middle
`

**Current behavior**: 
- Since all alpha = 1.0, always chooses alphaMode = OPAQUE
- Material.alphaCutoff = None
- Texture is RGB (no alpha channel)

---

## CRITICAL FINDINGS FOR VOLTAGE ENCODING IN ALPHA

### 1. Raw Voltage IS Available
Location: **carto_mapper.py:513-559**
`python
mesh_lat = mesh.vertex_color_values.get("lat")  # RAW VOLTAGE HERE [N] float64
# These are raw values BEFORE colormap converts them to RGB
`

### 2. Raw Voltage IS Lost After Colormap
- Input: [N] float64 raw values (e.g., 45.3 ms)
- Output: [N,4] float32 RGBA
- The normalization to [0,1] happens inside colormap
- Raw values are discarded after creating RGB

### 3. Alpha Channel Currently Unused
- Always set to 1.0 for valid vertices
- Always 1.0 for unmapped vertices
- PNG texture doesn't include alpha (RGB only)
- Material has alphaMode=OPAQUE globally

### 4. Clamp Range IS Available
`python
carto_mapper.py:559
vertex_colors = colormap_fn(active_values, clamp_range=clamp_range)
# clamp_range passed to colormap, used for normalization
`

---

## IMPLEMENTATION ROADMAP: VOLTAGE IN ALPHA

### Proposed Change

**File: carto_colormaps.py**
`python
def lat_colormap(
    values: np.ndarray,
    clamp_range: tuple[float, float] | None = None,
    encode_raw_to_alpha: bool = False,  # NEW
) -> np.ndarray:
    """Map LAT to RGBA, optionally encoding raw values in alpha."""
    return _apply_colormap(values, _LAT_STOPS, clamp_range, encode_raw_to_alpha)

def _apply_colormap(values, stops, clamp_range, encode_raw=False):
    colors = np.zeros((len(values), 4), dtype=np.float32)
    
    # Determine normalization range
    if clamp_range is not None:
        lo, hi = clamp_range
    else:
        lo = float(np.nanmin(values[~np.isnan(values)]))
        hi = float(np.nanmax(values[~np.isnan(values)]))
    
    # Normalize for color interpolation
    t_color = (values - lo) / (hi - lo)
    t_color = np.clip(t_color, 0.0, 1.0)
    
    # RGB from color stops (existing logic)
    # ...
    colors[valid, 0] = np.interp(t_color[valid], positions, r_stops)
    colors[valid, 1] = np.interp(t_color[valid], positions, g_stops)
    colors[valid, 2] = np.interp(t_color[valid], positions, b_stops)
    
    # Alpha: raw value encoding or constant 1.0
    if encode_raw:
        # Alpha = normalized raw value [0, 1]
        t_alpha = (values - lo) / (hi - lo)
        t_alpha = np.clip(t_alpha, 0.0, 1.0)
        colors[valid, 3] = t_alpha[valid].astype(np.float32)
        colors[~valid, 3] = 1.0  # Unmapped: opaque
    else:
        # Current behavior
        colors[valid, 3] = 1.0
        colors[~valid, 3] = 1.0
    
    return colors
`

### Secondary Changes Needed

1. **vertex_color_bake.py**: Update PNG encoding to include alpha
   `python
   # Currently: mode="RGB"
   # Change to: mode="RGBA" when encode_raw_to_alpha
   img = Image.fromarray(texture_u8, mode="RGBA")
   `

2. **builder.py**: Alpha mode detection recognizes encoding
   `python
   if encode_raw_to_alpha:
       alpha_mode = pygltflib.BLEND  # Alpha is meaningful
   else:
       alpha_mode = pygltflib.OPAQUE  # Alpha ignored
   `

3. **GLB extras**: Store decoding metadata
   `python
   gltf.extras["voltage_encoding"] = {
       "enabled": encode_raw_to_alpha,
       "coloring": coloring,  # "lat", "bipolar", "unipolar"
       "clamp_range": list(clamp_range),
       "units": "ms" if coloring=="lat" else "mV"
   }
   `

---

## KEY FILE REFERENCE

| File | Lines | Key Function | Purpose |
|------|-------|--------------|---------|
| **carto_reader.py** | 116-179 | parse_mesh_file() | Parse .mesh: geometry, normals, vertex colors |
| **carto_reader.py** | 182-255 | parse_car_file() | Parse _car.txt: sparse electrical measurements |
| **carto_reader.py** | 305-361 | _parse_vertices_colors_section() | Extract [VerticesColorsSection] LAT/bipolar/unipolar |
| **carto_mapper.py** | 46-130 | map_points_to_vertices_idw() | Interpolate sparse points to mesh vertices |
| **carto_mapper.py** | 504-584 | carto_mesh_to_mesh_data() | Main: extract voltages, apply colormap → vertex_colors |
| **carto_colormaps.py** | 16-30 | lat_colormap() | Convert LAT (ms) → RGBA via gradient |
| **carto_colormaps.py** | 33-43 | bipolar_colormap() | Convert bipolar voltage → RGBA |
| **carto_colormaps.py** | 46-55 | unipolar_colormap() | Convert unipolar voltage → RGBA |
| **carto_colormaps.py** | 94-157 | _apply_colormap() | Core: normalize, interpolate colors, set alpha |
| **vertex_color_bake.py** | 310-441 | rasterize_vertex_colors() | Texture baking: rasterize vertex colors to PNG |
| **builder.py** | 113-357 | _add_mesh_to_gltf() | Create GLB mesh with texture and material |
| **builder.py** | 180-190 | alpha mode detection | Choose OPAQUE/BLEND/MASK from alpha values |

---

## VOLTAGE AVAILABILITY TIMELINE

| Stage | Location | Form | Preserved? |
|-------|----------|------|------------|
| Raw in file | carto_reader.py:parse_car_file | CartoPoint.{bipolar,unipolar,lat} | ✓ Yes |
| Mesh pre-computed | carto_reader.py:parse_mesh_file | CartoMesh.vertex_color_values | ✓ Yes |
| Before colormap | carto_mapper.py:513-559 | active_values array [N] float64 | ✓ Yes |
| During normalization | carto_colormaps.py:114-125 | t = (v - lo)/(hi - lo) | ✓ Yes (normalized) |
| After colormap | carto_colormaps.py:145-148 | vertex_colors[N,4] float32 RGBA | ✗ No (converted to RGB) |
| In PNG texture | vertex_color_bake.py:439 | texture_u8[:,:,:3] RGB mode | ✗ No (alpha dropped) |
| In final GLB | builder.py, animation.py | baseColorTexture RGB | ✗ No |

---

## CONCLUSION

**Raw voltage data CAN be encoded in the alpha channel**, but requires:

1. **Capture point**: carto_colormaps.py, inside _apply_colormap()
2. **Normalization**: Use same clamp_range as RGB color interpolation
3. **Storage**: As float32 [0, 1] in alpha channel of vertex colors
4. **Propagation**: Through texture rasterization with RGBA output
5. **Metadata**: Store clamp_range + coloring in GLB extras for shader decoding

**Current status**: Alpha is unused (always 1.0), making this a clean addition with minimal impact on existing code.
