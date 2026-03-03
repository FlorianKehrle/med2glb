"""Color legend and metadata panel nodes for CARTO GLB exports.

Renders a color scale legend and a metadata info card as PNG textures using
Pillow, then embeds them as textured geometry nodes (children of root)
positioned next to the mesh.  The legend uses a cylinder with a wrap-around
gradient texture and opaque label strips; the info card uses a double-sided
flat panel readable from both directions.  Uses ``KHR_materials_unlit``
so panels are always readable regardless of scene lighting.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pygltflib
from PIL import Image, ImageDraw, ImageFont

from med2glb.glb.builder import _pad_to_4, write_accessor

logger = logging.getLogger("med2glb")

# Colormap stop definitions (imported lazily to avoid circular deps)
_COLORMAP_STOPS: dict[str, list[tuple[float, float, float, float]]] = {}

_TITLES: dict[str, str] = {
    "lat": "LAT (ms)",
    "bipolar": "Bipolar (mV)",
    "unipolar": "Unipolar (mV)",
}


def _get_stops(coloring: str) -> list[tuple[float, float, float, float]]:
    """Get colormap stops for a coloring mode."""
    if not _COLORMAP_STOPS:
        from med2glb.io.carto_colormaps import (
            _LAT_STOPS,
            _BIPOLAR_STOPS,
            _UNIPOLAR_STOPS,
        )
        _COLORMAP_STOPS["lat"] = _LAT_STOPS
        _COLORMAP_STOPS["bipolar"] = _BIPOLAR_STOPS
        _COLORMAP_STOPS["unipolar"] = _UNIPOLAR_STOPS
    return _COLORMAP_STOPS.get(coloring, _COLORMAP_STOPS["lat"])


def _interpolate_color(
    t: float, stops: list[tuple[float, float, float, float]],
) -> tuple[int, int, int]:
    """Interpolate through color stops at normalized position *t* (0..1)."""
    t = max(0.0, min(1.0, t))
    for i in range(len(stops) - 1):
        p0, r0, g0, b0 = stops[i]
        p1, r1, g1, b1 = stops[i + 1]
        if t <= p1:
            f = (t - p0) / (p1 - p0) if p1 > p0 else 0.0
            r = r0 + f * (r1 - r0)
            g = g0 + f * (g1 - g0)
            b = b0 + f * (b1 - b0)
            return int(r * 255), int(g * 255), int(b * 255)
    # Past last stop
    _, r, g, b = stops[-1]
    return int(r * 255), int(g * 255), int(b * 255)


def render_legend_wrap_image(
    coloring: str,
    clamp_range: tuple[float, float],
    width: int = 512,
    height: int = 256,
) -> bytes:
    """Render a cylinder wrap-around color scale legend as a PNG image.

    The entire texture is the vertical gradient (top = max, bottom = min).
    Opaque dark label strips at U=0.25 and U=0.75 show the title and 5 tick
    values — these map to opposite sides of the cylinder so the scale is
    readable from any viewing angle.  The UV seam (U=0/1) is label-free.

    Returns PNG bytes.
    """
    stops = _get_stops(coloring)
    title = _TITLES.get(coloring, coloring)
    vmin, vmax = clamp_range

    img = Image.new("RGBA", (width, height))

    # Fill every pixel column with the same vertical gradient
    for y in range(height):
        t = 1.0 - y / (height - 1)
        color = _interpolate_color(t, stops)
        for x in range(width):
            img.putpixel((x, y), color + (255,))

    # Opaque label strips at U=0.25 and U=0.75 (front & back of cylinder)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=14)
    font_sm = ImageFont.load_default(size=12)

    strip_width = 80
    strip_centers = [width // 4, 3 * width // 4]  # U=0.25 and U=0.75

    for cx in strip_centers:
        x0 = cx - strip_width // 2
        x1 = cx + strip_width // 2
        draw.rectangle([(x0, 0), (x1, height - 1)], fill=(20, 20, 20, 255))

        # Title at top
        draw.text((cx, 8), title, fill=(255, 255, 255, 255),
                  font=font, anchor="mt")

        # 5 tick labels along gradient height
        n_ticks = 5
        margin_top = 28
        margin_bottom = 8
        for i in range(n_ticks):
            frac = i / (n_ticks - 1)
            y = (height - margin_bottom) - frac * (height - margin_top - margin_bottom)
            val = vmin + frac * (vmax - vmin)

            if abs(val) >= 100:
                label = f"{val:.0f}"
            elif abs(val) >= 1:
                label = f"{val:.1f}"
            else:
                label = f"{val:.2f}"

            draw.text((cx, int(y)), label,
                      fill=(220, 220, 220, 255), font=font_sm, anchor="mm")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_info_image(
    metadata: dict,
    width: int = 256,
    height: int = 256,
) -> bytes:
    """Render an opaque metadata info card as a PNG image.

    Returns PNG bytes.
    """
    img = Image.new("RGBA", (width, height), (30, 30, 30, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=13)
    font_title = ImageFont.load_default(size=15)

    # Title
    draw.text((width // 2, 10), "Study Info", fill=(255, 255, 255, 255),
              font=font_title, anchor="mt")

    # Build lines from metadata
    lines: list[tuple[str, str]] = []
    _field_map = [
        ("study_name", "Study"),
        ("structure", "Map"),
        ("coloring", "Coloring"),
        ("unit", None),  # combined with coloring
        ("clamp_range", "Range"),
        ("mapping_points", "Points"),
        ("carto_version", "CARTO"),
        ("export_date", "Date"),
    ]

    for key, label in _field_map:
        if key == "unit":
            continue  # handled with coloring
        val = metadata.get(key)
        if val is None:
            continue
        if key == "coloring":
            unit = metadata.get("unit", "")
            if unit:
                lines.append((label, f"{val} ({unit})"))
            else:
                lines.append((label, str(val)))
        elif key == "clamp_range":
            unit = metadata.get("unit", "")
            if isinstance(val, (list, tuple)) and len(val) == 2:
                v0 = round(val[0], 1)
                v1 = round(val[1], 1)
                lines.append((label, f"{v0} \u2013 {v1} {unit}".strip()))
            else:
                lines.append((label, str(val)))
        elif key == "carto_version":
            # Split "(file format ...)" onto its own line
            s = str(val)
            paren_idx = s.find(" (")
            if paren_idx >= 0:
                lines.append((label, s[:paren_idx]))
                lines.append(("", s[paren_idx + 1:]))
            else:
                lines.append((label, s))
        elif key == "mapping_points":
            lines.append((label, f"{val:,}" if isinstance(val, int) else str(val)))
        else:
            lines.append((label, str(val)))

    # Render lines
    y = 34
    line_height = 22
    for label, value in lines:
        if y + line_height > height - 8:
            break
        if label:
            draw.text((12, y), f"{label}:", fill=(180, 180, 180, 255), font=font)
            draw.text((92, y), value, fill=(240, 240, 240, 255), font=font)
        else:
            # Continuation line (e.g. "(file format v5)")
            draw.text((92, y), value, fill=(240, 240, 240, 255), font=font)
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _setup_texture_material(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    name: str,
    png_bytes: bytes,
    *,
    double_sided: bool = True,
) -> int:
    """Embed a PNG texture and create an unlit material. Returns material index."""
    if not hasattr(gltf, "images") or gltf.images is None:
        gltf.images = []
    if not hasattr(gltf, "textures") or gltf.textures is None:
        gltf.textures = []
    if not hasattr(gltf, "samplers") or gltf.samplers is None:
        gltf.samplers = []

    # Embed PNG image
    img_offset = len(binary_data)
    binary_data.extend(png_bytes)
    _pad_to_4(binary_data)

    img_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=img_offset, byteLength=len(png_bytes),
    ))

    img_idx = len(gltf.images)
    gltf.images.append(pygltflib.Image(
        bufferView=img_bv_idx, mimeType="image/png",
    ))

    if len(gltf.samplers) == 0:
        gltf.samplers.append(pygltflib.Sampler(
            magFilter=pygltflib.LINEAR,
            minFilter=pygltflib.LINEAR,
            wrapS=pygltflib.CLAMP_TO_EDGE,
            wrapT=pygltflib.CLAMP_TO_EDGE,
        ))

    tex_idx = len(gltf.textures)
    gltf.textures.append(pygltflib.Texture(sampler=0, source=img_idx))

    mat_idx = len(gltf.materials)
    gltf.materials.append(pygltflib.Material(
        name=name,
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorTexture=pygltflib.TextureInfo(index=tex_idx),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
        alphaMode=pygltflib.BLEND,
        doubleSided=double_sided,
        extensions={"KHR_materials_unlit": {}},
    ))

    if not hasattr(gltf, "extensionsUsed") or gltf.extensionsUsed is None:
        gltf.extensionsUsed = []
    if "KHR_materials_unlit" not in gltf.extensionsUsed:
        gltf.extensionsUsed.append("KHR_materials_unlit")

    return mat_idx


def _create_mesh_node(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    name: str,
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    mat_idx: int,
    translation: list[float],
) -> int:
    """Write geometry arrays and create a mesh node. Returns node index."""
    pos_acc = write_accessor(
        gltf, binary_data, vertices, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC3, with_minmax=True,
    )
    norm_acc = write_accessor(
        gltf, binary_data, normals, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC3,
    )
    uv_acc = write_accessor(
        gltf, binary_data, uvs, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC2,
    )
    idx_acc = write_accessor(
        gltf, binary_data, faces, pygltflib.ELEMENT_ARRAY_BUFFER,
        pygltflib.UNSIGNED_INT, pygltflib.SCALAR, with_minmax=True,
    )

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(pygltflib.Mesh(
        name=name,
        primitives=[pygltflib.Primitive(
            attributes=pygltflib.Attributes(
                POSITION=pos_acc,
                NORMAL=norm_acc,
                TEXCOORD_0=uv_acc,
            ),
            indices=idx_acc,
            material=mat_idx,
        )],
    ))

    node_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(
        name=name,
        mesh=mesh_idx,
        translation=translation,
    ))
    return node_idx


def _create_cylinder(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    name: str,
    png_bytes: bytes,
    translation: list[float],
    radius: float,
    height: float,
) -> int:
    """Create a 32-segment open-ended textured cylinder. Returns node index.

    The texture wraps around the cylinder horizontally (U = angle / 2pi)
    and spans top-to-bottom vertically (V = 0 at top, V = 1 at bottom).
    An extra vertex column at the seam ensures UV continuity.
    """
    segments = 32
    cols = segments + 1  # extra column for UV seam
    hh = height / 2.0

    # Build vertices, normals, UVs: 2 rows (top, bottom) × cols
    n_verts = cols * 2
    vertices = np.empty((n_verts, 3), dtype=np.float32)
    normals = np.empty((n_verts, 3), dtype=np.float32)
    uvs = np.empty((n_verts, 2), dtype=np.float32)

    for col in range(cols):
        u = col / segments
        angle = u * 2.0 * np.pi
        nx = np.cos(angle)
        nz = np.sin(angle)
        x = radius * nx
        z = radius * nz

        # Top row (V=0)
        top = col
        vertices[top] = [x, hh, z]
        normals[top] = [nx, 0.0, nz]
        uvs[top] = [u, 0.0]

        # Bottom row (V=1)
        bot = col + cols
        vertices[bot] = [x, -hh, z]
        normals[bot] = [nx, 0.0, nz]
        uvs[bot] = [u, 1.0]

    # Triangle indices: 2 triangles per segment
    faces_list: list[int] = []
    for col in range(segments):
        tl = col
        tr = col + 1
        bl = col + cols
        br = col + 1 + cols
        faces_list.extend([tl, bl, tr, tr, bl, br])

    faces = np.array(faces_list, dtype=np.uint32)

    mat_idx = _setup_texture_material(
        gltf, binary_data, name, png_bytes, double_sided=False,
    )
    return _create_mesh_node(
        gltf, binary_data, name,
        vertices, normals, uvs, faces, mat_idx, translation,
    )


def _create_panel(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    name: str,
    png_bytes: bytes,
    translation: list[float],
    width: float,
    height: float,
) -> int:
    """Create a double-sided flat panel readable from both sides. Returns node index.

    Front face faces +Z with normal UVs.  Back face faces -Z with
    horizontally mirrored UVs so text reads correctly from behind.
    Uses ``doubleSided=False`` since both faces are explicit geometry.
    """
    hw = width / 2.0
    hh = height / 2.0

    vertices = np.array([
        # Front face (+Z)
        [-hw, -hh, 0.0],
        [ hw, -hh, 0.0],
        [ hw,  hh, 0.0],
        [-hw,  hh, 0.0],
        # Back face (-Z) — mirrored X for opposite winding
        [ hw, -hh, 0.0],
        [-hw, -hh, 0.0],
        [-hw,  hh, 0.0],
        [ hw,  hh, 0.0],
    ], dtype=np.float32)

    normals = np.array([
        [0.0, 0.0,  1.0],
        [0.0, 0.0,  1.0],
        [0.0, 0.0,  1.0],
        [0.0, 0.0,  1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float32)

    uvs = np.array([
        # Front UVs (normal)
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        # Back UVs (same mapping — vertex positions are already mirrored)
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)

    faces = np.array([
        0, 1, 2, 0, 2, 3,  # Front
        4, 5, 6, 4, 6, 7,  # Back
    ], dtype=np.uint32)

    mat_idx = _setup_texture_material(
        gltf, binary_data, name, png_bytes, double_sided=False,
    )
    return _create_mesh_node(
        gltf, binary_data, name,
        vertices, normals, uvs, faces, mat_idx, translation,
    )


def add_legend_nodes(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    mesh_vertices: np.ndarray,
    coloring: str,
    clamp_range: tuple[float, float],
    centroid: list[float],
    metadata: dict | None = None,
) -> list[int]:
    """Add color legend and metadata panel nodes to a glTF document.

    Both panels are positioned to the right of the mesh bounding box.
    Returns a list of node indices for the created panels.

    Args:
        gltf: The glTF document being built.
        binary_data: Binary buffer being assembled.
        mesh_vertices: Centered mesh vertices (for bounding box calculation).
        coloring: Colormap mode ("lat", "bipolar", "unipolar").
        clamp_range: (min, max) values for the color scale.
        centroid: Translation offset applied to mesh nodes.
        metadata: Optional dict of metadata fields for the info card.
    """
    node_indices: list[int] = []

    # Compute bounding box and bounding sphere from centered mesh vertices
    bbox_min = mesh_vertices.min(axis=0)
    bbox_max = mesh_vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    mesh_height = float(bbox_size[1])

    # Bounding-sphere radius — robust across varying patient anatomies
    half_extents = bbox_size / 2.0
    bsphere_radius = float(np.linalg.norm(half_extents))
    panel_gap = bsphere_radius * 0.35

    panel_height = mesh_height * 0.30
    cylinder_radius = panel_height * 0.5 / np.pi  # circumference ≈ legend_width
    info_height = panel_height * 0.75
    info_width = info_height  # square aspect matching 256×256 texture

    # Legend cylinder: right of mesh, centered at Y=0, Z=0
    legend_png = render_legend_wrap_image(coloring, clamp_range)
    legend_x = float(bbox_max[0]) + panel_gap + cylinder_radius
    legend_translation = [legend_x, 0.0, 0.0]
    legend_idx = _create_cylinder(
        gltf, binary_data, "legend", legend_png,
        legend_translation, cylinder_radius, panel_height,
    )
    node_indices.append(legend_idx)

    # Info panel: flat double-sided quad, next to legend
    if metadata:
        info_png = render_info_image(metadata)
        info_x = legend_x + cylinder_radius + info_width / 2.0 + panel_gap * 0.3
        info_translation = [info_x, 0.0, 0.0]
        info_idx = _create_panel(
            gltf, binary_data, "info", info_png,
            info_translation, info_width, info_height,
        )
        node_indices.append(info_idx)

    # Write structured metadata into gltf.extras for programmatic access
    extras: dict = {
        "coloring": coloring,
        "clamp_range": list(clamp_range),
    }
    if metadata:
        extras.update(metadata)
    gltf.extras = extras

    logger.info(
        "Added %d legend/info panel(s) for %s coloring [%.1f–%.1f]",
        len(node_indices), coloring, clamp_range[0], clamp_range[1],
    )

    return node_indices
