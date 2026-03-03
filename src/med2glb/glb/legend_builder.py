"""Color legend and metadata panel nodes for CARTO GLB exports.

Renders a color scale legend and a metadata info card as PNG images using
Pillow, then embeds them as textured quad nodes (children of root) positioned
next to the mesh.  Uses ``KHR_materials_unlit`` so panels are always readable
regardless of scene lighting.
"""

from __future__ import annotations

import io
import logging
import struct

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


def render_legend_image(
    coloring: str,
    clamp_range: tuple[float, float],
    width: int = 128,
    height: int = 256,
) -> bytes:
    """Render a vertical color scale legend as a PNG image.

    Returns PNG bytes.
    """
    stops = _get_stops(coloring)
    title = _TITLES.get(coloring, coloring)
    vmin, vmax = clamp_range

    img = Image.new("RGBA", (width, height), (30, 30, 30, 200))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=14)
    font_sm = ImageFont.load_default(size=12)

    # Title
    draw.text((width // 2, 8), title, fill=(255, 255, 255, 255),
              font=font, anchor="mt")

    # Gradient bar area
    bar_left = 14
    bar_right = 38
    bar_top = 32
    bar_bottom = height - 16

    # Draw vertical gradient (top = max, bottom = min)
    for y in range(bar_top, bar_bottom):
        t = 1.0 - (y - bar_top) / (bar_bottom - bar_top)
        color = _interpolate_color(t, stops)
        draw.line([(bar_left, y), (bar_right, y)], fill=color + (255,))

    # Border around gradient bar
    draw.rectangle(
        [(bar_left - 1, bar_top - 1), (bar_right + 1, bar_bottom + 1)],
        outline=(200, 200, 200, 255),
    )

    # Tick marks and labels (5 ticks: min, 25%, 50%, 75%, max)
    n_ticks = 5
    tick_x = bar_right + 4
    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        y = bar_bottom - frac * (bar_bottom - bar_top)
        val = vmin + frac * (vmax - vmin)

        # Format value
        if abs(val) >= 100:
            label = f"{val:.0f}"
        elif abs(val) >= 1:
            label = f"{val:.1f}"
        else:
            label = f"{val:.2f}"

        # Tick line
        draw.line([(bar_right + 1, int(y)), (tick_x, int(y))],
                  fill=(200, 200, 200, 255))
        # Label
        draw.text((tick_x + 3, int(y)), label,
                  fill=(220, 220, 220, 255), font=font_sm, anchor="lm")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_info_image(
    metadata: dict,
    width: int = 256,
    height: int = 256,
) -> bytes:
    """Render a metadata info card as a PNG image.

    Returns PNG bytes.
    """
    img = Image.new("RGBA", (width, height), (30, 30, 30, 200))
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
                lines.append((label, f"{val[0]} \u2013 {val[1]} {unit}".strip()))
            else:
                lines.append((label, str(val)))
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
        draw.text((12, y), f"{label}:", fill=(180, 180, 180, 255), font=font)
        draw.text((12 + 80, y), value, fill=(240, 240, 240, 255), font=font)
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_quad(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    name: str,
    png_bytes: bytes,
    translation: list[float],
    quad_width: float,
    quad_height: float,
) -> int:
    """Create a textured quad node with an unlit material. Returns node index."""
    # Ensure lists exist
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

    # Sampler (add if none exist yet)
    if len(gltf.samplers) == 0:
        gltf.samplers.append(pygltflib.Sampler(
            magFilter=pygltflib.LINEAR,
            minFilter=pygltflib.LINEAR,
            wrapS=pygltflib.CLAMP_TO_EDGE,
            wrapT=pygltflib.CLAMP_TO_EDGE,
        ))

    tex_idx = len(gltf.textures)
    gltf.textures.append(pygltflib.Texture(sampler=0, source=img_idx))

    # Material: unlit, alpha blend, double-sided
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
        doubleSided=True,
        extensions={"KHR_materials_unlit": {}},
    ))

    # Ensure KHR_materials_unlit is registered
    if not hasattr(gltf, "extensionsUsed") or gltf.extensionsUsed is None:
        gltf.extensionsUsed = []
    if "KHR_materials_unlit" not in gltf.extensionsUsed:
        gltf.extensionsUsed.append("KHR_materials_unlit")

    # Quad geometry: 4 vertices, 2 triangles
    # Centered at origin, extends in X and Y
    hw = quad_width / 2.0
    hh = quad_height / 2.0
    vertices = np.array([
        [-hw, -hh, 0.0],
        [hw, -hh, 0.0],
        [hw, hh, 0.0],
        [-hw, hh, 0.0],
    ], dtype=np.float32)

    uvs = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)

    faces = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    # Write geometry
    pos_acc = write_accessor(
        gltf, binary_data, vertices, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC3, with_minmax=True,
    )
    uv_acc = write_accessor(
        gltf, binary_data, uvs, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC2,
    )
    idx_acc = write_accessor(
        gltf, binary_data, faces, pygltflib.ELEMENT_ARRAY_BUFFER,
        pygltflib.UNSIGNED_INT, pygltflib.SCALAR, with_minmax=True,
    )

    # Mesh + node
    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(pygltflib.Mesh(
        name=name,
        primitives=[pygltflib.Primitive(
            attributes=pygltflib.Attributes(
                POSITION=pos_acc,
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

    # Compute bounding box from centered mesh vertices
    bbox_min = mesh_vertices.min(axis=0)
    bbox_max = mesh_vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    mesh_height = float(bbox_size[1])
    panel_height = mesh_height * 0.30
    panel_gap = mesh_height * 0.08

    # Legend quad dimensions (narrower than info card)
    legend_width = panel_height * 0.5  # 128:256 aspect ratio
    info_width = panel_height * 1.0     # 256:256 aspect ratio

    # Position: right of mesh, vertically centered
    x_base = float(bbox_max[0]) + panel_gap + centroid[0]

    # Legend panel: upper position
    legend_png = render_legend_image(coloring, clamp_range)
    legend_translation = [
        x_base + legend_width / 2.0,
        centroid[1] + mesh_height * 0.12,
        centroid[2],
    ]
    legend_idx = _create_quad(
        gltf, binary_data, "legend", legend_png,
        legend_translation, legend_width, panel_height,
    )
    node_indices.append(legend_idx)

    # Info panel: lower position (only if metadata provided)
    if metadata:
        info_png = render_info_image(metadata)
        info_translation = [
            x_base + info_width / 2.0,
            centroid[1] - mesh_height * 0.20,
            centroid[2],
        ]
        info_idx = _create_quad(
            gltf, binary_data, "info", info_png,
            info_translation, info_width, panel_height,
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
