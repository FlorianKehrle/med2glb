"""Bake per-vertex colors into a texture via xatlas UV unwrap.

HoloLens 2 (MRTK / glTFast) does not render the glTF COLOR_0 vertex attribute.
This module uses xatlas for proper UV parameterization, then rasterizes vertex
colors into a PNG texture (baseColorTexture) with barycentric interpolation and
gutter bleeding to eliminate seam artifacts.
"""

from __future__ import annotations

import io
import logging
import time

import numpy as np
import xatlas
from PIL import Image

logger = logging.getLogger("med2glb")


def xatlas_unwrap(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UV unwrap a mesh with xatlas.

    Args:
        vertices: (V, 3) float vertex positions.
        faces: (F, 3) uint32 triangle indices.
        normals: (V, 3) float vertex normals (optional, improves chart quality).

    Returns:
        (vmapping, new_faces, uvs) where:
        - vmapping: (V', ) int32 — maps each new vertex to its original index.
        - new_faces: (F, 3) uint32 — triangle indices into the new vertex array.
        - uvs: (V', 2) float32 — UV coordinates in [0, 1].
    """
    verts_f32 = np.ascontiguousarray(vertices, dtype=np.float32)
    faces_u32 = np.ascontiguousarray(faces, dtype=np.uint32)

    atlas = xatlas.Atlas()

    if normals is not None:
        norms_f32 = np.ascontiguousarray(normals, dtype=np.float32)
        atlas.add_mesh(verts_f32, faces_u32, norms_f32)
    else:
        atlas.add_mesh(verts_f32, faces_u32)

    t0 = time.monotonic()
    atlas.generate()
    elapsed = time.monotonic() - t0
    vmapping, new_faces, uvs = atlas[0]

    elapsed_str = f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed)//60}m {int(elapsed)%60}s"
    logger.info(
        "xatlas unwrap: %d → %d verts, %d charts (%s)",
        len(vertices), len(vmapping), atlas.chart_count, elapsed_str,
    )

    return vmapping, new_faces.astype(np.uint32), uvs.astype(np.float32)


# ---------------------------------------------------------------------------
# Core rasterization: compute pixel→triangle mapping
# ---------------------------------------------------------------------------

def _compute_pixel_mapping(
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute which pixel belongs to which triangle with what weights.

    Enumerates all bbox pixels for all triangles in a single vectorized pass
    using np.repeat to handle variable-size bounding boxes.

    Returns:
        (pixel_y, pixel_x, v0, v1, v2, bw0, bw1, bw2) — flat arrays where
        each entry is one pixel-to-triangle mapping with barycentric weights.
    """
    tex_w = tex_h = texture_size

    # Convert UVs to pixel coordinates (glTF UV: origin top-left)
    uv_px = uvs.copy()
    uv_px[:, 0] *= tex_w
    uv_px[:, 1] *= tex_h

    # Gather per-triangle vertex pixel coords: (F, 2) each
    p0x = uv_px[faces[:, 0], 0]
    p0y = uv_px[faces[:, 0], 1]
    p1x = uv_px[faces[:, 1], 0]
    p1y = uv_px[faces[:, 1], 1]
    p2x = uv_px[faces[:, 2], 0]
    p2y = uv_px[faces[:, 2], 1]

    # Per-triangle bounding boxes (clamped to texture)
    bb_min_x = np.clip(np.floor(np.minimum(np.minimum(p0x, p1x), p2x)).astype(np.int32), 0, tex_w - 1)
    bb_max_x = np.clip(np.ceil(np.maximum(np.maximum(p0x, p1x), p2x)).astype(np.int32), 0, tex_w - 1)
    bb_min_y = np.clip(np.floor(np.minimum(np.minimum(p0y, p1y), p2y)).astype(np.int32), 0, tex_h - 1)
    bb_max_y = np.clip(np.ceil(np.maximum(np.maximum(p0y, p1y), p2y)).astype(np.int32), 0, tex_h - 1)

    bb_w = (bb_max_x - bb_min_x + 1).astype(np.int64)
    bb_h = (bb_max_y - bb_min_y + 1).astype(np.int64)
    n_pixels_per_tri = bb_w * bb_h  # (F,)

    # Filter degenerate triangles
    denom = (p1y - p2y) * (p0x - p2x) + (p2x - p1x) * (p0y - p2y)
    valid = (np.abs(denom) > 1e-10) & (n_pixels_per_tri > 0)

    # Work only with valid triangles
    valid_idx = np.nonzero(valid)[0]
    if len(valid_idx) == 0:
        empty = np.array([], dtype=np.intp)
        empty_f = np.array([], dtype=np.float32)
        empty_u = np.array([], dtype=np.uint32)
        return empty, empty, empty_u, empty_u, empty_u, empty_f, empty_f, empty_f

    n_pix = n_pixels_per_tri[valid_idx]
    total_pixels = int(n_pix.sum())

    # Process in chunks to avoid extreme memory usage
    # Each pixel needs ~40 bytes of working memory
    max_pixels_per_chunk = 20_000_000  # ~800MB peak
    if total_pixels <= max_pixels_per_chunk:
        return _rasterize_chunk(
            valid_idx, n_pix, bb_min_x, bb_min_y, bb_w,
            p0x, p0y, p1x, p1y, p2x, p2y, denom, faces, tex_w, tex_h,
        )

    # Split into chunks
    all_py, all_px = [], []
    all_v0, all_v1, all_v2 = [], [], []
    all_bw0, all_bw1, all_bw2 = [], [], []

    cumsum = np.cumsum(n_pix)
    chunk_start = 0
    while chunk_start < len(valid_idx):
        # Find how many triangles fit in this chunk
        target = cumsum[chunk_start - 1] + max_pixels_per_chunk if chunk_start > 0 else max_pixels_per_chunk
        chunk_end = int(np.searchsorted(cumsum, target, side="right"))
        chunk_end = max(chunk_end, chunk_start + 1)  # at least 1 triangle

        chunk_idx = valid_idx[chunk_start:chunk_end]
        chunk_npix = n_pix[chunk_start:chunk_end]

        result = _rasterize_chunk(
            chunk_idx, chunk_npix, bb_min_x, bb_min_y, bb_w,
            p0x, p0y, p1x, p1y, p2x, p2y, denom, faces, tex_w, tex_h,
        )
        py, px, v0, v1, v2, bw0, bw1, bw2 = result
        if len(py) > 0:
            all_py.append(py)
            all_px.append(px)
            all_v0.append(v0)
            all_v1.append(v1)
            all_v2.append(v2)
            all_bw0.append(bw0)
            all_bw1.append(bw1)
            all_bw2.append(bw2)

        chunk_start = chunk_end

    if all_py:
        return (
            np.concatenate(all_py), np.concatenate(all_px),
            np.concatenate(all_v0), np.concatenate(all_v1), np.concatenate(all_v2),
            np.concatenate(all_bw0), np.concatenate(all_bw1), np.concatenate(all_bw2),
        )
    empty = np.array([], dtype=np.intp)
    empty_f = np.array([], dtype=np.float32)
    empty_u = np.array([], dtype=np.uint32)
    return empty, empty, empty_u, empty_u, empty_u, empty_f, empty_f, empty_f


def _rasterize_chunk(
    tri_indices: np.ndarray,
    n_pixels_per_tri: np.ndarray,
    bb_min_x: np.ndarray,
    bb_min_y: np.ndarray,
    bb_w: np.ndarray,
    p0x: np.ndarray, p0y: np.ndarray,
    p1x: np.ndarray, p1y: np.ndarray,
    p2x: np.ndarray, p2y: np.ndarray,
    denom: np.ndarray,
    faces: np.ndarray,
    tex_w: int, tex_h: int,
) -> tuple[np.ndarray, ...]:
    """Rasterize a chunk of triangles. Returns (py, px, v0, v1, v2, bw0, bw1, bw2)."""
    total = int(n_pixels_per_tri.sum())

    # Triangle index for each pixel (which triangle does this pixel belong to)
    tri_for_pixel = np.repeat(np.arange(len(tri_indices)), n_pixels_per_tri)  # (total,)

    # Compute pixel-within-triangle index
    pixel_offset = np.arange(total) - np.repeat(
        np.concatenate([[0], np.cumsum(n_pixels_per_tri)[:-1]]),
        n_pixels_per_tri,
    )

    # Convert flat pixel offset to (row, col) within bbox
    t = tri_indices[tri_for_pixel]
    w = bb_w[t]
    local_y = pixel_offset // w
    local_x = pixel_offset % w

    # Absolute pixel coordinates (center of pixel)
    px = (bb_min_x[t] + local_x).astype(np.float32) + 0.5
    py = (bb_min_y[t] + local_y).astype(np.float32) + 0.5

    # Barycentric coordinates
    inv_d = 1.0 / denom[t]
    dx = px - p2x[t]
    dy = py - p2y[t]

    w0 = ((p1y[t] - p2y[t]) * dx + (p2x[t] - p1x[t]) * dy) * inv_d
    w1 = ((p2y[t] - p0y[t]) * dx + (p0x[t] - p2x[t]) * dy) * inv_d
    w2 = 1.0 - w0 - w1

    # Inside triangle test
    inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)

    if not np.any(inside):
        empty = np.array([], dtype=np.intp)
        empty_f = np.array([], dtype=np.float32)
        empty_u = np.array([], dtype=np.uint32)
        return empty, empty, empty_u, empty_u, empty_u, empty_f, empty_f, empty_f

    # Filter to inside pixels only
    t_in = t[inside]
    px_in = (px[inside] - 0.5).astype(np.intp)
    py_in = (py[inside] - 0.5).astype(np.intp)

    # Bounds check
    in_bounds = (px_in >= 0) & (px_in < tex_w) & (py_in >= 0) & (py_in < tex_h)
    t_in = t_in[in_bounds]
    px_in = px_in[in_bounds]
    py_in = py_in[in_bounds]

    # Clamp barycentric for interpolation
    bw0 = np.maximum(w0[inside][in_bounds], 0.0).astype(np.float32)
    bw1 = np.maximum(w1[inside][in_bounds], 0.0).astype(np.float32)
    bw2 = np.maximum(w2[inside][in_bounds], 0.0).astype(np.float32)
    s = bw0 + bw1 + bw2
    s = np.where(s < 1e-10, 1.0, s)
    bw0 /= s
    bw1 /= s
    bw2 /= s

    v0 = faces[t_in, 0]
    v1 = faces[t_in, 1]
    v2 = faces[t_in, 2]

    return py_in, px_in, v0, v1, v2, bw0, bw1, bw2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rasterize_vertex_colors(
    faces: np.ndarray,
    uvs: np.ndarray,
    vertex_colors: np.ndarray,
    texture_size: int,
) -> bytes:
    """Rasterize per-vertex colors into a texture using UV coordinates.

    Args:
        faces: (F, 3) uint32 triangle indices into uvs/vertex_colors.
        uvs: (V, 2) float32 UV coordinates in [0, 1].
        vertex_colors: (V, 4) float32 RGBA colors in [0, 1].
        texture_size: Width/height of the output square texture.

    Returns:
        PNG bytes of the RGBA texture.
    """
    pixel_y, pixel_x, v0, v1, v2, bw0, bw1, bw2 = _compute_pixel_mapping(
        faces, uvs, texture_size,
    )
    return _apply_and_encode(
        pixel_y, pixel_x, v0, v1, v2, bw0, bw1, bw2,
        vertex_colors, texture_size,
    )


def precompute_rasterization_map(
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_size: int,
) -> dict:
    """Precompute pixel-to-triangle mapping for repeated rasterization.

    Call once after UV unwrap, then use ``apply_rasterization_map`` per frame.
    """
    pixel_y, pixel_x, v0, v1, v2, bw0, bw1, bw2 = _compute_pixel_mapping(
        faces, uvs, texture_size,
    )

    # Build mask for gutter bleeding (used by _bleed_gutter in _apply_and_encode)
    mask = np.zeros((texture_size, texture_size), dtype=bool)
    if len(pixel_y) > 0:
        mask[pixel_y, pixel_x] = True

    logger.info(
        "Rasterization map: %d pixels mapped from %d faces (%dx%d texture)",
        len(pixel_y), len(faces), texture_size, texture_size,
    )

    return {
        "pixel_y": pixel_y, "pixel_x": pixel_x,
        "v0": v0, "v1": v1, "v2": v2,
        "bw0": bw0, "bw1": bw1, "bw2": bw2,
        "mask": mask, "tex_size": texture_size,
    }


def apply_rasterization_map(
    raster_map: dict,
    vertex_colors: np.ndarray,
    image_format: str = "PNG",
    jpeg_quality: int = 85,
) -> bytes:
    """Apply precomputed rasterization map with new vertex colors.

    Vectorized gather + weighted sum + scatter for pixel colors,
    then inline gutter bleeding and image encoding.
    """
    return _apply_and_encode(
        raster_map["pixel_y"], raster_map["pixel_x"],
        raster_map["v0"], raster_map["v1"], raster_map["v2"],
        raster_map["bw0"], raster_map["bw1"], raster_map["bw2"],
        vertex_colors, raster_map["tex_size"],
        mask=raster_map["mask"],
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )


def _apply_and_encode(
    pixel_y: np.ndarray, pixel_x: np.ndarray,
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    bw0: np.ndarray, bw1: np.ndarray, bw2: np.ndarray,
    vertex_colors: np.ndarray,
    texture_size: int,
    mask: np.ndarray | None = None,
    image_format: str = "PNG",
    jpeg_quality: int = 85,
) -> bytes:
    """Interpolate colors, scatter into texture, bleed, encode to image bytes."""
    tex_size = texture_size
    texture = np.zeros((tex_size, tex_size, 4), dtype=np.float32)

    if mask is None:
        mask = np.zeros((tex_size, tex_size), dtype=bool)
        if len(pixel_y) > 0:
            mask[pixel_y, pixel_x] = True

    if len(pixel_y) > 0:
        colors = (
            bw0[:, np.newaxis] * vertex_colors[v0]
            + bw1[:, np.newaxis] * vertex_colors[v1]
            + bw2[:, np.newaxis] * vertex_colors[v2]
        )
        texture[pixel_y, pixel_x] = colors

    _bleed_gutter(texture, mask.copy(), tex_size, tex_size)

    texture_u8 = np.clip(texture * 255 + 0.5, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    if image_format == "JPEG":
        img = Image.fromarray(texture_u8[:, :, :3], mode="RGB")
        img.save(buf, format="JPEG", quality=jpeg_quality)
    else:
        # Output RGB (no alpha) — textures use alphaMode OPAQUE in glTF,
        # and unrasterized pixels would otherwise have alpha=0 causing
        # transparency artifacts.
        img = Image.fromarray(texture_u8[:, :, :3], mode="RGB")
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _bleed_gutter(
    texture: np.ndarray, mask: np.ndarray, tex_h: int, tex_w: int,
) -> None:
    """Expand filled pixels into empty neighbors (in-place).

    10 iterations provides enough padding for mipmap sampling at lower
    resolution levels (distant viewing in AR). With fewer iterations,
    UV chart boundaries bleed into unfilled black pixels at lower mip
    levels, causing visible dark spots on distant meshes.
    """
    for _ in range(10):
        empty = ~mask
        if not np.any(empty):
            break

        accum = np.zeros_like(texture)
        count = np.zeros((tex_h, tex_w), dtype=np.float32)

        # Up
        accum[1:] += np.where(mask[:-1, :, np.newaxis], texture[:-1], 0)
        count[1:] += mask[:-1].astype(np.float32)
        # Down
        accum[:-1] += np.where(mask[1:, :, np.newaxis], texture[1:], 0)
        count[:-1] += mask[1:].astype(np.float32)
        # Left
        accum[:, 1:] += np.where(mask[:, :-1, np.newaxis], texture[:, :-1], 0)
        count[:, 1:] += mask[:, :-1].astype(np.float32)
        # Right
        accum[:, :-1] += np.where(mask[:, 1:, np.newaxis], texture[:, 1:], 0)
        count[:, :-1] += mask[:, 1:].astype(np.float32)

        fill = empty & (count > 0)
        if not np.any(fill):
            break
        texture[fill] = accum[fill] / count[fill, np.newaxis]
        mask[fill] = True


def bake_vertex_colors_to_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    texture_size: int = 1024,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, bytes]:
    """Full pipeline: xatlas UV unwrap + rasterize vertex colors.

    Args:
        vertices: (V, 3) float vertex positions.
        faces: (F, 3) uint32 triangle indices.
        vertex_colors: (V, 4) float32 RGBA colors in [0, 1].
        texture_size: Width/height of the output square texture.
        normals: (V, 3) float vertex normals (optional).

    Returns:
        (new_verts, new_faces, new_normals, uvs, png_bytes) where geometry
        arrays match the xatlas-unwrapped topology.
    """
    vmapping, new_faces, uvs = xatlas_unwrap(vertices, faces, normals)

    new_verts = vertices[vmapping].astype(np.float32)
    new_normals = normals[vmapping].astype(np.float32) if normals is not None else None
    new_colors = vertex_colors[vmapping].astype(np.float32)

    png_bytes = rasterize_vertex_colors(new_faces, uvs, new_colors, texture_size)

    logger.info(
        "Baked vertex colors: %d faces → %dx%d texture (%d KB)",
        len(faces), texture_size, texture_size, len(png_bytes) // 1024,
    )

    return new_verts, new_faces, new_normals, uvs, png_bytes


def compute_texture_size(n_faces: int) -> int:
    """Choose a texture resolution based on face count."""
    if n_faces <= 5000:
        return 512
    if n_faces <= 20000:
        return 1024
    if n_faces <= 80000:
        return 2048
    return 4096
