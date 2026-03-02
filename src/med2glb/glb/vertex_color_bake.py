"""Bake per-vertex colors into a texture atlas for HoloLens 2 compatibility.

HoloLens 2 (MRTK / glTFast) does not render the glTF COLOR_0 vertex attribute.
This module generates UV coordinates and rasterizes vertex colors into a PNG
texture (baseColorTexture) which is universally supported.

Strategy: pack each triangle into a texture atlas with 1px gutter, then
rasterize vertex colors via barycentric interpolation.  The mesh is "unwelded"
so each face gets its own 3 vertices with unique UVs.
"""

from __future__ import annotations

import io
import logging
import math

import numpy as np
from PIL import Image

logger = logging.getLogger("med2glb")


def bake_vertex_colors_to_texture(
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    texture_size: int = 1024,
) -> tuple[np.ndarray, bytes]:
    """Bake per-vertex RGBA colors into a texture atlas.

    The mesh is unwelded: each face gets 3 unique vertices in UV space.
    The caller must also unweld positions/normals to match.

    Args:
        faces: (F, 3) triangle indices into vertex_colors.
        vertex_colors: (N, 4) RGBA float32 colors in [0, 1].
        texture_size: Width/height of the output square texture.

    Returns:
        (uvs, png_bytes) where uvs is (F*3, 2) float32 per-vertex UV
        coordinates (for the unwelded mesh) and png_bytes is RGBA PNG data.
    """
    n_faces = len(faces)
    tex_w = tex_h = texture_size

    # --- Step 1: Grid layout ---
    gutter = 1
    cols = max(1, int(math.ceil(math.sqrt(n_faces))))
    rows = max(1, int(math.ceil(n_faces / cols)))
    cell_w = tex_w // cols
    cell_h = tex_h // rows
    inner_w = max(2, cell_w - 2 * gutter)
    inner_h = max(2, cell_h - 2 * gutter)

    # --- Step 2: Assign UVs (vectorized) ---
    fi = np.arange(n_faces)
    col_idx = fi % cols
    row_idx = fi // cols
    cx = col_idx * cell_w + gutter
    cy = row_idx * cell_h + gutter

    # 3 UV positions per cell: bottom-left, bottom-right, top-center
    uvs = np.zeros((n_faces * 3, 2), dtype=np.float32)
    uvs[0::3, 0] = (cx + 0.5) / tex_w
    uvs[0::3, 1] = (cy + 0.5) / tex_h
    uvs[1::3, 0] = (cx + inner_w - 0.5) / tex_w
    uvs[1::3, 1] = (cy + 0.5) / tex_h
    uvs[2::3, 0] = (cx + inner_w / 2) / tex_w
    uvs[2::3, 1] = (cy + inner_h - 0.5) / tex_h

    # --- Step 3: Vectorized rasterization ---
    texture = _rasterize_all_faces(
        faces, vertex_colors, cx, cy, inner_w, inner_h, tex_w, tex_h,
    )

    # Flip Y: glTF UV origin is bottom-left, image origin is top-left
    texture = np.flipud(texture)

    # --- Step 4: Encode PNG ---
    img = Image.fromarray(texture, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    logger.info(
        "Baked vertex colors: %d faces → %dx%d texture (%d KB)",
        n_faces, tex_w, tex_h, len(png_bytes) // 1024,
    )

    return uvs, png_bytes


def compute_texture_size(n_faces: int) -> int:
    """Choose a texture resolution based on face count."""
    if n_faces <= 5000:
        return 512
    if n_faces <= 20000:
        return 1024
    if n_faces <= 80000:
        return 2048
    return 4096


def _rasterize_all_faces(
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    inner_w: int,
    inner_h: int,
    tex_w: int,
    tex_h: int,
) -> np.ndarray:
    """Rasterize all triangle cells into the texture using vectorized numpy.

    Each triangle is mapped to a fixed shape within its cell:
    v0 at (0.5, 0.5), v1 at (inner_w-0.5, 0.5), v2 at (inner_w/2, inner_h-0.5).

    We compute a pixel grid for one cell, compute barycentric weights for
    every pixel in that grid, then apply per-face colors in bulk.
    """
    texture = np.zeros((tex_h, tex_w, 4), dtype=np.uint8)
    n_faces = len(faces)

    # Convert vertex colors to uint8 for all face vertices at once
    vc_u8 = np.clip(vertex_colors * 255 + 0.5, 0, 255).astype(np.float32)

    # Per-face colors: (F, 3_verts, 4_rgba) in float for interpolation
    c0 = vc_u8[faces[:, 0]]  # (F, 4)
    c1 = vc_u8[faces[:, 1]]  # (F, 4)
    c2 = vc_u8[faces[:, 2]]  # (F, 4)

    # Build pixel grid for one cell (same for all faces)
    px = np.arange(inner_w, dtype=np.float32) + 0.5  # (W,)
    py = np.arange(inner_h, dtype=np.float32) + 0.5  # (H,)
    gx, gy = np.meshgrid(px, py)  # (H, W) each
    gx_flat = gx.ravel()  # (H*W,)
    gy_flat = gy.ravel()  # (H*W,)
    n_pixels = len(gx_flat)

    # Triangle reference vertices in cell-local coords
    t0x, t0y = 0.5, 0.5
    t1x, t1y = inner_w - 0.5, 0.5
    t2x, t2y = inner_w / 2, inner_h - 0.5

    # Barycentric weights for the fixed triangle (same for all faces)
    denom = (t1y - t2y) * (t0x - t2x) + (t2x - t1x) * (t0y - t2y)
    if abs(denom) < 1e-10:
        # Degenerate cell layout — shouldn't happen with inner_w >= 2
        return texture
    inv_d = 1.0 / denom

    w0 = ((t1y - t2y) * (gx_flat - t2x) + (t2x - t1x) * (gy_flat - t2y)) * inv_d  # (P,)
    w1 = ((t2y - t0y) * (gx_flat - t2x) + (t0x - t2x) * (gy_flat - t2y)) * inv_d  # (P,)
    w2 = 1.0 - w0 - w1  # (P,)

    # Clamp barycentric coords to fill entire cell (extrapolate outside triangle)
    w0c = np.maximum(w0, 0.0)
    w1c = np.maximum(w1, 0.0)
    w2c = np.maximum(w2, 0.0)
    s = w0c + w1c + w2c
    s = np.where(s < 1e-10, 1.0, s)
    w0c /= s
    w1c /= s
    w2c /= s

    # Process faces in batches to limit memory usage
    batch_size = max(1, min(2000, n_faces))
    for start in range(0, n_faces, batch_size):
        end = min(start + batch_size, n_faces)
        b = end - start

        # Interpolate: (B, P, 4) = w0(P) * c0(B,4) + w1(P) * c1(B,4) + w2(P) * c2(B,4)
        # Use einsum for efficiency
        colors_bp4 = (
            np.outer(w0c, np.ones(4)).reshape(1, n_pixels, 4) * c0[start:end, np.newaxis, :]
            + np.outer(w1c, np.ones(4)).reshape(1, n_pixels, 4) * c1[start:end, np.newaxis, :]
            + np.outer(w2c, np.ones(4)).reshape(1, n_pixels, 4) * c2[start:end, np.newaxis, :]
        )  # (B, P, 4)
        colors_bp4 = np.clip(colors_bp4, 0, 255).astype(np.uint8)

        # Reshape to (B, H, W, 4)
        colors_bhw4 = colors_bp4.reshape(b, inner_h, inner_w, 4)

        # Scatter into texture
        for i in range(b):
            fi = start + i
            y0 = int(cy[fi])
            x0 = int(cx[fi])
            texture[y0:y0 + inner_h, x0:x0 + inner_w] = colors_bhw4[i]

    return texture
