"""CARTO animated GLB: excitation wavefront via COLOR_0 vertex color morph targets.

Single-mesh animation using glTF morph targets.  The static LAT heatmap
is stored as base COLOR_0; per-frame wavefront colors (glow + ring model)
are stored as COLOR_0 morph target deltas.  A morph weight animation cycles
through frames for a seamless loop matching real CARTO 3 behavior (~4.5s).

The same cache (AnimatedBakeCache) pre-computes xatlas UV unwrap once for the
static heatmap GLBs, and computes wavefront frame colors for the animated GLB.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygltflib

from med2glb._utils import fmt_duration
from med2glb.core.types import MeshData
from med2glb.glb.builder import _center_vertices, _pad_to_4, write_accessor
from med2glb.glb.vertex_color_bake import compute_texture_size

logger = logging.getLogger("med2glb")

# Excitation wavefront parameters (tuned to match real CARTO3 behavior).
#
# The wavefront has two components:
#   1. Leading-edge ring: narrow bright band at the activation frontier
#   2. Trailing glow: broad diffuse region behind the ring (recently activated tissue)
#
# Combined coverage: ~25-30% of mesh surface per frame (matching real CARTO3).
# The ring is primarily white with subtle base-color tint.
_RING_SIGMA = 0.03       # σ of Gaussian ring (narrow leading edge)
_GLOW_DECAY = 8.0        # exponential decay rate for trailing glow
_GLOW_EXTENT = 0.25      # how far behind the ring the glow extends (in LAT-norm units)
_RING_INTENSITY = 0.70   # peak brightness of the ring (additive white)
_GLOW_INTENSITY = 0.25   # peak brightness of the glow (dimmer than ring)
_BASE_TINT = 0.10        # subtle base-color tint on the highlight

# Backward-compat aliases used by prepare_animated_cache() (removed in T065-T068)
_RING_WIDTH = _RING_SIGMA
_RING_WHITE = 1.0 - _BASE_TINT
_RING_COLOR = _BASE_TINT

# xatlas time estimation — empirical power-law model: t = k * n_faces^exp
# Fitted from 5 real CARTO meshes (v7.1 + v7.2, subdiv 0-2, 18K-506K faces).
# Topology affects timing heavily, so ±50% error is expected.
_XATLAS_K = 6.50e-08
_XATLAS_EXP = 1.79


_fmt_duration = fmt_duration  # backward-compat alias


def compute_wavefront_colors(
    lat_norm: np.ndarray,
    base_colors: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """Compute per-frame per-vertex excitation wavefront colors.

    The wavefront combines a bright leading-edge ring (narrow Gaussian)
    with a broad diffuse trailing glow (exponential decay), matching
    real CARTO 3 excitation behavior (~25-30% surface coverage per frame).

    Parameters
    ----------
    lat_norm : ndarray, shape (N,)
        Normalized LAT values in [0, 1].  NaN for unmapped vertices.
    base_colors : ndarray, shape (N, 4)
        Static heatmap RGBA colors (float32, range [0, 1]).
    n_frames : int
        Number of animation frames.

    Returns
    -------
    ndarray, shape (n_frames, N, 4)
        Per-frame RGBA colors (float32, range [0, 1]).  Unmapped
        vertices get the base color unchanged across all frames.
    """
    n_verts = len(lat_norm)
    valid = ~np.isnan(lat_norm)

    # Time steps wrap around: last frame transitions smoothly back to first.
    # We use n_frames+1 points to avoid duplicating the last=first boundary,
    # then take the first n_frames values.
    t_values = np.linspace(0, 1, n_frames, endpoint=False).reshape(-1, 1)  # (F, 1)
    lat_v = lat_norm[np.newaxis, :]  # (1, N)

    # --- Ring: narrow Gaussian at activation frontier ---
    sigma_sq_2 = 2.0 * _RING_SIGMA ** 2
    # Wrap-aware distance: accounts for seamless loop
    delta = lat_v - t_values  # (F, N)
    # Wrap to [-0.5, 0.5] for seamless looping
    delta = delta - np.round(delta)
    ring = np.exp(-(delta ** 2) / sigma_sq_2)  # (F, N)

    # --- Glow: broad exponential decay behind the ring ---
    # "Behind" = tissue that activated before current time (delta > 0 means
    # LAT < current_time after wrapping, i.e. already activated).
    # Use the wrapped delta: positive delta means vertex activated before t.
    glow_raw = np.where(
        delta > 0,
        np.exp(-_GLOW_DECAY * delta / _GLOW_EXTENT),
        0.0,
    )
    # Taper the glow so it fades to zero beyond _GLOW_EXTENT
    glow = np.where(delta <= _GLOW_EXTENT, glow_raw, 0.0)

    # --- Combine ring + glow ---
    intensity = _RING_INTENSITY * ring + _GLOW_INTENSITY * glow  # (F, N)
    intensity[:, ~valid] = 0.0
    intensity = np.clip(intensity, 0.0, 1.0)

    # Highlight color: white-dominant with subtle base-color tint
    base_rgb = base_colors[:, :3]  # (N, 3)
    highlight_rgb = np.full_like(base_rgb, 1.0) * (1.0 - _BASE_TINT) + base_rgb * _BASE_TINT

    # Per-frame colors: blend base color with highlight based on intensity
    # result = base_color * (1 - intensity) + highlight * intensity
    base_expanded = base_colors[np.newaxis, :, :]  # (1, N, 4)
    highlight_expanded = highlight_rgb[np.newaxis, :, :]  # (1, N, 3)
    intensity_3 = intensity[:, :, np.newaxis]  # (F, N, 1)

    frame_colors = np.empty((n_frames, n_verts, 4), dtype=np.float32)
    frame_colors[:, :, :3] = (
        base_expanded[:, :, :3] * (1.0 - intensity_3) + highlight_expanded * intensity_3
    )
    frame_colors[:, :, 3] = base_expanded[:, :, 3]  # preserve alpha

    return frame_colors


def _estimate_xatlas_time(n_faces: int) -> str:
    """Human-readable time estimate for xatlas UV unwrap."""
    secs = _XATLAS_K * (n_faces ** _XATLAS_EXP)
    if secs < 30:
        return "<1 min"
    if secs < 90:
        return "~1 min"
    mins = secs / 60
    if mins < 60:
        return f"~{mins:.0f} min"
    hours = mins / 60
    return f"~{hours:.1f} h"


@dataclass
class AnimatedBakeCache:
    """Pre-computed intermediates shared between animated CARTO variants.

    xatlas UV fields (unwelded_verts, unwelded_normals, unwelded_faces,
    shared_uvs, vmapping, base_texture) are used by the static GLB builders.

    frame_colors stores per-frame per-vertex RGBA for the wavefront animation
    (shape: n_frames × n_verts × 4, float32).  Replaces the old emissive
    texture approach (30 PNG images).
    """

    mesh_data: MeshData
    lat_values: np.ndarray
    unwelded_verts: np.ndarray
    unwelded_normals: np.ndarray | None
    unwelded_faces: np.ndarray
    shared_uvs: np.ndarray
    centroid: list[float]
    base_texture: bytes
    frame_colors: np.ndarray          # (n_frames, n_verts, 4) float32
    n_frames: int
    vmapping: np.ndarray | None = None
    step_times: dict[str, float] | None = None


def prepare_animated_cache(
    mesh_data: MeshData,
    lat_values: np.ndarray,
    n_frames: int = 30,
    target_faces: int = 20000,
    max_size_mb: float = 50.0,
    progress: Callable[[str, int, int], None] | None = None,
) -> AnimatedBakeCache | None:
    """Compute expensive intermediates for animated CARTO GLB variants.

    Uses the full mesh (no decimation).  Bakes one shared base color
    texture and N small emissive ring textures.
    Returns None if LAT values are invalid (all-NaN or zero range).

    The *progress* callback receives ``(description, current, total)``.
    For blocking steps (xatlas, rasterization), ``current=0, total=0`` —
    the caller should treat this as a status message.
    For frame-based steps, ``current`` and ``total`` are meaningful.
    """
    def _status(desc: str) -> None:
        """Report a blocking step (no progress counter)."""
        if progress:
            progress(desc, 0, 0)

    def _frame(desc: str, current: int, total: int) -> None:
        """Report frame progress (counter is meaningful)."""
        if progress:
            progress(desc, current, total)

    valid_lat = ~np.isnan(lat_values)
    if not np.any(valid_lat):
        return None

    lat_min = float(np.nanmin(lat_values))
    lat_max = float(np.nanmax(lat_values))
    lat_range = lat_max - lat_min
    if lat_range < 1e-6:
        return None

    # Normalize LAT to [0, 1]
    lat_norm = (lat_values - lat_min) / lat_range
    lat_norm[~valid_lat] = np.nan

    # Base colors from static coloring (mesh_data.vertex_colors)
    n_verts = len(mesh_data.vertices)
    if mesh_data.vertex_colors is not None:
        base_colors = mesh_data.vertex_colors.astype(np.float32)
    else:
        base_colors = np.full((n_verts, 4), [0.7, 0.7, 0.7, 1.0], dtype=np.float32)

    # Compute per-frame wavefront colors (glow + ring model, ~25-30% surface coverage)
    _status("Computing wavefront colors...")
    frame_colors = compute_wavefront_colors(lat_norm, base_colors, n_frames)

    # UV unwrap ONCE with xatlas — still needed for base color texture
    # used by the static heatmap GLB builders.
    from med2glb.glb.vertex_color_bake import (
        xatlas_unwrap_with_timer,
        precompute_rasterization_map,
        apply_rasterization_map,
    )

    n_faces = len(mesh_data.faces)
    base_tex_size = compute_texture_size(n_faces)

    _step_times: dict[str, float] = {}

    eta_str = _estimate_xatlas_time(n_faces)
    eta_secs = _XATLAS_K * (n_faces ** _XATLAS_EXP)

    def _xatlas_tick(elapsed: float, eta: float) -> None:
        pct = min(elapsed / eta * 100, 99) if eta > 0 else 0
        _status(
            f"UV unwrapping {n_faces:,} faces with xatlas — "
            f"{_fmt_duration(elapsed)} / ~{eta_str} ({pct:.0f}%)"
        )

    _status(f"UV unwrapping {n_faces:,} faces with xatlas (estimated {eta_str})...")
    t0 = time.monotonic()
    vmapping, new_faces, shared_uvs = xatlas_unwrap_with_timer(
        mesh_data.vertices, mesh_data.faces, mesh_data.normals,
        eta_seconds=eta_secs,
        on_tick=_xatlas_tick,
    )
    _step_times["xatlas"] = time.monotonic() - t0
    _status(f"UV unwrap done ({_fmt_duration(_step_times['xatlas'])}). Rasterizing {base_tex_size}x{base_tex_size} texture...")

    unwelded_verts, centroid = _center_vertices(
        mesh_data.vertices[vmapping].astype(np.float32),
    )
    unwelded_normals = None
    if mesh_data.normals is not None:
        unwelded_normals = mesh_data.normals[vmapping].astype(np.float32)

    # Precompute rasterization maps — full-res for base, smaller for emissive
    t0 = time.monotonic()
    base_raster_map = precompute_rasterization_map(new_faces, shared_uvs, base_tex_size)
    _step_times["Rasterize"] = time.monotonic() - t0
    _status(f"Rasterization done ({_fmt_duration(_step_times['Rasterize'])}). Baking textures...")

    # Bake ONE base color texture (for static GLB reuse)
    t0 = time.monotonic()
    base_colors_remapped = base_colors[vmapping]
    base_texture = apply_rasterization_map(
        base_raster_map, base_colors_remapped,
    )
    _step_times["Textures"] = time.monotonic() - t0
    del base_raster_map

    return AnimatedBakeCache(
        mesh_data=mesh_data,
        lat_values=lat_values,
        unwelded_verts=unwelded_verts,
        unwelded_normals=unwelded_normals,
        unwelded_faces=new_faces,
        shared_uvs=shared_uvs,
        centroid=centroid,
        base_texture=base_texture,
        frame_colors=frame_colors,
        n_frames=n_frames,
        vmapping=vmapping,
        step_times=_step_times,
    )


def build_carto_static_glb(
    cache: AnimatedBakeCache,
    output_path: Path,
    legend_info: dict | None = None,
    lesion_points: list | None = None,
) -> None:
    """Build a static GLB reusing geometry and base texture from the animated cache.

    Avoids a redundant xatlas unwrap + rasterization pass by reusing the
    pre-computed UV layout and base color texture from the animated cache.
    """
    mesh_data = cache.mesh_data
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        images=[],
        textures=[],
        samplers=[],
    )
    if mesh_data.material.unlit:
        gltf.extensionsUsed = ["KHR_materials_unlit"]
    binary_data = bytearray()

    # Sampler
    gltf.samplers.append(pygltflib.Sampler(
        magFilter=pygltflib.LINEAR,
        minFilter=pygltflib.LINEAR,
        wrapS=pygltflib.CLAMP_TO_EDGE,
        wrapT=pygltflib.CLAMP_TO_EDGE,
    ))

    # Base color texture
    img_offset = len(binary_data)
    binary_data.extend(cache.base_texture)
    _pad_to_4(binary_data)

    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=img_offset, byteLength=len(cache.base_texture),
    ))
    gltf.images.append(pygltflib.Image(bufferView=0, mimeType="image/png"))
    gltf.textures.append(pygltflib.Texture(sampler=0, source=0))

    # Material
    mat_kwargs: dict = dict(
        name="carto_static",
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorTexture=pygltflib.TextureInfo(index=0),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.45,
        ),
        doubleSided=True,
    )
    if mesh_data.material.unlit:
        mat_kwargs["extensions"] = {"KHR_materials_unlit": {}}
    gltf.materials.append(pygltflib.Material(**mat_kwargs))

    # Geometry
    pos_acc = write_accessor(
        gltf, binary_data, cache.unwelded_verts, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC3, with_minmax=True,
    )
    norm_acc = None
    if cache.unwelded_normals is not None:
        norm_acc = write_accessor(
            gltf, binary_data, cache.unwelded_normals, pygltflib.ARRAY_BUFFER,
            pygltflib.FLOAT, pygltflib.VEC3,
        )
    uv_acc = write_accessor(
        gltf, binary_data, cache.shared_uvs, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC2,
    )
    idx_acc = write_accessor(
        gltf, binary_data, cache.unwelded_faces.ravel(), pygltflib.ELEMENT_ARRAY_BUFFER,
        pygltflib.UNSIGNED_INT, pygltflib.SCALAR, with_minmax=True,
    )

    attrs = pygltflib.Attributes(POSITION=pos_acc)
    if norm_acc is not None:
        attrs.NORMAL = norm_acc
    attrs.TEXCOORD_0 = uv_acc

    gltf.meshes.append(pygltflib.Mesh(
        name="carto_static",
        primitives=[pygltflib.Primitive(
            attributes=attrs, indices=idx_acc, material=0,
        )],
    ))

    child_nodes = [0]
    gltf.nodes.append(pygltflib.Node(name="mesh", mesh=0))

    if legend_info:
        from med2glb.glb.legend_builder import add_legend_nodes
        centered_verts = (
            mesh_data.vertices - np.array(cache.centroid, dtype=np.float32)
        ).astype(np.float32)
        legend_nodes = add_legend_nodes(
            gltf, binary_data, centered_verts,
            coloring=legend_info["coloring"],
            clamp_range=tuple(legend_info["clamp_range"]),
            centroid=[0.0, 0.0, 0.0],
            metadata=legend_info.get("metadata"),
        )
        child_nodes.extend(legend_nodes)

    if lesion_points:
        from med2glb.glb.lesion_builder import add_lesion_nodes
        lesion_nodes = add_lesion_nodes(
            gltf, binary_data, lesion_points, list(cache.centroid),
        )
        child_nodes.extend(lesion_nodes)

    # Root node: mm -> m + 10x AR scale
    root_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(
        name="root", children=child_nodes, scale=[0.01, 0.01, 0.01],
    ))
    gltf.scenes[0].nodes = [root_idx]

    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))

    logger.debug(
        f"CARTO static GLB (from cache): {len(mesh_data.vertices)} verts, "
        f"{output_path.stat().st_size / 1024:.0f} KB"
    )


def build_carto_recolored_static_glb(
    cache: AnimatedBakeCache,
    mesh_data: MeshData,
    output_path: Path,
    legend_info: dict | None = None,
    lesion_points: list | None = None,
) -> None:
    """Build a static GLB reusing xatlas geometry from the cache with new vertex colors.

    For non-LAT colorings (bipolar/unipolar) that share the same mesh geometry
    but have different vertex colors.  Reuses the cached UV unwrap and
    rasterization map, only re-baking the base color texture (~1s instead of
    re-running xatlas which takes minutes).
    """
    from med2glb.glb.vertex_color_bake import (
        precompute_rasterization_map,
        apply_rasterization_map,
    )

    vmapping = cache.vmapping
    if vmapping is None:
        raise ValueError("Cache has no vmapping — cannot recolor")

    n_faces = len(cache.unwelded_faces)
    tex_size = compute_texture_size(n_faces)

    # Re-bake base color texture with the new vertex colors
    if mesh_data.vertex_colors is not None:
        new_colors = mesh_data.vertex_colors.astype(np.float32)[vmapping]
    else:
        new_colors = np.full((len(cache.unwelded_verts), 4), [0.7, 0.7, 0.7, 1.0], dtype=np.float32)

    raster_map = precompute_rasterization_map(cache.unwelded_faces, cache.shared_uvs, tex_size)
    new_texture = apply_rasterization_map(raster_map, new_colors)
    del raster_map

    # Build GLB using cached geometry + new texture
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        images=[],
        textures=[],
        samplers=[],
    )
    if mesh_data.material.unlit:
        gltf.extensionsUsed = ["KHR_materials_unlit"]
    binary_data = bytearray()

    gltf.samplers.append(pygltflib.Sampler(
        magFilter=pygltflib.LINEAR,
        minFilter=pygltflib.LINEAR,
        wrapS=pygltflib.CLAMP_TO_EDGE,
        wrapT=pygltflib.CLAMP_TO_EDGE,
    ))

    img_offset = len(binary_data)
    binary_data.extend(new_texture)
    _pad_to_4(binary_data)

    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=img_offset, byteLength=len(new_texture),
    ))
    gltf.images.append(pygltflib.Image(bufferView=0, mimeType="image/png"))
    gltf.textures.append(pygltflib.Texture(sampler=0, source=0))

    mat_kwargs: dict = dict(
        name="carto_recolored",
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorTexture=pygltflib.TextureInfo(index=0),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.45,
        ),
        doubleSided=True,
    )
    if mesh_data.material.unlit:
        mat_kwargs["extensions"] = {"KHR_materials_unlit": {}}
    gltf.materials.append(pygltflib.Material(**mat_kwargs))

    pos_acc = write_accessor(
        gltf, binary_data, cache.unwelded_verts, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC3, with_minmax=True,
    )
    norm_acc = None
    if cache.unwelded_normals is not None:
        norm_acc = write_accessor(
            gltf, binary_data, cache.unwelded_normals, pygltflib.ARRAY_BUFFER,
            pygltflib.FLOAT, pygltflib.VEC3,
        )
    uv_acc = write_accessor(
        gltf, binary_data, cache.shared_uvs, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC2,
    )
    idx_acc = write_accessor(
        gltf, binary_data, cache.unwelded_faces.ravel(), pygltflib.ELEMENT_ARRAY_BUFFER,
        pygltflib.UNSIGNED_INT, pygltflib.SCALAR, with_minmax=True,
    )

    attrs = pygltflib.Attributes(POSITION=pos_acc)
    if norm_acc is not None:
        attrs.NORMAL = norm_acc
    attrs.TEXCOORD_0 = uv_acc

    gltf.meshes.append(pygltflib.Mesh(
        name="carto_recolored",
        primitives=[pygltflib.Primitive(
            attributes=attrs, indices=idx_acc, material=0,
        )],
    ))

    child_nodes = [0]
    gltf.nodes.append(pygltflib.Node(name="mesh", mesh=0))

    if legend_info:
        from med2glb.glb.legend_builder import add_legend_nodes
        centered_verts = (
            mesh_data.vertices - np.array(cache.centroid, dtype=np.float32)
        ).astype(np.float32)
        legend_nodes = add_legend_nodes(
            gltf, binary_data, centered_verts,
            coloring=legend_info["coloring"],
            clamp_range=tuple(legend_info["clamp_range"]),
            centroid=[0.0, 0.0, 0.0],
            metadata=legend_info.get("metadata"),
        )
        child_nodes.extend(legend_nodes)

    if lesion_points:
        from med2glb.glb.lesion_builder import add_lesion_nodes
        lesion_nodes = add_lesion_nodes(
            gltf, binary_data, lesion_points, list(cache.centroid),
        )
        child_nodes.extend(lesion_nodes)

    root_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(
        name="root", children=child_nodes, scale=[0.01, 0.01, 0.01],
    ))
    gltf.scenes[0].nodes = [root_idx]

    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))

    logger.debug(
        f"CARTO recolored static GLB: {len(mesh_data.vertices)} verts, "
        f"{output_path.stat().st_size / 1024:.0f} KB"
    )


def build_carto_animated_glb(
    mesh_data: MeshData,
    lat_values: np.ndarray,
    output_path: Path,
    n_frames: int = 30,
    loop_duration_s: float = 4.5,
    target_faces: int = 20000,
    max_size_mb: float = 50.0,
    vectors: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
    legend_info: dict | None = None,
    cache: AnimatedBakeCache | None = None,
    lesion_points: list | None = None,
) -> bool:
    """Build animated GLB with CARTO excitation wavefront using morph targets.

    Single mesh + COLOR_0 vertex color morph targets (one per frame).
    Base COLOR_0 = static LAT heatmap colors; each morph target stores the
    per-frame wavefront delta so the runtime interpolates smoothly.

    Loop duration default is 4.5s to match real CARTO 3 cycle timing.

    Returns:
        True if the GLB was written, False if skipped.
    """
    from med2glb.glb.animation import _add_animated_mesh_to_gltf, _add_morph_animation

    def _report(desc: str, current: int = 0, total: int = 0) -> None:
        if progress:
            progress(desc, current, total)

    if cache is not None:
        mesh_data = cache.mesh_data
        lat_values = cache.lat_values
        n_frames = cache.n_frames
        frame_colors = cache.frame_colors
        centroid = cache.centroid
    else:
        # Compute everything from scratch (non-cached path)
        tmp_cache = prepare_animated_cache(
            mesh_data, lat_values, n_frames=n_frames,
            target_faces=target_faces, max_size_mb=max_size_mb,
            progress=progress,
        )
        if tmp_cache is None:
            from med2glb.glb.builder import build_glb
            build_glb([mesh_data], output_path, source_units="mm")
            return True

        mesh_data = tmp_cache.mesh_data
        lat_values = tmp_cache.lat_values
        n_frames = tmp_cache.n_frames
        frame_colors = tmp_cache.frame_colors
        centroid = tmp_cache.centroid

    # Compute animated arrow dashes if vectors enabled
    arrow_frame_dashes = None
    arrow_speed_factors = None
    if vectors:
        from med2glb.mesh.lat_vectors import (
            trace_all_streamlines, compute_animated_dashes,
            compute_dash_speed_factors,
            assess_streamline_quality,
        )
        _report("Tracing streamlines...")
        streamlines, face_grads, face_centers = trace_all_streamlines(
            mesh_data.vertices, mesh_data.faces, lat_values,
            mesh_data.normals, target_count=300,
        )
        if not streamlines:
            logger.debug("Vectors skipped: no streamlines traced")
            return False
        bbox_diag = float(np.linalg.norm(
            mesh_data.vertices.max(0) - mesh_data.vertices.min(0)))
        good, reason = assess_streamline_quality(streamlines, bbox_diag)
        if not good:
            logger.debug("Vectors skipped: %s", reason)
            return False
        if streamlines:
            _report(f"Generating dash animation for {len(streamlines)} streamlines...")
            arrow_frame_dashes = compute_animated_dashes(
                streamlines, n_frames=n_frames,
            )
            arrow_speed_factors = compute_dash_speed_factors(
                arrow_frame_dashes, face_grads, face_centers,
            )

    # Build morph target color deltas from base colors.
    # frame_colors[f] = absolute RGBA per vertex for frame f.
    # Morph target delta = frame_colors[f] - base_colors so the glTF
    # runtime computes: result = base + weight * delta = frame_colors[f].
    n_verts = len(mesh_data.vertices)
    if mesh_data.vertex_colors is not None:
        base_colors = mesh_data.vertex_colors.astype(np.float32)
    else:
        base_colors = np.full((n_verts, 4), [0.7, 0.7, 0.7, 1.0], dtype=np.float32)

    color_deltas = [
        (frame_colors[f] - base_colors).astype(np.float32)
        for f in range(n_frames)
    ]

    # Compute lat_norm for TEXCOORD_1 so ColorMorphApplicator in Unity can
    # recompute wavefront colors without needing morph accessor data.
    valid_lat = ~np.isnan(lat_values)
    lat_min = float(np.nanmin(lat_values)) if np.any(valid_lat) else 0.0
    lat_max = float(np.nanmax(lat_values)) if np.any(valid_lat) else 1.0
    lat_range = lat_max - lat_min if (lat_max - lat_min) > 1e-6 else 1.0
    lat_norm_arr = np.where(valid_lat, (lat_values - lat_min) / lat_range, 0.0).astype(np.float32)
    # Pack as float2: x=lat_norm, y=0
    lat_uv1 = np.column_stack([lat_norm_arr, np.zeros(n_verts, dtype=np.float32)])

    # Create centered mesh data for the single animated mesh
    centered_verts, cent_offset = _center_vertices(mesh_data.vertices.astype(np.float32))
    anim_mesh = MeshData(
        structure_name=mesh_data.structure_name or "carto_animated",
        vertices=centered_verts,
        faces=mesh_data.faces,
        normals=mesh_data.normals.astype(np.float32) if mesh_data.normals is not None else None,
        material=mesh_data.material,
    )

    # glTF document
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        animations=[],
    )
    binary_data = bytearray()

    # No positional displacement morph targets — color-only animation
    pos_mts = [np.zeros((n_verts, 3), dtype=np.float32)] * n_frames

    _report("Assembling morph target GLB...", 0, n_frames)
    mesh_node_idx, _ = _add_animated_mesh_to_gltf(
        gltf, anim_mesh, pos_mts, binary_data, 0,
        color_morph_targets=color_deltas,
        base_vertex_colors=base_colors,
        uv1_data=lat_uv1,
    )

    # Set material name and PBR properties. "carto_wavefront" is the marker
    # used by Loader.cs (IsCartoAnimatedMesh) to detect and attach
    # ColorMorphApplicator on HoloLens. Using PBR (roughness=1.0) instead of
    # KHR_materials_unlit means external PC/web viewers render with correct
    # scene lighting rather than at full unlit brightness.
    anim_mat = gltf.materials[gltf.meshes[-1].primitives[0].material]
    anim_mat.name = "carto_wavefront"
    anim_mat.pbrMetallicRoughness = pygltflib.PbrMetallicRoughness(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )

    # Morph weight animation: cycle through frames seamlessly
    dt = loop_duration_s / n_frames
    frame_times = [i * dt for i in range(n_frames)]
    _add_morph_animation(gltf, frame_times, [pos_mts], binary_data, 1)

    child_node_indices = [mesh_node_idx]

    # Arrow nodes (vectors) if enabled
    if arrow_frame_dashes is not None:
        _report("Building arrow geometry...")
        from med2glb.glb.arrow_builder import build_animated_arrow_nodes
        from med2glb.glb.animation import _add_scale_toggle_channels
        arrow_node_indices = build_animated_arrow_nodes(
            arrow_frame_dashes,
            mesh_data.vertices, mesh_data.normals,
            gltf, binary_data, n_frames,
            speed_factors=arrow_speed_factors,
            unlit=mesh_data.material.unlit,
            centroid_offset=cent_offset,
        )
        child_node_indices.extend(arrow_node_indices)
        # Add scale-toggle animation channels (synced to morph target timeline)
        _add_scale_toggle_channels(
            gltf, gltf.animations[0], arrow_node_indices, frame_times, binary_data,
        )

    if legend_info:
        from med2glb.glb.legend_builder import add_legend_nodes
        centered_for_legend = (
            mesh_data.vertices - np.array(cent_offset, dtype=np.float32)
        ).astype(np.float32)
        legend_nodes = add_legend_nodes(
            gltf, binary_data, centered_for_legend,
            coloring=legend_info["coloring"],
            clamp_range=tuple(legend_info["clamp_range"]),
            centroid=[0.0, 0.0, 0.0],
            metadata=legend_info.get("metadata"),
        )
        child_node_indices.extend(legend_nodes)

    if lesion_points:
        from med2glb.glb.lesion_builder import add_lesion_nodes
        lesion_nodes = add_lesion_nodes(
            gltf, binary_data, lesion_points, list(cent_offset),
        )
        child_node_indices.extend(lesion_nodes)

    # Root node: mm → m + 10x AR display scale
    root_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(
        name="root",
        children=child_node_indices,
        scale=[0.01, 0.01, 0.01],
    ))
    gltf.scenes[0].nodes = [root_idx]

    # Finalize
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))

    logger.debug(
        f"CARTO animated GLB: {n_frames} frames (morph targets), "
        f"{n_verts} verts, "
        f"{output_path.stat().st_size / 1024:.0f} KB"
    )
    return True
