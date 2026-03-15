"""CARTO animated GLB: excitation highlight ring using emissive overlay.

Uses the same full-quality mesh as the static GLB.  The static colormap
is baked into a shared baseColorTexture, and the sweeping highlight ring
is rendered via per-frame emissiveTextures (mostly black PNGs, compressed to KTX2).
Animation switches the visible frame via scale [1,1,1] / [0,0,0] —
universally supported in glTF viewers including HoloLens 2 (MRTK/glTFast).
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

# Highlight ring parameters (tuned to match CARTO3 excitation ring)
_RING_WIDTH = 0.10  # sigma of Gaussian — wide, soft gradient like the real CARTO system
_RING_WHITE_BLEND = 0.55  # how much to brighten LAT color toward white (0=pure LAT, 1=pure white)

# xatlas time estimation — empirical power-law model: t = k * n_faces^exp
# Fitted from 5 real CARTO meshes (v7.1 + v7.2, subdiv 0-2, 18K-506K faces).
# Topology affects timing heavily, so ±50% error is expected.
_XATLAS_K = 6.50e-08
_XATLAS_EXP = 1.79


_fmt_duration = fmt_duration  # backward-compat alias


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
    """Pre-computed intermediates shared between animated CARTO variants."""

    mesh_data: MeshData
    lat_values: np.ndarray
    unwelded_verts: np.ndarray
    unwelded_normals: np.ndarray | None
    unwelded_faces: np.ndarray
    shared_uvs: np.ndarray
    centroid: list[float]
    base_texture: bytes
    emissive_textures: list[bytes]
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

    # Generate emissive ring colors for each frame.
    # The ring adapts to the local LAT color: each vertex's base color is
    # brightened toward white, so the ring glows red-white in red regions,
    # blue-white in blue regions, etc. — matching CARTO3 reference behavior.
    t_values = np.linspace(0, 1, n_frames).reshape(-1, 1)  # (F, 1)
    sigma_sq_2 = 2 * _RING_WIDTH ** 2
    ring_all = np.exp(-((lat_norm[np.newaxis, :] - t_values) ** 2) / sigma_sq_2)  # (F, N)
    ring_all[:, ~valid_lat] = 0.0
    # Per-vertex highlight color: lerp base LAT color toward white
    base_rgb = base_colors[:, :3]  # (N, 3)
    highlight_rgb = base_rgb + _RING_WHITE_BLEND * (1.0 - base_rgb)  # (N, 3)
    # Emissive = ring_intensity * highlight_color, as RGBA with A=1
    emissive_all = ring_all[:, :, np.newaxis] * highlight_rgb[np.newaxis, :, :]  # (F, N, 3)
    emissive_rgba = np.zeros((n_frames, n_verts, 4), dtype=np.float32)
    emissive_rgba[:, :, :3] = emissive_all
    emissive_rgba[:, :, 3] = 1.0
    del ring_all, emissive_all, highlight_rgb

    # UV unwrap ONCE with xatlas
    from med2glb.glb.vertex_color_bake import (
        xatlas_unwrap_with_timer,
        precompute_rasterization_map,
        apply_rasterization_map,
    )

    n_faces = len(mesh_data.faces)
    base_tex_size = compute_texture_size(n_faces)
    # Emissive ring textures are mostly black (thin gaussian band) —
    # 1024 is plenty for the ring highlight even on large meshes.
    emissive_tex_size = min(base_tex_size, 1024)

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

    # Bake ONE base color texture (same quality as static GLB)
    t0 = time.monotonic()
    base_colors_remapped = base_colors[vmapping]
    base_texture = apply_rasterization_map(
        base_raster_map, base_colors_remapped,
    )

    # Bake per-frame emissive ring textures (mostly black, compressed by KTX2)
    if emissive_tex_size == base_tex_size:
        emissive_raster_map = base_raster_map
    else:
        del base_raster_map  # free memory before building smaller map
        emissive_raster_map = precompute_rasterization_map(
            new_faces, shared_uvs, emissive_tex_size,
        )

    emissive_textures: list[bytes] = []
    for fi in range(n_frames):
        _frame(f"Baking textures ({fi + 1}/{n_frames})...", fi, n_frames)
        ring_colors = emissive_rgba[fi, vmapping]
        img_bytes = apply_rasterization_map(
            emissive_raster_map, ring_colors,
        )
        emissive_textures.append(img_bytes)
    _step_times["Textures"] = time.monotonic() - t0
    del emissive_raster_map

    del emissive_rgba

    return AnimatedBakeCache(
        mesh_data=mesh_data,
        lat_values=lat_values,
        unwelded_verts=unwelded_verts,
        unwelded_normals=unwelded_normals,
        unwelded_faces=new_faces,
        shared_uvs=shared_uvs,
        centroid=centroid,
        base_texture=base_texture,
        emissive_textures=emissive_textures,
        n_frames=n_frames,
        vmapping=vmapping,
        step_times=_step_times,
    )


def build_carto_static_glb(
    cache: AnimatedBakeCache,
    output_path: Path,
    legend_info: dict | None = None,
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
    loop_duration_s: float = 2.0,
    target_faces: int = 20000,
    max_size_mb: float = 50.0,
    vectors: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
    legend_info: dict | None = None,
    cache: AnimatedBakeCache | None = None,
) -> bool:
    """Build animated GLB with CARTO-style highlight ring over static colormap.

    Uses the full mesh (no decimation).  The static vertex colors are baked
    into a shared baseColorTexture, and the highlight ring is rendered via
    per-frame emissiveTextures.

    Returns:
        True if the GLB was written, False if skipped.
    """
    def _report(desc: str, current: int = 0, total: int = 0) -> None:
        if progress:
            progress(desc, current, total)

    if cache is not None:
        mesh_data = cache.mesh_data
        lat_values = cache.lat_values
        n_frames = cache.n_frames
        unwelded_verts = cache.unwelded_verts
        unwelded_normals = cache.unwelded_normals
        unwelded_faces = cache.unwelded_faces
        shared_uvs = cache.shared_uvs
        centroid = cache.centroid
        base_texture = cache.base_texture
        emissive_textures = cache.emissive_textures
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
        unwelded_verts = tmp_cache.unwelded_verts
        unwelded_normals = tmp_cache.unwelded_normals
        unwelded_faces = tmp_cache.unwelded_faces
        shared_uvs = tmp_cache.shared_uvs
        centroid = tmp_cache.centroid
        base_texture = tmp_cache.base_texture
        emissive_textures = tmp_cache.emissive_textures

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

    # Build glTF
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
        images=[],
        textures=[],
        samplers=[],
    )
    if mesh_data.material.unlit:
        gltf.extensionsUsed = ["KHR_materials_unlit"]
    binary_data = bytearray()

    # Shared sampler
    gltf.samplers.append(pygltflib.Sampler(
        magFilter=pygltflib.LINEAR,
        minFilter=pygltflib.LINEAR,
        wrapS=pygltflib.CLAMP_TO_EDGE,
        wrapT=pygltflib.CLAMP_TO_EDGE,
    ))

    # Embed base color texture (shared across all frames)
    base_offset = len(binary_data)
    binary_data.extend(base_texture)
    _pad_to_4(binary_data)

    base_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=base_offset, byteLength=len(base_texture),
    ))
    base_img_idx = len(gltf.images)
    gltf.images.append(pygltflib.Image(
        bufferView=base_bv_idx, mimeType="image/png",
    ))
    base_tex_idx = len(gltf.textures)
    gltf.textures.append(pygltflib.Texture(sampler=0, source=base_img_idx))

    # Embed per-frame emissive textures
    emissive_tex_indices: list[int] = []
    for fi in range(n_frames):
        img_offset = len(binary_data)
        binary_data.extend(emissive_textures[fi])
        _pad_to_4(binary_data)

        img_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=img_offset, byteLength=len(emissive_textures[fi]),
        ))
        img_idx = len(gltf.images)
        gltf.images.append(pygltflib.Image(
            bufferView=img_bv_idx, mimeType="image/png",
        ))
        tex_idx = len(gltf.textures)
        gltf.textures.append(pygltflib.Texture(sampler=0, source=img_idx))
        emissive_tex_indices.append(tex_idx)

    del emissive_textures  # free memory

    # Shared geometry: positions, normals, UVs, indices (written once)
    pos_acc = write_accessor(
        gltf, binary_data, unwelded_verts, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC3, with_minmax=True,
    )

    norm_acc = None
    if unwelded_normals is not None:
        norm_acc = write_accessor(
            gltf, binary_data, unwelded_normals, pygltflib.ARRAY_BUFFER,
            pygltflib.FLOAT, pygltflib.VEC3,
        )

    uv_acc = write_accessor(
        gltf, binary_data, shared_uvs, pygltflib.ARRAY_BUFFER,
        pygltflib.FLOAT, pygltflib.VEC2,
    )

    idx_acc = write_accessor(
        gltf, binary_data, unwelded_faces.ravel(), pygltflib.ELEMENT_ARRAY_BUFFER,
        pygltflib.UNSIGNED_INT, pygltflib.SCALAR, with_minmax=True,
    )

    # Per-frame: material (shared base + per-frame emissive) + mesh + node
    _report("Assembling GLB...", n_frames, n_frames)
    for fi in range(n_frames):
        mat_idx = len(gltf.materials)
        mat_kwargs: dict = dict(
            name=f"wavefront_{fi}",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=base_tex_idx),
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.45,
            ),
            emissiveTexture=pygltflib.TextureInfo(index=emissive_tex_indices[fi]),
            emissiveFactor=[1.0, 1.0, 1.0],
            doubleSided=True,
        )
        if mesh_data.material.unlit:
            mat_kwargs["extensions"] = {"KHR_materials_unlit": {}}
        gltf.materials.append(pygltflib.Material(**mat_kwargs))

        # Primitive with shared geometry
        attrs = pygltflib.Attributes(POSITION=pos_acc)
        if norm_acc is not None:
            attrs.NORMAL = norm_acc
        attrs.TEXCOORD_0 = uv_acc

        mesh_idx = len(gltf.meshes)
        gltf.meshes.append(pygltflib.Mesh(
            name=f"wavefront_{fi}",
            primitives=[pygltflib.Primitive(
                attributes=attrs,
                indices=idx_acc,
                material=mat_idx,
            )],
        ))

        # Node: first frame visible, rest hidden
        # Tiny z-offset per frame prevents z-fighting on HoloLens when
        # zero-scaled meshes aren't fully culled by the renderer.
        scale = [1.0, 1.0, 1.0] if fi == 0 else [0.0, 0.0, 0.0]
        z_offset = fi * 0.0001  # 0.1mm per frame — invisible but separates geometry
        gltf.nodes.append(pygltflib.Node(
            name=f"wavefront_{fi}", mesh=mesh_idx, scale=scale,
            translation=[0.0, 0.0, z_offset],
        ))

    # Collect all child node indices for the root node
    child_node_indices = list(range(n_frames))

    # Add arrow nodes if vectors enabled
    arrow_node_indices: list[int] = []
    if arrow_frame_dashes is not None:
        _report("Building arrow geometry...")
        from med2glb.glb.arrow_builder import build_animated_arrow_nodes
        arrow_node_indices = build_animated_arrow_nodes(
            arrow_frame_dashes,
            mesh_data.vertices, mesh_data.normals,
            gltf, binary_data, n_frames,
            speed_factors=arrow_speed_factors,
            unlit=mesh_data.material.unlit,
            centroid_offset=centroid,
        )
        child_node_indices.extend(arrow_node_indices)

    if legend_info:
        from med2glb.glb.legend_builder import add_legend_nodes
        centered_verts = (
            mesh_data.vertices - np.array(centroid, dtype=np.float32)
        ).astype(np.float32)
        legend_nodes = add_legend_nodes(
            gltf, binary_data, centered_verts,
            coloring=legend_info["coloring"],
            clamp_range=tuple(legend_info["clamp_range"]),
            centroid=[0.0, 0.0, 0.0],
            metadata=legend_info.get("metadata"),
        )
        child_node_indices.extend(legend_nodes)

    # Animation: switch visible frame via scale keyframes
    dt = loop_duration_s / n_frames
    keyframe_times = np.array([i * dt for i in range(n_frames)], dtype=np.float32)
    time_acc = write_accessor(
        gltf, binary_data, keyframe_times, None,
        pygltflib.FLOAT, pygltflib.SCALAR, with_minmax=True,
    )

    channels = []
    samplers = []

    for fi in range(n_frames):
        scales = np.zeros((n_frames, 3), dtype=np.float32)
        scales[fi] = [1.0, 1.0, 1.0]

        scale_acc = write_accessor(
            gltf, binary_data, scales, None,
            pygltflib.FLOAT, pygltflib.VEC3,
        )

        sampler_idx = len(samplers)
        samplers.append(pygltflib.AnimationSampler(
            input=time_acc,
            output=scale_acc,
            interpolation=pygltflib.ANIM_STEP,
        ))
        channels.append(pygltflib.AnimationChannel(
            sampler=sampler_idx,
            target=pygltflib.AnimationChannelTarget(node=fi, path="scale"),
        ))

    # Arrow frame visibility channels (synced with wavefront)
    for fi, node_idx in enumerate(arrow_node_indices):
        scales = np.zeros((n_frames, 3), dtype=np.float32)
        scales[fi] = [1.0, 1.0, 1.0]

        scale_acc = write_accessor(
            gltf, binary_data, scales, None,
            pygltflib.FLOAT, pygltflib.VEC3,
        )

        sampler_idx = len(samplers)
        samplers.append(pygltflib.AnimationSampler(
            input=time_acc,
            output=scale_acc,
            interpolation=pygltflib.ANIM_STEP,
        ))
        channels.append(pygltflib.AnimationChannel(
            sampler=sampler_idx,
            target=pygltflib.AnimationChannelTarget(node=node_idx, path="scale"),
        ))

    gltf.animations.append(pygltflib.Animation(
        name="excitation_ring",
        channels=channels,
        samplers=samplers,
    ))

    # Root node: mm -> m conversion + 10x AR display scale.
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
        f"CARTO animated GLB: {n_frames} frames, "
        f"{len(mesh_data.vertices)} verts, "
        f"{output_path.stat().st_size / 1024:.0f} KB"
    )
    return True
