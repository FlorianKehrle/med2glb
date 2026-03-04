"""CARTO animated GLB: excitation highlight ring using frame-based visibility.

Creates N mesh copies with different vertex colors. The static colormap
(LAT, bipolar, or unipolar) is always visible; a bright highlight ring
sweeps across the surface following LAT activation timing.  Animation
switches the visible frame via scale [1,1,1] / [0,0,0] — universally
supported in glTF viewers including HoloLens 2 (MRTK/glTFast).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pygltflib

from med2glb.core.types import MeshData
from med2glb.glb.builder import _center_vertices, _pad_to_4, write_accessor

logger = logging.getLogger("med2glb")

# Highlight ring parameters (tuned to match CARTO PRIME V7 "Map Replay" video)
_RING_WIDTH = 0.025  # σ of Gaussian — narrow, sharp band like the real CARTO system
_HIGHLIGHT_ADD = np.array([0.55, 0.55, 0.55], dtype=np.float32)  # additive white brightness


def _compute_anim_target_faces(
    n_faces: int,
    n_frames: int,
    max_size_bytes: int,
) -> int:
    """Compute the max face count that keeps animated GLB under a size limit.

    Animated GLBs share geometry but duplicate COLOR_0 per frame.
    Per-frame colors use uint8 RGBA (4 bytes/vert) instead of float32 (16).
    Approximate bytes: F/2 * (60 + 4*N_frames) where F/2 ≈ vertex count.
    """
    bytes_per_face = (60 + 4 * n_frames) / 2  # ≈90 for 30 frames
    max_faces = int(max_size_bytes / bytes_per_face)
    # Clamp: never exceed original, never go below 10K
    return max(10000, min(max_faces, n_faces))


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
) -> bool:
    """Build animated GLB with CARTO-style highlight ring over static colormap.

    The static vertex colors (from any coloring mode) remain visible at all
    times.  A bright ring sweeps across the surface following LAT activation
    timing.

    Args:
        mesh_data: MeshData with vertex_colors set (LAT/bipolar/unipolar colormap).
        lat_values: Per-vertex LAT values (ms) for ring timing. NaN for unknown.
        output_path: Where to write the .glb file.
        n_frames: Number of animation frames.
        loop_duration_s: Total animation loop time in seconds.
        target_faces: Explicit face limit (legacy). Overridden by max_size_mb
            when max_size_mb yields a higher limit.
        max_size_mb: Target max file size in MB. Used to compute an appropriate
            face count that balances quality and file size.
        vectors: If True, add animated LAT streamline arrows.
        progress: Optional callback(description, current, total) for progress.

    Returns:
        True if the GLB was written, False if skipped (vectors requested but
        streamline quality was insufficient — the non-vectors variant already
        covers this case).
    """
    def _report(desc: str, current: int = 0, total: int = 0) -> None:
        if progress:
            progress(desc, current, total)

    # Use whichever limit allows more faces (better quality)
    size_based_target = _compute_anim_target_faces(
        len(mesh_data.faces), n_frames, int(max_size_mb * 1024 * 1024),
    )
    effective_target = max(target_faces, size_based_target)

    # Decimate if mesh is large (animation duplicates mesh N times)
    if len(mesh_data.faces) > effective_target:
        _report(f"Decimating {len(mesh_data.faces):,} → {effective_target:,} faces...")
        from med2glb.mesh.processing import decimate, compute_normals

        decimated = decimate(mesh_data, target_faces=effective_target)
        decimated = compute_normals(decimated)
        # Resample LAT values to decimated mesh via nearest neighbor
        # (vertex_colors are already preserved inside decimate())
        from scipy.spatial import KDTree
        tree = KDTree(mesh_data.vertices)
        _, idx = tree.query(decimated.vertices)
        lat_values = lat_values[idx]
        mesh_data = decimated

    valid_lat = ~np.isnan(lat_values)
    if not np.any(valid_lat):
        # No valid LAT — fall back to static export
        from med2glb.glb.builder import build_glb
        build_glb([mesh_data], output_path, source_units="mm")
        return True

    lat_min = float(np.nanmin(lat_values))
    lat_max = float(np.nanmax(lat_values))
    lat_range = lat_max - lat_min
    if lat_range < 1e-6:
        from med2glb.glb.builder import build_glb
        build_glb([mesh_data], output_path, source_units="mm")
        return True

    # Normalize LAT to [0, 1]
    lat_norm = (lat_values - lat_min) / lat_range
    lat_norm[~valid_lat] = np.nan

    # Base colors from static coloring (mesh_data.vertex_colors)
    n_verts = len(mesh_data.vertices)
    if mesh_data.vertex_colors is not None:
        base_colors = mesh_data.vertex_colors.astype(np.float32)
    else:
        # Fallback: neutral light gray if no colormap was applied
        base_colors = np.full((n_verts, 4), [0.7, 0.7, 0.7, 1.0], dtype=np.float32)

    # Generate per-frame vertex colors with highlight ring (store as uint8
    # immediately to cut memory from 16 bytes/vert to 4 bytes/vert per frame)
    frame_colors_u8: list[np.ndarray] = []
    for fi in range(n_frames):
        _report(f"Generating frame {fi + 1}/{n_frames}...", fi, n_frames)
        t = fi / max(n_frames - 1, 1)  # ring position in normalized LAT space
        colors = base_colors.copy()

        # Gaussian ring: bright band centered at current wavefront position
        ring = np.exp(-((lat_norm - t) ** 2) / (2 * _RING_WIDTH ** 2))
        ring[~valid_lat] = 0.0

        # Additive brightening: preserves base hue, adds white light
        # (red→yellow, green→bright green, blue→cyan — matches real CARTO)
        add = ring.reshape(-1, 1) * _HIGHLIGHT_ADD
        colors[:, :3] = np.minimum(colors[:, :3] + add, 1.0)
        colors[:, 3] = 1.0

        frame_colors_u8.append(np.clip(colors * 255 + 0.5, 0, 255).astype(np.uint8))

    # Compute animated arrow dashes if vectors enabled
    arrow_frame_dashes = None
    arrow_speed_factors = None
    if vectors:
        _report("Tracing streamlines...")
        from med2glb.mesh.lat_vectors import (
            trace_all_streamlines, compute_animated_dashes,
            compute_face_gradients, compute_dash_speed_factors,
            assess_streamline_quality,
        )
        streamlines = trace_all_streamlines(
            mesh_data.vertices, mesh_data.faces, lat_values,
            mesh_data.normals, target_count=300,
        )
        if not streamlines:
            logger.warning("Vectors skipped: no streamlines traced")
            return False
        bbox_diag = float(np.linalg.norm(
            mesh_data.vertices.max(0) - mesh_data.vertices.min(0)))
        good, reason = assess_streamline_quality(streamlines, bbox_diag)
        if not good:
            logger.warning("Vectors skipped: %s", reason)
            return False
        if streamlines:
            _report(f"Generating dash animation for {len(streamlines)} streamlines...")
            arrow_frame_dashes = compute_animated_dashes(
                streamlines, n_frames=n_frames,
            )
            # Speed-dependent sizing
            face_grads, face_centers, _ = compute_face_gradients(
                mesh_data.vertices, mesh_data.faces, lat_values,
            )
            arrow_speed_factors = compute_dash_speed_factors(
                arrow_frame_dashes, face_grads, face_centers,
            )

    # UV unwrap ONCE with xatlas, precompute raster map, apply per frame
    from med2glb.glb.vertex_color_bake import (
        xatlas_unwrap,
        precompute_rasterization_map,
        apply_rasterization_map,
        compute_texture_size,
    )

    tex_size = compute_texture_size(len(mesh_data.faces))

    _report("UV unwrapping with xatlas...", 0, n_frames)
    vmapping, new_faces, shared_uvs = xatlas_unwrap(
        mesh_data.vertices, mesh_data.faces, mesh_data.normals,
    )
    unwelded_verts, centroid = _center_vertices(
        mesh_data.vertices[vmapping].astype(np.float32),
    )
    unwelded_normals = None
    if mesh_data.normals is not None:
        unwelded_normals = mesh_data.normals[vmapping].astype(np.float32)
    unwelded_faces = new_faces

    # Precompute pixel→triangle mapping once (expensive), then apply per frame (cheap)
    _report("Precomputing rasterization map...", 0, n_frames)
    raster_map = precompute_rasterization_map(new_faces, shared_uvs, tex_size)

    _report("Baking frame textures...", 0, n_frames)
    frame_textures: list[bytes] = []
    for fi in range(n_frames):
        _report(f"Baking frame {fi + 1}/{n_frames}...", fi, n_frames)
        colors_remapped = frame_colors_u8[fi][vmapping].astype(np.float32) / 255.0
        png_bytes = apply_rasterization_map(raster_map, colors_remapped)
        frame_textures.append(png_bytes)

    # Free per-frame color arrays and raster map (textures are baked now)
    del frame_colors_u8, raster_map

    # Build glTF with N frame nodes under a root mm→m scale node
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],  # set after building nodes
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

    # Embed per-frame textures into binary buffer
    for fi in range(n_frames):
        img_offset = len(binary_data)
        binary_data.extend(frame_textures[fi])
        _pad_to_4(binary_data)

        img_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=img_offset, byteLength=len(frame_textures[fi]),
        ))
        img_idx = len(gltf.images)
        gltf.images.append(pygltflib.Image(
            bufferView=img_bv_idx, mimeType="image/png",
        ))
        gltf.textures.append(pygltflib.Texture(sampler=0, source=img_idx))

    del frame_textures  # free memory

    # Shared geometry: positions, normals, UVs, indices (written once, unwelded)
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

    # Per-frame: material (with texture) + mesh + node
    _report("Assembling GLB...", n_frames, n_frames)
    for fi in range(n_frames):
        mat_idx = len(gltf.materials)
        mat_kwargs: dict = dict(
            name=f"wavefront_{fi}",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=fi),
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.7,
            ),
            doubleSided=True,
        )
        if mesh_data.material.unlit:
            mat_kwargs["extensions"] = {"KHR_materials_unlit": {}}
        gltf.materials.append(pygltflib.Material(**mat_kwargs))

        # Primitive with shared geometry + per-frame texture
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
        scale = [1.0, 1.0, 1.0] if fi == 0 else [0.0, 0.0, 0.0]
        gltf.nodes.append(pygltflib.Node(
            name=f"wavefront_{fi}", mesh=mesh_idx, scale=scale,
        ))

    # Collect all child node indices for the root node
    child_node_indices = list(range(n_frames))  # wavefront frame nodes

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

    # Wavefront frame visibility channels
    for fi in range(n_frames):
        # Scale output: [1,1,1] at this frame's keyframe, [0,0,0] at others
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

    # Root node: mm → m conversion + 10x AR display scale.
    # Cardiac structures (~12cm) render at ~120cm for comfortable AR inspection.
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

    logger.info(
        f"CARTO animated GLB: {n_frames} frames, "
        f"{len(mesh_data.vertices)} verts, "
        f"{output_path.stat().st_size / 1024:.0f} KB"
    )
    return True
