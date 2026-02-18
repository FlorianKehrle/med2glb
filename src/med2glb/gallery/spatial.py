"""Spatial fan GLB builder: position quads using DICOM spatial metadata."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pygltflib

from med2glb.core.types import GallerySlice
from med2glb.gallery._glb_utils import (
    add_parent_node,
    add_quad_geometry,
    add_scale_animation,
    add_textured_quad_node,
    create_base_gltf,
    finalize_gltf,
    group_by_position,
    make_png_material,
    quad_vertices_for_slice,
)

logger = logging.getLogger(__name__)


def build_spatial_glb(
    slices: list[GallerySlice],
    output_path: Path,
    animate: bool = True,
    temporal_resolution: float | None = None,
) -> bool:
    """Build a GLB with quads at their real-world spatial positions.

    Returns True if the file was created, False if skipped (no spatial metadata).
    """
    if not slices:
        return False

    has_positions = any(s.image_position is not None for s in slices)
    if not has_positions:
        logger.info("Skipping spatial output — no ImagePositionPatient metadata available.")
        return False

    has_temporal = any(s.temporal_index is not None for s in slices)

    gltf, binary_data = create_base_gltf()
    if has_temporal and animate:
        gltf.animations = []

    # Shared geometry sized for first slice (all quads use same unit quad)
    vertices = quad_vertices_for_slice(slices[0])
    geom = add_quad_geometry(gltf, binary_data, vertices)

    if has_temporal and animate:
        _build_animated_spatial(gltf, binary_data, slices, geom, temporal_resolution)
    else:
        _build_static_spatial(gltf, binary_data, slices, geom)

    finalize_gltf(gltf, binary_data, output_path)
    return True


def _build_static_spatial(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    slices: list[GallerySlice],
    geom,
) -> None:
    """Place one quad per spatial position using 4x4 matrix transforms."""
    has_temporal = any(s.temporal_index is not None for s in slices)
    if has_temporal:
        groups = group_by_position(slices)
        display_slices = [
            sorted(g, key=lambda s: (s.temporal_index or 0))[0]
            for g in groups.values()
        ]
        display_slices.sort(key=lambda s: s.instance_number)
    else:
        display_slices = slices

    for idx, sl in enumerate(display_slices):
        matrix = _compute_spatial_matrix(sl)
        mat_idx = make_png_material(gltf, binary_data, sl.pixel_data, f"spatial_{idx}")
        add_textured_quad_node(
            gltf, binary_data, geom, mat_idx, f"spatial_{idx}",
            matrix=matrix,
        )


def _build_animated_spatial(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    slices: list[GallerySlice],
    geom,
    temporal_resolution: float | None,
) -> None:
    """Place animated cells at spatial positions; each toggles temporal frames."""
    groups = group_by_position(slices)
    all_cell_nodes: list[list[int]] = []

    for idx, (key, group) in enumerate(sorted(groups.items())):
        group.sort(key=lambda s: (s.temporal_index or 0))
        matrix = _compute_spatial_matrix(group[0])

        parent_idx = add_parent_node(gltf, f"spatial_{idx}", matrix=matrix)

        cell_nodes: list[int] = []
        for fi, sl in enumerate(group):
            mat_idx = make_png_material(
                gltf, binary_data, sl.pixel_data, f"spatial_{idx}_f{fi}",
            )
            scale = [1.0, 1.0, 1.0] if fi == 0 else [0.0, 0.0, 0.0]
            node_idx = add_textured_quad_node(
                gltf, binary_data, geom, mat_idx, f"spatial_{idx}_f{fi}",
                scale=scale, parent_node_idx=parent_idx,
            )
            cell_nodes.append(node_idx)
        all_cell_nodes.append(cell_nodes)

    # Synchronized animation
    if all_cell_nodes:
        _add_sync_animation(gltf, binary_data, all_cell_nodes, temporal_resolution)


def _add_sync_animation(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    all_cell_nodes: list[list[int]],
    temporal_resolution: float | None,
) -> None:
    """Add synchronized scale animation across all spatial positions."""
    from med2glb.glb.builder import write_accessor

    max_frames = max(len(nodes) for nodes in all_cell_nodes)
    if max_frames < 2:
        return

    dt = (temporal_resolution or 33.3) / 1000.0
    keyframe_times = np.array([i * dt for i in range(max_frames)], dtype=np.float32)

    time_acc = write_accessor(
        gltf, binary_data, keyframe_times,
        None, pygltflib.FLOAT, pygltflib.SCALAR, True,
    )

    channels: list[pygltflib.AnimationChannel] = []
    samplers: list[pygltflib.AnimationSampler] = []

    for cell_nodes in all_cell_nodes:
        n = len(cell_nodes)
        for i, node_idx in enumerate(cell_nodes):
            scales = np.zeros((max_frames, 3), dtype=np.float32)
            for k in range(max_frames):
                if k % n == i:
                    scales[k] = [1.0, 1.0, 1.0]

            s_acc = write_accessor(
                gltf, binary_data, scales,
                None, pygltflib.FLOAT, pygltflib.VEC3,
            )
            sampler_idx = len(samplers)
            samplers.append(
                pygltflib.AnimationSampler(
                    input=time_acc, output=s_acc,
                    interpolation=pygltflib.ANIM_STEP,
                )
            )
            channels.append(
                pygltflib.AnimationChannel(
                    sampler=sampler_idx,
                    target=pygltflib.AnimationChannelTarget(node=node_idx, path="scale"),
                )
            )

    if not hasattr(gltf, "animations") or gltf.animations is None:
        gltf.animations = []
    gltf.animations.append(
        pygltflib.Animation(name="spatial_cycle", channels=channels, samplers=samplers)
    )


def _compute_spatial_matrix(sl: GallerySlice) -> list[float]:
    """Build a column-major 4x4 matrix from DICOM position + orientation.

    Converts mm positions to metres for glTF.
    """
    if sl.image_orientation and len(sl.image_orientation) == 6:
        row_cos = np.array(sl.image_orientation[:3], dtype=np.float64)
        col_cos = np.array(sl.image_orientation[3:], dtype=np.float64)
    else:
        row_cos = np.array([1.0, 0.0, 0.0])
        col_cos = np.array([0.0, 1.0, 0.0])

    normal = np.cross(row_cos, col_cos)

    if sl.image_position is not None:
        pos = np.array(sl.image_position, dtype=np.float64) / 1000.0  # mm → m
    else:
        pos = np.array([0.0, 0.0, 0.0])

    # glTF uses column-major: [col0, col1, col2, col3] each as 4 elements
    return [
        float(row_cos[0]), float(row_cos[1]), float(row_cos[2]), 0.0,
        float(col_cos[0]), float(col_cos[1]), float(col_cos[2]), 0.0,
        float(normal[0]),  float(normal[1]),  float(normal[2]),  0.0,
        float(pos[0]),     float(pos[1]),     float(pos[2]),     1.0,
    ]
