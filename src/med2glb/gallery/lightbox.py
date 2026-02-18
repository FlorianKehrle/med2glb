"""Lightbox GLB builder: arrange slices in a grid layout."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygltflib

from med2glb.core.types import GallerySlice
from med2glb.gallery._glb_utils import (
    QuadGeometry,
    add_parent_node,
    add_quad_geometry,
    add_scale_animation,
    add_textured_quad_node,
    create_base_gltf,
    finalize_gltf,
    group_by_position,
    make_png_material,
)


def build_lightbox_glb(
    slices: list[GallerySlice],
    output_path: Path,
    columns: int = 6,
    animate: bool = True,
    temporal_resolution: float | None = None,
) -> None:
    """Build a single GLB with all slices laid out in a grid.

    Each cell is sized to the maximum physical dimensions across all slices
    with a 5 % gap between cells.
    """
    if not slices:
        return

    has_temporal = any(s.temporal_index is not None for s in slices)

    # Compute uniform cell size from maximum physical dimensions
    cell_w, cell_h = _compute_cell_size(slices)
    gap = 0.05  # 5 % of cell size
    step_x = cell_w * (1 + gap)
    step_y = cell_h * (1 + gap)

    gltf, binary_data = create_base_gltf()
    if has_temporal and animate:
        gltf.animations = []

    # Shared geometry at max cell size
    hw, hh = cell_w / 2, cell_h / 2
    vertices = np.array(
        [[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]],
        dtype=np.float32,
    )
    geom = add_quad_geometry(gltf, binary_data, vertices)

    if has_temporal and animate:
        _build_animated_lightbox(
            gltf, binary_data, slices, geom, columns, step_x, step_y,
            temporal_resolution,
        )
    else:
        _build_static_lightbox(gltf, binary_data, slices, geom, columns, step_x, step_y)

    finalize_gltf(gltf, binary_data, output_path)


def _build_static_lightbox(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    slices: list[GallerySlice],
    geom: QuadGeometry,
    columns: int,
    step_x: float,
    step_y: float,
) -> None:
    """Place one textured quad per slice in a grid."""
    # For temporal data without animation, pick first frame per position
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
        col = idx % columns
        row = idx // columns
        tx = col * step_x
        ty = -(row * step_y)
        mat_idx = make_png_material(gltf, binary_data, sl.pixel_data, f"cell_{idx}")
        add_textured_quad_node(
            gltf, binary_data, geom, mat_idx, f"cell_{idx}",
            translation=[tx, ty, 0.0],
        )


def _build_animated_lightbox(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    slices: list[GallerySlice],
    geom: QuadGeometry,
    columns: int,
    step_x: float,
    step_y: float,
    temporal_resolution: float | None,
) -> None:
    """Place animated cells in a grid; each cell toggles temporal frames."""
    groups = group_by_position(slices)
    all_node_indices: list[list[int]] = []

    for idx, (key, group) in enumerate(sorted(groups.items())):
        group.sort(key=lambda s: (s.temporal_index or 0))
        col = idx % columns
        row = idx // columns
        tx = col * step_x
        ty = -(row * step_y)

        parent_idx = add_parent_node(
            gltf, f"cell_{idx}", translation=[tx, ty, 0.0],
        )

        cell_nodes: list[int] = []
        for fi, sl in enumerate(group):
            mat_idx = make_png_material(
                gltf, binary_data, sl.pixel_data, f"cell_{idx}_f{fi}",
            )
            scale = [1.0, 1.0, 1.0] if fi == 0 else [0.0, 0.0, 0.0]
            node_idx = add_textured_quad_node(
                gltf, binary_data, geom, mat_idx, f"cell_{idx}_f{fi}",
                scale=scale, parent_node_idx=parent_idx,
            )
            cell_nodes.append(node_idx)
        all_node_indices.append(cell_nodes)

    # Synchronize animation across all cells
    if all_node_indices:
        _add_sync_animation(gltf, binary_data, all_node_indices, temporal_resolution)


def _add_sync_animation(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    all_cell_nodes: list[list[int]],
    temporal_resolution: float | None,
) -> None:
    """Add synchronized scale animation across all cells."""
    from med2glb.glb.builder import _pad_to_4, write_accessor

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
            # This frame is visible at keyframe i, hidden otherwise
            # For cells with fewer frames, cycle
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
        pygltflib.Animation(name="lightbox_cycle", channels=channels, samplers=samplers)
    )


def _compute_cell_size(slices: list[GallerySlice]) -> tuple[float, float]:
    """Compute uniform cell size (metres) from max physical dims across slices."""
    max_w = 0.0
    max_h = 0.0
    for sl in slices:
        row_sp, col_sp = sl.pixel_spacing
        w = sl.cols * col_sp / 1000.0
        h = sl.rows * row_sp / 1000.0
        max_w = max(max_w, w)
        max_h = max(max_h, h)
    return max_w, max_h
