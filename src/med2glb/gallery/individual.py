"""Individual GLB builder: one GLB file per gallery slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib

from med2glb.core.types import GallerySlice
from med2glb.gallery._glb_utils import (
    add_quad_geometry,
    add_scale_animation,
    add_textured_quad_node,
    create_base_gltf,
    finalize_gltf,
    group_by_position,
    make_png_material,
    quad_vertices_for_slice,
)


def build_individual_glbs(
    slices: list[GallerySlice],
    output_dir: Path,
    animate: bool = True,
    temporal_resolution: float | None = None,
) -> list[Path]:
    """Create one GLB per unique spatial position.

    For static data each file is a single textured quad.
    For temporal data each file contains animated frame-switching.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    has_temporal = any(s.temporal_index is not None for s in slices)

    if has_temporal and animate:
        groups = group_by_position(slices)
        pad = len(str(len(groups)))
        for idx, (_, group) in enumerate(sorted(groups.items())):
            group.sort(key=lambda s: (s.temporal_index or 0))
            out_path = output_dir / f"slice_{idx + 1:0{pad}d}.glb"
            _build_animated_individual(group, out_path, temporal_resolution)
            paths.append(out_path)
    else:
        # Static: pick first temporal frame per position, or all if no temporal
        if has_temporal:
            groups = group_by_position(slices)
            static_slices = [
                sorted(g, key=lambda s: (s.temporal_index or 0))[0]
                for g in groups.values()
            ]
            static_slices.sort(key=lambda s: s.instance_number)
        else:
            static_slices = slices

        pad = len(str(len(static_slices)))
        for idx, sl in enumerate(static_slices):
            out_path = output_dir / f"slice_{idx + 1:0{pad}d}.glb"
            _build_static_individual(sl, out_path)
            paths.append(out_path)

    return paths


def _build_static_individual(sl: GallerySlice, output_path: Path) -> None:
    """Build a single-quad static GLB for one slice."""
    gltf, binary_data = create_base_gltf()

    mat_idx = make_png_material(gltf, binary_data, sl.pixel_data, "slice")
    vertices = quad_vertices_for_slice(sl)
    geom = add_quad_geometry(gltf, binary_data, vertices)
    add_textured_quad_node(gltf, binary_data, geom, mat_idx, "slice")

    finalize_gltf(gltf, binary_data, output_path)


def _build_animated_individual(
    frames: list[GallerySlice],
    output_path: Path,
    temporal_resolution: float | None,
) -> None:
    """Build an animated GLB toggling between temporal frames via scale."""
    n_frames = len(frames)
    gltf, binary_data = create_base_gltf()
    gltf.animations = []

    # Use first frame for geometry dimensions
    vertices = quad_vertices_for_slice(frames[0])
    geom = add_quad_geometry(gltf, binary_data, vertices)

    node_indices: list[int] = []
    for i, sl in enumerate(frames):
        mat_idx = make_png_material(gltf, binary_data, sl.pixel_data, f"frame_{i}")
        scale = [1.0, 1.0, 1.0] if i == 0 else [0.0, 0.0, 0.0]
        node_idx = add_textured_quad_node(
            gltf, binary_data, geom, mat_idx, f"frame_{i}", scale=scale,
        )
        node_indices.append(node_idx)

    add_scale_animation(gltf, binary_data, node_indices, temporal_resolution)
    finalize_gltf(gltf, binary_data, output_path)
