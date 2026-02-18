"""Temporal mesh processing: consistent topology, temporal smoothing, morph targets."""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

from med2glb.core.types import AnimatedResult, ConversionResult, MeshData

logger = logging.getLogger(__name__)


def build_morph_targets_from_frames(
    frame_results: list[ConversionResult],
    temporal_resolution: float | None = None,
) -> AnimatedResult:
    """Build morph target animation from per-frame conversion results.

    Takes the first frame as the base topology and deforms vertices
    for subsequent frames using nearest-surface-point correspondence.
    """
    if not frame_results:
        raise ValueError("No frame results to build morph targets from")

    base_result = frame_results[0]
    n_meshes = len(base_result.meshes)

    # Build morph targets per mesh
    all_morph_targets: list[list[np.ndarray]] = [[] for _ in range(n_meshes)]
    base_meshes = list(base_result.meshes)

    for mesh_idx in range(n_meshes):
        base_mesh = base_meshes[mesh_idx]
        base_verts = base_mesh.vertices

        for frame_idx in range(1, len(frame_results)):
            frame_mesh = frame_results[frame_idx].meshes[mesh_idx] if mesh_idx < len(frame_results[frame_idx].meshes) else None

            if frame_mesh is None:
                # No corresponding mesh — zero displacement
                all_morph_targets[mesh_idx].append(
                    np.zeros_like(base_verts, dtype=np.float32)
                )
                continue

            # Find corresponding vertex positions via nearest surface point
            deformed_verts = _find_corresponding_vertices(
                base_verts, frame_mesh.vertices
            )

            # Compute displacement (morph target = displacement from base)
            displacement = deformed_verts - base_verts
            all_morph_targets[mesh_idx].append(displacement.astype(np.float32))

    # Apply temporal smoothing to morph targets
    for mesh_idx in range(n_meshes):
        if all_morph_targets[mesh_idx]:
            all_morph_targets[mesh_idx] = _smooth_morph_targets(
                all_morph_targets[mesh_idx]
            )

    # Compute frame times
    frame_times = _compute_frame_times(len(frame_results), temporal_resolution)
    loop_duration = frame_times[-1] if frame_times else 1.0

    return AnimatedResult(
        base_meshes=base_meshes,
        morph_targets=all_morph_targets,
        frame_times=frame_times,
        loop_duration=loop_duration,
        method_name=base_result.method_name,
        processing_time=sum(r.processing_time for r in frame_results),
        warnings=[w for r in frame_results for w in r.warnings],
    )


def _find_corresponding_vertices(
    base_vertices: np.ndarray, target_vertices: np.ndarray
) -> np.ndarray:
    """Find nearest-surface-point correspondence between base and target mesh.

    For each vertex in the base mesh, find the closest vertex in the target mesh.
    This ensures consistent vertex count across all morph targets.
    """
    tree = cKDTree(target_vertices)
    distances, indices = tree.query(base_vertices, k=1)

    return target_vertices[indices].astype(np.float32)


def _smooth_morph_targets(
    targets: list[np.ndarray], window_size: int = 3
) -> list[np.ndarray]:
    """Apply temporal smoothing to morph target displacements.

    Uses a weighted moving average across frames to prevent
    animation flickering while preserving motion dynamics.
    """
    if len(targets) < window_size:
        return targets

    smoothed = []
    half_win = window_size // 2

    for i in range(len(targets)):
        start = max(0, i - half_win)
        end = min(len(targets), i + half_win + 1)
        window = targets[start:end]

        # Weighted average: center frame gets highest weight
        weights = np.array([1.0] * len(window))
        center_idx = i - start
        weights[center_idx] = 2.0
        weights = weights / weights.sum()

        avg = np.zeros_like(targets[i])
        for w, t in zip(weights, window):
            avg += w * t

        smoothed.append(avg.astype(np.float32))

    return smoothed


def _compute_frame_times(
    n_frames: int, temporal_resolution: float | None
) -> list[float]:
    """Compute keyframe timestamps in seconds."""
    if temporal_resolution:
        # temporal_resolution is in ms
        dt = temporal_resolution / 1000.0
    else:
        # Default: assume 1 second cardiac cycle
        dt = 1.0 / max(n_frames - 1, 1)

    return [i * dt for i in range(n_frames)]
