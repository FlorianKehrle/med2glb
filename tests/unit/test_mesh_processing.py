"""Unit tests for mesh processing."""

from __future__ import annotations

import numpy as np
import pytest

from med2glb.core.types import MaterialConfig, MeshData
from med2glb.mesh.processing import (
    compute_normals,
    decimate,
    fill_holes,
    process_mesh,
    taubin_smooth,
)


@pytest.fixture
def sphere_mesh():
    """Create a sphere mesh with known properties."""
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    return MeshData(
        vertices=np.array(sphere.vertices, dtype=np.float32),
        faces=np.array(sphere.faces, dtype=np.int32),
        structure_name="test_sphere",
        material=MaterialConfig(),
    )


def test_taubin_smooth_preserves_shape(sphere_mesh):
    smoothed = taubin_smooth(sphere_mesh, iterations=10)
    assert smoothed.vertices.shape == sphere_mesh.vertices.shape
    assert smoothed.faces.shape == sphere_mesh.faces.shape

    # Volume should be roughly preserved
    import trimesh

    orig = trimesh.Trimesh(sphere_mesh.vertices, sphere_mesh.faces)
    result = trimesh.Trimesh(smoothed.vertices, smoothed.faces)
    assert abs(result.volume - orig.volume) / orig.volume < 0.1  # <10% change


def test_taubin_smooth_zero_iterations(sphere_mesh):
    smoothed = taubin_smooth(sphere_mesh, iterations=0)
    np.testing.assert_array_equal(smoothed.vertices, sphere_mesh.vertices)


def test_decimate_reduces_faces(sphere_mesh):
    target = len(sphere_mesh.faces) // 2
    decimated = decimate(sphere_mesh, target_faces=target)
    assert len(decimated.faces) <= target + target * 0.1  # Allow 10% tolerance


def test_decimate_noop_below_target(sphere_mesh):
    target = len(sphere_mesh.faces) * 2
    decimated = decimate(sphere_mesh, target_faces=target)
    assert len(decimated.faces) == len(sphere_mesh.faces)


def test_compute_normals(sphere_mesh):
    result = compute_normals(sphere_mesh)
    assert result.normals is not None
    assert result.normals.shape == result.vertices.shape


def test_fill_holes(synthetic_mesh):
    result = fill_holes(synthetic_mesh)
    assert result.vertices.shape[1] == 3
    assert result.faces.shape[1] == 3


def test_process_mesh_full_pipeline(sphere_mesh):
    result = process_mesh(
        sphere_mesh, smoothing_iterations=5, target_faces=100
    )
    assert result.normals is not None
    assert len(result.faces) <= 120  # Rough target
