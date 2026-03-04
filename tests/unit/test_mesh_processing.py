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
    remove_small_components,
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


def test_compute_normals_preserves_vertex_colors(sphere_mesh):
    """compute_normals must forward vertex_colors (Issue A fix)."""
    colors = np.random.rand(len(sphere_mesh.vertices), 4).astype(np.float32)
    sphere_mesh.vertex_colors = colors
    result = compute_normals(sphere_mesh)
    assert result.vertex_colors is not None
    np.testing.assert_array_equal(result.vertex_colors, colors)


def test_compute_normals_none_colors(sphere_mesh):
    """compute_normals with no vertex_colors keeps None."""
    assert sphere_mesh.vertex_colors is None
    result = compute_normals(sphere_mesh)
    assert result.vertex_colors is None


class TestRemoveSmallComponents:
    def test_removes_tiny_fragment(self):
        """A small disconnected triangle should be removed."""
        import trimesh

        # Main component: icosphere (~1280 faces)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
        main_verts = np.array(sphere.vertices, dtype=np.float32)
        main_faces = np.array(sphere.faces, dtype=np.int32)

        # Tiny fragment: one triangle far away
        frag_verts = np.array([[100, 0, 0], [101, 0, 0], [100, 1, 0]], dtype=np.float32)
        frag_faces = np.array([[len(main_verts), len(main_verts) + 1, len(main_verts) + 2]], dtype=np.int32)

        all_verts = np.vstack([main_verts, frag_verts])
        all_faces = np.vstack([main_faces, frag_faces])

        mesh = MeshData(
            vertices=all_verts, faces=all_faces,
            structure_name="test", material=MaterialConfig(),
        )
        result = remove_small_components(mesh)
        # Fragment (1 face) is <1% of total → removed
        assert len(result.faces) == len(main_faces)

    def test_preserves_vertex_colors(self):
        """vertex_colors should survive small component removal."""
        import trimesh

        sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
        n = len(sphere.vertices)
        colors = np.random.rand(n + 3, 4).astype(np.float32)

        frag_verts = np.array([[100, 0, 0], [101, 0, 0], [100, 1, 0]], dtype=np.float32)
        frag_faces = np.array([[n, n + 1, n + 2]], dtype=np.int32)

        mesh = MeshData(
            vertices=np.vstack([np.array(sphere.vertices, dtype=np.float32), frag_verts]),
            faces=np.vstack([np.array(sphere.faces, dtype=np.int32), frag_faces]),
            vertex_colors=colors,
            structure_name="test", material=MaterialConfig(),
        )
        result = remove_small_components(mesh)
        assert result.vertex_colors is not None
        assert result.vertex_colors.shape == (n, 4)

    def test_no_colors_ok(self):
        """Works fine when vertex_colors is None."""
        import trimesh

        sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
        n = len(sphere.vertices)

        frag_verts = np.array([[100, 0, 0], [101, 0, 0], [100, 1, 0]], dtype=np.float32)
        frag_faces = np.array([[n, n + 1, n + 2]], dtype=np.int32)

        mesh = MeshData(
            vertices=np.vstack([np.array(sphere.vertices, dtype=np.float32), frag_verts]),
            faces=np.vstack([np.array(sphere.faces, dtype=np.int32), frag_faces]),
            structure_name="test", material=MaterialConfig(),
        )
        result = remove_small_components(mesh)
        assert result.vertex_colors is None

    def test_single_component_noop(self, sphere_mesh):
        """Single-component mesh is returned unchanged."""
        result = remove_small_components(sphere_mesh)
        assert result is sphere_mesh
