"""Tests for glb/arrow_builder.py — dash mesh generation and GLB node building."""

from __future__ import annotations

import numpy as np
import pygltflib
import pytest

from med2glb.core.types import MeshData
from med2glb.glb.arrow_builder import (
    ArrowParams,
    _auto_scale_params,
    _teardrop_radius,
    build_animated_arrow_nodes,
    build_frame_dashes,
    generate_dash_mesh,
)


@pytest.fixture
def default_params():
    return ArrowParams()


@pytest.fixture
def simple_mesh():
    """A flat square mesh for surface normal lookup."""
    vertices = np.array([
        [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
    ], dtype=np.float32)
    normals = np.array([
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
    ], dtype=np.float32)
    return vertices, normals


class TestGenerateDashMesh:
    def test_produces_valid_geometry(self, default_params):
        start = np.array([0, 0, 0], dtype=np.float64)
        end = np.array([5, 0, 0], dtype=np.float64)
        normal = np.array([0, 0, 1], dtype=np.float64)

        verts, faces = generate_dash_mesh(start, end, normal, default_params)
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        assert len(verts) > 0
        assert len(faces) > 0

    def test_vertex_count_matches_segments(self):
        params = ArrowParams(segments=8, n_rings=6)
        start = np.array([0, 0, 0], dtype=np.float64)
        end = np.array([5, 0, 0], dtype=np.float64)
        normal = np.array([0, 0, 1], dtype=np.float64)

        verts, faces = generate_dash_mesh(start, end, normal, params)
        # Teardrop: 2 tips + (n_rings - 2) * segments = 2 + 4*8 = 34
        assert len(verts) == 2 + (params.n_rings - 2) * params.segments

    def test_zero_length_dash(self, default_params):
        start = np.array([5, 5, 5], dtype=np.float64)
        end = np.array([5, 5, 5], dtype=np.float64)
        normal = np.array([0, 0, 1], dtype=np.float64)

        verts, faces = generate_dash_mesh(start, end, normal, default_params)
        assert len(verts) == 0
        assert len(faces) == 0

    def test_face_indices_valid(self, default_params):
        start = np.array([0, 0, 0], dtype=np.float64)
        end = np.array([3, 4, 0], dtype=np.float64)
        normal = np.array([0, 0, 1], dtype=np.float64)

        verts, faces = generate_dash_mesh(start, end, normal, default_params)
        assert faces.min() >= 0
        assert faces.max() < len(verts)


class TestTeardropRadius:
    def test_zero_at_tips(self):
        assert _teardrop_radius(0.0, 1.0) == 0.0
        assert _teardrop_radius(1.0, 1.0) == 0.0

    def test_positive_interior(self):
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert _teardrop_radius(t, 1.0) > 0.0

    def test_peak_near_62_percent(self):
        # Sample at 1% intervals and find peak
        ts = [i / 100.0 for i in range(1, 100)]
        radii = [_teardrop_radius(t, 1.0) for t in ts]
        peak_t = ts[radii.index(max(radii))]
        assert 0.5 < peak_t < 0.75  # peak near 62%

    def test_scales_with_max_r(self):
        r1 = _teardrop_radius(0.5, 1.0)
        r2 = _teardrop_radius(0.5, 2.0)
        assert abs(r2 - 2.0 * r1) < 1e-10


class TestAutoScaleParams:
    def test_scales_with_diagonal(self):
        small = _auto_scale_params(10.0)
        large = _auto_scale_params(100.0)
        assert large.shaft_radius > small.shaft_radius
        assert large.head_radius > small.head_radius


class TestBuildFrameDashes:
    def test_produces_mesh_data(self, simple_mesh):
        vertices, normals = simple_mesh
        dashes = [
            (np.array([2, 2, 0.0]), np.array([5, 2, 0.0])),
            (np.array([2, 5, 0.0]), np.array([5, 5, 0.0])),
        ]
        params = ArrowParams()
        result = build_frame_dashes(dashes, vertices, normals, params)
        assert result is not None
        assert isinstance(result, MeshData)
        assert len(result.vertices) > 0
        assert len(result.faces) > 0
        assert result.vertex_colors is not None
        assert result.structure_name == "lat_vectors"

    def test_empty_dashes(self, simple_mesh):
        vertices, normals = simple_mesh
        result = build_frame_dashes([], vertices, normals, ArrowParams())
        assert result is None

    def test_no_normals(self, simple_mesh):
        vertices, _ = simple_mesh
        dashes = [(np.array([2, 2, 0.0]), np.array([5, 2, 0.0]))]
        result = build_frame_dashes(dashes, vertices, None, ArrowParams())
        assert result is None


class TestBuildAnimatedArrowNodes:
    def test_creates_correct_number_of_nodes(self, simple_mesh):
        vertices, normals = simple_mesh
        n_frames = 3
        all_dashes = [
            [(np.array([2, 2, 0.0]), np.array([5, 2, 0.0]))]
            for _ in range(n_frames)
        ]

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[])],
            nodes=[], meshes=[], accessors=[],
            bufferViews=[], buffers=[], materials=[],
        )
        binary_data = bytearray()

        node_indices = build_animated_arrow_nodes(
            all_dashes, vertices, normals,
            gltf, binary_data, n_frames,
        )

        assert len(node_indices) == n_frames
        # First frame visible, others hidden
        assert gltf.nodes[node_indices[0]].scale == [1.0, 1.0, 1.0]
        assert gltf.nodes[node_indices[1]].scale == [0.0, 0.0, 0.0]

    def test_unlit_extension(self, simple_mesh):
        """unlit=True should add KHR_materials_unlit to arrow material."""
        vertices, normals = simple_mesh
        all_dashes = [
            [(np.array([2, 2, 0.0]), np.array([5, 2, 0.0]))]
            for _ in range(2)
        ]

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[])],
            nodes=[], meshes=[], accessors=[],
            bufferViews=[], buffers=[], materials=[],
        )
        binary_data = bytearray()

        build_animated_arrow_nodes(
            all_dashes, vertices, normals,
            gltf, binary_data, 2, unlit=True,
        )

        assert "KHR_materials_unlit" in (gltf.extensionsUsed or [])
        # The arrow material (first one added) should have the extension
        assert "KHR_materials_unlit" in (gltf.materials[0].extensions or {})

    def test_handles_empty_frames(self, simple_mesh):
        vertices, normals = simple_mesh
        all_dashes = [[], [], []]

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[])],
            nodes=[], meshes=[], accessors=[],
            bufferViews=[], buffers=[], materials=[],
        )
        binary_data = bytearray()

        node_indices = build_animated_arrow_nodes(
            all_dashes, vertices, normals,
            gltf, binary_data, 3,
        )

        assert len(node_indices) == 3
