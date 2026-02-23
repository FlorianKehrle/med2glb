"""Tests for glb/arrow_builder.py — dash mesh generation and GLB node building."""

from __future__ import annotations

import numpy as np
import pygltflib
import pytest

from med2glb.core.types import MeshData
from med2glb.glb.arrow_builder import (
    ArrowParams,
    _auto_scale_params,
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
        params = ArrowParams(segments=8)
        start = np.array([0, 0, 0], dtype=np.float64)
        end = np.array([5, 0, 0], dtype=np.float64)
        normal = np.array([0, 0, 1], dtype=np.float64)

        verts, faces = generate_dash_mesh(start, end, normal, params)
        # 3 * segments (bottom, top, cone_base) + 1 tip = 3*8 + 1 = 25
        assert len(verts) == 3 * 8 + 1

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
