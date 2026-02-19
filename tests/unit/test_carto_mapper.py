"""Unit tests for CARTO point-to-vertex mapping and colormaps."""

from __future__ import annotations

import numpy as np
import pytest

from med2glb.io.carto_colormaps import bipolar_colormap, lat_colormap, unipolar_colormap
from med2glb.io.carto_mapper import (
    build_inactive_mask,
    carto_mesh_to_mesh_data,
    interpolate_sparse_values,
    map_points_to_vertices,
    map_points_to_vertices_idw,
    subdivide_carto_mesh,
)


class TestMapPointsToVertices:
    def test_basic_mapping(self, synthetic_carto_mesh, synthetic_carto_points):
        values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="lat"
        )
        assert len(values) == len(synthetic_carto_mesh.vertices)
        # At least some values should be non-NaN (nearest-neighbor assigns to all)
        assert not np.all(np.isnan(values))

    def test_bipolar_field(self, synthetic_carto_mesh, synthetic_carto_points):
        values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="bipolar"
        )
        assert not np.all(np.isnan(values))
        # Bipolar voltages should be positive
        assert np.all(values[~np.isnan(values)] >= 0)

    def test_unipolar_field(self, synthetic_carto_mesh, synthetic_carto_points):
        values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="unipolar"
        )
        assert not np.all(np.isnan(values))

    def test_empty_points(self, synthetic_carto_mesh):
        values = map_points_to_vertices(synthetic_carto_mesh, [], field="lat")
        assert np.all(np.isnan(values))

    def test_invalid_field(self, synthetic_carto_mesh, synthetic_carto_points):
        with pytest.raises(ValueError, match="Unknown field"):
            map_points_to_vertices(
                synthetic_carto_mesh, synthetic_carto_points, field="invalid"
            )


class TestInterpolateSparseValues:
    def test_fills_gaps(self, synthetic_carto_mesh):
        values = np.full(len(synthetic_carto_mesh.vertices), np.nan)
        # Set a few known values
        values[0] = 10.0
        values[5] = 20.0
        values[10] = 30.0

        result = interpolate_sparse_values(synthetic_carto_mesh, values)
        # Original values should be preserved
        assert result[0] == pytest.approx(10.0)
        assert result[5] == pytest.approx(20.0)
        assert result[10] == pytest.approx(30.0)

    def test_all_valid_unchanged(self, synthetic_carto_mesh):
        values = np.arange(len(synthetic_carto_mesh.vertices), dtype=np.float64)
        result = interpolate_sparse_values(synthetic_carto_mesh, values)
        np.testing.assert_array_equal(result, values)


class TestBuildInactiveMask:
    def test_all_active(self, synthetic_carto_mesh):
        mask = build_inactive_mask(synthetic_carto_mesh)
        assert not np.any(mask)

    def test_with_inactive(self, synthetic_carto_mesh):
        synthetic_carto_mesh.group_ids[0] = -1000000
        synthetic_carto_mesh.group_ids[3] = -1000000
        mask = build_inactive_mask(synthetic_carto_mesh)
        assert mask[0] is np.True_
        assert mask[3] is np.True_
        assert np.sum(mask) == 2


class TestCartoMeshToMeshData:
    def test_basic_conversion(self, synthetic_carto_mesh, synthetic_carto_points):
        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, synthetic_carto_points, coloring="lat", subdivide=0,
        )
        assert mesh_data.vertices is not None
        assert mesh_data.faces is not None
        assert mesh_data.vertex_colors is not None
        assert mesh_data.vertex_colors.shape[1] == 4  # RGBA

    def test_filters_inactive(self, carto_mesh_dir):
        from med2glb.io.carto_reader import parse_mesh_file, parse_car_file

        mesh = parse_mesh_file(carto_mesh_dir / "1-TestMap.mesh")
        _, points = parse_car_file(carto_mesh_dir / "1-TestMap_car.txt")

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
        # Original has 4 verts (1 inactive), so should have 3 active
        assert len(mesh_data.vertices) == 3
        # Original has 2 faces (1 inactive), so should have 1
        assert len(mesh_data.faces) == 1

    def test_no_points_fallback(self, synthetic_carto_mesh):
        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, None, coloring="lat", subdivide=0,
        )
        assert mesh_data.vertex_colors is not None
        # Should be mesh default color (green)
        assert mesh_data.vertex_colors[0, 1] == pytest.approx(1.0)  # green channel

    def test_vertex_colors_are_float32(self, synthetic_carto_mesh, synthetic_carto_points):
        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, synthetic_carto_points, subdivide=0,
        )
        assert mesh_data.vertex_colors.dtype == np.float32


class TestColormaps:
    def test_lat_colormap_range(self):
        values = np.array([-200, -100, 0, 50, 100], dtype=np.float64)
        colors = lat_colormap(values)
        assert colors.shape == (5, 4)
        assert np.all(colors[:, :3] >= 0)
        assert np.all(colors[:, :3] <= 1)
        assert np.all(colors[:, 3] == 1.0)

    def test_lat_nan_transparent(self):
        values = np.array([0, np.nan, 100], dtype=np.float64)
        colors = lat_colormap(values)
        assert colors[1, 3] < 1.0  # NaN vertex is semi-transparent

    def test_bipolar_default_range(self):
        values = np.array([0.05, 0.5, 1.0, 1.5], dtype=np.float64)
        colors = bipolar_colormap(values)
        # Low voltage (scar) should be red
        assert colors[0, 0] > 0.8  # high red
        assert colors.shape == (4, 4)

    def test_unipolar_colormap(self):
        values = np.array([3.0, 6.5, 10.0], dtype=np.float64)
        colors = unipolar_colormap(values)
        assert colors.shape == (3, 4)
        assert np.all(colors[:, 3] == 1.0)

    def test_all_same_value(self):
        values = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        colors = lat_colormap(values)
        # Should not crash — all get mid-range color
        assert colors.shape == (3, 4)

    def test_custom_clamp_range(self):
        values = np.array([-300, -100, 100, 300], dtype=np.float64)
        colors = lat_colormap(values, clamp_range=(-200, 200))
        # Values outside range should be clamped (first red, last purple)
        assert colors[0, 0] > 0.8  # red


@pytest.fixture
def _manifold_carto_mesh():
    """Create a closed manifold CARTO mesh (octahedron) suitable for Loop subdivision."""
    from med2glb.core.types import CartoMesh

    # Regular octahedron — 6 vertices, 8 faces, fully closed manifold
    vertices = np.array([
        [0, 0, 1], [1, 0, 0], [0, 1, 0],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    ], dtype=np.float64) * 30

    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
        [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4],
    ], dtype=np.int32)

    normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    group_ids = np.zeros(len(vertices), dtype=np.int32)
    face_group_ids = np.zeros(len(faces), dtype=np.int32)

    return CartoMesh(
        mesh_id=1,
        vertices=vertices,
        faces=faces,
        normals=normals,
        group_ids=group_ids,
        face_group_ids=face_group_ids,
        mesh_color=(0.0, 1.0, 0.0, 1.0),
        color_names=["Unipolar", "Bipolar", "LAT"],
        structure_name="test_octa",
    )


class TestSubdivideCartoMesh:
    def test_increases_vertex_count(self, _manifold_carto_mesh):
        result = subdivide_carto_mesh(_manifold_carto_mesh, iterations=1)
        assert len(result.vertices) > len(_manifold_carto_mesh.vertices)
        assert len(result.faces) > len(_manifold_carto_mesh.faces)

    def test_zero_iterations_noop(self, _manifold_carto_mesh):
        result = subdivide_carto_mesh(_manifold_carto_mesh, iterations=0)
        assert result is _manifold_carto_mesh

    def test_group_id_propagation(self, _manifold_carto_mesh):
        _manifold_carto_mesh.group_ids[0] = -1000000
        result = subdivide_carto_mesh(_manifold_carto_mesh, iterations=1)
        assert np.any(result.group_ids == -1000000)
        assert np.any(result.group_ids != -1000000)

    def test_normals_computed(self, _manifold_carto_mesh):
        result = subdivide_carto_mesh(_manifold_carto_mesh, iterations=1)
        norms = np.linalg.norm(result.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)

    def test_preserves_metadata(self, _manifold_carto_mesh):
        result = subdivide_carto_mesh(_manifold_carto_mesh, iterations=1)
        assert result.mesh_id == _manifold_carto_mesh.mesh_id
        assert result.structure_name == _manifold_carto_mesh.structure_name
        assert result.mesh_color == _manifold_carto_mesh.mesh_color


class TestMapPointsToVerticesIdw:
    def test_basic_mapping(self, synthetic_carto_mesh, synthetic_carto_points):
        values = map_points_to_vertices_idw(
            synthetic_carto_mesh, synthetic_carto_points, field="lat"
        )
        assert len(values) == len(synthetic_carto_mesh.vertices)
        assert not np.all(np.isnan(values))

    def test_differs_from_nn(self, synthetic_carto_mesh, synthetic_carto_points):
        nn_values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="lat"
        )
        idw_values = map_points_to_vertices_idw(
            synthetic_carto_mesh, synthetic_carto_points, field="lat"
        )
        assert not np.allclose(nn_values, idw_values)

    def test_empty_points(self, synthetic_carto_mesh):
        values = map_points_to_vertices_idw(synthetic_carto_mesh, [], field="lat")
        assert np.all(np.isnan(values))

    def test_k_clamping(self, synthetic_carto_mesh):
        """When fewer points than k exist, k is clamped without error."""
        from med2glb.core.types import CartoPoint

        points = [
            CartoPoint(0, [0, 0, 0], [0, 0, 0], 1.0, 1.0, -50.0),
            CartoPoint(1, [30, 0, 0], [0, 0, 0], 2.0, 2.0, 50.0),
        ]
        values = map_points_to_vertices_idw(
            synthetic_carto_mesh, points, field="lat", k=6,
        )
        assert len(values) == len(synthetic_carto_mesh.vertices)
        assert not np.all(np.isnan(values))


class TestCartoMeshToMeshDataWithSubdivide:
    def test_subdivide_increases_vertices(self, _manifold_carto_mesh, synthetic_carto_points):
        mesh_no_sub = carto_mesh_to_mesh_data(
            _manifold_carto_mesh, synthetic_carto_points, coloring="lat", subdivide=0,
        )
        mesh_sub = carto_mesh_to_mesh_data(
            _manifold_carto_mesh, synthetic_carto_points, coloring="lat", subdivide=1,
        )
        assert len(mesh_sub.vertices) > len(mesh_no_sub.vertices)

    def test_subdivide_zero_matches_original(self, _manifold_carto_mesh, synthetic_carto_points):
        mesh_data = carto_mesh_to_mesh_data(
            _manifold_carto_mesh, synthetic_carto_points, coloring="lat", subdivide=0,
        )
        assert len(mesh_data.vertices) == len(_manifold_carto_mesh.vertices)
