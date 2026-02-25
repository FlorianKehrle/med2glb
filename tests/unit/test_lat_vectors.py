"""Tests for mesh/lat_vectors.py — gradient computation, streamline tracing, animated dashes."""

from __future__ import annotations

import numpy as np
import pytest

from med2glb.mesh.lat_vectors import (
    _smooth_streamline,
    assess_streamline_quality,
    build_face_adjacency,
    compute_animated_dashes,
    compute_dash_speed_factors,
    compute_face_gradients,
    compute_vertex_gradients,
    generate_streamline_seeds,
    trace_all_streamlines,
    trace_streamline,
)


@pytest.fixture
def flat_triangle_mesh():
    """A flat 2-triangle mesh in the XY plane."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    normals = np.array([
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
    ], dtype=np.float64)
    return vertices, faces, normals


@pytest.fixture
def larger_mesh():
    """A 4x4 grid mesh for more substantial tests."""
    # 5x5 = 25 vertices, 32 faces
    verts = []
    for j in range(5):
        for i in range(5):
            verts.append([float(i) * 10, float(j) * 10, 0.0])
    vertices = np.array(verts, dtype=np.float64)

    faces = []
    for j in range(4):
        for i in range(4):
            v0 = j * 5 + i
            v1 = v0 + 1
            v2 = v0 + 6
            v3 = v0 + 5
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    faces = np.array(faces, dtype=np.int32)

    normals = np.zeros_like(vertices)
    normals[:, 2] = 1.0

    # LAT values: linear gradient in X direction
    lat = vertices[:, 0].copy()  # 0 to 40

    return vertices, faces, normals, lat


class TestComputeFaceGradients:
    def test_uniform_lat_gives_zero_gradient(self, flat_triangle_mesh):
        vertices, faces, _ = flat_triangle_mesh
        lat = np.array([5.0, 5.0, 5.0, 5.0])
        grads, centers, valid = compute_face_gradients(vertices, faces, lat)
        assert grads.shape == (2, 3)
        assert np.allclose(grads, 0.0, atol=1e-10)
        assert np.all(valid)

    def test_linear_gradient_in_x(self, flat_triangle_mesh):
        vertices, faces, _ = flat_triangle_mesh
        # LAT = x coordinate -> gradient should point in +X
        lat = vertices[:, 0].copy()
        grads, centers, valid = compute_face_gradients(vertices, faces, lat)
        assert np.all(valid)
        for g in grads:
            if np.linalg.norm(g) > 1e-10:
                direction = g / np.linalg.norm(g)
                assert abs(direction[0]) > 0.9  # points in X

    def test_nan_lat_marks_face_invalid(self, flat_triangle_mesh):
        vertices, faces, _ = flat_triangle_mesh
        lat = np.array([0.0, 10.0, np.nan, 5.0])
        grads, centers, valid = compute_face_gradients(vertices, faces, lat)
        # Face 0 has vertex 2 which is NaN -> invalid
        assert not valid[0]
        assert np.allclose(grads[0], 0.0)

    def test_face_centers(self, flat_triangle_mesh):
        vertices, faces, _ = flat_triangle_mesh
        lat = np.array([0.0, 10.0, 20.0, 5.0])
        _, centers, _ = compute_face_gradients(vertices, faces, lat)
        # Face 0 centroid: (0+10+10)/3, (0+0+10)/3, 0
        expected_0 = np.array([20.0 / 3, 10.0 / 3, 0.0])
        assert np.allclose(centers[0], expected_0, atol=0.01)


class TestComputeVertexGradients:
    def test_tangent_plane_projection(self, flat_triangle_mesh):
        vertices, faces, normals = flat_triangle_mesh
        lat = vertices[:, 0].copy()
        vg = compute_vertex_gradients(vertices, faces, lat, normals)
        # For flat mesh with normals in Z, gradient should have no Z component
        assert np.allclose(vg[:, 2], 0.0, atol=1e-10)

    def test_gradient_direction(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        vg = compute_vertex_gradients(vertices, faces, lat, normals)
        # Most vertex gradients should point roughly in +X
        for i in range(len(vg)):
            mag = np.linalg.norm(vg[i])
            if mag > 1e-6:
                assert vg[i, 0] > 0  # positive X component


class TestBuildFaceAdjacency:
    def test_adjacency_structure(self, flat_triangle_mesh):
        _, faces, _ = flat_triangle_mesh
        adj = build_face_adjacency(faces)
        # Edge (0, 2) in face 1 -> opposite is (2, 0) which is in face 0
        assert (0, 2) in adj
        assert adj[(0, 2)] == 1
        assert (2, 0) in adj
        assert adj[(2, 0)] == 0


class TestTraceStreamline:
    def test_trace_produces_points(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        grads, _, valid = compute_face_gradients(vertices, faces, lat)
        adj = build_face_adjacency(faces)

        bary = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        path = trace_streamline(
            start_face=0, start_bary=bary,
            faces=faces, vertices=vertices,
            face_gradients=grads, face_adjacency=adj,
            max_steps=50, step_size=1.0,
        )
        assert len(path) >= 2

    def test_streamline_follows_gradient(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        grads, _, valid = compute_face_gradients(vertices, faces, lat)
        adj = build_face_adjacency(faces)

        bary = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        path = trace_streamline(
            start_face=0, start_bary=bary,
            faces=faces, vertices=vertices,
            face_gradients=grads, face_adjacency=adj,
            max_steps=100, step_size=0.5,
        )
        # The X coordinate should generally increase along the path
        # (gradient is in +X direction)
        if len(path) >= 3:
            x_coords = [p[0] for p in path]
            assert x_coords[-1] > x_coords[0]


class TestGenerateStreamlineSeeds:
    def test_seeds_from_valid_mesh(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        grads, _, valid = compute_face_gradients(vertices, faces, lat)
        seeds = generate_streamline_seeds(
            vertices, faces, lat, grads, valid, target_count=10,
        )
        assert len(seeds) > 0
        for face_idx, bary in seeds:
            assert 0 <= face_idx < len(faces)
            assert np.allclose(bary.sum(), 1.0, atol=1e-6)

    def test_no_seeds_from_nan_mesh(self, larger_mesh):
        vertices, faces, normals, _ = larger_mesh
        lat = np.full(len(vertices), np.nan)
        grads, _, valid = compute_face_gradients(vertices, faces, lat)
        seeds = generate_streamline_seeds(
            vertices, faces, lat, grads, valid, target_count=10,
        )
        assert len(seeds) == 0


class TestTraceAllStreamlines:
    def test_traces_produce_polylines(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        streamlines = trace_all_streamlines(
            vertices, faces, lat, normals, target_count=5, max_steps=50,
        )
        assert len(streamlines) > 0
        for sl in streamlines:
            assert len(sl) >= 3
            for pt in sl:
                assert pt.shape == (3,)


class TestComputeAnimatedDashes:
    def test_frame_count(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        streamlines = trace_all_streamlines(
            vertices, faces, lat, normals, target_count=5,
        )
        n_frames = 10
        frames = compute_animated_dashes(streamlines, n_frames=n_frames)
        assert len(frames) == n_frames

    def test_dashes_are_valid_segments(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        streamlines = trace_all_streamlines(
            vertices, faces, lat, normals, target_count=5,
        )
        frames = compute_animated_dashes(streamlines, n_frames=5)
        for frame_dashes in frames:
            for start, end in frame_dashes:
                assert start.shape == (3,)
                assert end.shape == (3,)
                assert np.linalg.norm(end - start) > 0

    def test_empty_streamlines(self):
        frames = compute_animated_dashes([], n_frames=5)
        assert len(frames) == 5
        assert all(len(f) == 0 for f in frames)

    def test_dashes_move_between_frames(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        streamlines = trace_all_streamlines(
            vertices, faces, lat, normals, target_count=5,
        )
        frames = compute_animated_dashes(streamlines, n_frames=10)
        if len(frames[0]) > 0 and len(frames[1]) > 0:
            # Dashes should be at different positions in different frames
            pos_0 = np.array([d[0] for d in frames[0]])
            pos_1 = np.array([d[0] for d in frames[1]])
            # Not identical (they've moved)
            if len(pos_0) == len(pos_1):
                assert not np.allclose(pos_0, pos_1, atol=1e-6)


class TestSmoothStreamline:
    def test_short_path_unchanged(self):
        """Paths with < 4 points should be returned as-is."""
        path = [np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([2, 0, 0.0])]
        result = _smooth_streamline(path)
        assert len(result) == 3
        for a, b in zip(path, result):
            assert np.allclose(a, b)

    def test_smoothing_reduces_sharp_corners(self):
        """A zigzag path should become smoother after filtering."""
        path = [np.array([float(i), float(i % 2) * 2.0, 0.0]) for i in range(20)]
        smoothed = _smooth_streamline(path, sigma=2.0)
        assert len(smoothed) == len(path)
        # Measure total angular variation: smoothed should have less
        def _total_angle(pts):
            total = 0.0
            for i in range(1, len(pts) - 1):
                d1 = pts[i] - pts[i - 1]
                d2 = pts[i + 1] - pts[i]
                n1 = np.linalg.norm(d1)
                n2 = np.linalg.norm(d2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_a = np.clip(np.dot(d1, d2) / (n1 * n2), -1, 1)
                    total += np.arccos(cos_a)
            return total
        orig_pts = [np.array(p) for p in path]
        assert _total_angle(smoothed) < _total_angle(orig_pts)

    def test_preserves_general_direction(self):
        """Smoothing should not reverse the path direction."""
        path = [np.array([float(i), 0.5 * np.sin(i), 0.0]) for i in range(30)]
        smoothed = _smooth_streamline(path, sigma=2.0)
        # Start-to-end direction should be similar
        orig_dir = path[-1] - path[0]
        smooth_dir = smoothed[-1] - smoothed[0]
        assert np.dot(orig_dir, smooth_dir) > 0  # same general direction


class TestComputeDashSpeedFactors:
    def test_output_structure(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        streamlines = trace_all_streamlines(
            vertices, faces, lat, normals, target_count=5,
        )
        frames = compute_animated_dashes(streamlines, n_frames=5)
        grads, centers, _ = compute_face_gradients(vertices, faces, lat)
        speed_factors = compute_dash_speed_factors(frames, grads, centers)
        assert len(speed_factors) == len(frames)
        for sf, dashes in zip(speed_factors, frames):
            assert len(sf) == len(dashes)

    def test_values_in_unit_range(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        streamlines = trace_all_streamlines(
            vertices, faces, lat, normals, target_count=5,
        )
        frames = compute_animated_dashes(streamlines, n_frames=5)
        grads, centers, _ = compute_face_gradients(vertices, faces, lat)
        speed_factors = compute_dash_speed_factors(frames, grads, centers)
        for sf in speed_factors:
            for val in sf:
                assert 0.0 <= val <= 1.0 + 1e-10

    def test_empty_frames(self):
        grads = np.array([[1, 0, 0]], dtype=np.float64)
        centers = np.array([[0, 0, 0]], dtype=np.float64)
        speed_factors = compute_dash_speed_factors([[], []], grads, centers)
        assert speed_factors == [[], []]


class TestAssessStreamlineQuality:
    def test_good_streamlines_pass(self, larger_mesh):
        vertices, faces, normals, lat = larger_mesh
        streamlines = trace_all_streamlines(
            vertices, faces, lat, normals, target_count=20, max_steps=100,
        )
        bbox_diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
        # Small test mesh only produces a few streamlines, so lower min_count
        ok, reason = assess_streamline_quality(
            streamlines, bbox_diag, min_count=2,
        )
        assert ok, f"Expected pass but got: {reason}"
        assert reason == ""

    def test_too_few_streamlines_fail(self):
        # Only 3 streamlines — below default min_count=10
        streamlines = [
            [np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([2, 0, 0.0])]
            for _ in range(3)
        ]
        ok, reason = assess_streamline_quality(streamlines, mesh_bbox_diag=100.0)
        assert not ok
        assert "3 usable" in reason

    def test_short_streamlines_fail(self):
        # 20 streamlines but very short (length ~0.02 vs bbox diag 1000)
        streamlines = [
            [np.array([0, 0, 0.0]), np.array([0.01, 0, 0.0]), np.array([0.02, 0, 0.0])]
            for _ in range(20)
        ]
        ok, reason = assess_streamline_quality(streamlines, mesh_bbox_diag=1000.0)
        assert not ok
        assert "median" in reason

    def test_empty_streamlines_fail(self):
        ok, reason = assess_streamline_quality([], mesh_bbox_diag=100.0)
        assert not ok

    def test_degenerate_bbox_fail(self):
        streamlines = [
            [np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([2, 0, 0.0])]
            for _ in range(20)
        ]
        ok, reason = assess_streamline_quality(streamlines, mesh_bbox_diag=0.0)
        assert not ok
        assert "degenerate" in reason

    def test_custom_thresholds(self):
        # 5 streamlines, length ~2 each, bbox 100 → ratio 2%
        streamlines = [
            [np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([2, 0, 0.0])]
            for _ in range(5)
        ]
        # With relaxed thresholds, should pass
        ok, _ = assess_streamline_quality(
            streamlines, mesh_bbox_diag=100.0,
            min_count=3, min_median_length_ratio=0.01,
        )
        assert ok
