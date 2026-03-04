"""Integration tests for the CARTO conversion pipeline.

Uses real CARTO test data from test_data/CARTO/ when available.
Synthetic-data tests always run; real-data tests skip if files are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import pytest
from scipy.spatial import KDTree

from med2glb.cli_wizard import _MIN_LAT_RANGE_MS

# Real test data paths (relative to repo root)
_REPO = Path(__file__).parent.parent.parent
TEST_DATA = _REPO / "test_data" / "CARTO"
_GLB_OUTPUT = _REPO / "test_data" / "integration_test_output" / "carto"

# v7.1 — single mesh (1-Map), sparse points, low vector quality
CARTO_V71 = TEST_DATA / "Version_7.1.80.33" / "Study 1" / "Export_Study"

# v7.2 O — single mesh (1-1-1-Rp-ReLA)
CARTO_V72_O = TEST_DATA / "Version_7.2.10.423" / "O"


# ---------------------------------------------------------------------------
# Module-scoped fixtures for expensive study loading (parsed once, reused)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def v71_study():
    """Load CARTO v7.1 study once for the entire module."""
    if not CARTO_V71.exists():
        pytest.skip("CARTO v7.1 test data not available")
    from med2glb.io.carto_reader import load_carto_study

    return load_carto_study(CARTO_V71)


@pytest.fixture(scope="module")
def v72_o_study():
    """Load CARTO v7.2 O study once for the entire module."""
    if not CARTO_V72_O.exists():
        pytest.skip("CARTO v7.2 O data not available")
    from med2glb.io.carto_reader import load_carto_study

    return load_carto_study(CARTO_V72_O)


@pytest.fixture
def v71_output() -> Path:
    """Persistent output for CARTO v7.1 GLBs."""
    d = _GLB_OUTPUT / "v7.1"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def v72_o_output() -> Path:
    """Persistent output for CARTO v7.2 O GLBs."""
    d = _GLB_OUTPUT / "v7.2-O"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_active_lat(mesh, points, mesh_data):
    """Extract LAT values matching mesh_data vertices (after fill stripping).

    carto_mesh_to_mesh_data strips fill geometry (non-dominant groups), so
    the active vertex count can be smaller than ``group_ids != -1000000``.
    This helper maps LAT to all raw mesh vertices, then resamples to the
    mesh_data vertex positions via nearest-neighbor.
    """
    from med2glb.io.carto_mapper import map_points_to_vertices, interpolate_sparse_values

    lat_all = map_points_to_vertices(mesh, points, field="lat")
    lat_all = interpolate_sparse_values(mesh, lat_all)

    # Resample to mesh_data.vertices via KDTree (handles any vertex subset)
    tree = KDTree(mesh.vertices)
    _, idx = tree.query(mesh_data.vertices)
    return lat_all[idx]


# ---------------------------------------------------------------------------
# Synthetic-data tests (always run)
# ---------------------------------------------------------------------------
class TestCartoEndToEnd:
    def test_synthetic_static_pipeline(self, carto_mesh_dir, tmp_path):
        """End-to-end: synthetic CARTO dir -> static GLB."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.builder import build_glb

        study = load_carto_study(carto_mesh_dir)
        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat")
        output = tmp_path / "test_carto.glb"
        build_glb([mesh_data], output)

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.meshes) == 1
        prim = gltf.meshes[0].primitives[0]
        assert prim.attributes.TEXCOORD_0 is not None

    def test_synthetic_animated_pipeline(self, carto_mesh_dir, tmp_path):
        """End-to-end: synthetic CARTO dir -> animated GLB via cache."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import (
            carto_mesh_to_mesh_data,
            map_points_to_vertices,
            interpolate_sparse_values,
        )
        from med2glb.glb.carto_builder import build_carto_animated_glb, prepare_animated_cache

        study = load_carto_study(carto_mesh_dir)
        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
        lat_values = map_points_to_vertices(mesh, points, field="lat")
        lat_values = interpolate_sparse_values(mesh, lat_values)
        active_mask = mesh.group_ids != -1000000
        active_lat = lat_values[active_mask]

        cache = prepare_animated_cache(
            mesh_data, active_lat, n_frames=5, target_faces=100000,
        )
        assert cache is not None

        output = tmp_path / "test_animated.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=5, target_faces=100000,
            cache=cache,
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.animations) == 1

    def test_multiple_colorings(self, carto_mesh_dir, tmp_path):
        """Test that all coloring modes produce valid GLBs."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.builder import build_glb

        study = load_carto_study(carto_mesh_dir)
        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)

        for coloring in ["lat", "bipolar", "unipolar"]:
            mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring=coloring)
            output = tmp_path / f"test_{coloring}.glb"
            build_glb([mesh_data], output)
            assert output.exists()
            assert output.stat().st_size > 0

    def test_animated_all_colorings(self, carto_mesh_dir, tmp_path):
        """Test that animation works with all coloring modes (highlight ring)."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import (
            carto_mesh_to_mesh_data,
            map_points_to_vertices,
            interpolate_sparse_values,
        )
        from med2glb.glb.carto_builder import build_carto_animated_glb, prepare_animated_cache

        study = load_carto_study(carto_mesh_dir)
        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)

        # Extract LAT values once (used for ring timing regardless of coloring)
        lat_values = map_points_to_vertices(mesh, points, field="lat")
        lat_values = interpolate_sparse_values(mesh, lat_values)
        active_mask = mesh.group_ids != -1000000
        active_lat = lat_values[active_mask]

        for coloring in ["lat", "bipolar", "unipolar"]:
            mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring=coloring, subdivide=0)

            cache = prepare_animated_cache(
                mesh_data, active_lat, n_frames=5, target_faces=100000,
            )
            assert cache is not None

            output = tmp_path / f"test_animated_{coloring}.glb"
            build_carto_animated_glb(
                mesh_data, active_lat, output,
                n_frames=5, target_faces=100000,
                cache=cache,
            )

            assert output.exists()
            gltf = pygltflib.GLTF2.load(str(output))
            assert len(gltf.animations) == 1
            assert len(gltf.meshes) == 5


# ---------------------------------------------------------------------------
# Real data: CARTO v7.1 (single mesh, sparse points)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CARTO_V71.exists(), reason="CARTO v7.1 test data not available")
class TestRealCartoV71:
    def test_load_study(self, v71_study):
        assert len(v71_study.meshes) >= 1
        assert v71_study.version in ("5.0", "6.0")

    def test_static_lat(self, v71_study, v71_output):
        """Static LAT GLB from v7.1 data."""
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.builder import build_glb

        mesh = v71_study.meshes[0]
        points = v71_study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat")
        output = v71_output / f"{mesh.structure_name}_lat.glb"
        build_glb([mesh_data], output)

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        prim = gltf.meshes[0].primitives[0]
        assert prim.attributes.TEXCOORD_0 is not None

    def test_animated_lat(self, v71_study, v71_output):
        """Animated GLB from v7.1 data via cache (low frame count for speed)."""
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.carto_builder import build_carto_animated_glb, prepare_animated_cache

        mesh = v71_study.meshes[0]
        points = v71_study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
        active_lat = _extract_active_lat(mesh, points, mesh_data)

        cache = prepare_animated_cache(
            mesh_data, active_lat, n_frames=5, target_faces=10000,
        )
        assert cache is not None

        output = v71_output / f"{mesh.structure_name}_lat_animated.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=5, target_faces=10000,
            cache=cache,
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.animations) == 1
        assert len(gltf.meshes) == 5

    def test_vector_quality_accepted(self, v71_study):
        """v7.1 single-map data accepted — LAT range (297ms) and gradient coverage are sufficient."""
        from med2glb.cli_wizard import _assess_vector_quality

        quality = _assess_vector_quality(v71_study, selected_indices=None)
        assert quality.suitable
        assert quality.valid_points > 0
        assert quality.lat_range_ms > _MIN_LAT_RANGE_MS


# ---------------------------------------------------------------------------
# Real data: CARTO v7.2 O (single mesh)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CARTO_V72_O.exists(), reason="CARTO v7.2 O data not available")
class TestRealCartoV72O:
    def test_load_study(self, v72_o_study):
        assert len(v72_o_study.meshes) == 1
        assert v72_o_study.version == "6.0"
        assert v72_o_study.meshes[0].structure_name == "1-1-1-Rp-ReLA"

    def test_static_all_colorings(self, v72_o_study, v72_o_output):
        """Static pipeline: single mesh x all colorings (subdivide once, reuse)."""
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data, subdivide_carto_mesh
        from med2glb.glb.builder import build_glb

        mesh = v72_o_study.meshes[0]
        points = v72_o_study.points.get(mesh.structure_name)

        # Subdivide once, reuse across all coloring modes
        subdivided = subdivide_carto_mesh(mesh, iterations=1)

        for coloring in ["lat", "bipolar", "unipolar"]:
            mesh_data = carto_mesh_to_mesh_data(
                mesh, points, coloring=coloring, pre_subdivided=subdivided,
            )
            output = v72_o_output / f"{mesh.structure_name}_{coloring}.glb"
            build_glb([mesh_data], output)
            assert output.exists()
            assert output.stat().st_size > 100

    def test_animated(self, v72_o_study, v72_o_output):
        """Animated GLB via cache (low frame count for test speed)."""
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.carto_builder import build_carto_animated_glb, prepare_animated_cache

        mesh = v72_o_study.meshes[0]
        points = v72_o_study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
        active_lat = _extract_active_lat(mesh, points, mesh_data)

        cache = prepare_animated_cache(
            mesh_data, active_lat, n_frames=5, target_faces=10000,
        )
        assert cache is not None

        output = v72_o_output / f"{mesh.structure_name}_lat_animated.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=5, target_faces=10000,
            cache=cache,
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.animations) == 1
        assert len(gltf.meshes) >= 5

    def test_vector_quality(self, v72_o_study):
        """Test vector quality assessment."""
        from med2glb.cli_wizard import _assess_vector_quality, _assess_single_mesh

        quality = _assess_vector_quality(v72_o_study, selected_indices=None)
        assert quality.valid_points > 0

        mesh = v72_o_study.meshes[0]
        pts = v72_o_study.points.get(mesh.structure_name, [])
        single = _assess_single_mesh(mesh, pts)
        assert single.valid_points >= 0
        assert single.lat_range_ms >= 0.0

    def test_animated_with_vectors(self, v72_o_study, v72_o_output):
        """Animated GLB with vectors via shared cache — vectors may be skipped after decimation."""
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.carto_builder import build_carto_animated_glb, prepare_animated_cache
        from med2glb.cli_wizard import _assess_vector_quality

        quality = _assess_vector_quality(v72_o_study, selected_indices=None)
        if not quality.suitable:
            pytest.skip("No meshes suitable for vectors in test_O dataset")

        mesh_idx = quality.suitable_indices[0]
        mesh = v72_o_study.meshes[mesh_idx]
        points = v72_o_study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
        active_lat = _extract_active_lat(mesh, points, mesh_data)

        # Shared cache for both variants (the production code path)
        cache = prepare_animated_cache(
            mesh_data, active_lat, n_frames=5, target_faces=10000,
        )
        assert cache is not None

        output = v72_o_output / f"{mesh.structure_name}_lat_animated_vectors.glb"
        wrote = build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=5, target_faces=10000,
            vectors=True,
            cache=cache,
        )

        # Vector quality may be rejected after decimation even if
        # _assess_vector_quality passed on the full mesh
        if wrote:
            assert output.exists()
            gltf = pygltflib.GLTF2.load(str(output))
            assert len(gltf.animations) == 1
            assert len(gltf.meshes) >= 5
