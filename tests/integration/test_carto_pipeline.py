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

# Real test data paths (relative to repo root)
_REPO = Path(__file__).parent.parent.parent
TEST_DATA = _REPO / "test_data" / "CARTO"

# v7.1 — single mesh (1-Map), sparse points, low vector quality
CARTO_V71 = TEST_DATA / "Version_7.1.80.33" / "Study 1" / "Export_Study"

# v7.2 test_B — 2 meshes (ReBS)
CARTO_V72_B = TEST_DATA / "Version_7.2.10.423" / "test_B"

# v7.2 test_O — 4 meshes (LA, RA, remaps) with XML study file
CARTO_V72_O = TEST_DATA / "Version_7.2.10.423" / "test_O"


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
        assert prim.attributes.COLOR_0 is not None

    def test_synthetic_animated_pipeline(self, carto_mesh_dir, tmp_path):
        """End-to-end: synthetic CARTO dir -> animated GLB."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import (
            carto_mesh_to_mesh_data,
            map_points_to_vertices,
            interpolate_sparse_values,
        )
        from med2glb.glb.carto_builder import build_carto_animated_glb

        study = load_carto_study(carto_mesh_dir)
        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
        lat_values = map_points_to_vertices(mesh, points, field="lat")
        lat_values = interpolate_sparse_values(mesh, lat_values)
        active_mask = mesh.group_ids != -1000000
        active_lat = lat_values[active_mask]

        output = tmp_path / "test_animated.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=5, target_faces=100000,
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
        from med2glb.glb.carto_builder import build_carto_animated_glb

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
            output = tmp_path / f"test_animated_{coloring}.glb"
            build_carto_animated_glb(
                mesh_data, active_lat, output,
                n_frames=5, target_faces=100000,
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
    def test_load_study(self):
        from med2glb.io.carto_reader import load_carto_study

        study = load_carto_study(CARTO_V71)
        assert len(study.meshes) >= 1
        assert study.version in ("5.0", "6.0")

    def test_static_lat(self, tmp_path):
        """Static LAT GLB from v7.1 data."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.builder import build_glb

        study = load_carto_study(CARTO_V71)
        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat")
        output = tmp_path / f"{mesh.structure_name}_lat.glb"
        build_glb([mesh_data], output)

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        prim = gltf.meshes[0].primitives[0]
        assert prim.attributes.COLOR_0 is not None

    def test_animated_lat(self, tmp_path):
        """Animated GLB from v7.1 data (low frame count for speed)."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.carto_builder import build_carto_animated_glb

        study = load_carto_study(CARTO_V71)
        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
        active_lat = _extract_active_lat(mesh, points, mesh_data)

        output = tmp_path / f"{mesh.structure_name}_lat_animated.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=5, target_faces=10000,
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.animations) == 1
        assert len(gltf.meshes) == 5

    def test_vector_quality_rejected(self):
        """v7.1 single-map data should be rejected for vectors (low density)."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.cli_wizard import _assess_vector_quality

        study = load_carto_study(CARTO_V71)
        quality = _assess_vector_quality(study, selected_indices=None)
        # Known low point density — vectors not suitable
        assert not quality.suitable


# ---------------------------------------------------------------------------
# Real data: CARTO v7.2 test_B (2 meshes)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CARTO_V72_B.exists(), reason="CARTO v7.2 test_B data not available")
class TestRealCartoV72B:
    def test_load_study(self):
        from med2glb.io.carto_reader import load_carto_study

        study = load_carto_study(CARTO_V72_B)
        assert len(study.meshes) == 2
        assert study.version == "6.0"

    def test_static_all_meshes(self, tmp_path):
        """Convert both meshes to static GLB with all colorings."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.builder import build_glb

        study = load_carto_study(CARTO_V72_B)

        for mesh in study.meshes:
            points = study.points.get(mesh.structure_name)
            for coloring in ["lat", "bipolar", "unipolar"]:
                mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring=coloring)
                output = tmp_path / f"{mesh.structure_name}_{coloring}.glb"
                build_glb([mesh_data], output)
                assert output.exists()
                assert output.stat().st_size > 100

    def test_animated_first_mesh(self, tmp_path):
        """Animated GLB from first mesh in test_B."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.carto_builder import build_carto_animated_glb

        study = load_carto_study(CARTO_V72_B)
        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
        active_lat = _extract_active_lat(mesh, points, mesh_data)

        output = tmp_path / f"{mesh.structure_name}_lat_animated.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=5, target_faces=10000,
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.animations) == 1

    def test_vector_quality_per_mesh(self):
        """Test per-mesh vector quality assessment on multi-mesh study."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.cli_wizard import _assess_vector_quality

        study = load_carto_study(CARTO_V72_B)
        quality = _assess_vector_quality(study, selected_indices=None)
        # Per-mesh assessment should return suitable_indices (possibly empty)
        assert quality.suitable_indices is not None or not quality.suitable


# ---------------------------------------------------------------------------
# Real data: CARTO v7.2 test_O (4 meshes with XML study)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CARTO_V72_O.exists(), reason="CARTO v7.2 test_O data not available")
class TestRealCartoV72O:
    def test_load_study(self):
        from med2glb.io.carto_reader import load_carto_study

        study = load_carto_study(CARTO_V72_O)
        assert len(study.meshes) == 2
        assert study.version == "6.0"

        # Verify all expected mesh names are present
        names = {m.structure_name for m in study.meshes}
        for expected in ["1-1-1-Rp-ReLA", "2-RA"]:
            assert expected in names, f"Expected mesh '{expected}' not found in {names}"

    def test_static_all_meshes_all_colorings(self, tmp_path):
        """Full static pipeline: all meshes x all colorings."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.builder import build_glb

        study = load_carto_study(CARTO_V72_O)

        for mesh in study.meshes:
            points = study.points.get(mesh.structure_name)
            for coloring in ["lat", "bipolar", "unipolar"]:
                mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring=coloring)
                output = tmp_path / f"{mesh.structure_name}_{coloring}.glb"
                build_glb([mesh_data], output)
                assert output.exists()
                assert output.stat().st_size > 100

    def test_animated_all_meshes(self, tmp_path):
        """Animated GLB for each mesh (low frame count for test speed)."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.carto_builder import build_carto_animated_glb

        study = load_carto_study(CARTO_V72_O)

        for mesh in study.meshes:
            points = study.points.get(mesh.structure_name)
            if not points:
                continue

            mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
            active_lat = _extract_active_lat(mesh, points, mesh_data)

            # Skip if no valid LAT data
            if np.all(np.isnan(active_lat)):
                continue

            output = tmp_path / f"{mesh.structure_name}_lat_animated.glb"
            build_carto_animated_glb(
                mesh_data, active_lat, output,
                n_frames=5, target_faces=10000,
            )

            assert output.exists()
            gltf = pygltflib.GLTF2.load(str(output))
            assert len(gltf.animations) == 1
            assert len(gltf.meshes) >= 5  # 5 wavefront frames + possible arrow frames

    def test_vector_quality_per_mesh(self):
        """Test per-mesh vector assessment on multi-mesh study."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.cli_wizard import _assess_vector_quality, _assess_single_mesh

        study = load_carto_study(CARTO_V72_O)

        # Overall assessment
        quality = _assess_vector_quality(study, selected_indices=None)
        assert quality.valid_points > 0

        # Per-mesh assessment
        for i, mesh in enumerate(study.meshes):
            pts = study.points.get(mesh.structure_name, [])
            single = _assess_single_mesh(mesh, pts)
            # Each mesh should have some diagnostic info
            assert single.valid_points >= 0
            assert single.point_density >= 0.0

    def test_animated_with_vectors(self, tmp_path):
        """Animated GLB with vectors on a suitable mesh."""
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.carto_builder import build_carto_animated_glb
        from med2glb.cli_wizard import _assess_vector_quality

        study = load_carto_study(CARTO_V72_O)

        quality = _assess_vector_quality(study, selected_indices=None)
        if not quality.suitable:
            pytest.skip("No meshes suitable for vectors in test_O dataset")

        # Pick the first suitable mesh
        mesh_idx = quality.suitable_indices[0]
        mesh = study.meshes[mesh_idx]
        points = study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
        active_lat = _extract_active_lat(mesh, points, mesh_data)

        output = tmp_path / f"{mesh.structure_name}_lat_animated_vectors.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=5, target_faces=10000,
            vectors=True,
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.animations) == 1
        # Should have wavefront frames + arrow frames
        assert len(gltf.meshes) >= 5
