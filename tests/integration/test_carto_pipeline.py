"""Integration tests for the CARTO conversion pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import pytest

CARTO_DATA = Path(__file__).parent.parent.parent / "CARTO_Example_Data"
CARTO_OLD = CARTO_DATA / "older CARTO versions" / "Study 1" / "Export_Study-1-01_16_2015-15-31-41"
CARTO_V71 = CARTO_DATA / "Version_7.1.80.33" / "Study 1" / "Export_Study"
CARTO_V72 = CARTO_DATA / "Version_7.2.10.423" / "Export_Study-1-01_09_2023-20-30-09"


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


@pytest.mark.skipif(not CARTO_OLD.exists(), reason="CARTO old data not available")
class TestRealCartoOld:
    def test_load_and_convert(self, tmp_path):
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.builder import build_glb

        study = load_carto_study(CARTO_OLD)
        assert len(study.meshes) >= 1
        assert study.version == "4.0"

        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)
        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat")

        output = tmp_path / "old_carto.glb"
        build_glb([mesh_data], output)
        assert output.exists()
        assert output.stat().st_size > 100


@pytest.mark.skipif(not CARTO_V71.exists(), reason="CARTO v7.1 data not available")
class TestRealCartoV71:
    def test_load_and_convert(self, tmp_path):
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.builder import build_glb

        study = load_carto_study(CARTO_V71)
        assert study.version == "5.0"

        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)
        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat")

        output = tmp_path / "v71_carto.glb"
        build_glb([mesh_data], output)
        assert output.exists()
        # v7.1 had vertex color issues — verify COLOR_0 is present
        gltf = pygltflib.GLTF2.load(str(output))
        prim = gltf.meshes[0].primitives[0]
        assert prim.attributes.COLOR_0 is not None


@pytest.mark.skipif(not CARTO_V72.exists(), reason="CARTO v7.2 data not available")
class TestRealCartoV72:
    def test_load_and_convert_all_meshes(self, tmp_path):
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.builder import build_glb

        study = load_carto_study(CARTO_V72)
        assert study.version == "6.0"
        assert len(study.meshes) >= 2

        for mesh in study.meshes:
            points = study.points.get(mesh.structure_name)
            mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat")

            output = tmp_path / f"{mesh.structure_name}.glb"
            build_glb([mesh_data], output)
            assert output.exists()

    def test_bipolar_coloring(self, tmp_path):
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data
        from med2glb.glb.builder import build_glb

        study = load_carto_study(CARTO_V72)
        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)
        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="bipolar")

        output = tmp_path / "v72_bipolar.glb"
        build_glb([mesh_data], output)
        assert output.exists()

    def test_animated_lat(self, tmp_path):
        from med2glb.io.carto_reader import load_carto_study
        from med2glb.io.carto_mapper import (
            carto_mesh_to_mesh_data,
            map_points_to_vertices,
            interpolate_sparse_values,
        )
        from med2glb.glb.carto_builder import build_carto_animated_glb

        study = load_carto_study(CARTO_V72)
        mesh = study.meshes[0]
        points = study.points.get(mesh.structure_name)

        mesh_data = carto_mesh_to_mesh_data(mesh, points, coloring="lat", subdivide=0)
        lat_values = map_points_to_vertices(mesh, points, field="lat")
        lat_values = interpolate_sparse_values(mesh, lat_values)
        active_mask = mesh.group_ids != -1000000
        active_lat = lat_values[active_mask]

        output = tmp_path / "v72_animated.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=10, target_faces=10000,
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.animations) == 1
        assert len(gltf.meshes) == 10
