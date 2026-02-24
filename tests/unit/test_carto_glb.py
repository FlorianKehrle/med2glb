"""Unit tests for CARTO GLB builder (static + animated)."""

from __future__ import annotations

import numpy as np
import pygltflib
import pytest

from med2glb.core.types import MaterialConfig, MeshData
from med2glb.glb.builder import build_glb


class TestGlbVertexColors:
    def test_build_glb_with_vertex_colors(self, synthetic_mesh, tmp_path):
        """Test that COLOR_0 attribute is written when vertex_colors is set."""
        n_verts = len(synthetic_mesh.vertices)
        synthetic_mesh.vertex_colors = np.random.rand(n_verts, 4).astype(np.float32)
        synthetic_mesh.vertex_colors[:, 3] = 1.0  # opaque

        output = tmp_path / "colored.glb"
        build_glb([synthetic_mesh], output)

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        prim = gltf.meshes[0].primitives[0]
        assert prim.attributes.COLOR_0 is not None

        # Material should have white base color
        mat = gltf.materials[0]
        bcf = mat.pbrMetallicRoughness.baseColorFactor
        assert bcf == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_build_glb_without_vertex_colors(self, synthetic_mesh, tmp_path):
        """Existing meshes without vertex_colors should work unchanged."""
        output = tmp_path / "no_colors.glb"
        build_glb([synthetic_mesh], output)

        gltf = pygltflib.GLTF2.load(str(output))
        prim = gltf.meshes[0].primitives[0]
        assert prim.attributes.COLOR_0 is None

        # Material should use original base color
        mat = gltf.materials[0]
        bcf = mat.pbrMetallicRoughness.baseColorFactor
        assert bcf[0] == pytest.approx(0.8)  # red

    def test_vertex_colors_with_transparency(self, synthetic_mesh, tmp_path):
        """GLB should use BLEND mode when vertex colors have alpha < 1."""
        n_verts = len(synthetic_mesh.vertices)
        synthetic_mesh.vertex_colors = np.ones((n_verts, 4), dtype=np.float32)
        synthetic_mesh.vertex_colors[:, 3] = 0.5  # semi-transparent

        output = tmp_path / "transparent_vc.glb"
        build_glb([synthetic_mesh], output)

        gltf = pygltflib.GLTF2.load(str(output))
        mat = gltf.materials[0]
        assert mat.alphaMode == pygltflib.BLEND


class TestCartoAnimatedGlb:
    def test_build_animated_glb(self, synthetic_carto_mesh, synthetic_carto_points, tmp_path):
        """Test animated GLB creation from CARTO data."""
        from med2glb.glb.carto_builder import build_carto_animated_glb
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data, map_points_to_vertices

        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, synthetic_carto_points, coloring="lat", subdivide=0
        )
        lat_values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="lat"
        )
        # Filter to active vertices (all active in synthetic mesh)
        active_mask = synthetic_carto_mesh.group_ids != -1000000
        active_lat = lat_values[active_mask]

        output = tmp_path / "animated_carto.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=5, loop_duration_s=1.0,
            target_faces=100000,  # no decimation needed for tiny mesh
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))

        # Should have 5 frames (meshes, nodes)
        assert len(gltf.meshes) == 5
        assert len(gltf.nodes) == 5

        # First node visible, rest hidden
        assert gltf.nodes[0].scale == [1.0, 1.0, 1.0]
        assert gltf.nodes[1].scale == [0.0, 0.0, 0.0]

        # Should have animation
        assert len(gltf.animations) == 1
        assert gltf.animations[0].name == "excitation_ring"

        # Each mesh should have COLOR_0
        for mesh in gltf.meshes:
            prim = mesh.primitives[0]
            assert prim.attributes.COLOR_0 is not None

        # CARTO meshes should use KHR_materials_unlit
        assert "KHR_materials_unlit" in (gltf.extensionsUsed or [])
        for mat in gltf.materials:
            assert "KHR_materials_unlit" in (mat.extensions or {})

    def test_single_frame_no_division_error(self, synthetic_carto_mesh, synthetic_carto_points, tmp_path):
        """n_frames=1 must not raise ZeroDivisionError."""
        from med2glb.glb.carto_builder import build_carto_animated_glb
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data, map_points_to_vertices

        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, synthetic_carto_points, coloring="lat", subdivide=0
        )
        lat_values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="lat"
        )
        active_mask = synthetic_carto_mesh.group_ids != -1000000
        active_lat = lat_values[active_mask]

        output = tmp_path / "single_frame.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=1, loop_duration_s=1.0,
            target_faces=100000,
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.meshes) == 1
        assert len(gltf.nodes) == 1

    def test_all_nan_lat_falls_back_to_static(self, synthetic_carto_mesh, tmp_path):
        """When all LAT values are NaN, should produce a static GLB."""
        from med2glb.glb.carto_builder import build_carto_animated_glb
        from med2glb.io.carto_mapper import carto_mesh_to_mesh_data

        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, None, coloring="lat"
        )
        lat_values = np.full(len(mesh_data.vertices), np.nan)

        output = tmp_path / "fallback.glb"
        build_carto_animated_glb(mesh_data, lat_values, output, n_frames=5)

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        # Static fallback: 1 mesh, no animation
        assert len(gltf.meshes) == 1
        assert len(gltf.animations) == 0
