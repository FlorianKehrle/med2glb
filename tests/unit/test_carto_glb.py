"""Unit tests for CARTO GLB builder (static + animated)."""

from __future__ import annotations

import numpy as np
import pygltflib
import pytest

from med2glb.core.types import MaterialConfig, MeshData
from med2glb.glb.builder import build_glb


class TestGlbVertexColors:
    def test_build_glb_with_vertex_colors(self, synthetic_mesh, tmp_path):
        """Test that vertex colors are baked into a texture (TEXCOORD_0 + baseColorTexture)."""
        n_verts = len(synthetic_mesh.vertices)
        synthetic_mesh.vertex_colors = np.random.rand(n_verts, 4).astype(np.float32)
        synthetic_mesh.vertex_colors[:, 3] = 1.0  # opaque

        output = tmp_path / "colored.glb"
        build_glb([synthetic_mesh], output)

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        prim = gltf.meshes[0].primitives[0]
        # Vertex colors are now baked into a texture, not stored as COLOR_0
        assert prim.attributes.TEXCOORD_0 is not None
        assert prim.attributes.COLOR_0 is None

        # Material should have baseColorTexture
        mat = gltf.materials[0]
        assert mat.pbrMetallicRoughness.baseColorTexture is not None
        bcf = mat.pbrMetallicRoughness.baseColorFactor
        assert bcf == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_build_glb_without_vertex_colors(self, synthetic_mesh, tmp_path):
        """Existing meshes without vertex_colors should work unchanged."""
        output = tmp_path / "no_colors.glb"
        build_glb([synthetic_mesh], output)

        gltf = pygltflib.GLTF2.load(str(output))
        prim = gltf.meshes[0].primitives[0]
        assert prim.attributes.COLOR_0 is None
        assert prim.attributes.TEXCOORD_0 is None

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
        """Test animated GLB creation — single mesh + COLOR_0 morph targets."""
        from med2glb.glb.carto_builder import build_carto_animated_glb
        from med2glb.io.carto_mapper import (
            carto_mesh_to_mesh_data,
            map_points_to_vertices,
            interpolate_sparse_values,
        )
        from scipy.spatial import KDTree

        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, synthetic_carto_points, coloring="lat", subdivide=0
        )
        lat_values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="lat"
        )
        lat_values = interpolate_sparse_values(synthetic_carto_mesh, lat_values)
        tree = KDTree(synthetic_carto_mesh.vertices)
        _, idx = tree.query(mesh_data.vertices)
        active_lat = lat_values[idx]

        output = tmp_path / "animated_carto.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=5, loop_duration_s=1.0,
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))

        # Single mesh (not 30 copies) + 1 root node
        assert len(gltf.meshes) == 1
        assert len(gltf.nodes) == 2  # 1 mesh node + 1 root

        # Root node: mm→m + 10x AR scale
        root = gltf.nodes[-1]
        assert root.scale == [0.01, 0.01, 0.01]
        assert gltf.scenes[0].nodes == [len(gltf.nodes) - 1]

        # Single mesh has COLOR_0 base attribute and 5 morph targets
        prim = gltf.meshes[0].primitives[0]
        assert prim.attributes.COLOR_0 is not None
        assert len(prim.targets) == 5
        for target in prim.targets:
            # pygltflib returns morph targets as plain dicts when loaded
            color_accessor = target.get("COLOR_0") if isinstance(target, dict) else target.COLOR_0
            assert color_accessor is not None

        # No textures (vertex color morph targets, no emissive textures)
        assert len(gltf.images) == 0
        assert len(gltf.textures) == 0

        # Has morph weight animation
        assert len(gltf.animations) == 1

    def test_single_frame_no_division_error(self, synthetic_carto_mesh, synthetic_carto_points, tmp_path):
        """n_frames=1 must not raise ZeroDivisionError."""
        from med2glb.glb.carto_builder import build_carto_animated_glb
        from med2glb.io.carto_mapper import (
            carto_mesh_to_mesh_data, map_points_to_vertices,
            interpolate_sparse_values,
        )
        from scipy.spatial import KDTree

        mesh_data = carto_mesh_to_mesh_data(
            synthetic_carto_mesh, synthetic_carto_points, coloring="lat", subdivide=0
        )
        lat_values = map_points_to_vertices(
            synthetic_carto_mesh, synthetic_carto_points, field="lat"
        )
        lat_values = interpolate_sparse_values(synthetic_carto_mesh, lat_values)
        tree = KDTree(synthetic_carto_mesh.vertices)
        _, idx = tree.query(mesh_data.vertices)
        active_lat = lat_values[idx]

        output = tmp_path / "single_frame.glb"
        build_carto_animated_glb(
            mesh_data, active_lat, output,
            n_frames=1, loop_duration_s=1.0,
        )

        assert output.exists()
        gltf = pygltflib.GLTF2.load(str(output))
        assert len(gltf.meshes) == 1
        assert len(gltf.nodes) == 2  # 1 frame + 1 root node

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
