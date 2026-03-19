"""Unit tests for COLOR_0 morph target support in animation.py — T064."""

from __future__ import annotations

import numpy as np
import pygltflib
import pytest

from med2glb.core.types import MeshData
from med2glb.glb.animation import _add_animated_mesh_to_gltf, _add_morph_animation


@pytest.fixture
def simple_mesh_data() -> MeshData:
    """A tiny 4-vertex, 2-face mesh for testing."""
    return MeshData(
        structure_name="test_color_morph",
        vertices=np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        ], dtype=np.float64),
        faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
        normals=np.array([
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        ], dtype=np.float64),
    )


class TestColorMorphTargets:
    """Test that COLOR_0 morph targets are correctly added to glTF."""

    def test_color_only_morph_targets(self, simple_mesh_data):
        """Pure color animation — no positional displacement."""
        gltf = pygltflib.GLTF2()
        gltf.scenes = [pygltflib.Scene(nodes=[])]
        gltf.scene = 0
        gltf.materials = [pygltflib.Material(
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(),
        )]
        binary = bytearray()

        n_verts = 4
        n_frames = 3
        base_colors = np.tile([0.5, 0.0, 0.0, 1.0], (n_verts, 1)).astype(np.float32)

        # Color morph targets: each frame has different per-vertex colors
        color_mts = [
            np.tile([1.0, 0.0, 0.0, 1.0], (n_verts, 1)).astype(np.float32),
            np.tile([0.0, 1.0, 0.0, 1.0], (n_verts, 1)).astype(np.float32),
            np.tile([0.0, 0.0, 1.0, 1.0], (n_verts, 1)).astype(np.float32),
        ]

        # No positional morph targets
        pos_mts = [np.zeros((n_verts, 3), dtype=np.float32)] * n_frames

        node_idx, _ = _add_animated_mesh_to_gltf(
            gltf, simple_mesh_data, pos_mts, binary, 0,
            color_morph_targets=color_mts,
            base_vertex_colors=base_colors,
        )

        mesh = gltf.meshes[0]
        prim = mesh.primitives[0]

        # Base attribute should have COLOR_0
        assert prim.attributes.COLOR_0 is not None
        base_color_acc = gltf.accessors[prim.attributes.COLOR_0]
        assert base_color_acc.type == pygltflib.VEC4
        assert base_color_acc.count == n_verts

        # Each morph target should have both POSITION and COLOR_0
        assert len(prim.targets) == n_frames
        for target in prim.targets:
            assert target.POSITION is not None
            assert target.COLOR_0 is not None
            color_acc = gltf.accessors[target.COLOR_0]
            assert color_acc.type == pygltflib.VEC4
            assert color_acc.count == n_verts

        # Weights should match number of targets
        assert len(mesh.weights) == n_frames

    def test_color_morph_without_positional(self, simple_mesh_data):
        """Color-only morph targets (empty positional list)."""
        gltf = pygltflib.GLTF2()
        gltf.scenes = [pygltflib.Scene(nodes=[])]
        gltf.scene = 0
        gltf.materials = [pygltflib.Material(
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(),
        )]
        binary = bytearray()

        n_verts = 4
        n_frames = 2
        base_colors = np.full((n_verts, 4), [0.3, 0.3, 0.3, 1.0], dtype=np.float32)
        color_mts = [
            np.full((n_verts, 4), [1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.full((n_verts, 4), [0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ]

        node_idx, _ = _add_animated_mesh_to_gltf(
            gltf, simple_mesh_data, [], binary, 0,
            color_morph_targets=color_mts,
            base_vertex_colors=base_colors,
        )

        mesh = gltf.meshes[0]
        prim = mesh.primitives[0]
        assert len(prim.targets) == n_frames
        for target in prim.targets:
            # No POSITION since empty positional list
            assert target.POSITION is None
            assert target.COLOR_0 is not None

    def test_backward_compat_no_colors(self, simple_mesh_data):
        """Without color args, function works exactly as before (positional only)."""
        gltf = pygltflib.GLTF2()
        gltf.scenes = [pygltflib.Scene(nodes=[])]
        gltf.scene = 0
        gltf.materials = [pygltflib.Material(
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(),
        )]
        binary = bytearray()

        n_verts = 4
        pos_mts = [np.random.rand(n_verts, 3).astype(np.float32) * 0.01 for _ in range(2)]

        node_idx, _ = _add_animated_mesh_to_gltf(
            gltf, simple_mesh_data, pos_mts, binary, 0,
        )

        mesh = gltf.meshes[0]
        prim = mesh.primitives[0]
        assert prim.attributes.COLOR_0 is None
        assert len(prim.targets) == 2
        for target in prim.targets:
            assert target.POSITION is not None
            assert target.COLOR_0 is None


class TestColorMorphAnimation:
    """Test full animation pipeline with COLOR_0 morph targets."""

    def test_animated_glb_roundtrip(self, simple_mesh_data, tmp_path):
        """Build a complete animated GLB with color morph targets and verify."""
        gltf = pygltflib.GLTF2()
        gltf.scenes = [pygltflib.Scene(nodes=[])]
        gltf.scene = 0
        gltf.materials = [pygltflib.Material(
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(),
        )]
        binary = bytearray()

        n_verts = 4
        n_frames = 5
        base_colors = np.full((n_verts, 4), [0.5, 0.0, 0.0, 1.0], dtype=np.float32)

        # Color morph targets: gradient from base to white
        color_mts = []
        for f in range(n_frames):
            t = f / n_frames
            c = np.full((n_verts, 4), [0.5 + 0.5 * t, t, t, 1.0], dtype=np.float32)
            color_mts.append(c)

        pos_mts = [np.zeros((n_verts, 3), dtype=np.float32)] * n_frames

        node_idx, _ = _add_animated_mesh_to_gltf(
            gltf, simple_mesh_data, pos_mts, binary, 0,
            color_morph_targets=color_mts,
            base_vertex_colors=base_colors,
        )
        gltf.scenes[0].nodes.append(node_idx)

        # Add morph weight animation — 1 node
        frame_times = [i * 0.15 for i in range(n_frames)]
        _add_morph_animation(gltf, frame_times, [pos_mts], binary, 1)

        # Finalize and save
        gltf.buffers = [pygltflib.Buffer(byteLength=len(binary))]
        gltf.set_binary_blob(bytes(binary))
        out = tmp_path / "color_morph.glb"
        gltf.save(str(out))
        assert out.exists()

        # Reload and verify
        loaded = pygltflib.GLTF2.load(str(out))
        assert len(loaded.meshes) == 1
        prim = loaded.meshes[0].primitives[0]
        assert prim.attributes.COLOR_0 is not None
        assert len(prim.targets) == n_frames
        assert len(loaded.animations) == 1


class TestUv1Data:
    """Test TEXCOORD_1 (lat_norm) support in _add_animated_mesh_to_gltf."""

    def test_uv1_data_written_to_primitive(self, simple_mesh_data):
        """TEXCOORD_1 accessor is created and referenced when uv1_data is provided."""
        gltf = pygltflib.GLTF2()
        gltf.scenes = [pygltflib.Scene(nodes=[])]
        gltf.scene = 0
        binary = bytearray()

        n_verts = 4
        lat_norm = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float32)
        uv1 = np.column_stack([lat_norm, np.zeros(n_verts, dtype=np.float32)])

        node_idx, _ = _add_animated_mesh_to_gltf(
            gltf, simple_mesh_data,
            morph_targets=[],
            binary_data=binary,
            index=0,
            uv1_data=uv1,
        )

        prim = gltf.meshes[0].primitives[0]
        assert prim.attributes.TEXCOORD_0 is not None, "dummy TEXCOORD_0 required for glTFast UV loading"
        assert prim.attributes.TEXCOORD_1 is not None

    def test_uv1_data_absent_when_not_provided(self, simple_mesh_data):
        """TEXCOORD_0 and TEXCOORD_1 are absent when uv1_data is not passed (backward compat)."""
        gltf = pygltflib.GLTF2()
        gltf.scenes = [pygltflib.Scene(nodes=[])]
        gltf.scene = 0
        binary = bytearray()

        _add_animated_mesh_to_gltf(
            gltf, simple_mesh_data,
            morph_targets=[],
            binary_data=binary,
            index=0,
        )

        prim = gltf.meshes[0].primitives[0]
        assert prim.attributes.TEXCOORD_0 is None
        assert prim.attributes.TEXCOORD_1 is None

    def test_uv1_roundtrip(self, simple_mesh_data, tmp_path):
        """TEXCOORD_1 survives save/reload cycle."""
        gltf = pygltflib.GLTF2()
        gltf.scenes = [pygltflib.Scene(nodes=[])]
        gltf.scene = 0
        binary = bytearray()

        n_verts = 4
        lat_norm = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float32)
        uv1 = np.column_stack([lat_norm, np.zeros(n_verts, dtype=np.float32)])

        node_idx, _ = _add_animated_mesh_to_gltf(
            gltf, simple_mesh_data,
            morph_targets=[],
            binary_data=binary,
            index=0,
            uv1_data=uv1,
        )
        gltf.scenes[0].nodes.append(node_idx)
        gltf.buffers = [pygltflib.Buffer(byteLength=len(binary))]
        gltf.set_binary_blob(bytes(binary))
        out = tmp_path / "uv1_roundtrip.glb"
        gltf.save(str(out))

        loaded = pygltflib.GLTF2.load(str(out))
        prim = loaded.meshes[0].primitives[0]
        assert prim.attributes.TEXCOORD_0 is not None
        assert prim.attributes.TEXCOORD_1 is not None
