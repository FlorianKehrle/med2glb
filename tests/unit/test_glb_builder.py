"""Unit tests for GLB builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import pytest

from med2glb.core.types import MaterialConfig, MeshData
from med2glb.glb.builder import build_glb


def test_build_glb_single_mesh(synthetic_mesh, tmp_path):
    output = tmp_path / "test.glb"
    build_glb([synthetic_mesh], output)

    assert output.exists()
    assert output.stat().st_size > 0

    # Validate it's a valid glTF
    gltf = pygltflib.GLTF2.load(str(output))
    assert len(gltf.meshes) == 1
    assert len(gltf.materials) == 1
    assert len(gltf.nodes) == 1


def test_build_glb_multiple_meshes(synthetic_mesh, tmp_path):
    mesh2 = MeshData(
        vertices=synthetic_mesh.vertices + 2.0,
        faces=synthetic_mesh.faces.copy(),
        structure_name="second_mesh",
        material=MaterialConfig(base_color=(0.2, 0.4, 0.8), alpha=0.7),
    )

    output = tmp_path / "multi.glb"
    build_glb([synthetic_mesh, mesh2], output)

    gltf = pygltflib.GLTF2.load(str(output))
    assert len(gltf.meshes) == 2
    assert len(gltf.materials) == 2


def test_build_glb_with_transparency(synthetic_mesh, tmp_path):
    synthetic_mesh.material = MaterialConfig(
        base_color=(0.8, 0.2, 0.2), alpha=0.5
    )
    output = tmp_path / "transparent.glb"
    build_glb([synthetic_mesh], output)

    gltf = pygltflib.GLTF2.load(str(output))
    mat = gltf.materials[0]
    assert mat.alphaMode == pygltflib.BLEND


def test_build_glb_mask_mode_for_zero_alpha(synthetic_mesh, tmp_path):
    """Vertex colors with alpha=0.0 should use MASK mode, not BLEND."""
    n_verts = len(synthetic_mesh.vertices)
    synthetic_mesh.vertex_colors = np.ones((n_verts, 4), dtype=np.float32)
    # Some vertices fully transparent (NaN colormap pattern)
    synthetic_mesh.vertex_colors[0, 3] = 0.0

    output = tmp_path / "mask_mode.glb"
    build_glb([synthetic_mesh], output)

    gltf = pygltflib.GLTF2.load(str(output))
    mat = gltf.materials[0]
    assert mat.alphaMode == pygltflib.MASK
    assert mat.alphaCutoff == pytest.approx(0.5)


def test_build_glb_with_normals(synthetic_mesh, tmp_path):
    synthetic_mesh.normals = np.random.randn(
        len(synthetic_mesh.vertices), 3
    ).astype(np.float32)

    output = tmp_path / "normals.glb"
    build_glb([synthetic_mesh], output)

    gltf = pygltflib.GLTF2.load(str(output))
    prim = gltf.meshes[0].primitives[0]
    assert prim.attributes.NORMAL is not None
