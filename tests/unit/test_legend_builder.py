"""Unit tests for legend_builder: color legend + metadata panel nodes."""

from __future__ import annotations

import numpy as np
import pygltflib
import pytest
from PIL import Image
import io

from med2glb.core.types import MaterialConfig, MeshData
from med2glb.glb.legend_builder import (
    add_legend_nodes,
    render_legend_wrap_image,
    render_info_image,
    _interpolate_color,
)


# ---------------------------------------------------------------------------
# Image rendering tests
# ---------------------------------------------------------------------------


def test_render_legend_wrap_image_returns_valid_png():
    png = render_legend_wrap_image("lat", (0.0, 100.0))
    assert isinstance(png, bytes)
    assert len(png) > 100
    img = Image.open(io.BytesIO(png))
    assert img.size == (1024, 512)


def test_render_legend_wrap_image_custom_size():
    png = render_legend_wrap_image("unipolar", (3.0, 10.0), width=256, height=128)
    img = Image.open(io.BytesIO(png))
    assert img.size == (256, 128)


def test_render_info_image_returns_valid_png():
    metadata = {
        "study_name": "Test Study",
        "structure": "LA",
        "coloring": "lat",
        "unit": "ms",
        "clamp_range": [-50, 100],
        "mapping_points": 1234,
        "carto_version": "CARTO 3 v7.2+ (file format v6.0)",
        "export_date": "2026-03-03",
    }
    png = render_info_image(metadata)
    assert isinstance(png, bytes)
    img = Image.open(io.BytesIO(png))
    assert img.size == (640, 448)
    assert img.mode == "RGBA"


def test_render_info_image_partial_metadata():
    """Missing fields should be skipped without error."""
    metadata = {"study_name": "Partial"}
    png = render_info_image(metadata)
    img = Image.open(io.BytesIO(png))
    assert img.size == (640, 448)


def test_render_info_image_empty_metadata():
    png = render_info_image({})
    img = Image.open(io.BytesIO(png))
    assert img.size == (640, 448)


# ---------------------------------------------------------------------------
# Color interpolation tests
# ---------------------------------------------------------------------------


def test_interpolate_color_at_stops():
    stops = [(0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)]
    assert _interpolate_color(0.0, stops) == (255, 0, 0)
    assert _interpolate_color(1.0, stops) == (0, 0, 255)


def test_interpolate_color_midpoint():
    stops = [(0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)]
    r, g, b = _interpolate_color(0.5, stops)
    assert 120 <= r <= 135  # ~127
    assert g == 0
    assert 120 <= b <= 135


def test_interpolate_color_clamps():
    stops = [(0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0)]
    assert _interpolate_color(-0.5, stops) == (255, 0, 0)
    assert _interpolate_color(1.5, stops) == (0, 255, 0)


# ---------------------------------------------------------------------------
# glTF node creation tests
# ---------------------------------------------------------------------------


def _make_test_gltf() -> tuple[pygltflib.GLTF2, bytearray]:
    """Create a minimal glTF for testing."""
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
    )
    return gltf, bytearray()


def test_add_legend_nodes_creates_nodes():
    gltf, binary_data = _make_test_gltf()
    vertices = np.array([
        [-10, -20, -5],
        [10, 20, 5],
        [0, 0, 0],
    ], dtype=np.float32)

    node_indices = add_legend_nodes(
        gltf, binary_data, vertices,
        coloring="lat",
        clamp_range=(-50.0, 100.0),
        centroid=[0.0, 0.0, 0.0],
    )

    # Should create at least the legend node (no info without metadata)
    assert len(node_indices) == 1
    assert gltf.nodes[node_indices[0]].name == "legend"


def test_add_legend_nodes_with_metadata_creates_two_nodes():
    gltf, binary_data = _make_test_gltf()
    vertices = np.array([
        [-10, -20, -5],
        [10, 20, 5],
        [0, 0, 0],
    ], dtype=np.float32)

    metadata = {
        "study_name": "My Study",
        "structure": "LA",
        "coloring": "bipolar",
        "unit": "mV",
        "clamp_range": [0.05, 1.5],
        "mapping_points": 500,
    }

    node_indices = add_legend_nodes(
        gltf, binary_data, vertices,
        coloring="bipolar",
        clamp_range=(0.05, 1.5),
        centroid=[0.0, 0.0, 0.0],
        metadata=metadata,
    )

    assert len(node_indices) == 2
    assert gltf.nodes[node_indices[0]].name == "legend"
    assert gltf.nodes[node_indices[1]].name == "info"


def test_add_legend_nodes_uses_unlit_material():
    gltf, binary_data = _make_test_gltf()
    vertices = np.array([[-5, -5, -5], [5, 5, 5]], dtype=np.float32)

    add_legend_nodes(
        gltf, binary_data, vertices,
        coloring="lat",
        clamp_range=(0.0, 100.0),
        centroid=[0.0, 0.0, 0.0],
    )

    # Material should be unlit
    assert len(gltf.materials) >= 1
    mat = gltf.materials[0]
    assert "KHR_materials_unlit" in (mat.extensions or {})
    assert mat.alphaMode == pygltflib.BLEND
    assert mat.doubleSided is False

    # Extension registered
    assert "KHR_materials_unlit" in (gltf.extensionsUsed or [])


def test_add_legend_nodes_sets_extras():
    gltf, binary_data = _make_test_gltf()
    vertices = np.array([[-5, -5, -5], [5, 5, 5]], dtype=np.float32)

    metadata = {"study_name": "Test", "coloring": "unipolar"}
    add_legend_nodes(
        gltf, binary_data, vertices,
        coloring="unipolar",
        clamp_range=(3.0, 10.0),
        centroid=[0.0, 0.0, 0.0],
        metadata=metadata,
    )

    assert gltf.extras is not None
    assert gltf.extras["coloring"] == "unipolar"
    assert gltf.extras["clamp_range"] == [3.0, 10.0]
    assert gltf.extras["study_name"] == "Test"


def test_add_legend_nodes_positions_right_of_mesh():
    gltf, binary_data = _make_test_gltf()
    # Mesh extends from x=-50 to x=50
    vertices = np.array([
        [-50, -30, 0],
        [50, 30, 0],
    ], dtype=np.float32)

    node_indices = add_legend_nodes(
        gltf, binary_data, vertices,
        coloring="lat",
        clamp_range=(0.0, 200.0),
        centroid=[0.0, 0.0, 0.0],
    )

    legend_node = gltf.nodes[node_indices[0]]
    # Legend should be to the right of the mesh (x > bbox_max_x = 50)
    assert legend_node.translation[0] > 50.0


def test_add_legend_nodes_cylinder_geometry():
    """Legend should be a 32-segment cylinder: 66 vertices, 192 indices."""
    gltf, binary_data = _make_test_gltf()
    vertices = np.array([[-10, -10, 0], [10, 10, 0]], dtype=np.float32)

    node_indices = add_legend_nodes(
        gltf, binary_data, vertices,
        coloring="lat",
        clamp_range=(0.0, 100.0),
        centroid=[0.0, 0.0, 0.0],
    )

    legend_node = gltf.nodes[node_indices[0]]
    mesh = gltf.meshes[legend_node.mesh]
    prim = mesh.primitives[0]

    assert prim.attributes.POSITION is not None
    assert prim.attributes.NORMAL is not None
    assert prim.attributes.TEXCOORD_0 is not None

    # 33 columns × 2 rows = 66 vertices
    pos_acc = gltf.accessors[prim.attributes.POSITION]
    assert pos_acc.count == 66

    # 32 segments × 2 triangles × 3 indices = 192 indices
    idx_acc = gltf.accessors[prim.indices]
    assert idx_acc.count == 192


def test_add_legend_nodes_info_panel_geometry():
    """Info panel should be a double-sided quad: 8 vertices, 12 indices."""
    gltf, binary_data = _make_test_gltf()
    vertices = np.array([[-10, -10, 0], [10, 10, 0]], dtype=np.float32)

    metadata = {"study_name": "Test", "coloring": "lat"}
    node_indices = add_legend_nodes(
        gltf, binary_data, vertices,
        coloring="lat",
        clamp_range=(0.0, 100.0),
        centroid=[0.0, 0.0, 0.0],
        metadata=metadata,
    )

    info_node = gltf.nodes[node_indices[1]]
    mesh = gltf.meshes[info_node.mesh]
    prim = mesh.primitives[0]

    assert prim.attributes.POSITION is not None
    assert prim.attributes.NORMAL is not None
    assert prim.attributes.TEXCOORD_0 is not None

    # 2 faces × 4 vertices = 8 vertices
    pos_acc = gltf.accessors[prim.attributes.POSITION]
    assert pos_acc.count == 8

    # 2 faces × 2 triangles × 3 indices = 12 indices
    idx_acc = gltf.accessors[prim.indices]
    assert idx_acc.count == 12


# ---------------------------------------------------------------------------
# Integration with build_glb
# ---------------------------------------------------------------------------


def test_build_glb_with_legend_info(synthetic_mesh, tmp_path):
    """build_glb with legend_info should add legend nodes."""
    from med2glb.glb.builder import build_glb

    output = tmp_path / "with_legend.glb"
    legend_info = {
        "coloring": "bipolar",
        "clamp_range": [0.05, 1.5],
        "metadata": {
            "study_name": "Integration Test",
            "structure": "LA",
            "coloring": "bipolar",
            "unit": "mV",
        },
    }

    build_glb(
        [synthetic_mesh], output, source_units="mm",
        legend_info=legend_info,
    )

    gltf = pygltflib.GLTF2.load(str(output))
    node_names = [n.name for n in gltf.nodes]
    assert "legend" in node_names
    assert "info" in node_names
    assert gltf.extras is not None
    assert gltf.extras["coloring"] == "bipolar"


def test_build_glb_without_legend_info_unchanged(synthetic_mesh, tmp_path):
    """build_glb without legend_info should not add extra nodes."""
    from med2glb.glb.builder import build_glb

    output = tmp_path / "no_legend.glb"
    build_glb([synthetic_mesh], output)

    gltf = pygltflib.GLTF2.load(str(output))
    node_names = [n.name for n in gltf.nodes]
    assert "legend" not in node_names
    assert "info" not in node_names
