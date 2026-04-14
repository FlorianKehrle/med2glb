"""EML/SCAR overlay node for CARTO animated GLBs.

Embeds a transparent overlay mesh as a named child node inside the animated
GLB.  Only EML, ExtEML, and SCAR flagged vertices are visible (per-vertex
alpha); unflagged vertices are fully transparent so the LAT animation shows
through.

The overlay node uses ``scale = [1.001, 1.001, 1.001]`` — 0.1% larger than
the heart mesh — to push the surface slightly outward and prevent Z-fighting
on HoloLens.

Unity / VeldtAR side: ``ModelData.cs`` detects the ``"eml_"`` node name prefix
and excludes the overlay from the explodable subpart list (same mechanism used
for ``"ablation_"`` nodes).  The material's ``alphaMode="BLEND"`` is handled
automatically by the existing ``Loader.cs`` transparent path:
``_ALPHABLEND_ON → SetStandardAlphaMode(3) → _ALPHAPREMULTIPLY_ON``.
"""

from __future__ import annotations

import logging

import numpy as np
import pygltflib

from med2glb.core.types import MeshData
from med2glb.glb.builder import _center_vertices, _pad_to_4, write_accessor

logger = logging.getLogger("med2glb")

_EML_MATERIAL_NAME = "carto_eml"

# Scale relative to the heart mesh: 0.1% outward prevents Z-fighting on HL2.
_EML_SCALE: float = 1.001


def add_eml_overlay_node(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    eml_mesh_data: MeshData,
    centroid_offset: list[float] | None = None,
) -> list[int]:
    """Add an EML/SCAR overlay mesh as a child node in the animated GLB.

    The overlay uses the same geometry as the heart mesh with per-vertex alpha:

    * Normal (unflagged) vertices: α = 0 (transparent, LAT animation visible)
    * EML vertices: orange, α = 0.85
    * ExtEML vertices: yellow, α = 0.85
    * SCAR vertices: red, α = 0.95

    The node is named ``"eml_overlay"`` and placed at the mesh centroid with
    scale 1.001 to prevent Z-fighting.

    Args:
        gltf:             The glTF document being built.
        binary_data:      Binary buffer being assembled.
        eml_mesh_data:    MeshData with ``vertex_colors`` containing per-vertex
                          RGBA (with per-vertex alpha from ``eml_scar_colormap``).
        centroid_offset:  The same centroid subtracted from the animated mesh
                          vertices in ``carto_builder.py``.  When provided the
                          EML vertices are shifted by the same offset so the
                          overlay lands exactly on the heart mesh.  When
                          ``None``, ``_center_vertices`` is used as fallback.

    Returns:
        List containing the single EML overlay node index, or [] if skipped.
    """
    if eml_mesh_data.vertex_colors is None:
        return []

    # Use the same centroid as the animated mesh so both nodes sit at origin.
    # Passing centroid_offset avoids re-computing the centroid from potentially
    # slightly different floating-point vertices (subdivision can shift the
    # mean by a few ULPs) and guarantees perfect alignment.
    if centroid_offset is not None:
        vertices_centered = (
            eml_mesh_data.vertices.astype(np.float32)
            - np.array(centroid_offset, dtype=np.float32)
        )
    else:
        vertices_centered, _ = _center_vertices(
            eml_mesh_data.vertices.astype(np.float32)
        )

    pos_acc = write_accessor(
        gltf, binary_data, vertices_centered,
        pygltflib.ARRAY_BUFFER, pygltflib.FLOAT, pygltflib.VEC3,
        with_minmax=True,
    )
    norm_acc = write_accessor(
        gltf, binary_data, eml_mesh_data.normals.astype(np.float32),
        pygltflib.ARRAY_BUFFER, pygltflib.FLOAT, pygltflib.VEC3,
    )
    color_acc = write_accessor(
        gltf, binary_data, eml_mesh_data.vertex_colors.astype(np.float32),
        pygltflib.ARRAY_BUFFER, pygltflib.FLOAT, pygltflib.VEC4,
    )
    idx_acc = write_accessor(
        gltf, binary_data, eml_mesh_data.faces.astype(np.uint32).ravel(),
        pygltflib.ELEMENT_ARRAY_BUFFER, pygltflib.UNSIGNED_INT, pygltflib.SCALAR,
        with_minmax=True,
    )

    # Transparent material — alphaMode BLEND triggers the Loader.cs
    # transparent path automatically (no explicit Loader.cs change required).
    mat_idx = len(gltf.materials)
    gltf.materials.append(pygltflib.Material(
        name=_EML_MATERIAL_NAME,
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 0.9],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
        alphaMode=pygltflib.BLEND,
        doubleSided=True,
        extensions={"KHR_materials_unlit": {}},
    ))
    if not hasattr(gltf, "extensionsUsed") or gltf.extensionsUsed is None:
        gltf.extensionsUsed = []
    if "KHR_materials_unlit" not in gltf.extensionsUsed:
        gltf.extensionsUsed.append("KHR_materials_unlit")

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(pygltflib.Mesh(
        name="eml_overlay",
        primitives=[pygltflib.Primitive(
            attributes=pygltflib.Attributes(
                POSITION=pos_acc,
                NORMAL=norm_acc,
                COLOR_0=color_acc,
            ),
            indices=idx_acc,
            material=mat_idx,
        )],
    ))

    node_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(
        name="eml_overlay",
        mesh=mesh_idx,
        translation=[0.0, 0.0, 0.0],
        scale=[_EML_SCALE, _EML_SCALE, _EML_SCALE],
    ))

    n_flagged = int(np.sum(eml_mesh_data.vertex_colors[:, 3] > 0.01))
    logger.debug(
        "EML overlay: %d / %d vertices flagged (EML/ExtEML/SCAR)",
        n_flagged, len(eml_mesh_data.vertices),
    )
    return [node_idx]
