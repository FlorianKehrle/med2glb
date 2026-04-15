"""EML/SCAR overlay node for CARTO animated GLBs.

Embeds a transparent overlay mesh as a named child node inside the animated
GLB.  Only triangles that contain at least one EML, ExtEML, or SCAR vertex
are included; unflagged-only triangles are stripped entirely so the LAT
animation underneath shows through without requiring per-vertex alpha.

The overlay node uses ``scale = [1.001, 1.001, 1.001]`` — 0.1% larger than
the heart mesh — to push the surface slightly outward and prevent Z-fighting
on HoloLens.

**HL2 transparency strategy** — per-vertex alpha (COLOR_0) is *not* used
because the Unity Standard shader (applied by ``ConvertGltfMaterialsToStandard``
in Loader.cs) ignores vertex colors entirely and glTFast shader variants
(``_ALPHABLEND_ON``) can be stripped from HL2 builds.  Instead three
separate mesh primitives are built — one per EML type — each with a
*uniform* ``baseColorFactor`` and ``alphaMode=BLEND``.
``ConvertGltfMaterialsToStandard`` copies ``baseColorFactor`` → ``_Color``
and promotes ``_ALPHABLEND_ON`` → ``_ALPHAPREMULTIPLY_ON`` (preserved in
HL2 builds), producing correct semi-transparent colored regions.

Unity / VeldtAR side: ``ModelData.cs`` detects the ``"eml_"`` node name
prefix and excludes the overlay from the explodable subpart list (same
mechanism used for ``"ablation_"`` nodes).
"""

from __future__ import annotations

import logging

import numpy as np
import pygltflib

from med2glb.core.types import MeshData
from med2glb.glb.builder import _center_vertices, write_accessor

logger = logging.getLogger("med2glb")

# Scale relative to the heart mesh: 0.1% outward prevents Z-fighting on HL2.
_EML_SCALE: float = 1.001

# Per-type material definitions: (material name, RGBA baseColorFactor)
# Colors match CARTO 8 display conventions (white=EML, magenta=ExtEML, gray=SCAR).
_EML_TYPE_DEFS: dict[int, tuple[str, list[float]]] = {
    1: ("eml_overlay_eml",    [1.0,  1.0,  1.0,  0.85]),  # EML    — white
    2: ("eml_overlay_exteml", [0.63, 0.16, 0.68, 0.85]),  # ExtEML — magenta/purple
    3: ("eml_overlay_scar",   [0.45, 0.45, 0.45, 0.90]),  # SCAR   — gray
}


def _vertex_types(vertex_colors: np.ndarray) -> np.ndarray:
    """Return per-vertex EML type label (0=normal, 1=EML, 2=ExtEML, 3=SCAR).

    The type is recovered from the baked RGBA produced by ``eml_scar_colormap``:
    * alpha ≈ 0     → normal (0)
    * alpha ≈ 0.90  → SCAR   (3, gray)
    * alpha ≈ 0.85, green < 0.30 → ExtEML (2, magenta/purple)
    * alpha ≈ 0.85, green ≥ 0.30 → EML    (1, white)
    """
    alpha = vertex_colors[:, 3]
    green = vertex_colors[:, 1]
    vtype = np.zeros(len(vertex_colors), dtype=np.int32)
    active = alpha > 0.01
    vtype[active & (alpha > 0.88)] = 3                      # SCAR (gray, α=0.90)
    vtype[active & (alpha <= 0.88) & (green < 0.30)] = 2   # ExtEML (magenta, low green)
    vtype[active & (alpha <= 0.88) & (green >= 0.30)] = 1  # EML (white, high green)
    return vtype


def add_eml_overlay_node(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    eml_mesh_data: MeshData,
    centroid_offset: list[float] | None = None,
) -> list[int]:
    """Add EML/SCAR overlay as a child node in the animated GLB.

    Creates up to three mesh primitives (EML / ExtEML / SCAR) each covering
    only the triangles that belong to that tissue type.  Each primitive has its
    own uniform-color material (no vertex colors) so ``ConvertGltfMaterialsToStandard``
    in Loader.cs produces correct semi-transparent Standard materials on HL2.

    Triangle assignment: a triangle is assigned to the highest-priority type
    present among its three vertices (SCAR=3 > ExtEML=2 > EML=1 > normal=0).
    Triangles where all three vertices are normal (type=0) are excluded.

    Args:
        gltf:             The glTF document being built.
        binary_data:      Binary buffer being assembled.
        eml_mesh_data:    MeshData whose ``vertex_colors`` was produced by
                          ``eml_scar_colormap`` (RGBA with per-vertex type).
        centroid_offset:  Centroid offset used for the animated mesh in
                          ``carto_builder.py``.  When provided the EML
                          vertices are shifted by the same amount so the
                          overlay lands exactly on the heart mesh.

    Returns:
        List with the single EML overlay node index, or ``[]`` if no
        EML/SCAR triangles are found.
    """
    if eml_mesh_data.vertex_colors is None:
        return []

    # Center vertices using the same offset as the animated mesh.
    if centroid_offset is not None:
        verts = (
            eml_mesh_data.vertices.astype(np.float32)
            - np.array(centroid_offset, dtype=np.float32)
        )
    else:
        verts, _ = _center_vertices(eml_mesh_data.vertices.astype(np.float32))

    normals = (
        eml_mesh_data.normals.astype(np.float32)
        if eml_mesh_data.normals is not None
        else None
    )
    faces = eml_mesh_data.faces  # (F, 3) int array

    # Per-vertex type label (0–3)
    vtype = _vertex_types(eml_mesh_data.vertex_colors)

    # Per-face dominant type: max of the 3 vertex types
    face_type = vtype[faces].max(axis=1)  # (F,) int

    primitives: list[pygltflib.Primitive] = []

    for etype, (mat_name, base_color) in _EML_TYPE_DEFS.items():
        mask = face_type == etype
        if not np.any(mask):
            continue

        type_faces = faces[mask]  # (K, 3)

        # Remap to a compact vertex buffer for this primitive
        unique_idx, remapped = np.unique(type_faces.ravel(), return_inverse=True)
        sub_verts = verts[unique_idx]
        sub_faces = remapped.reshape(-1, 3).astype(np.uint32)

        pos_acc = write_accessor(
            gltf, binary_data, sub_verts,
            pygltflib.ARRAY_BUFFER, pygltflib.FLOAT, pygltflib.VEC3,
            with_minmax=True,
        )
        idx_acc = write_accessor(
            gltf, binary_data, sub_faces.ravel(),
            pygltflib.ELEMENT_ARRAY_BUFFER, pygltflib.UNSIGNED_INT, pygltflib.SCALAR,
            with_minmax=True,
        )

        attrs = pygltflib.Attributes(POSITION=pos_acc)
        if normals is not None:
            norm_acc = write_accessor(
                gltf, binary_data, normals[unique_idx],
                pygltflib.ARRAY_BUFFER, pygltflib.FLOAT, pygltflib.VEC3,
            )
            attrs.NORMAL = norm_acc

        mat_idx = len(gltf.materials)
        gltf.materials.append(pygltflib.Material(
            name=mat_name,
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=base_color,
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            alphaMode=pygltflib.BLEND,
            doubleSided=True,
        ))

        primitives.append(pygltflib.Primitive(
            attributes=attrs,
            indices=idx_acc,
            material=mat_idx,
        ))

        logger.debug(
            "EML overlay type %d (%s): %d triangles, %d vertices",
            etype, mat_name, int(mask.sum()), len(unique_idx),
        )

    if not primitives:
        return []

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(pygltflib.Mesh(name="eml_overlay", primitives=primitives))

    node_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(
        name="eml_overlay",
        mesh=mesh_idx,
        translation=[0.0, 0.0, 0.0],
        scale=[_EML_SCALE, _EML_SCALE, _EML_SCALE],
    ))

    return [node_idx]
