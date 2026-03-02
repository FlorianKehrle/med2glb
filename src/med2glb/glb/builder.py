"""GLB builder: create glTF scenes with PBR materials via pygltflib."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pygltflib

from med2glb.core.types import MeshData


def build_glb(
    meshes: list[MeshData],
    output_path: Path,
    extra_meshes: list[MeshData] | None = None,
    source_units: str = "m",
) -> None:
    """Build a GLB file from one or more meshes with PBR materials.

    Args:
        meshes: Primary meshes to include in the GLB.
        output_path: Where to write the .glb file.
        extra_meshes: Additional meshes (e.g. vector arrows) added as separate nodes.
        source_units: Unit of vertex coordinates. ``"mm"`` adds a root node
            with scale 0.001 so the GLB is in glTF-standard metres.
    """
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

    all_binary_data = bytearray()
    child_nodes: list[int] = []

    for i, mesh_data in enumerate(meshes):
        node_idx = _add_mesh_to_gltf(gltf, mesh_data, all_binary_data, i)
        child_nodes.append(node_idx)

    if extra_meshes:
        for i, mesh_data in enumerate(extra_meshes):
            node_idx = _add_mesh_to_gltf(
                gltf, mesh_data, all_binary_data, len(meshes) + i,
            )
            child_nodes.append(node_idx)

    # Wrap in a root node that converts mm → m when needed
    if source_units == "mm":
        root_idx = len(gltf.nodes)
        gltf.nodes.append(pygltflib.Node(
            name="root",
            children=child_nodes,
            scale=[0.001, 0.001, 0.001],
        ))
        gltf.scenes[0].nodes = [root_idx]
    else:
        gltf.scenes[0].nodes = child_nodes

    # Set buffer
    gltf.buffers.append(
        pygltflib.Buffer(byteLength=len(all_binary_data))
    )
    gltf.set_binary_blob(bytes(all_binary_data))

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))


def _add_mesh_to_gltf(
    gltf: pygltflib.GLTF2,
    mesh_data: MeshData,
    binary_data: bytearray,
    index: int,
) -> int:
    """Add a single mesh with material to the glTF document. Returns node index."""
    mat = mesh_data.material

    # Create material
    material_idx = len(gltf.materials)
    has_vertex_colors = mesh_data.vertex_colors is not None
    # When vertex colors are present, use white base so COLOR_0 drives appearance
    alpha_cutoff = None
    if has_vertex_colors:
        base_color_factor = [1.0, 1.0, 1.0, 1.0]
        min_alpha = float(mesh_data.vertex_colors[:, 3].min())
        if min_alpha > 0.99:
            alpha_mode = pygltflib.OPAQUE
        elif min_alpha < 0.01:
            # Fully transparent vertices → MASK is faster than BLEND on HoloLens
            alpha_mode = pygltflib.MASK
            alpha_cutoff = 0.5
        else:
            alpha_mode = pygltflib.BLEND
    else:
        base_color_factor = [
            mat.base_color[0],
            mat.base_color[1],
            mat.base_color[2],
            mat.alpha,
        ]
        alpha_mode = pygltflib.BLEND if mat.alpha < 1.0 else pygltflib.OPAQUE
    mat_kwargs: dict = dict(
        name=mat.name or mesh_data.structure_name,
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=base_color_factor,
            metallicFactor=mat.metallic,
            roughnessFactor=mat.roughness,
        ),
        alphaMode=alpha_mode,
        doubleSided=True,
    )
    if alpha_cutoff is not None:
        mat_kwargs["alphaCutoff"] = alpha_cutoff
    if mat.unlit:
        mat_kwargs["extensions"] = {"KHR_materials_unlit": {}}
    material = pygltflib.Material(**mat_kwargs)
    gltf.materials.append(material)
    if mat.unlit:
        if not hasattr(gltf, "extensionsUsed") or gltf.extensionsUsed is None:
            gltf.extensionsUsed = []
        if "KHR_materials_unlit" not in gltf.extensionsUsed:
            gltf.extensionsUsed.append("KHR_materials_unlit")

    # Add vertex position data
    vertices = mesh_data.vertices.astype(np.float32)
    pos_data = vertices.tobytes()
    pos_offset = len(binary_data)
    binary_data.extend(pos_data)
    _pad_to_4(binary_data)

    pos_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=pos_offset,
            byteLength=len(pos_data),
            target=pygltflib.ARRAY_BUFFER,
        )
    )

    pos_min = vertices.min(axis=0).tolist()
    pos_max = vertices.max(axis=0).tolist()
    pos_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=pos_bv_idx,
            componentType=pygltflib.FLOAT,
            count=len(vertices),
            type=pygltflib.VEC3,
            max=pos_max,
            min=pos_min,
        )
    )

    # Add normals if available
    normal_acc_idx = None
    if mesh_data.normals is not None:
        normals = mesh_data.normals.astype(np.float32)
        norm_data = normals.tobytes()
        norm_offset = len(binary_data)
        binary_data.extend(norm_data)
        _pad_to_4(binary_data)

        norm_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=norm_offset,
                byteLength=len(norm_data),
                target=pygltflib.ARRAY_BUFFER,
            )
        )

        normal_acc_idx = len(gltf.accessors)
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=norm_bv_idx,
                componentType=pygltflib.FLOAT,
                count=len(normals),
                type=pygltflib.VEC3,
            )
        )

    # Add vertex colors (COLOR_0) if available — stored as uint8 normalized
    # to reduce file size by 75% vs float32 (8-bit precision is sufficient
    # for clinical colormaps).
    color_acc_idx = None
    if has_vertex_colors:
        colors_u8 = np.clip(mesh_data.vertex_colors * 255.0, 0, 255).astype(np.uint8)
        color_data = colors_u8.tobytes()
        color_offset = len(binary_data)
        binary_data.extend(color_data)
        _pad_to_4(binary_data)

        color_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=color_offset,
                byteLength=len(color_data),
                target=pygltflib.ARRAY_BUFFER,
            )
        )

        color_acc_idx = len(gltf.accessors)
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=color_bv_idx,
                componentType=pygltflib.UNSIGNED_BYTE,
                count=len(colors_u8),
                type=pygltflib.VEC4,
                normalized=True,
            )
        )

    # Add face indices
    faces = mesh_data.faces.astype(np.uint32)
    idx_data = faces.tobytes()
    idx_offset = len(binary_data)
    binary_data.extend(idx_data)
    _pad_to_4(binary_data)

    idx_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=idx_offset,
            byteLength=len(idx_data),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )

    idx_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=idx_bv_idx,
            componentType=pygltflib.UNSIGNED_INT,
            count=faces.size,
            type=pygltflib.SCALAR,
            max=[int(faces.max())],
            min=[int(faces.min())],
        )
    )

    # Create primitive
    attributes = pygltflib.Attributes(POSITION=pos_acc_idx)
    if normal_acc_idx is not None:
        attributes.NORMAL = normal_acc_idx
    if color_acc_idx is not None:
        attributes.COLOR_0 = color_acc_idx

    primitive = pygltflib.Primitive(
        attributes=attributes,
        indices=idx_acc_idx,
        material=material_idx,
    )

    # Create mesh and node
    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(
        pygltflib.Mesh(
            name=mesh_data.structure_name,
            primitives=[primitive],
        )
    )

    node_idx = len(gltf.nodes)
    gltf.nodes.append(
        pygltflib.Node(
            name=mesh_data.structure_name,
            mesh=mesh_idx,
        )
    )

    return node_idx


def write_accessor(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    data_array: np.ndarray,
    target: int | None,
    comp_type: int,
    acc_type: str,
    with_minmax: bool = False,
    normalized: bool = False,
) -> int:
    """Write array data as a bufferView + accessor. Returns accessor index."""
    raw = data_array.tobytes()
    off = len(binary_data)
    binary_data.extend(raw)
    _pad_to_4(binary_data)
    bv_idx = len(gltf.bufferViews)
    bv_kwargs: dict = dict(buffer=0, byteOffset=off, byteLength=len(raw))
    if target is not None:
        bv_kwargs["target"] = target
    gltf.bufferViews.append(pygltflib.BufferView(**bv_kwargs))
    acc_idx = len(gltf.accessors)
    kwargs: dict = dict(
        bufferView=bv_idx,
        componentType=comp_type,
        count=len(data_array),
        type=acc_type,
    )
    if normalized:
        kwargs["normalized"] = True
    if with_minmax:
        mx = data_array.max(axis=0).tolist()
        mn = data_array.min(axis=0).tolist()
        if not isinstance(mx, list):
            mx = [mx]
            mn = [mn]
        kwargs["max"] = mx
        kwargs["min"] = mn
    gltf.accessors.append(pygltflib.Accessor(**kwargs))
    return acc_idx


def _pad_to_4(data: bytearray) -> None:
    """Pad binary data to 4-byte alignment."""
    remainder = len(data) % 4
    if remainder:
        data.extend(b"\x00" * (4 - remainder))
