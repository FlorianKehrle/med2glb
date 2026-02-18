"""GLB builder: create glTF scenes with PBR materials via pygltflib."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pygltflib

from med2glb.core.types import MeshData


def build_glb(meshes: list[MeshData], output_path: Path) -> None:
    """Build a GLB file from one or more meshes with PBR materials."""
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

    for i, mesh_data in enumerate(meshes):
        node_idx = _add_mesh_to_gltf(gltf, mesh_data, all_binary_data, i)
        gltf.scenes[0].nodes.append(node_idx)

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
    alpha_mode = pygltflib.BLEND if mat.alpha < 1.0 else pygltflib.OPAQUE
    gltf.materials.append(
        pygltflib.Material(
            name=mat.name or mesh_data.structure_name,
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=[
                    mat.base_color[0],
                    mat.base_color[1],
                    mat.base_color[2],
                    mat.alpha,
                ],
                metallicFactor=mat.metallic,
                roughnessFactor=mat.roughness,
            ),
            alphaMode=alpha_mode,
            doubleSided=True,
        )
    )

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
