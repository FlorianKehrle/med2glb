"""Morph target animation: build animated GLB with morph targets via pygltflib."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pygltflib

from dicom2glb.core.types import AnimatedResult, MeshData
from dicom2glb.glb.builder import _add_mesh_to_gltf, _pad_to_4


def build_animated_glb(result: AnimatedResult, output_path: Path) -> None:
    """Build an animated GLB with morph target animation."""
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        animations=[],
    )

    all_binary_data = bytearray()

    # Add base meshes with morph targets
    for mesh_idx, mesh_data in enumerate(result.base_meshes):
        morph_targets = result.morph_targets[mesh_idx] if mesh_idx < len(result.morph_targets) else []

        node_idx = _add_animated_mesh_to_gltf(
            gltf, mesh_data, morph_targets, all_binary_data, mesh_idx
        )
        gltf.scenes[0].nodes.append(node_idx)

    # Add animation (targeting morph weights)
    if result.morph_targets and any(len(mt) > 0 for mt in result.morph_targets):
        _add_morph_animation(
            gltf,
            result.frame_times,
            result.morph_targets,
            all_binary_data,
            len(gltf.scenes[0].nodes),
        )

    # Set buffer
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(all_binary_data)))
    gltf.set_binary_blob(bytes(all_binary_data))

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))


def _add_animated_mesh_to_gltf(
    gltf: pygltflib.GLTF2,
    mesh_data: MeshData,
    morph_targets: list[np.ndarray],
    binary_data: bytearray,
    index: int,
) -> int:
    """Add a mesh with morph targets to the glTF document."""
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

    # Add base vertex positions
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

    pos_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=pos_bv_idx,
            componentType=pygltflib.FLOAT,
            count=len(vertices),
            type=pygltflib.VEC3,
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist(),
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

    # Add morph targets
    morph_target_accessors = []
    for mt_idx, displacement in enumerate(morph_targets):
        disp = displacement.astype(np.float32)
        disp_data = disp.tobytes()
        disp_offset = len(binary_data)
        binary_data.extend(disp_data)
        _pad_to_4(binary_data)

        disp_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=disp_offset,
                byteLength=len(disp_data),
            )
        )
        disp_acc_idx = len(gltf.accessors)
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=disp_bv_idx,
                componentType=pygltflib.FLOAT,
                count=len(disp),
                type=pygltflib.VEC3,
                max=disp.max(axis=0).tolist(),
                min=disp.min(axis=0).tolist(),
            )
        )
        morph_target_accessors.append(disp_acc_idx)

    # Create primitive
    attributes = pygltflib.Attributes(POSITION=pos_acc_idx)
    if normal_acc_idx is not None:
        attributes.NORMAL = normal_acc_idx

    # Build morph target list for primitive
    targets = []
    for acc_idx in morph_target_accessors:
        targets.append(pygltflib.Attributes(POSITION=acc_idx))

    primitive = pygltflib.Primitive(
        attributes=attributes,
        indices=idx_acc_idx,
        material=material_idx,
        targets=targets if targets else None,
    )

    # Create mesh with morph target weights (all zero initially)
    mesh_idx = len(gltf.meshes)
    weights = [0.0] * len(morph_targets) if morph_targets else None
    gltf.meshes.append(
        pygltflib.Mesh(
            name=mesh_data.structure_name,
            primitives=[primitive],
            weights=weights,
        )
    )

    # Create node
    node_idx = len(gltf.nodes)
    gltf.nodes.append(
        pygltflib.Node(
            name=mesh_data.structure_name,
            mesh=mesh_idx,
        )
    )

    return node_idx


def _add_morph_animation(
    gltf: pygltflib.GLTF2,
    frame_times: list[float],
    morph_targets_per_mesh: list[list[np.ndarray]],
    binary_data: bytearray,
    n_nodes: int,
) -> None:
    """Add morph weight animation to the glTF document."""
    n_morph_targets = max(len(mt) for mt in morph_targets_per_mesh) if morph_targets_per_mesh else 0
    if n_morph_targets == 0:
        return

    channels = []
    samplers = []

    for node_idx in range(n_nodes):
        if node_idx >= len(morph_targets_per_mesh):
            continue
        mesh_morph_targets = morph_targets_per_mesh[node_idx]
        if not mesh_morph_targets:
            continue

        n_mt = len(mesh_morph_targets)

        # Build keyframe times: one for base (t=0) + one per morph target
        # For morph animation: at each keyframe, activate one morph target
        n_keyframes = n_mt + 1  # base + morph targets
        keyframe_times = np.array(frame_times[:n_keyframes], dtype=np.float32)
        if len(keyframe_times) < n_keyframes:
            # Pad with evenly spaced times
            dt = frame_times[-1] / n_keyframes if frame_times else 1.0 / n_keyframes
            keyframe_times = np.array([i * dt for i in range(n_keyframes)], dtype=np.float32)

        # Write keyframe times
        time_data = keyframe_times.tobytes()
        time_offset = len(binary_data)
        binary_data.extend(time_data)
        _pad_to_4(binary_data)

        time_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=time_offset,
                byteLength=len(time_data),
            )
        )
        time_acc_idx = len(gltf.accessors)
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=time_bv_idx,
                componentType=pygltflib.FLOAT,
                count=len(keyframe_times),
                type=pygltflib.SCALAR,
                max=[float(keyframe_times.max())],
                min=[float(keyframe_times.min())],
            )
        )

        # Build morph weights: at each keyframe, one target is 1.0, rest are 0.0
        # Frame 0: base (all weights 0) → Frame 1: target 0 active → Frame 2: target 1 active...
        weights = np.zeros((n_keyframes, n_mt), dtype=np.float32)
        for i in range(n_mt):
            weights[i + 1, i] = 1.0  # Activate target i at keyframe i+1

        weight_data = weights.tobytes()
        weight_offset = len(binary_data)
        binary_data.extend(weight_data)
        _pad_to_4(binary_data)

        weight_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=weight_offset,
                byteLength=len(weight_data),
            )
        )
        weight_acc_idx = len(gltf.accessors)
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=weight_bv_idx,
                componentType=pygltflib.FLOAT,
                count=n_keyframes * n_mt,
                type=pygltflib.SCALAR,
            )
        )

        sampler_idx = len(samplers)
        samplers.append(
            pygltflib.AnimationSampler(
                input=time_acc_idx,
                output=weight_acc_idx,
                interpolation=pygltflib.LINEAR,
            )
        )

        channels.append(
            pygltflib.AnimationChannel(
                sampler=sampler_idx,
                target=pygltflib.AnimationChannelTarget(
                    node=node_idx,
                    path="weights",
                ),
            )
        )

    if channels:
        gltf.animations.append(
            pygltflib.Animation(
                name="cardiac_cycle",
                channels=channels,
                samplers=samplers,
            )
        )
