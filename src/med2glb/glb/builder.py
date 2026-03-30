"""GLB builder: create glTF scenes with PBR materials via pygltflib."""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np
import pygltflib

from med2glb.core.types import MeshData

logger = logging.getLogger("med2glb")


def _center_vertices(vertices: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Subtract centroid from vertices, return (centered, translation)."""
    centroid = vertices.mean(axis=0).astype(np.float64)
    centered = (vertices - centroid).astype(np.float32)
    return centered, centroid.tolist()


def build_glb(
    meshes: list[MeshData],
    output_path: Path,
    extra_meshes: list[MeshData] | None = None,
    source_units: str = "m",
    legend_info: dict | None = None,
    lesion_points: list | None = None,
) -> None:
    """Build a GLB file from one or more meshes with PBR materials.

    Args:
        meshes: Primary meshes to include in the GLB.
        output_path: Where to write the .glb file.
        extra_meshes: Additional meshes (e.g. vector arrows) added as separate nodes.
        source_units: Unit of vertex coordinates. ``"mm"`` adds a root node
            with scale 0.001 so the GLB is in glTF-standard metres.
        legend_info: Optional dict with ``coloring``, ``clamp_range``, and
            ``metadata`` keys to add color legend + info panel nodes.
        lesion_points: Optional list of :class:`~med2glb.core.types.LesionPoint`
            objects to embed as orange-red sphere nodes named ``lesion_000…N``.
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

    centroid = None
    for i, mesh_data in enumerate(meshes):
        node_idx, mesh_centroid = _add_mesh_to_gltf(
            gltf, mesh_data, all_binary_data, i, centroid_override=centroid,
        )
        if centroid is None:
            centroid = mesh_centroid
        child_nodes.append(node_idx)

    if extra_meshes:
        for i, mesh_data in enumerate(extra_meshes):
            node_idx, _ = _add_mesh_to_gltf(
                gltf, mesh_data, all_binary_data, len(meshes) + i,
                centroid_override=centroid,
            )
            child_nodes.append(node_idx)

    if legend_info and centroid is not None:
        from med2glb.glb.legend_builder import add_legend_nodes
        centered_verts = (
            meshes[0].vertices - np.array(centroid, dtype=np.float32)
        ).astype(np.float32)
        legend_nodes = add_legend_nodes(
            gltf, all_binary_data, centered_verts,
            coloring=legend_info["coloring"],
            clamp_range=tuple(legend_info["clamp_range"]),
            centroid=[0.0, 0.0, 0.0],
            metadata=legend_info.get("metadata"),
        )
        child_nodes.extend(legend_nodes)

    if lesion_points and centroid is not None:
        from med2glb.glb.lesion_builder import add_lesion_nodes
        lesion_nodes = add_lesion_nodes(
            gltf, all_binary_data, lesion_points, centroid,
        )
        child_nodes.extend(lesion_nodes)

    # Wrap in a root node that converts mm → m when needed.
    # 10x real-world scale so cardiac structures (~12cm) render at ~120cm
    # in AR — large enough to inspect comfortably on HoloLens 2.
    if source_units == "mm":
        root_idx = len(gltf.nodes)
        gltf.nodes.append(pygltflib.Node(
            name="root",
            children=child_nodes,
            scale=[0.01, 0.01, 0.01],
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
    centroid_override: list[float] | None = None,
) -> tuple[int, list[float]]:
    """Add a single mesh with material to the glTF document. Returns node index.

    When vertex colors are present, they are baked into a baseColorTexture
    (PNG atlas) for universal compatibility — HoloLens 2 / MRTK / glTFast
    do not render the COLOR_0 vertex attribute.
    """
    mat = mesh_data.material
    has_vertex_colors = mesh_data.vertex_colors is not None

    # --- Bake vertex colors into a texture ---
    tex_info = None
    if has_vertex_colors:
        from med2glb.glb.vertex_color_bake import (
            bake_vertex_colors_to_texture,
            compute_texture_size,
        )

        tex_size = compute_texture_size(len(mesh_data.faces))
        vertices, faces, normals, uvs, png_bytes = bake_vertex_colors_to_texture(
            mesh_data.vertices, mesh_data.faces, mesh_data.vertex_colors,
            texture_size=tex_size, normals=mesh_data.normals,
        )

        # Embed PNG image into binary buffer
        img_offset = len(binary_data)
        binary_data.extend(png_bytes)
        _pad_to_4(binary_data)

        img_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=img_offset, byteLength=len(png_bytes),
        ))

        # Ensure images/textures/samplers lists exist
        if not hasattr(gltf, "images") or gltf.images is None:
            gltf.images = []
        if not hasattr(gltf, "textures") or gltf.textures is None:
            gltf.textures = []
        if not hasattr(gltf, "samplers") or gltf.samplers is None:
            gltf.samplers = []

        img_idx = len(gltf.images)
        gltf.images.append(pygltflib.Image(
            bufferView=img_bv_idx, mimeType="image/png",
        ))

        # One shared sampler (add only once)
        if len(gltf.samplers) == 0:
            gltf.samplers.append(pygltflib.Sampler(
                magFilter=pygltflib.LINEAR,
                minFilter=pygltflib.LINEAR,
                wrapS=pygltflib.CLAMP_TO_EDGE,
                wrapT=pygltflib.CLAMP_TO_EDGE,
            ))

        tex_idx = len(gltf.textures)
        gltf.textures.append(pygltflib.Texture(sampler=0, source=img_idx))

        tex_info = pygltflib.TextureInfo(index=tex_idx)

        # Alpha mode from vertex colors
        min_alpha = float(mesh_data.vertex_colors[:, 3].min())
        if min_alpha > 0.99:
            alpha_mode = pygltflib.OPAQUE
            alpha_cutoff = None
        elif min_alpha < 0.01:
            alpha_mode = pygltflib.MASK
            alpha_cutoff = 0.5
        else:
            alpha_mode = pygltflib.BLEND
            alpha_cutoff = None

        logger.info(
            "Texture-baked mesh %d: %d verts (xatlas from %d), %dx%d texture",
            index, len(vertices), len(mesh_data.vertices), tex_size, tex_size,
        )
    else:
        vertices = mesh_data.vertices.astype(np.float32)
        normals = mesh_data.normals.astype(np.float32) if mesh_data.normals is not None else None
        faces = mesh_data.faces.astype(np.uint32)
        uvs = None
        alpha_mode = pygltflib.BLEND if mat.alpha < 1.0 else pygltflib.OPAQUE
        alpha_cutoff = None

    # --- Center vertices at origin ---
    if centroid_override is not None:
        centroid = centroid_override
        vertices = (vertices - np.array(centroid, dtype=np.float32)).astype(np.float32)
    else:
        vertices, centroid = _center_vertices(vertices)

    # --- Create material ---
    material_idx = len(gltf.materials)
    pbr_kwargs: dict = dict(
        metallicFactor=mat.metallic,
        roughnessFactor=mat.roughness,
    )
    if tex_info is not None:
        pbr_kwargs["baseColorTexture"] = tex_info
        pbr_kwargs["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
    else:
        pbr_kwargs["baseColorFactor"] = [
            mat.base_color[0], mat.base_color[1], mat.base_color[2], mat.alpha,
        ]

    mat_kwargs: dict = dict(
        name=mat.name or mesh_data.structure_name,
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(**pbr_kwargs),
        alphaMode=alpha_mode,
        doubleSided=True,
    )
    if alpha_cutoff is not None:
        mat_kwargs["alphaCutoff"] = alpha_cutoff
    if mat.unlit:
        mat_kwargs["extensions"] = {"KHR_materials_unlit": {}}
    gltf.materials.append(pygltflib.Material(**mat_kwargs))
    if mat.unlit:
        if not hasattr(gltf, "extensionsUsed") or gltf.extensionsUsed is None:
            gltf.extensionsUsed = []
        if "KHR_materials_unlit" not in gltf.extensionsUsed:
            gltf.extensionsUsed.append("KHR_materials_unlit")

    # --- Write geometry ---
    # Positions
    pos_data = vertices.tobytes()
    pos_offset = len(binary_data)
    binary_data.extend(pos_data)
    _pad_to_4(binary_data)

    pos_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=pos_offset, byteLength=len(pos_data),
        target=pygltflib.ARRAY_BUFFER,
    ))

    pos_acc_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=pos_bv_idx,
        componentType=pygltflib.FLOAT,
        count=len(vertices),
        type=pygltflib.VEC3,
        max=vertices.max(axis=0).tolist(),
        min=vertices.min(axis=0).tolist(),
    ))

    # Normals
    normal_acc_idx = None
    if normals is not None:
        norm_data = normals.tobytes()
        norm_offset = len(binary_data)
        binary_data.extend(norm_data)
        _pad_to_4(binary_data)

        norm_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=norm_offset, byteLength=len(norm_data),
            target=pygltflib.ARRAY_BUFFER,
        ))

        normal_acc_idx = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=norm_bv_idx,
            componentType=pygltflib.FLOAT,
            count=len(normals),
            type=pygltflib.VEC3,
        ))

    # UVs (only when texture-baked)
    uv_acc_idx = None
    if uvs is not None:
        uv_data = uvs.tobytes()
        uv_offset = len(binary_data)
        binary_data.extend(uv_data)
        _pad_to_4(binary_data)

        uv_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=uv_offset, byteLength=len(uv_data),
            target=pygltflib.ARRAY_BUFFER,
        ))

        uv_acc_idx = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=uv_bv_idx,
            componentType=pygltflib.FLOAT,
            count=len(uvs),
            type=pygltflib.VEC2,
        ))

    # Indices
    idx_data = faces.tobytes()
    idx_offset = len(binary_data)
    binary_data.extend(idx_data)
    _pad_to_4(binary_data)

    idx_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=idx_offset, byteLength=len(idx_data),
        target=pygltflib.ELEMENT_ARRAY_BUFFER,
    ))

    idx_acc_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=idx_bv_idx,
        componentType=pygltflib.UNSIGNED_INT,
        count=faces.size,
        type=pygltflib.SCALAR,
        max=[int(faces.max())],
        min=[int(faces.min())],
    ))

    # --- Primitive ---
    attributes = pygltflib.Attributes(POSITION=pos_acc_idx)
    if normal_acc_idx is not None:
        attributes.NORMAL = normal_acc_idx
    if uv_acc_idx is not None:
        attributes.TEXCOORD_0 = uv_acc_idx

    primitive = pygltflib.Primitive(
        attributes=attributes,
        indices=idx_acc_idx,
        material=material_idx,
    )

    # Create mesh and node
    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(pygltflib.Mesh(
        name=mesh_data.structure_name,
        primitives=[primitive],
    ))

    node_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(
        name=mesh_data.structure_name,
        mesh=mesh_idx,
    ))

    return node_idx, centroid


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
