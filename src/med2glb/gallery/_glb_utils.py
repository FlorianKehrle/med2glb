"""Shared GLB construction helpers for gallery builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygltflib

from med2glb.core.types import GallerySlice
from med2glb.glb.builder import _pad_to_4, write_accessor
from med2glb.glb.texture import pixel_data_to_png


@dataclass
class QuadGeometry:
    """Accessor indices for shared quad geometry."""

    pos_acc: int
    norm_acc: int
    tc_acc: int
    idx_acc: int


def create_base_gltf() -> tuple[pygltflib.GLTF2, bytearray]:
    """Create an empty glTF document with one sampler."""
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        textures=[],
        images=[],
        samplers=[],
    )
    gltf.samplers.append(
        pygltflib.Sampler(
            magFilter=pygltflib.LINEAR,
            minFilter=pygltflib.LINEAR,
            wrapS=pygltflib.CLAMP_TO_EDGE,
            wrapT=pygltflib.CLAMP_TO_EDGE,
        )
    )
    return gltf, bytearray()


def quad_vertices_for_slice(sl: GallerySlice) -> np.ndarray:
    """Compute quad vertex positions from slice physical dimensions (in metres)."""
    row_sp, col_sp = sl.pixel_spacing
    width = sl.cols * col_sp / 1000.0
    height = sl.rows * row_sp / 1000.0
    hw, hh = width / 2, height / 2
    return np.array(
        [[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]],
        dtype=np.float32,
    )


def add_quad_geometry(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    vertices: np.ndarray,
) -> QuadGeometry:
    """Write shared quad buffers (positions, normals, texcoords, indices)."""
    texcoords = np.array(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32,
    )
    normals = np.array(
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

    pos_acc = write_accessor(
        gltf, binary_data, vertices,
        pygltflib.ARRAY_BUFFER, pygltflib.FLOAT, pygltflib.VEC3, True,
    )
    norm_acc = write_accessor(
        gltf, binary_data, normals,
        pygltflib.ARRAY_BUFFER, pygltflib.FLOAT, pygltflib.VEC3,
    )
    tc_acc = write_accessor(
        gltf, binary_data, texcoords,
        pygltflib.ARRAY_BUFFER, pygltflib.FLOAT, pygltflib.VEC2,
    )
    idx_acc = write_accessor(
        gltf, binary_data, indices,
        pygltflib.ELEMENT_ARRAY_BUFFER, pygltflib.UNSIGNED_SHORT, pygltflib.SCALAR, True,
    )
    return QuadGeometry(pos_acc=pos_acc, norm_acc=norm_acc, tc_acc=tc_acc, idx_acc=idx_acc)


def make_png_material(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    pixel_data: np.ndarray,
    name: str,
) -> int:
    """Encode pixel data as PNG texture and add material. Returns material index."""
    png_bytes = pixel_data_to_png(pixel_data)
    img_offset = len(binary_data)
    binary_data.extend(png_bytes)
    _pad_to_4(binary_data)

    bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(buffer=0, byteOffset=img_offset, byteLength=len(png_bytes))
    )

    img_idx = len(gltf.images)
    gltf.images.append(pygltflib.Image(bufferView=bv_idx, mimeType="image/png"))

    tex_idx = len(gltf.textures)
    gltf.textures.append(pygltflib.Texture(sampler=0, source=img_idx))

    mat_idx = len(gltf.materials)
    gltf.materials.append(
        pygltflib.Material(
            name=name,
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=tex_idx),
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            doubleSided=True,
        )
    )
    return mat_idx


def add_textured_quad_node(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    geom: QuadGeometry,
    material_idx: int,
    name: str,
    scale: list[float] | None = None,
    translation: list[float] | None = None,
    matrix: list[float] | None = None,
    parent_node_idx: int | None = None,
) -> int:
    """Add a mesh + node referencing shared quad geometry. Returns node index."""
    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(
        pygltflib.Mesh(
            name=name,
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(
                        POSITION=geom.pos_acc,
                        NORMAL=geom.norm_acc,
                        TEXCOORD_0=geom.tc_acc,
                    ),
                    indices=geom.idx_acc,
                    material=material_idx,
                )
            ],
        )
    )

    node_idx = len(gltf.nodes)
    node_kwargs: dict = dict(name=name, mesh=mesh_idx)
    if matrix is not None:
        node_kwargs["matrix"] = matrix
    else:
        if scale is not None:
            node_kwargs["scale"] = scale
        if translation is not None:
            node_kwargs["translation"] = translation
    gltf.nodes.append(pygltflib.Node(**node_kwargs))

    if parent_node_idx is not None:
        parent = gltf.nodes[parent_node_idx]
        if parent.children is None:
            parent.children = []
        parent.children.append(node_idx)
    else:
        gltf.scenes[0].nodes.append(node_idx)

    return node_idx


def add_parent_node(
    gltf: pygltflib.GLTF2,
    name: str,
    translation: list[float] | None = None,
    matrix: list[float] | None = None,
) -> int:
    """Add a non-mesh parent node (for grouping). Returns node index."""
    node_idx = len(gltf.nodes)
    kwargs: dict = dict(name=name)
    if matrix is not None:
        kwargs["matrix"] = matrix
    elif translation is not None:
        kwargs["translation"] = translation
    gltf.nodes.append(pygltflib.Node(**kwargs))
    gltf.scenes[0].nodes.append(node_idx)
    return node_idx


def add_scale_animation(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    node_indices: list[int],
    temporal_resolution: float | None = None,
) -> None:
    """Add STEP scale animation toggling visibility across *node_indices*."""
    n = len(node_indices)
    if n < 2:
        return

    dt = (temporal_resolution or 33.3) / 1000.0
    keyframe_times = np.array([i * dt for i in range(n)], dtype=np.float32)

    time_acc = write_accessor(
        gltf, binary_data, keyframe_times,
        None, pygltflib.FLOAT, pygltflib.SCALAR, True,
    )

    channels: list[pygltflib.AnimationChannel] = []
    samplers: list[pygltflib.AnimationSampler] = []

    for i, node_idx in enumerate(node_indices):
        scales = np.zeros((n, 3), dtype=np.float32)
        scales[i] = [1.0, 1.0, 1.0]

        s_acc = write_accessor(
            gltf, binary_data, scales,
            None, pygltflib.FLOAT, pygltflib.VEC3,
        )

        sampler_idx = len(samplers)
        samplers.append(
            pygltflib.AnimationSampler(
                input=time_acc,
                output=s_acc,
                interpolation=pygltflib.ANIM_STEP,
            )
        )
        channels.append(
            pygltflib.AnimationChannel(
                sampler=sampler_idx,
                target=pygltflib.AnimationChannelTarget(node=node_idx, path="scale"),
            )
        )

    if not hasattr(gltf, "animations") or gltf.animations is None:
        gltf.animations = []
    gltf.animations.append(
        pygltflib.Animation(name="gallery_cycle", channels=channels, samplers=samplers)
    )


def group_by_position(slices: list[GallerySlice]) -> dict[str, list[GallerySlice]]:
    """Group slices by spatial position (rounded to 0.1 mm)."""
    groups: dict[str, list[GallerySlice]] = {}
    for sl in slices:
        key = _position_key(sl)
        groups.setdefault(key, []).append(sl)
    return groups


def _position_key(sl: GallerySlice) -> str:
    """Create a hashable key from image position rounded to 0.1 mm."""
    if sl.image_position is not None:
        return ",".join(f"{v:.1f}" for v in sl.image_position)
    return f"inst_{sl.instance_number}"


def finalize_gltf(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    output_path: Path,
) -> None:
    """Set buffer size, binary blob, and save to disk."""
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))
