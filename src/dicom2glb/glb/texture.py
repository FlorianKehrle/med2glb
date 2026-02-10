"""Textured plane GLB: embed a DICOM image as a textured quad via pygltflib."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pygltflib

from dicom2glb.core.volume import DicomVolume
from dicom2glb.glb.builder import _pad_to_4


def build_textured_plane_glb(volume: DicomVolume, output_path: Path) -> None:
    """Build a GLB containing a textured quad from a single DICOM slice."""
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(name="image_plane", mesh=0)],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
        textures=[],
        images=[],
        samplers=[],
    )

    binary_data = bytearray()

    # Get 2D pixel data (first slice of the volume)
    pixel_data = volume.voxels[0]  # [Y, X]
    rows, cols = pixel_data.shape

    # Compute physical dimensions from pixel spacing
    row_spacing, col_spacing = volume.pixel_spacing
    width = cols * col_spacing / 1000.0  # Convert mm to meters for glTF
    height = rows * row_spacing / 1000.0

    # Create quad vertices (two triangles forming a rectangle)
    half_w = width / 2
    half_h = height / 2
    vertices = np.array(
        [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ],
        dtype=np.float32,
    )

    # UV coordinates
    texcoords = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Two triangles
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

    # Normals (all pointing +Z)
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # --- Encode image as PNG ---
    png_bytes = _pixel_data_to_png(pixel_data)

    # Write image data first
    img_offset = len(binary_data)
    binary_data.extend(png_bytes)
    _pad_to_4(binary_data)

    img_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=img_offset,
            byteLength=len(png_bytes),
        )
    )

    gltf.images.append(
        pygltflib.Image(bufferView=img_bv_idx, mimeType="image/png")
    )

    gltf.samplers.append(
        pygltflib.Sampler(
            magFilter=pygltflib.LINEAR,
            minFilter=pygltflib.LINEAR,
            wrapS=pygltflib.CLAMP_TO_EDGE,
            wrapT=pygltflib.CLAMP_TO_EDGE,
        )
    )

    gltf.textures.append(pygltflib.Texture(sampler=0, source=0))

    # Material with base color texture
    gltf.materials.append(
        pygltflib.Material(
            name="dicom_image",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=0),
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            doubleSided=True,
        )
    )

    # --- Write vertex data ---
    # Positions
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
            count=4,
            type=pygltflib.VEC3,
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist(),
        )
    )

    # Normals
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
    norm_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=norm_bv_idx,
            componentType=pygltflib.FLOAT,
            count=4,
            type=pygltflib.VEC3,
        )
    )

    # Texcoords
    tc_data = texcoords.tobytes()
    tc_offset = len(binary_data)
    binary_data.extend(tc_data)
    _pad_to_4(binary_data)

    tc_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=tc_offset,
            byteLength=len(tc_data),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    tc_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=tc_bv_idx,
            componentType=pygltflib.FLOAT,
            count=4,
            type=pygltflib.VEC2,
        )
    )

    # Indices
    idx_data = indices.tobytes()
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
            componentType=pygltflib.UNSIGNED_SHORT,
            count=6,
            type=pygltflib.SCALAR,
            max=[3],
            min=[0],
        )
    )

    # Mesh primitive
    gltf.meshes.append(
        pygltflib.Mesh(
            name="image_plane",
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(
                        POSITION=pos_acc_idx,
                        NORMAL=norm_acc_idx,
                        TEXCOORD_0=tc_acc_idx,
                    ),
                    indices=idx_acc_idx,
                    material=0,
                )
            ],
        )
    )

    # Buffer
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_data)))
    gltf.set_binary_blob(bytes(binary_data))

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save(str(output_path))


def _pixel_data_to_png(pixel_data: np.ndarray) -> bytes:
    """Convert 2D pixel array to PNG bytes."""
    from PIL import Image

    # Normalize to 0-255 range
    data = pixel_data.astype(np.float64)
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    else:
        data = np.zeros_like(data, dtype=np.uint8)

    img = Image.fromarray(data, mode="L")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
